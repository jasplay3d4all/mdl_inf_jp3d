import sys
sys.path.append('../../../optical_flow/DenseMatching')

def warp_frame_latent(latent, flow) :
    latent = einops.rearrange(latent.cpu().numpy().squeeze(0), 'c h w -> h w c')
    lh, lw = latent.shape[:2]
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    latent = cv2.resize(latent, (w, h), interpolation=cv2.INTER_CUBIC)
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_latent = cv2.remap(latent, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    remapped_latent = cv2.resize(remapped_latent, (lw, lh), interpolation=cv2.INTER_CUBIC)
    remapped_latent = torch.from_numpy(einops.rearrange(remapped_latent, 'h w c -> 1 c h w'))
    return remapped_latent

def confidence_to_mask(confidence, flow, dist, mask_aux) :
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = np.zeros((confidence.shape[0], confidence.shape[1]), dtype = np.uint8)
    mask[confidence < 0.9] = 255
    mask_aux.pixel_travel_dist = warp_frame_pdcnet(mask_aux.pixel_travel_dist, flow) + dist
    mask_aux.pixel_travel_dist[confidence < 0.9] = 0
    mask[mask_aux.pixel_travel_dist > mask_aux.thres] = 255
    mask_aux.pixel_travel_dist[mask_aux.pixel_travel_dist > mask_aux.thres] = 0
    mask = cv2.dilate(mask, kern)
    return mask


def warp_frame(frame, flow) :
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return frame

class PDCNetPlus() :
    def __init__(self, ckpt_path = 'pre_trained_models/PDCNet_plus_m.pth.tar') -> None:
        local_optim_iter = 14
        global_gocor_arguments = {'optim_iter': 6, 'steplength_reg': 0.1, 'train_label_map': False,
                                    'apply_query_loss': True,
                                    'reg_kernel_size': 3, 'reg_inter_dim': 16, 'reg_output_dim': 16}
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}
        network = PDCNet_vgg16(global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
                                    normalize='leakyrelu', same_local_corr_at_all_levels=True,
                                    local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
                                    local_decoder_type='OpticalFlowEstimatorResidualConnection',
                                    global_decoder_type='CMDTopResidualConnection',
                                    corr_for_corr_uncertainty_decoder='corr',
                                    give_layer_before_flow_to_uncertainty_decoder=True,
                                    var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0,
                                    make_two_feature_copies=True)
        network = load_network(network, checkpoint_path=ckpt_path).cuda()
        network.eval()
        self.network = network

    @torch.no_grad()
    def calc(self, frame1, frame2) :
        source_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w')
        target_img = einops.rearrange(torch.from_numpy(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), 'h w c -> 1 c h w')

        flow_est, uncertainty_est = self.network.estimate_flow_and_confidence_map(source_img, target_img)

        flow_est = flow_est.permute(0, 2, 3, 1)[0].cpu().numpy()
        confidence = uncertainty_est['weight_map'].softmax(dim=1).cpu().numpy()[0][0]
        log_confidence = uncertainty_est['weight_map'].log_softmax(dim=1).cpu().numpy()[0][0]
        return flow_est, confidence, log_confidence

    @torch.no_grad()
    def of_calc(self, frame1, frame2) :
        flow, confidence, log_confidence = self.calc(frame1, frame2)
        h, w = flow.shape[:2]
        disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
        X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                        np.linspace(0, h - 1, h))
        map_x = (X+disp_x).astype(np.float32)
        map_y = (Y+disp_y).astype(np.float32)
        map_x -= np.arange(w)
        map_y -= np.arange(h)[:,np.newaxis]
        v = np.sqrt(map_x*map_x+map_y*map_y)
        v[confidence < 0.9] = 0
        print('v.max()', v.max(), 'v.min()', v.min())
        return flow, confidence, v, log_confidence

    def generate_mask(self, cum_confidence: np.ndarray, log_confidence: np.ndarray, thres = 0.8) :
        mask = np.zeros((cum_confidence.shape[0], cum_confidence.shape[1]), dtype = np.uint8)
        mask[cum_confidence < thres] = 255
        log_confidence[cum_confidence < thres] = 0 # reset pixels to full confidence that will be inpainted
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.dilate(mask, kern), log_confidence