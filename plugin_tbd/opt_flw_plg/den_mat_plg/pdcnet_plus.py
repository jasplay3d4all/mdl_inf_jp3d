import sys
sys.path.append('../optical_flow/DenseMatching/')
import numpy as np
import cv2
import einops
import torch
import os
from PIL import Image


from models.PDCNet.PDCNet import PDCNet_vgg16

from utils_data.image_transforms import ArrayToTensor
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.util_optical_flow import flow_to_image
from model_selection import load_network

def get_img_lst(in_fld_path):
    img_file_lst = os.listdir(in_fld_path)
    img_file_lst.sort()
    img_file_lst = [ x for x in img_file_lst if x.endswith(".png") ]

    return img_file_lst

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

def mix_propagated_ai_frame(raw_ai_frame, warped_propagated_ai_frame, mask, propagated_pixel_weight = 1.0) :
    if propagated_pixel_weight < 0.001 :
        return raw_ai_frame
    weights = np.zeros((raw_ai_frame.shape[0], raw_ai_frame.shape[1]), dtype = np.float32)
    weights[mask <= 127] = propagated_pixel_weight
    weights[mask > 127] = 1 - propagated_pixel_weight
    weights = weights[:, :, None]
    # TODO: employ poisson blending
    ai_frame = raw_ai_frame.astype(np.float32) * (1 - weights) + warped_propagated_ai_frame.astype(np.float32) * weights
    return np.clip(ai_frame, 0, 255).astype(np.uint8)


def warp_frame(frame, flow) :
    h, w = flow.shape[:2]
    disp_x, disp_y = flow[:, :, 0], flow[:, :, 1]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return frame
def generate_mask(cum_confidence: np.ndarray, log_confidence: np.ndarray, thres = 0.8) :
    mask = np.zeros((cum_confidence.shape[0], cum_confidence.shape[1]), dtype = np.uint8)
    mask[cum_confidence < thres] = 255
    log_confidence[cum_confidence < thres] = 0 # reset pixels to full confidence that will be inpainted
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kern), log_confidence

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

    
    def gen_msk_all_imgs(self, ip_fld, op_fld, conf_thres=0.8):
        os.makedirs(op_fld, exist_ok=True)
        img_lst = get_img_lst(ip_fld)

        for i, img_file in enumerate(img_lst):
            print("img ", img_file)
            ref_img = cv2.imread(os.path.join(ip_fld, img_lst[i+1])) #Image.open(img_pth)

            img_pth = os.path.join(ip_fld, img_file)

            tgt_img = cv2.imread(img_pth) #Image.open(img_pth)
            flow, confidence, v, log_confidence = self.of_calc(ref_img, tgt_img)
            warped_ai_frame = warp_frame(ref_img, flow)
            mask, log_confidence = generate_mask(confidence, log_confidence, thres = conf_thres)
            op_pth = os.path.join(op_fld, img_file)

            confidence_u8 = np.clip(confidence * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(f'{op_fld}/pixel_confidence_{i:06d}.png', confidence_u8)
            cv2.imwrite(f'{op_fld}/warped_ai_{i:06d}.png', warped_ai_frame)
            cv2.imwrite(f'{op_fld}/mask_{i:06d}.png', mask)

        return 
