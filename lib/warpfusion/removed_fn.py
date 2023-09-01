




#@title Generate captions for keyframes
#@markdown Automatically generate captions for every n-th frame, \
#@markdown or keyframe list: at keyframe, at offset from keyframe, between keyframes.\
#@markdown keyframe source: Every n-th frame, user-input, Content-aware scheduling keyframes
inputFrames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
make_captions = False #@param {'type':'boolean'}
keyframe_source = 'Every n-th frame' #@param ['Content-aware scheduling keyframes', 'User-defined keyframe list', 'Every n-th frame']
#@markdown This option only works with  keyframe source == User-defined keyframe list
user_defined_keyframes = [3,4,5] #@param
#@markdown This option only works with  keyframe source == Content-aware scheduling keyframes
diff_thresh = 0.33 #@param {'type':'number'}
#@markdown This option only works with  keyframe source == Every n-th frame
nth_frame = 60 #@param {'type':'number'}
if keyframe_source == 'Content-aware scheduling keyframes': 
  if diff in [None, '', []]:
    print('ERROR: Keyframes were not generated. Please go back to Content-aware scheduling cell, enable analyze_video nad run it or choose a different caption keyframe source.')
    caption_keyframes = None
  else:  
    caption_keyframes = [1]+[i+1 for i,o in enumerate(diff) if o>=diff_thresh]
if keyframe_source == 'User-defined keyframe list':
  caption_keyframes = user_defined_keyframes
if keyframe_source == 'Every n-th frame':
  caption_keyframes = list(range(1, len(inputFrames), nth_frame))
#@markdown Remaps keyframes based on selected offset mode
offset_mode = 'Fixed' #@param ['Fixed', 'Between Keyframes', 'None']
#@markdown Only works with offset_mode == Fixed
fixed_offset = 0 #@param {'type':'number'}

videoFramesCaptions = videoFramesFolder+'Captions'
if make_captions and caption_keyframes is not None:
  try:
    blip_model
  except: 

    os.chdir('./BLIP')
    from models.blip import blip_decoder
    os.chdir('../')
    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'# -O /content/model_base_caption_capfilt_large.pth'

    blip_model = blip_decoder(pretrained=model_url, image_size=384, vit='base',med_config='./BLIP/configs/med_config.json')
    blip_model.eval()
    blip_model = blip_model.to(device) 
  finally:
    print('Using keyframes: ', caption_keyframes[:20], ' (first 20 keyframes displyed')
    if offset_mode == 'None':
      keyframes = caption_keyframes
    if offset_mode == 'Fixed':
      keyframes = caption_keyframes
      for i in range(len(caption_keyframes)):
        if keyframes[i] >= max(caption_keyframes):
          keyframes[i] = caption_keyframes[i]
        else: keyframes[i] = min(caption_keyframes[i]+fixed_offset, caption_keyframes[i+1])
      print('Remapped keyframes to ', keyframes[:20])
    if offset_mode == 'Between Keyframes':
      keyframes = caption_keyframes
      for i in range(len(caption_keyframes)):
        if keyframes[i] >= max(caption_keyframes):
          keyframes[i] = caption_keyframes[i]
        else: 
          keyframes[i] = caption_keyframes[i] + int((caption_keyframes[i+1]-caption_keyframes[i])/2)
      print('Remapped keyframes to ', keyframes[:20])
       
    videoFramesCaptions = videoFramesFolder+'Captions'
    createPath(videoFramesCaptions)

  
  from tqdm.notebook import trange

  for f in pathlib.Path(videoFramesCaptions).glob('*.txt'):
          f.unlink()
  for i in tqdm(keyframes):

    with torch.no_grad():
      keyFrameFilename = inputFrames[i-1]
      raw_image = Image.open(keyFrameFilename)
      image = transform(raw_image).unsqueeze(0).to(device) 
      caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5)
      captionFilename = os.path.join(videoFramesCaptions, keyFrameFilename.replace('\\','/').split('/')[-1][:-4]+'.txt')
      with open(captionFilename, 'w') as f:
        f.write(caption[0])

def load_caption(caption_file):
    caption = ''
    with open(caption_file, 'r') as f:
      caption = f.read()
    return caption

def get_caption(frame_num):
  caption_files = sorted(glob(os.path.join(videoFramesCaptions,'*.txt')))
  frame_num1 = frame_num+1
  if len(caption_files) == 0:
    return None
  frame_numbers = [int(o.replace('\\','/').split('/')[-1][:-4]) for o in caption_files]
  # print(frame_numbers, frame_num)
  if frame_num1 < frame_numbers[0]:
    return load_caption(caption_files[0])
  if frame_num1 >= frame_numbers[-1]:
    return load_caption(caption_files[-1])
  for i in range(len(frame_numbers)):
    if frame_num1 >= frame_numbers[i] and frame_num1 < frame_numbers[i+1]:
      return load_caption(caption_files[i])
  return None








from PIL import Image, ImageOps, ImageStat, ImageEnhance

def get_stats(image):
   stat = ImageStat.Stat(image)
   brightness = sum(stat.mean) / len(stat.mean)
   contrast = sum(stat.stddev) / len(stat.stddev)
   return brightness, contrast

#implemetation taken from https://github.com/lowfuel/progrockdiffusion

def adjust_brightness(image):

  brightness, contrast = get_stats(image)
  if brightness > high_brightness_threshold:
    print(" Brightness over threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(high_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image>high_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)
  if brightness < low_brightness_threshold:
    print(" Brightness below threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(low_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image<low_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)

  image = np.array(image)
  image = np.where(image>max_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = np.where(image<min_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = Image.fromarray(image)
  return image






# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
import PIL


def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif args.animation_mode == 'Video Input Legacy':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomPerspective(distortion_scale=0.4, p=0.7),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.15),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif  args.animation_mode == '2D' or args.animation_mode == 'Video Input':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.4),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
          ])
          

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)

                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts











def save_settings(skip_save=False):
  settings_out = batchFolder+f"/settings"
  os.makedirs(settings_out, exist_ok=True)
  setting_list = {
    'text_prompts': text_prompts,
    'user_comment':user_comment,
    'image_prompts': image_prompts,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'max_frames': max_frames,
    'interp_spline': interp_spline,
    'init_image': init_image,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'seed': seed,
    'width': width_height[0],
    'height': width_height[1],
    'diffusion_model': diffusion_model,
    'diffusion_steps': diffusion_steps,
    'max_frames': max_frames,
    'video_init_path':video_init_path,
    'extract_nth_frame':extract_nth_frame,
    'flow_video_init_path':flow_video_init_path,
    'flow_extract_nth_frame':flow_extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'turbo_mode':turbo_mode,
    'turbo_steps':turbo_steps,
    'turbo_preroll':turbo_preroll,
    'flow_warp':flow_warp,
    'check_consistency':check_consistency,
    'turbo_frame_skips_steps' : turbo_frame_skips_steps,
    'forward_weights_clip' : forward_weights_clip,
    'forward_weights_clip_turbo_step' : forward_weights_clip_turbo_step,
    'padding_ratio':padding_ratio,
    'padding_mode':padding_mode,
    'consistency_blur':consistency_blur,
    'inpaint_blend':inpaint_blend,
    'match_color_strength':match_color_strength,
    'high_brightness_threshold':high_brightness_threshold,
    'high_brightness_adjust_ratio':high_brightness_adjust_ratio,
    'low_brightness_threshold':low_brightness_threshold,
    'low_brightness_adjust_ratio':low_brightness_adjust_ratio,
    'stop_early': stop_early,
    'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
    'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
    'max_brightness_threshold':max_brightness_threshold,
    'min_brightness_threshold':min_brightness_threshold,
    'enable_adjust_brightness':enable_adjust_brightness,
    'dynamic_thresh':dynamic_thresh,
    'warp_interp':warp_interp,
    'fixed_code':fixed_code,
    'code_randomness':code_randomness,
    # 'normalize_code': normalize_code,
    'mask_result':mask_result,
    'reverse_cc_order':reverse_cc_order,
    'flow_lq':flow_lq,
    'use_predicted_noise':use_predicted_noise,
    'clip_guidance_scale':clip_guidance_scale,
    'clip_type':clip_type,
    'clip_pretrain':clip_pretrain,
    'missed_consistency_weight':missed_consistency_weight,
    'overshoot_consistency_weight':overshoot_consistency_weight,
    'edges_consistency_weight':edges_consistency_weight,
    'style_strength_schedule':style_strength_schedule_bkup,
    'flow_blend_schedule':flow_blend_schedule_bkup,
    'steps_schedule':steps_schedule_bkup,
    'init_scale_schedule':init_scale_schedule_bkup,
    'latent_scale_schedule':latent_scale_schedule_bkup,
    'latent_scale_template': latent_scale_template,
    'init_scale_template':init_scale_template,
    'steps_template':steps_template,
    'style_strength_template':style_strength_template,
    'flow_blend_template':flow_blend_template,
    'make_schedules':make_schedules,
    'normalize_latent':normalize_latent,
    'normalize_latent_offset':normalize_latent_offset,
    'colormatch_frame':colormatch_frame,
    'use_karras_noise':use_karras_noise,
    'end_karras_ramp_early':end_karras_ramp_early,
    'use_background_mask':use_background_mask,
    'apply_mask_after_warp':apply_mask_after_warp,
    'background':background,
    'background_source':background_source,
    'mask_source':mask_source,
    'extract_background_mask':extract_background_mask,
    'mask_video_path':mask_video_path,
    'negative_prompts':negative_prompts,
    'invert_mask':invert_mask,
    'warp_strength': warp_strength,
    'flow_override_map':flow_override_map,
    'cfg_scale_schedule':cfg_scale_schedule_bkup,
    'respect_sched':respect_sched,
    'color_match_frame_str':color_match_frame_str,
    'colormatch_offset':colormatch_offset,
    'latent_fixed_mean':latent_fixed_mean,
    'latent_fixed_std':latent_fixed_std,
    'colormatch_method':colormatch_method,
    'colormatch_regrain':colormatch_regrain,
    'warp_mode':warp_mode,
    'use_patchmatch_inpaiting':use_patchmatch_inpaiting,
    'blend_latent_to_init':blend_latent_to_init,
    'warp_towards_init':warp_towards_init,
    'init_grad':init_grad,
    'grad_denoised':grad_denoised,
    'colormatch_after':colormatch_after,
    'colormatch_turbo':colormatch_turbo,
    'model_version':model_version,
    'cond_image_src':cond_image_src,
    'warp_num_k':warp_num_k,
    'warp_forward':warp_forward,
    'sampler':sampler.__name__,
    'mask_clip':(mask_clip_low, mask_clip_high),
    'inpainting_mask_weight':inpainting_mask_weight , 
    'inverse_inpainting_mask':inverse_inpainting_mask,
    'mask_source':mask_source,
    'model_path':model_path,
    'diff_override':diff_override,
    'image_scale_schedule':image_scale_schedule_bkup,
    'image_scale_template':image_scale_template,
    'frame_range': frame_range,
    'detect_resolution' :detect_resolution, 
    'bg_threshold':bg_threshold, 
    'diffuse_inpaint_mask_blur':diffuse_inpaint_mask_blur, 
    'diffuse_inpaint_mask_thresh':diffuse_inpaint_mask_thresh,
    'add_noise_to_latent':add_noise_to_latent,
    'noise_upscale_ratio':noise_upscale_ratio,
    'fixed_seed':fixed_seed,
    'init_latent_fn':init_latent_fn.__name__,
    'value_threshold':value_threshold,
    'distance_threshold':distance_threshold,
    'masked_guidance':masked_guidance,
    'cc_masked_diffusion':cc_masked_diffusion,
    'alpha_masked_diffusion':alpha_masked_diffusion,
    'inverse_mask_order':inverse_mask_order,
    'invert_alpha_masked_diffusion':invert_alpha_masked_diffusion,
    'quantize':quantize,
    'cb_noise_upscale_ratio':cb_noise_upscale_ratio,  
    'cb_add_noise_to_latent':cb_add_noise_to_latent,
    'cb_use_start_code':cb_use_start_code,
    'cb_fixed_code':cb_fixed_code,
    'cb_norm_latent':cb_norm_latent,
    'guidance_use_start_code':guidance_use_start_code,
    'offload_model':offload_model,
    'controlnet_preprocess':controlnet_preprocess,
    'small_controlnet_model_path':small_controlnet_model_path,
    'use_scale':use_scale,
    'g_invert_mask':g_invert_mask,
    'controlnet_multimodel':json.dumps(controlnet_multimodel),
    'img_zero_uncond':img_zero_uncond,
    'do_softcap':do_softcap,
    'softcap_thresh':softcap_thresh,
    'softcap_q':softcap_q,
    'deflicker_latent_scale':deflicker_latent_scale,
    'deflicker_scale':deflicker_scale,
    'controlnet_multimodel_mode':controlnet_multimodel_mode,
    'no_half_vae':no_half_vae,
    'temporalnet_source':temporalnet_source,
    'temporalnet_skip_1st_frame':temporalnet_skip_1st_frame,
    'rec_randomness':rec_randomness,
    'rec_source':rec_source,
    'rec_cfg':rec_cfg,
    'rec_prompts':rec_prompts,
    'inpainting_mask_source':inpainting_mask_source,
    'rec_steps_pct':rec_steps_pct,
    'max_faces': max_faces,
    'num_flow_updates':num_flow_updates,
    'control_sd15_openpose_hands_face':control_sd15_openpose_hands_face,
    'control_sd15_depth_detector':control_sd15_depth_detector,
    'control_sd15_softedge_detector':control_sd15_softedge_detector,
    'control_sd15_seg_detector':control_sd15_seg_detector,
    'control_sd15_scribble_detector':control_sd15_scribble_detector,
    'control_sd15_lineart_coarse':control_sd15_lineart_coarse,
    'control_sd15_inpaint_mask_source':control_sd15_inpaint_mask_source,
    'control_sd15_shuffle_source':control_sd15_shuffle_source,
    'control_sd15_shuffle_1st_source':control_sd15_shuffle_1st_source,
    'overwrite_rec_noise':overwrite_rec_noise,
    'use_legacy_cc':use_legacy_cc,
    'missed_consistency_dilation':missed_consistency_dilation, 
    'edge_consistency_width':edge_consistency_width,
    'use_reference':use_reference,
    'reference_weight':reference_weight,
    'reference_source':reference_source,
    'reference_mode':reference_mode,
    'use_legacy_fixed_code':use_legacy_fixed_code,
    'consistency_dilate':consistency_dilate,
    'prompt_patterns_sched':prompt_patterns_sched
  }
  if not skip_save:
    try: 
      settings_fname = f"{settings_out}/{batch_name}({batchNum})_settings.txt"
      if os.path.exists(settings_fname):
        s_meta = os.path.getmtime(settings_fname) 
        os.rename(settings_fname,settings_fname[:-4]+str(s_meta)+'.txt' )
      with open(settings_fname, "w+") as f:   #save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
      print(e)
      print('Settings:', setting_list)
  return setting_list







#@title gui

#@markdown Load default settings
gui_difficulty_dict = {
    "I'm too young to die.":["flow_warp", "warp_strength","warp_mode","padding_mode","padding_ratio", 
      "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k","warp_forward", 
      "blend_json_schedules", "VERBOSE","offload_model", "do_softcap", "softcap_thresh", 
      "softcap_q", "user_comment","turbo_mode","turbo_steps", "colormatch_turbo",
      "turbo_frame_skips_steps","soften_consistency_mask_for_turbo_frames", "check_consistency", 
      "missed_consistency_weight","overshoot_consistency_weight", "edges_consistency_weight",
      "soften_consistency_mask","consistency_blur","match_color_strength","mask_result",
      "use_patchmatch_inpaiting","normalize_latent","normalize_latent_offset","latent_fixed_mean",
      "latent_fixed_std","latent_norm_4d","use_karras_noise", "cond_image_src", "inpainting_mask_source",
      "inverse_inpainting_mask", "inpainting_mask_weight", "init_grad", "grad_denoised", 
      "image_scale_schedule","blend_latent_to_init","dynamic_thresh","rec_cfg", "rec_source", 
      "rec_steps_pct", "controlnet_multimodel_mode",
      "overwrite_rec_noise", 
      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp"],
    "Hey, not too rough.":["flow_warp", "warp_strength","warp_mode", 
      "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k","warp_forward", 

      "check_consistency", 
      
      "use_patchmatch_inpaiting","init_grad", "grad_denoised", 
      "image_scale_schedule","blend_latent_to_init","rec_cfg", 
      
      "colormatch_after","sat_scale", "clamp_grad", "apply_mask_after_warp"],
    "Hurt me plenty.":"",
    "Ultra-Violence.":[]
}

gui_difficulty = "Hey, not too rough." #@param ["I'm too young to die.", "Hey, not too rough.", "Ultra-Violence."]
print(f'Using "{gui_difficulty}" gui difficulty. Please switch to another difficulty\nto unlock up to {len(gui_difficulty_dict[gui_difficulty])} more settings when you`re ready :D')
default_settings_path = '' #@param {'type':'string'}
load_default_settings = True #@param {'type':'boolean'}
#@markdown Disable to load settings into GUI from colab cells. You will need to re-run colab cells you've edited to apply changes, then re-run the gui cell.\
#@markdown Enable to keep GUI state.
keep_gui_state_on_cell_rerun = True #@param {'type':'boolean'}
settings_out = batchFolder+f"/settings"
from  ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, SelectionSlider, Valid

def desc_widget(widget, desc, width=80, h=True):
    if isinstance(widget, Checkbox): return widget
    if isinstance(width, str):
        if width.endswith('%') or width.endswith('px'):
            layout = Layout(width=width)
    else: layout = Layout(width=f'{width}') 

    text = Label(desc, layout = layout, tooltip = widget.tooltip, description_tooltip = widget.description_tooltip)
    return HBox([text, widget]) if h else VBox([text, widget])

class ControlNetControls(HBox):
    def __init__(self,  name, values, **kwargs):
        self.label  = HTML(
                description=name,
                description_tooltip=name,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px', width='200px'))
        
        self.enable = Checkbox(value=values['weight']>0,description='',indent=True, description_tooltip='Enable model.',
                               style={'description_width': '25px' },layout=Layout(width='70px', left='-25px'))
        self.weight = FloatText(value = values['weight'], description=' ', step=0.05, 
                                description_tooltip = 'Controlnet model weights. ', layout=Layout(width='100px', visibility= 'visible' if values['weight']>0 else 'hidden'),
                                style={'description_width': '25px' })
        self.start_end = FloatRangeSlider(
          value=[values['start'],values['end']],
          min=0,
          max=1,
          step=0.01,
          description=' ',
          description_tooltip='Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
          disabled=False,
          continuous_update=False,
          orientation='horizontal',
          readout=True,
          layout = Layout(width='300px', visibility= 'visible' if values['weight']>0 else 'hidden'),
          style={'description_width': '50px' }
        )
        
        self.enable.observe(self.on_change)
        self.weight.observe(self.on_change)

        super().__init__([self.enable, self.label, self.weight, self.start_end], layout = Layout(valign='center'))
    
    def on_change(self, change):
      # print(change)
      if change['name'] == 'value':

        if self.enable.value: 
              self.weight.disabled = False
              self.weight.layout.visibility = 'visible'
              if change['old'] == False and self.weight.value==0:
                self.weight.value = 1
              # if self.weight.value>0:
              self.start_end.disabled = False 
              self.label.disabled = False 
              self.start_end.layout.visibility = 'visible'
        else: 
              self.weight.disabled = True
              self.start_end.disabled = True 
              self.label.disabled = True 
              self.weight.layout.visibility = 'hidden'
              self.start_end.layout.visibility = 'hidden'

    def __getattr__(self, attr):
        if attr == 'value':
            weight = 0
            if self.weight.value>0 and self.enable.value: weight = self.weight.value
            (start,end) = self.start_end.value
            return {
                  "weight": weight,
                  "start":start,
                  "end":end
                }
        else:
            return super.__getattr__(attr)
    
class ControlGUI(VBox):
  def __init__(self, args):
    enable_label = HTML(
                    description='Enable',
                    description_tooltip='Enable',  style={'description_width': '50px' },
                    layout = Layout(width='40px', left='-50px', ))
    model_label = HTML(
                    description='Model name',
                    description_tooltip='Model name',  style={'description_width': '100px' },
                    layout = Layout(width='265px'))
    weight_label = HTML(
                    description='weight',
                    description_tooltip='Model weight. 0 weight effectively disables the model. The total sum of all the weights will be normalized to 1.',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='125px'))#65
    range_label = HTML(
                    description='active range (% or total steps)',
                    description_tooltip='Model`s active range. % of total steps when the model is active.\n Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='200px'))
    controls_list = [HBox([enable_label,model_label, weight_label, range_label ])]
    controls_dict = {}
    self.possible_controlnets = ['control_sd15_depth',
        'control_sd15_canny',
        'control_sd15_softedge',
        'control_sd15_mlsd',
        'control_sd15_normalbae',
        'control_sd15_openpose',
        'control_sd15_scribble',
        'control_sd15_seg',
        'control_sd15_temporalnet',
        'control_sd15_face',
        'control_sd15_ip2p',
        'control_sd15_inpaint',
        'control_sd15_lineart',
        'control_sd15_lineart_anime',
        'control_sd15_shuffle']
    for key in self.possible_controlnets:
      if key in args.keys():
        w = ControlNetControls(key, args[key])
      else: 
        w = ControlNetControls(key, {
            "weight":0,
            "start":0,
            "end":1
        })
      controls_list.append(w)
      controls_dict[key] = w
    
    self.args = args
    self.ws = controls_dict
    super().__init__(controls_list)

  def __getattr__(self, attr):
        if attr == 'value':
            res = {}
            for key in self.possible_controlnets:
              if self.ws[key].value['weight'] > 0:
                res[key] = self.ws[key].value
            return res
        else:
            return super.__getattr__(attr)

def set_visibility(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
          obj[key].layout.visibility = value


#try keep settings on occasional run cell 
if keep_gui_state_on_cell_rerun:
  try:

    latent_scale_schedule=eval(get_value('latent_scale_schedule',guis))
    init_scale_schedule=eval(get_value('init_scale_schedule',guis))
    steps_schedule=eval(get_value('steps_schedule',guis))
    style_strength_schedule=eval(get_value('style_strength_schedule',guis))
    cfg_scale_schedule=eval(get_value('cfg_scale_schedule',guis))
    flow_blend_schedule=eval(get_value('flow_blend_schedule',guis))
    image_scale_schedule=eval(get_value('image_scale_schedule',guis))

    user_comment= get_value('user_comment',guis)
    blend_json_schedules=get_value('blend_json_schedules',guis)
    VERBOSE=get_value('VERBOSE',guis)
    use_background_mask=get_value('use_background_mask',guis)
    invert_mask=get_value('invert_mask',guis)
    background=get_value('background',guis)
    background_source=get_value('background_source',guis)
    (mask_clip_low, mask_clip_high) = get_value('mask_clip',guis) 

    #turbo 
    turbo_mode=get_value('turbo_mode',guis)
    turbo_steps=get_value('turbo_steps',guis)
    colormatch_turbo=get_value('colormatch_turbo',guis)
    turbo_frame_skips_steps=get_value('turbo_frame_skips_steps',guis)
    soften_consistency_mask_for_turbo_frames=get_value('soften_consistency_mask_for_turbo_frames',guis)

    #warp
    flow_warp= get_value('flow_warp',guis)
    apply_mask_after_warp=get_value('apply_mask_after_warp',guis)
    warp_num_k=get_value('warp_num_k',guis)
    warp_forward=get_value('warp_forward',guis)
    warp_strength=get_value('warp_strength',guis)
    flow_override_map=eval(get_value('flow_override_map',guis))
    warp_mode=get_value('warp_mode',guis)
    warp_towards_init=get_value('warp_towards_init',guis)

    #cc
    check_consistency=get_value('check_consistency',guis)
    missed_consistency_weight=get_value('missed_consistency_weight',guis)
    overshoot_consistency_weight=get_value('overshoot_consistency_weight',guis)
    edges_consistency_weight=get_value('edges_consistency_weight',guis)
    consistency_blur=get_value('consistency_blur',guis)
    consistency_dilate=get_value('consistency_dilate',guis)
    padding_ratio=get_value('padding_ratio',guis)
    padding_mode=get_value('padding_mode',guis)
    match_color_strength=get_value('match_color_strength',guis)
    soften_consistency_mask=get_value('soften_consistency_mask',guis)
    mask_result=get_value('mask_result',guis)
    use_patchmatch_inpaiting=get_value('use_patchmatch_inpaiting',guis)

    #diffusion
    text_prompts=eval(get_value('text_prompts',guis))
    negative_prompts=eval(get_value('negative_prompts',guis))
    prompt_patterns_sched = eval(get_value('prompt_patterns_sched',guis))
    cond_image_src=get_value('cond_image_src',guis)
    set_seed=get_value('set_seed',guis)
    clamp_grad=get_value('clamp_grad',guis)
    clamp_max=get_value('clamp_max',guis)
    sat_scale=get_value('sat_scale',guis)
    init_grad=get_value('init_grad',guis)
    grad_denoised=get_value('grad_denoised',guis)
    blend_latent_to_init=get_value('blend_latent_to_init',guis)
    fixed_code=get_value('fixed_code',guis)
    code_randomness=get_value('code_randomness',guis)
    # normalize_code=get_value('normalize_code',guis)
    dynamic_thresh=get_value('dynamic_thresh',guis)
    sampler = get_value('sampler',guis)
    use_karras_noise = get_value('use_karras_noise',guis)
    inpainting_mask_weight = get_value('inpainting_mask_weight',guis)
    inverse_inpainting_mask = get_value('inverse_inpainting_mask',guis)
    inpainting_mask_source = get_value('mask_source',guis)

    #colormatch
    normalize_latent=get_value('normalize_latent',guis)
    normalize_latent_offset=get_value('normalize_latent_offset',guis)
    latent_fixed_mean=eval(str(get_value('latent_fixed_mean',guis)))
    latent_fixed_std=eval(str(get_value('latent_fixed_std',guis)))
    latent_norm_4d=get_value('latent_norm_4d',guis)
    colormatch_frame=get_value('colormatch_frame',guis)
    color_match_frame_str=get_value('color_match_frame_str',guis)
    colormatch_offset=get_value('colormatch_offset',guis)
    colormatch_method=get_value('colormatch_method',guis)
    colormatch_regrain=get_value('colormatch_regrain',guis)
    colormatch_after=get_value('colormatch_after',guis)
    image_prompts = {}

    fixed_seed = get_value('fixed_seed',guis)

    #rec noise
    rec_cfg = get_value('rec_cfg',guis)
    rec_steps_pct = get_value('rec_steps_pct',guis)
    rec_prompts = eval(get_value('rec_prompts',guis))
    rec_randomness = get_value('rec_randomness',guis)
    use_predicted_noise = get_value('use_predicted_noise',guis)
    overwrite_rec_noise  = get_value('overwrite_rec_noise',guis)

    #controlnet
    save_controlnet_annotations = get_value('save_controlnet_annotations',guis)
    control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face',guis)
    control_sd15_depth_detector  = get_value('control_sd15_depth_detector',guis)
    control_sd15_softedge_detector = get_value('control_sd15_softedge_detector',guis)
    control_sd15_seg_detector = get_value('control_sd15_seg_detector',guis)
    control_sd15_scribble_detector = get_value('control_sd15_scribble_detector',guis)
    control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse',guis)
    control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source',guis)
    control_sd15_shuffle_source = get_value('control_sd15_shuffle_source',guis)
    control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source',guis)
    controlnet_multimodel = get_value('controlnet_multimodel',guis)

    controlnet_preprocess = get_value('controlnet_preprocess',guis)
    detect_resolution  = get_value('detect_resolution',guis)
    bg_threshold = get_value('bg_threshold',guis)
    low_threshold = get_value('low_threshold',guis)
    high_threshold = get_value('high_threshold',guis)
    value_threshold = get_value('value_threshold',guis)
    distance_threshold = get_value('distance_threshold',guis)
    temporalnet_source = get_value('temporalnet_source',guis)
    temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame',guis)
    controlnet_multimodel_mode = get_value('controlnet_multimodel_mode',guis)
    max_faces = get_value('max_faces',guis)

    do_softcap = get_value('do_softcap',guis)
    softcap_thresh = get_value('softcap_thresh',guis)
    softcap_q = get_value('softcap_q',guis)

    masked_guidance = get_value('masked_guidance',guis)
    cc_masked_diffusion = get_value('cc_masked_diffusion',guis)
    alpha_masked_diffusion = get_value('alpha_masked_diffusion',guis)
    invert_alpha_masked_diffusion = get_value('invert_alpha_masked_diffusion',guis)
  except: 
    pass

gui_misc = {
    "user_comment": Textarea(value=user_comment,layout=Layout(width=f'80%'),  description = 'user_comment:',  description_tooltip = 'Enter a comment to differentiate between save files.'),
    "blend_json_schedules": Checkbox(value=blend_json_schedules, description='blend_json_schedules',indent=True, description_tooltip = 'Smooth values between keyframes.', tooltip = 'Smooth values between keyframes.'),
    "VERBOSE": Checkbox(value=VERBOSE,description='VERBOSE',indent=True, description_tooltip = 'Print all logs'),
    "offload_model": Checkbox(value=offload_model,description='offload_model',indent=True, description_tooltip = 'Offload unused models to CPU and back to GPU to save VRAM. May reduce speed.'),
    "do_softcap": Checkbox(value=do_softcap,description='do_softcap',indent=True, description_tooltip = 'Softly clamp latent excessive values. Reduces feedback loop effect a bit.'),
    "softcap_thresh":FloatSlider(value=softcap_thresh, min=0, max=1, step=0.05, description='softcap_thresh:', readout=True, readout_format='.1f', description_tooltip='Scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected'),
    "softcap_q":FloatSlider(value=softcap_q, min=0, max=1, step=0.05, description='softcap_q:', readout=True, readout_format='.1f', description_tooltip='Percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%'),

}

gui_mask = {
    "use_background_mask":Checkbox(value=use_background_mask,description='use_background_mask',indent=True, description_tooltip='Enable masking. In order to use it, you have to either extract or provide an existing mask in Video Masking cell.\n'),
    "invert_mask":Checkbox(value=invert_mask,description='invert_mask',indent=True, description_tooltip='Inverts the mask, allowing to process either backgroung or characters, depending on your mask.'),
    "background": Dropdown(description='background', 
                           options = ['image', 'color', 'init_video'], value = background, 
                           description_tooltip='Background type. Image - uses static image specified in background_source, color - uses fixed color specified in background_source, init_video - uses raw init video for masked areas.'), 
    "background_source": Text(value=background_source, description = 'background_source', description_tooltip='Specify image path or color name of hash.'),
    "apply_mask_after_warp": Checkbox(value=apply_mask_after_warp,description='apply_mask_after_warp',indent=True, description_tooltip='On to reduce ghosting. Apply mask after warping and blending warped image with current raw frame. If off, only current frame will be masked, previous frame will be warped and blended wuth masked current frame.'),
    "mask_clip" : IntRangeSlider(
      value=mask_clip,
      min=0,
      max=255,
      step=1,
      description='Mask clipping:',
      description_tooltip='Values below the selected range will be treated as black mask, values above - as white.',
      disabled=False,
      continuous_update=False,
      orientation='horizontal',
      readout=True)
    
}

gui_turbo = {
    "turbo_mode":Checkbox(value=turbo_mode,description='turbo_mode',indent=True, description_tooltip='Turbo mode skips diffusion process on turbo_steps number of frames. Frames are still being warped and blended. Speeds up the render at the cost of possible trails an ghosting.' ),
    "turbo_steps": IntText(value = turbo_steps, description='turbo_steps:', description_tooltip='Number of turbo frames'),
    "colormatch_turbo":Checkbox(value=colormatch_turbo,description='colormatch_turbo',indent=True, description_tooltip='Apply frame color matching during turbo frames. May increease rendering speed, but may add minor flickering.'),
    "turbo_frame_skips_steps" :  SelectionSlider(description='turbo_frame_skips_steps', 
                                                 options = ['70%','75%','80%','85%', '80%', '95%', '100% (don`t diffuse turbo frames, fastest)'], value = '100% (don`t diffuse turbo frames, fastest)', description_tooltip='Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.'),
    "soften_consistency_mask_for_turbo_frames": FloatSlider(value=soften_consistency_mask_for_turbo_frames, min=0, max=1, step=0.05, description='soften_consistency_mask_for_turbo_frames:', readout=True, readout_format='.1f', description_tooltip='Clips the consistency mask, reducing it`s effect'),
  
}

gui_warp = {
    "flow_warp":Checkbox(value=flow_warp,description='flow_warp',indent=True, description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
    
    "flow_blend_schedule" : Textarea(value=str(flow_blend_schedule),layout=Layout(width=f'80%'),  description = 'flow_blend_schedule:',  description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
    "warp_num_k": IntText(value = warp_num_k, description='warp_num_k:', description_tooltip='Nubmer of clusters in forward-warp mode. The more - the smoother is the motion. Lower values move larger chunks of image at a time.'),
    "warp_forward": Checkbox(value=warp_forward,description='warp_forward',indent=True,  description_tooltip='Experimental. Enable patch-based flow warping. Groups pixels by motion direction and moves them together, instead of moving individual pixels.'),
    # "warp_interp": Textarea(value='Image.LANCZOS',layout=Layout(width=f'80%'),  description = 'warp_interp:'),
    "warp_strength": FloatText(value = warp_strength, description='warp_strength:', description_tooltip='Experimental. Motion vector multiplier. Provides a glitchy effect.'),
    "flow_override_map":  Textarea(value=str(flow_override_map),layout=Layout(width=f'80%'),  description = 'flow_override_map:', description_tooltip='Experimental. Motion vector maps mixer. Allows changing frame-motion vetor indexes or repeating motion, provides a glitchy effect.'),
    "warp_mode": Dropdown(description='warp_mode', options = ['use_latent', 'use_image'],
                          value = warp_mode, description_tooltip='Experimental. Apply warp to latent vector. May get really blurry, but reduces feedback loop effect for slow movement'), 
    "warp_towards_init": Dropdown(description='warp_towards_init',
                                  options = ['stylized', 'off'] , value = warp_towards_init, description_tooltip='Experimental. After a frame is stylized, computes the difference between output and input for that frame, and warps the output back to input, preserving its shape.'),
    "padding_ratio": FloatSlider(value=padding_ratio, min=0, max=1, step=0.05, description='padding_ratio:', readout=True, readout_format='.1f', description_tooltip='Amount of padding. Padding is used to avoid black edges when the camera is moving out of the frame.'),
    "padding_mode": Dropdown(description='padding_mode', options = ['reflect','edge','wrap'],
                             value = padding_mode),
}

# warp_interp = Image.LANCZOS

gui_consistency = {
    "check_consistency":Checkbox(value=check_consistency,description='check_consistency',indent=True, description_tooltip='Enables consistency checking (CC). CC is used to avoid ghosting and trails, that appear due to lack of information while warping frames. It allows replacing motion edges, frame borders, incorrectly moved areas with raw init frame data.'),
    "missed_consistency_weight":FloatSlider(value=missed_consistency_weight, min=0, max=1, step=0.05, description='missed_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for incorrectly predicted\moved areas. For example, if an object moves and background appears behind it. We can predict what to put in that spot, so we can either duplicate the object, resulting in trail, or use init video data for that region.'),
    "overshoot_consistency_weight":FloatSlider(value=overshoot_consistency_weight, min=0, max=1, step=0.05, description='overshoot_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for areas that appeared out of the frame. We can either leave them black or use raw init video.'),
    "edges_consistency_weight":FloatSlider(value=edges_consistency_weight, min=0, max=1, step=0.05, description='edges_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for motion edges. Moving objects are most likely to leave trails, this option together with missed consistency weight helps prevent that, but in a more subtle manner.'),
    "soften_consistency_mask" :  FloatSlider(value=soften_consistency_mask, min=0, max=1, step=0.05, description='soften_consistency_mask:', readout=True, readout_format='.1f'),
    "consistency_blur": FloatText(value = consistency_blur, description='consistency_blur:'),
    "consistency_dilate": FloatText(value = consistency_dilate, description='consistency_dilate:', description_tooltip='expand consistency mask without blurring the edges'),
    "barely used": Label(' '),
    "match_color_strength" : FloatSlider(value=match_color_strength, min=0, max=1, step=0.05, description='match_color_strength:', readout=True, readout_format='.1f', description_tooltip='Enables colormathing raw init video pixls in inconsistent areas only to the stylized frame. May reduce flickering for inconsistent areas.'),
    "mask_result": Checkbox(value=mask_result,description='mask_result',indent=True, description_tooltip='Stylizes only inconsistent areas. Takes consistent areas from the previous frame.'),
    "use_patchmatch_inpaiting": FloatSlider(value=use_patchmatch_inpaiting, min=0, max=1, step=0.05, description='use_patchmatch_inpaiting:', readout=True, readout_format='.1f', description_tooltip='Uses patchmatch inapinting for inconsistent areas. Is slow.'),
}

gui_diffusion = {
    "use_karras_noise":Checkbox(value=use_karras_noise,description='use_karras_noise',indent=True, description_tooltip='Enable for samplers that have K at their name`s end.'),
    "sampler": Dropdown(description='sampler',options= [('sample_euler', sample_euler), 
                                  ('sample_euler_ancestral',sample_euler_ancestral), 
                                  ('sample_heun',sample_heun),
                                  ('sample_dpm_2', sample_dpm_2),
                                  ('sample_dpm_2_ancestral',sample_dpm_2_ancestral),
                                  ('sample_lms', sample_lms),
                                  ('sample_dpm_fast', sample_dpm_fast),
                                  ('sample_dpm_adaptive',sample_dpm_adaptive),
                                  ('sample_dpmpp_2s_ancestral', sample_dpmpp_2s_ancestral),
                                  ('sample_dpmpp_sde', sample_dpmpp_sde),
                                  ('sample_dpmpp_2m', sample_dpmpp_2m)], value = sampler),
    "prompt_patterns_sched": Textarea(value=str(prompt_patterns_sched),layout=Layout(width=f'80%'),  description = 'Replace patterns:'),
    "text_prompts" : Textarea(value=str(text_prompts),layout=Layout(width=f'80%'),  description = 'Prompt:'),
    "negative_prompts" :  Textarea(value=str(negative_prompts), layout=Layout(width=f'80%'), description = 'Negative Prompt:'),
    "cond_image_src":Dropdown(description='cond_image_src', options = ['init', 'stylized','cond_video'] , 
                            value = cond_image_src, description_tooltip='Depth map source for depth model. It can either take raw init video frame or previously stylized frame.'), 
    "inpainting_mask_source":Dropdown(description='inpainting_mask_source', options = ['none', 'consistency_mask', 'cond_video'] , 
                           value = inpainting_mask_source, description_tooltip='Inpainting model mask source. none - full white mask (inpaint whole image), consistency_mask - inpaint inconsistent areas only'),
    "inverse_inpainting_mask":Checkbox(value=inverse_inpainting_mask,description='inverse_inpainting_mask',indent=True, description_tooltip='Inverse inpainting mask'),
    "inpainting_mask_weight":FloatSlider(value=inpainting_mask_weight, min=0, max=1, step=0.05, description='inpainting_mask_weight:', readout=True, readout_format='.1f', 
                                         description_tooltip= 'Inpainting mask weight. 0 - Disables inpainting mask.'),
    "set_seed": IntText(value = set_seed, description='set_seed:', description_tooltip='Seed. Use -1 for random.'),
    "clamp_grad":Checkbox(value=clamp_grad,description='clamp_grad',indent=True, description_tooltip='Enable limiting the effect of external conditioning per diffusion step'),
    "clamp_max": FloatText(value = clamp_max, description='clamp_max:',description_tooltip='limit the effect of external conditioning per diffusion step'),
    "latent_scale_schedule":Textarea(value=str(latent_scale_schedule),layout=Layout(width=f'80%'),  description = 'latent_scale_schedule:', description_tooltip='Latents scale defines how much minimize difference between output and input stylized image in latent space.'),
    "init_scale_schedule": Textarea(value=str(init_scale_schedule),layout=Layout(width=f'80%'),  description = 'init_scale_schedule:', description_tooltip='Init scale defines how much minimize difference between output and input stylized image in RGB space.'),
    "sat_scale": FloatText(value = sat_scale, description='sat_scale:', description_tooltip='Saturation scale limits oversaturation.'),
    "init_grad": Checkbox(value=init_grad,description='init_grad',indent=True,  description_tooltip='On - compare output to real frame, Off - to stylized frame'),
    "grad_denoised" : Checkbox(value=grad_denoised,description='grad_denoised',indent=True, description_tooltip='Fastest, On by default, calculate gradients with respect to denoised image instead of input image per diffusion step.' ),
    "steps_schedule" : Textarea(value=str(steps_schedule),layout=Layout(width=f'80%'),  description = 'steps_schedule:', 
                               description_tooltip= 'Total diffusion steps schedule. Use list format like [50,70], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:50, 20:70} format to specify keyframes only.'),
    "style_strength_schedule" : Textarea(value=str(style_strength_schedule),layout=Layout(width=f'80%'),  description = 'style_strength_schedule:',
                                          description_tooltip= 'Diffusion (style) strength. Actual number of diffusion steps taken (at 50 steps with 0.3 or 30% style strength you get 15 steps, which also means 35 0r 70% skipped steps). Inverse of skep steps. Use list format like [0.5,0.35], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:0.5, 20:0.35} format to specify keyframes only.'),
    "cfg_scale_schedule": Textarea(value=str(cfg_scale_schedule),layout=Layout(width=f'80%'),  description = 'cfg_scale_schedule:', description_tooltip= 'Guidance towards text prompt. 7 is a good starting value, 1 is off (text prompt has no effect).'),
    "image_scale_schedule": Textarea(value=str(image_scale_schedule),layout=Layout(width=f'80%'),  description = 'image_scale_schedule:', description_tooltip= 'Only used with InstructPix2Pix Model. Guidance towards text prompt. 1.5 is a good starting value'),
    "blend_latent_to_init": FloatSlider(value=blend_latent_to_init, min=0, max=1, step=0.05, description='blend_latent_to_init:', readout=True, readout_format='.1f', description_tooltip = 'Blend latent vector with raw init'),
    # "use_karras_noise": Checkbox(value=False,description='use_karras_noise',indent=True),
    # "end_karras_ramp_early": Checkbox(value=False,description='end_karras_ramp_early',indent=True),
    "fixed_seed": Checkbox(value=fixed_seed,description='fixed_seed',indent=True, description_tooltip= 'Fixed seed.'),
    "fixed_code":  Checkbox(value=fixed_code,description='fixed_code',indent=True, description_tooltip= 'Fixed seed analog. Fixes diffusion noise.'),
    "code_randomness": FloatSlider(value=code_randomness, min=0, max=1, step=0.05, description='code_randomness:', readout=True, readout_format='.1f', description_tooltip= 'Fixed seed amount/effect strength.'),
    # "normalize_code":Checkbox(value=normalize_code,description='normalize_code',indent=True, description_tooltip= 'Whether to normalize the noise after adding fixed seed.'),
    "dynamic_thresh": FloatText(value = dynamic_thresh, description='dynamic_thresh:', description_tooltip= 'Limit diffusion model prediction output. Lower values may introduce clamping/feedback effect'),
    "use_predicted_noise":Checkbox(value=use_predicted_noise,description='use_predicted_noise',indent=True, description_tooltip='Reconstruct initial noise from init / stylized image.'),
    "rec_prompts" : Textarea(value=str(rec_prompts),layout=Layout(width=f'80%'),  description = 'Rec Prompt:'),
    "rec_randomness":   FloatSlider(value=rec_randomness, min=0, max=1, step=0.05, description='rec_randomness:', readout=True, readout_format='.1f', description_tooltip= 'Reconstructed noise randomness. 0 - reconstructed noise only. 1 - random noise.'),
    "rec_cfg": FloatText(value = rec_cfg, description='rec_cfg:', description_tooltip= 'CFG scale for noise reconstruction. 1-1.9 are the best values.'),
    "rec_source": Dropdown(description='rec_source', options = ['init', 'stylized'] , 
                            value = rec_source, description_tooltip='Source for noise reconstruction. Either raw init frame or stylized frame.'), 
    "rec_steps_pct":FloatSlider(value=rec_steps_pct, min=0, max=1, step=0.05, description='rec_steps_pct:', readout=True, readout_format='.2f', description_tooltip= 'Reconstructed noise steps in relation to total steps. 1 = 100% steps.'),
    "overwrite_rec_noise":Checkbox(value=overwrite_rec_noise,description='overwrite_rec_noise',indent=True, 
                               description_tooltip= 'Overwrite reconstructed noise cache. By default reconstructed noise is not calculated if the settings haven`t changed too much. You can eit prompt, neg prompt, cfg scale,  style strength, steps withot reconstructing the noise every time.'),

    "masked_guidance":Checkbox(value=masked_guidance,description='masked_guidance',indent=True, 
                               description_tooltip= 'Use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas.'),
    "cc_masked_diffusion": FloatSlider(value=cc_masked_diffusion, min=0, max=1, step=0.05, 
                                 description='cc_masked_diffusion:', readout=True, readout_format='.2f', description_tooltip= '0 - off. 0.5-0.7 are good values. Make inconsistent area passes only before this % of actual steps, then diffuse whole image.'),
    "alpha_masked_diffusion": FloatSlider(value=alpha_masked_diffusion, min=0, max=1, step=0.05, 
                                 description='alpha_masked_diffusion:', readout=True, readout_format='.2f', description_tooltip= '0 - off. 0.5-0.7 are good values. Make alpha masked area passes only before this % of actual steps, then diffuse whole image.'),
    "invert_alpha_masked_diffusion":Checkbox(value=invert_alpha_masked_diffusion,description='invert_alpha_masked_diffusion',indent=True, 
                               description_tooltip= 'invert alpha ask for masked diffusion'),

    
}
gui_colormatch = {
    "normalize_latent": Dropdown(description='normalize_latent',
                                 options = ['off', 'user_defined', 'color_video', 'color_video_offset',
    'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], value =normalize_latent ,description_tooltip= 'Normalize latent to prevent it from overflowing. User defined: use fixed input values (latent_fixed_*) Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset field below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset filed below).'),
    "normalize_latent_offset":IntText(value = normalize_latent_offset, description='normalize_latent_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "latent_fixed_mean": FloatText(value = latent_fixed_mean, description='latent_fixed_mean:', description_tooltip= 'User defined mean value for normalize_latent=user_Defined mode'),
    "latent_fixed_std": FloatText(value = latent_fixed_std, description='latent_fixed_std:', description_tooltip= 'User defined standard deviation value for normalize_latent=user_Defined mode'),
    "latent_norm_4d": Checkbox(value=latent_norm_4d,description='latent_norm_4d',indent=True, description_tooltip= 'Normalize on a per-channel basis (on by default)'),
    "colormatch_frame": Dropdown(description='colormatch_frame', options = ['off', 'stylized_frame', 'color_video', 'color_video_offset', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], 
                                 value = colormatch_frame,
                                 description_tooltip= 'Match frame colors to prevent it from overflowing.  Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset filed below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset field below).'),
    "color_match_frame_str": FloatText(value = color_match_frame_str, description='color_match_frame_str:', description_tooltip= 'Colormatching strength. 0 - no colormatching effect.'),
    "colormatch_offset":IntText(value =colormatch_offset, description='colormatch_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "colormatch_method": Dropdown(description='colormatch_method', options = ['LAB', 'PDF', 'mean'], value =colormatch_method ),
    # "colormatch_regrain": Checkbox(value=False,description='colormatch_regrain',indent=True),
    "colormatch_after":Checkbox(value=colormatch_after,description='colormatch_after',indent=True, description_tooltip= 'On - Colormatch output frames when saving to disk, may differ from the preview. Off - colormatch before stylizing.'),
    
}

gui_controlnet = {
    "controlnet_preprocess": Checkbox(value=controlnet_preprocess,description='controlnet_preprocess',indent=True, 
                                      description_tooltip= 'preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing.'),
    "detect_resolution":IntText(value = detect_resolution, description='detect_resolution:', description_tooltip= 'Control net conditioning image resolution. The size of the image passed into controlnet preprocessors. Suggest keeping this as high as you can fit into your VRAM for more details.'),
    "bg_threshold":FloatText(value = bg_threshold, description='bg_threshold:', description_tooltip='Control net depth/normal bg cutoff threshold'),
    "low_threshold":IntText(value = low_threshold, description='low_threshold:', description_tooltip= 'Control net canny filter parameters'), 
    "high_threshold":IntText(value = high_threshold, description='high_threshold:', description_tooltip= 'Control net canny filter parameters'),
    "value_threshold":FloatText(value = value_threshold, description='value_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "distance_threshold":FloatText(value = distance_threshold, description='distance_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "temporalnet_source":Dropdown(description ='temporalnet_source', options = ['init', 'stylized'] , 
                            value = temporalnet_source, description_tooltip='Temporalnet guidance source. Previous init or previous stylized frame'),
    "temporalnet_skip_1st_frame": Checkbox(value = temporalnet_skip_1st_frame,description='temporalnet_skip_1st_frame',indent=True, 
                                      description_tooltip='Skip temporalnet for 1st frame (if not skipped, will use raw init for guidance'),
    "controlnet_multimodel_mode":Dropdown(description='controlnet_multimodel_mode', options = ['internal','external'], value =controlnet_multimodel_mode, description_tooltip='internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.' ), 
    "max_faces":IntText(value = max_faces, description='max_faces:', description_tooltip= 'Max faces to detect. Control net face parameters'),
    "save_controlnet_annotations": Checkbox(value = save_controlnet_annotations,description='save_controlnet_annotations',indent=True, 
                                      description_tooltip='Save controlnet annotator predictions. They will be saved to your project dir /controlnetDebug folder.'),
    "control_sd15_openpose_hands_face":Checkbox(value = control_sd15_openpose_hands_face,description='control_sd15_openpose_hands_face',indent=True, 
                                      description_tooltip='Enable full openpose mode with hands and facial features.'),
    "control_sd15_depth_detector" :Dropdown(description='control_sd15_depth_detector', options = ['Zoe','Midas'], value =control_sd15_depth_detector, 
                                            description_tooltip='Depth annotator model.' ), 
    "control_sd15_softedge_detector":Dropdown(description='control_sd15_softedge_detector', options = ['HED','PIDI'], value =control_sd15_softedge_detector, 
                                            description_tooltip='Softedge annotator model.' ), 
    "control_sd15_seg_detector":Dropdown(description='control_sd15_seg_detector', options = ['Seg_OFCOCO', 'Seg_OFADE20K', 'Seg_UFADE20K'], value =control_sd15_seg_detector, 
                                            description_tooltip='Segmentation annotator model.' ),
    "control_sd15_scribble_detector":Dropdown(description='control_sd15_scribble_detector', options = ['HED','PIDI'], value =control_sd15_scribble_detector, 
                                            description_tooltip='Sccribble annotator model.' ), 
    "control_sd15_lineart_coarse":Checkbox(value = control_sd15_lineart_coarse,description='control_sd15_lineart_coarse',indent=True, 
                                      description_tooltip='Coarse strokes mode.'),
    "control_sd15_inpaint_mask_source":Dropdown(description='control_sd15_inpaint_mask_source', options = ['consistency_mask', 'None', 'cond_video'], value =control_sd15_inpaint_mask_source, 
                                            description_tooltip='Inpainting controlnet mask source. consistency_mask - inpaints inconsistent areas, None - whole image, cond_video - loads external mask' ), 
    "control_sd15_shuffle_source":Dropdown(description='control_sd15_shuffle_source', options = ['color_video', 'init', 'prev_frame', 'first_frame'], value =control_sd15_shuffle_source, 
                                            description_tooltip='Shuffle controlnet source. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (stylized+warped with consistency mask and flow_blend opacity), prev_frame - uses previously stylized frame (stylized, not warped), first_frame - first stylized frame' ), 
    "control_sd15_shuffle_1st_source":Dropdown(description='control_sd15_shuffle_1st_source', options = ['color_video', 'init', 'None'], value =control_sd15_shuffle_1st_source, 
                                            description_tooltip='Set 1st frame source for shuffle model. If you need to geet the 1st frame style from your image, and for the consecutive frames you want to use the resulting stylized images. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (raw video frame), None - skips this controlnet for the 1st frame. For example, if you like the 1st frame you`re getting and want to keep its style, but don`t want to use an external image as a source.'),
    "controlnet_multimodel":ControlGUI(controlnet_multimodel)               
    
}

colormatch_regrain = False

guis = [gui_diffusion, gui_controlnet, gui_warp, gui_consistency, gui_turbo, gui_mask, gui_colormatch, gui_misc]

for key in gui_difficulty_dict[gui_difficulty]:
  for gui in guis:
    set_visibility(key, 'hidden', gui)

class FilePath(HBox):
    def __init__(self,  **kwargs):
        self.model_path = Text(value='',  continuous_update = True,**kwargs)
        self.path_checker = Valid(
        value=False, layout=Layout(width='2000px')
        )
        
        self.model_path.observe(self.on_change)
        super().__init__([self.model_path, self.path_checker])
    
    def __getattr__(self, attr):
        if attr == 'value':
            return self.model_path.value
        else:
            return super.__getattr__(attr)
    
    def on_change(self, change):
        if change['name'] == 'value':
            if os.path.exists(change['new']):
                self.path_checker.value = True
                self.path_checker.description = ''
            else: 
                self.path_checker.value = False
                self.path_checker.description = 'The file does not exist. Please specify the correct path.'

def add_labels_dict(gui):
    style = {'description_width': '250px' }
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        gui[key].style = style
        # temp = gui[key]
        # temp.observe(dump_gui())
        # gui[key] = temp
        if isinstance(gui[key], ControlGUI): 
          continue
        if not isinstance(gui[key], Textarea) and not isinstance( gui[key],Checkbox ):
            # vis = gui[key].layout.visibility
            # gui[key].layout = layout
            gui[key].layout.width = '500px'
        if isinstance( gui[key],Checkbox ):
            html_label = HTML(
                description=gui[key].description,
                description_tooltip=gui[key].description_tooltip,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px'))
            gui_labels[key] = HBox([gui[key],html_label])
            gui_labels[key].layout.visibility = gui[key].layout.visibility
            gui[key].description = ''
            # gui_labels[key] = gui[key]

        else:

            gui_labels[key] = gui[key]
            # gui_labels[key].layout.visibility = gui[key].layout.visibility
        # gui_labels[key].observe(print('smth changed', time.time()))
 
    return gui_labels


gui_diffusion_label, gui_controlnet_label, gui_warp_label, gui_consistency_label, gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_label = [add_labels_dict(o) for o in guis]

cond_keys = ['latent_scale_schedule','init_scale_schedule','clamp_grad','clamp_max','init_grad','grad_denoised','masked_guidance' ]
conditioning_w = Accordion([VBox([gui_diffusion_label[o] for o in cond_keys])])
conditioning_w.set_title(0, 'External Conditioning...')

seed_keys = ['set_seed', 'fixed_seed', 'fixed_code', 'code_randomness']
seed_w = Accordion([VBox([gui_diffusion_label[o] for o in seed_keys])])
seed_w.set_title(0, 'Seed...')

rec_keys = ['use_predicted_noise','rec_prompts','rec_cfg','rec_randomness', 'rec_source', 'rec_steps_pct', 'overwrite_rec_noise']
rec_w = Accordion([VBox([gui_diffusion_label[o] for o in rec_keys])])
rec_w.set_title(0, 'Reconstructed noise...')

prompt_keys = ['text_prompts', 'negative_prompts', 'prompt_patterns_sched',
'steps_schedule', 'style_strength_schedule', 
'cfg_scale_schedule', 'blend_latent_to_init', 'dynamic_thresh',  
'cond_image_src', 'cc_masked_diffusion', 'alpha_masked_diffusion', 'invert_alpha_masked_diffusion']
if model_version == 'v1_instructpix2pix':
  prompt_keys.append('image_scale_schedule')
if  model_version == 'v1_inpainting':
  prompt_keys+=['inpainting_mask_source', 'inverse_inpainting_mask', 'inpainting_mask_weight']
prompt_keys = [o for o in prompt_keys if o not in seed_keys+cond_keys]
prompt_w = [gui_diffusion_label[o] for o in prompt_keys]

gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'], 
gui_diffusion_label['use_karras_noise'], conditioning_w, seed_w, rec_w]

control_annotator_keys = ['controlnet_preprocess','save_controlnet_annotations','detect_resolution','bg_threshold','low_threshold','high_threshold','value_threshold',
                          'distance_threshold', 'max_faces', 'control_sd15_openpose_hands_face','control_sd15_depth_detector' ,'control_sd15_softedge_detector',
'control_sd15_seg_detector','control_sd15_scribble_detector','control_sd15_lineart_coarse','control_sd15_inpaint_mask_source',
'control_sd15_shuffle_source','control_sd15_shuffle_1st_source', 'temporalnet_source', 'temporalnet_skip_1st_frame',]
control_annotator_w = Accordion([VBox([gui_controlnet_label[o] for o in control_annotator_keys])])
control_annotator_w.set_title(0, 'Controlnet annotator settings...')
controlnet_model_w = Accordion([gui_controlnet['controlnet_multimodel']])
controlnet_model_w.set_title(0, 'Controlnet models settings...')
control_keys = [ 'controlnet_multimodel_mode']
control_w = [gui_controlnet_label[o] for o in control_keys]
gui_control_list = [controlnet_model_w, control_annotator_w, *control_w]

#misc
misc_keys = ["user_comment","blend_json_schedules","VERBOSE","offload_model"]
misc_w = [gui_misc_label[o] for o in misc_keys]

softcap_keys = ['do_softcap','softcap_thresh','softcap_q']
softcap_w = Accordion([VBox([gui_misc_label[o] for o in softcap_keys])])
softcap_w.set_title(0, 'Softcap settings...')

load_settings_btn = Button(description='Load settings')
def btn_eventhandler(obj):
  load_settings(load_settings_path.value)
load_settings_btn.on_click(btn_eventhandler)
load_settings_path = FilePath(placeholder='Please specify the path to the settings file to load.', description_tooltip='Please specify the path to the settings file to load.')
settings_w = Accordion([VBox([load_settings_path, load_settings_btn])])
settings_w.set_title(0, 'Load settings...')
gui_misc_list = [*misc_w, softcap_w, settings_w]

guis_labels_source = [gui_diffusion_list]
guis_titles_source = ['diffusion']
if 'control' in model_version:
  guis_labels_source += [gui_control_list]
  guis_titles_source += ['controlnet']
  
guis_labels_source += [gui_warp_label, gui_consistency_label, 
gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_list]
guis_titles_source += ['warp', 'consistency', 'turbo', 'mask', 'colormatch', 'misc']

guis_labels = [VBox([*o.values()]) if isinstance(o, dict) else VBox(o) for o in guis_labels_source]

app = Tab(guis_labels)
for i,title in enumerate(guis_titles_source):
    app.set_title(i, title)

def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key].value
        else: 
            for o in obj.keys():
                res = get_value(key, obj[o])
                if res is not None: return res
    if isinstance(obj, list):
        for o in obj:
            res = get_value(key, o)
            if res is not None: return res
    return None

def set_value(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].value = value
        else: 
            for o in obj.keys():
                set_value(key, value, obj[o])
                 
    if isinstance(obj, list):
        for o in obj:
            set_value(key, value, o)














#@title Do the Run!
#@markdown Preview max size
from glob import glob
controlnet_multimodel = get_value('controlnet_multimodel',guis)
image_prompts = {}
controlnet_multimodel_temp = {}
for key in controlnet_multimodel.keys():
  weight = controlnet_multimodel[key]["weight"]
  if weight !=0 :
    controlnet_multimodel_temp[key] = controlnet_multimodel[key]
controlnet_multimodel = controlnet_multimodel_temp
 
inverse_mask_order = False 
can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(getattr(torch.nn.functional, "scaled_dot_product_attention")) # not everyone has torch 2.x to use sdp
if can_use_sdp:
  shared.opts.xformers = False 
  shared.cmd_opts.xformers = False 

import copy
apply_depth = None;
apply_canny = None; apply_mlsd = None; 
apply_hed = None; apply_openpose = None;
apply_seg = None;
loaded_controlnets = {}
torch.cuda.empty_cache(); gc.collect(); 
sd_model.control_scales = ([1]*13)
if model_version == 'control_multi':
  sd_model.control_model.cpu()
  print('Checking downloaded Annotator and ControlNet Models')
  for controlnet in controlnet_multimodel.keys():
    controlnet_settings = controlnet_multimodel[controlnet]
    weight = controlnet_settings["weight"]
    if weight!=0:
      small_url = control_model_urls[controlnet]
      local_filename = small_url.split('/')[-1]
      small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
      if use_small_controlnet and os.path.exists(model_path) and not os.path.exists(small_controlnet_model_path):
        print(f'Model found at {model_path}. Small model not found at {small_controlnet_model_path}.')
        if not os.path.exists(small_controlnet_model_path) or force_download:
          try:
            pathlib.Path(small_controlnet_model_path).unlink()
          except: pass
          print(f'Downloading small {controlnet} model... ')
          wget.download(small_url,  small_controlnet_model_path)
          print(f'Downloaded small {controlnet} model.')


      # helper_names = control_helpers[controlnet]
      # if helper_names is not None:
      #     if type(helper_names) == str: helper_names = [helper_names]
      #     for helper_name in helper_names:
      #       helper_model_url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/'+helper_name
      #       helper_model_path = f'{root_dir}/ControlNet/annotator/ckpts/'+helper_name
      #       if not os.path.exists(helper_model_path) or force_download:
      #         try:
      #           pathlib.Path(helper_model_path).unlink()
      #         except: pass
      #         wget.download(helper_model_url, helper_model_path)

  print('Loading ControlNet Models')
  loaded_controlnets = {}
  for controlnet in controlnet_multimodel.keys():
    controlnet_settings = controlnet_multimodel[controlnet]
    weight = controlnet_settings["weight"]
    if weight!=0:
      loaded_controlnets[controlnet] = copy.deepcopy(sd_model.control_model)
      small_url = control_model_urls[controlnet]
      local_filename = small_url.split('/')[-1]
      small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
      if os.path.exists(small_controlnet_model_path):
          ckpt = small_controlnet_model_path
          print(f"Loading model from {ckpt}")
          if ckpt.endswith('.safetensors'):
            pl_sd = {}
            with safe_open(ckpt, framework="pt", device=load_to) as f:
              for key in f.keys():
                  pl_sd[key] = f.get_tensor(key)
          else: pl_sd = torch.load(ckpt, map_location=load_to)

          if "global_step" in pl_sd:
              print(f"Global Step: {pl_sd['global_step']}")
          if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
          else: sd = pl_sd
          if "control_model.input_blocks.0.0.bias" in sd:


            sd = dict([(o.split('control_model.')[-1],sd[o]) for o in sd.keys() if o != 'difference'])

            # print('control_model in sd')
          del pl_sd

          gc.collect()
          m, u = loaded_controlnets[controlnet].load_state_dict(sd, strict=True)
          loaded_controlnets[controlnet].half()
          if len(m) > 0 and verbose:
              print("missing keys:")
              print(m, len(m))
          if len(u) > 0 and verbose:
              print("unexpected keys:")
              print(u, len(u))
      else: 
        print('Small controlnet model not found in path but specified in settings. Please adjust settings or check controlnet path.')
        sys.exit(0)


# print('Loading annotators.')
controlnet_keys = controlnet_multimodel.keys() if model_version == 'control_multi' else model_version
if "control_sd15_depth" in controlnet_keys or "control_sd15_normal" in controlnet_keys:
        if control_sd15_depth_detector == 'Midas' or "control_sd15_normal" in controlnet_keys:
          from annotator.midas import MidasDetector
          apply_depth = MidasDetector()
          print('Loaded MidasDetector')
        if control_sd15_depth_detector == 'Zoe':
          from annotator.zoe import ZoeDetector
          apply_depth = ZoeDetector()
          print('Loaded ZoeDetector')

if "control_sd15_normalbae" in controlnet_keys:
        from annotator.normalbae import NormalBaeDetector
        apply_normal = NormalBaeDetector()
        print('Loaded NormalBaeDetector')
if 'control_sd15_canny' in controlnet_keys :
        from annotator.canny import CannyDetector
        apply_canny = CannyDetector()
        print('Loaded CannyDetector')
if 'control_sd15_softedge' in controlnet_keys:
        if control_sd15_softedge_detector == 'HED':
          from annotator.hed import HEDdetector
          apply_softedge = HEDdetector()
          print('Loaded HEDdetector')
        if control_sd15_softedge_detector == 'PIDI':
          from annotator.pidinet import PidiNetDetector
          apply_softedge = PidiNetDetector()
          print('Loaded PidiNetDetector')
if 'control_sd15_scribble' in controlnet_keys:
        from annotator.util import nms
        if control_sd15_scribble_detector == 'HED':
          from annotator.hed import HEDdetector
          apply_scribble = HEDdetector()
          print('Loaded HEDdetector')
        if control_sd15_scribble_detector == 'PIDI':
          from annotator.pidinet import PidiNetDetector
          apply_scribble = PidiNetDetector()
          print('Loaded PidiNetDetector')

if "control_sd15_mlsd" in controlnet_keys:
        from annotator.mlsd import MLSDdetector
        apply_mlsd = MLSDdetector()
        print('Loaded MLSDdetector')
if "control_sd15_openpose" in controlnet_keys:
        from annotator.openpose import OpenposeDetector
        apply_openpose = OpenposeDetector()
        print('Loaded OpenposeDetector')
if "control_sd15_seg" in controlnet_keys:
        if control_sd15_seg_detector == 'Seg_OFCOCO':
          from annotator.oneformer import OneformerCOCODetector
          apply_seg = OneformerCOCODetector()
          print('Loaded OneformerCOCODetector')
        elif control_sd15_seg_detector == 'Seg_OFADE20K':
          from annotator.oneformer import OneformerADE20kDetector
          apply_seg = OneformerADE20kDetector()
          print('Loaded OneformerADE20kDetector')
        elif control_sd15_seg_detector == 'Seg_UFADE20K':
          from annotator.uniformer import UniformerDetector
          apply_seg = UniformerDetector()
          print('Loaded UniformerDetector')
if "control_sd15_shuffle" in controlnet_keys:
        from annotator.shuffle import ContentShuffleDetector
        apply_shuffle = ContentShuffleDetector()
        print('Loaded ContentShuffleDetector')

# if "control_sd15_ip2p" in controlnet_keys:
#   #no annotator
#   pass 
# if "control_sd15_inpaint" in controlnet_keys:
#   #no annotator
#   pass
if "control_sd15_lineart" in controlnet_keys:
  from annotator.lineart import LineartDetector
  apply_lineart = LineartDetector()
  print('Loaded LineartDetector')
if "control_sd15_lineart_anime" in controlnet_keys:
  from annotator.lineart_anime import LineartAnimeDetector
  apply_lineart_anime = LineartAnimeDetector()
  print('Loaded LineartAnimeDetector')

def deflicker_loss(processed2, processed1, raw1, raw2, criterion1, criterion2):
  raw_diff = criterion1(raw2, raw1)
  proc_diff = criterion1(processed1, processed2)
  return criterion2(raw_diff, proc_diff)


unload()
sd_model.cuda()
sd_hijack.model_hijack.hijack(sd_model)
sd_hijack.model_hijack.embedding_db.add_embedding_dir(custom_embed_dir)
sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(sd_model, force_reload=True)


latent_scale_schedule=eval(get_value('latent_scale_schedule',guis))
init_scale_schedule=eval(get_value('init_scale_schedule',guis))
steps_schedule=eval(get_value('steps_schedule',guis))
style_strength_schedule=eval(get_value('style_strength_schedule',guis))
cfg_scale_schedule=eval(get_value('cfg_scale_schedule',guis))
flow_blend_schedule=eval(get_value('flow_blend_schedule',guis))
image_scale_schedule=eval(get_value('image_scale_schedule',guis))

latent_scale_schedule_bkup = copy.copy(latent_scale_schedule)
init_scale_schedule_bkup = copy.copy(init_scale_schedule)
steps_schedule_bkup = copy.copy(steps_schedule)
style_strength_schedule_bkup = copy.copy(style_strength_schedule)
flow_blend_schedule_bkup = copy.copy(flow_blend_schedule)
cfg_scale_schedule_bkup = copy.copy(cfg_scale_schedule)
image_scale_schedule_bkup = copy.copy(image_scale_schedule)

if make_schedules:
  if diff is None and diff_override == []: sys.exit(f'\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous cell, run it, and then run this cell again\n')
  if diff_override != []: diff = diff_override

  print('Applied schedules:')
  latent_scale_schedule = check_and_adjust_sched(latent_scale_schedule, latent_scale_template, diff, respect_sched)
  init_scale_schedule = check_and_adjust_sched(init_scale_schedule, init_scale_template, diff, respect_sched)
  steps_schedule = check_and_adjust_sched(steps_schedule, steps_template, diff, respect_sched)
  style_strength_schedule = check_and_adjust_sched(style_strength_schedule, style_strength_template, diff, respect_sched)
  flow_blend_schedule = check_and_adjust_sched(flow_blend_schedule, flow_blend_template, diff, respect_sched)
  cfg_scale_schedule = check_and_adjust_sched(cfg_scale_schedule, cfg_scale_template, diff, respect_sched)
  image_scale_schedule = check_and_adjust_sched(image_scale_schedule, cfg_scale_template, diff, respect_sched)
  for sched, name in zip([latent_scale_schedule,   init_scale_schedule,  steps_schedule,  style_strength_schedule,  flow_blend_schedule,
  cfg_scale_schedule, image_scale_schedule], ['latent_scale_schedule',   'init_scale_schedule',  'steps_schedule',  'style_strength_schedule',  'flow_blend_schedule',
  'cfg_scale_schedule', 'image_scale_schedule']):
    if type(sched) == list:
      if len(sched)>2: 
        print(name, ': ', sched[:100])

use_karras_noise = False
end_karras_ramp_early = False
# use_predicted_noise = False
warp_interp = Image.LANCZOS
start_code_cb = None #variable for cb_code
guidance_start_code = None #variable for guidance code

display_size = 512 #@param

user_comment= get_value('user_comment',guis)
blend_json_schedules=get_value('blend_json_schedules',guis)
VERBOSE=get_value('VERBOSE',guis)
use_background_mask=get_value('use_background_mask',guis)
invert_mask=get_value('invert_mask',guis)
background=get_value('background',guis)
background_source=get_value('background_source',guis)
(mask_clip_low, mask_clip_high) = get_value('mask_clip',guis) 

#turbo 
turbo_mode=get_value('turbo_mode',guis)
turbo_steps=get_value('turbo_steps',guis)
colormatch_turbo=get_value('colormatch_turbo',guis)
turbo_frame_skips_steps=get_value('turbo_frame_skips_steps',guis)
soften_consistency_mask_for_turbo_frames=get_value('soften_consistency_mask_for_turbo_frames',guis)

#warp
flow_warp= get_value('flow_warp',guis)
apply_mask_after_warp=get_value('apply_mask_after_warp',guis)
warp_num_k=get_value('warp_num_k',guis)
warp_forward=get_value('warp_forward',guis)
warp_strength=get_value('warp_strength',guis)
flow_override_map=eval(get_value('flow_override_map',guis))
warp_mode=get_value('warp_mode',guis)
warp_towards_init=get_value('warp_towards_init',guis)

#cc
check_consistency=get_value('check_consistency',guis)
missed_consistency_weight=get_value('missed_consistency_weight',guis)
overshoot_consistency_weight=get_value('overshoot_consistency_weight',guis)
edges_consistency_weight=get_value('edges_consistency_weight',guis)
consistency_blur=get_value('consistency_blur',guis)
consistency_dilate=get_value('consistency_dilate',guis)
padding_ratio=get_value('padding_ratio',guis)
padding_mode=get_value('padding_mode',guis)
match_color_strength=get_value('match_color_strength',guis)
soften_consistency_mask=get_value('soften_consistency_mask',guis)
mask_result=get_value('mask_result',guis)
use_patchmatch_inpaiting=get_value('use_patchmatch_inpaiting',guis)

#diffusion
text_prompts=eval(get_value('text_prompts',guis))
negative_prompts=eval(get_value('negative_prompts',guis))
prompt_patterns_sched = eval(get_value('prompt_patterns_sched',guis))
cond_image_src=get_value('cond_image_src',guis)
set_seed=get_value('set_seed',guis)
clamp_grad=get_value('clamp_grad',guis)
clamp_max=get_value('clamp_max',guis)
sat_scale=get_value('sat_scale',guis)
init_grad=get_value('init_grad',guis)
grad_denoised=get_value('grad_denoised',guis)
blend_latent_to_init=get_value('blend_latent_to_init',guis)
fixed_code=get_value('fixed_code',guis)
code_randomness=get_value('code_randomness',guis)
# normalize_code=get_value('normalize_code',guis)
dynamic_thresh=get_value('dynamic_thresh',guis)
sampler = get_value('sampler',guis)
use_karras_noise = get_value('use_karras_noise',guis)
inpainting_mask_weight = get_value('inpainting_mask_weight',guis)
inverse_inpainting_mask = get_value('inverse_inpainting_mask',guis)
inpainting_mask_source = get_value('mask_source',guis)

#colormatch
normalize_latent=get_value('normalize_latent',guis)
normalize_latent_offset=get_value('normalize_latent_offset',guis)
latent_fixed_mean=eval(str(get_value('latent_fixed_mean',guis)))
latent_fixed_std=eval(str(get_value('latent_fixed_std',guis)))
latent_norm_4d=get_value('latent_norm_4d',guis)
colormatch_frame=get_value('colormatch_frame',guis)
color_match_frame_str=get_value('color_match_frame_str',guis)
colormatch_offset=get_value('colormatch_offset',guis)
colormatch_method=get_value('colormatch_method',guis)
colormatch_regrain=get_value('colormatch_regrain',guis)
colormatch_after=get_value('colormatch_after',guis)
image_prompts = {}

fixed_seed = get_value('fixed_seed',guis)

rec_cfg = get_value('rec_cfg',guis)
rec_steps_pct = get_value('rec_steps_pct',guis)
rec_prompts = eval(get_value('rec_prompts',guis))
rec_randomness = get_value('rec_randomness',guis)
use_predicted_noise = get_value('use_predicted_noise',guis)
overwrite_rec_noise  = get_value('overwrite_rec_noise',guis)

#controlnet
save_controlnet_annotations = get_value('save_controlnet_annotations',guis)
control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face',guis)
control_sd15_depth_detector  = get_value('control_sd15_depth_detector',guis)
control_sd15_softedge_detector = get_value('control_sd15_softedge_detector',guis)
control_sd15_seg_detector = get_value('control_sd15_seg_detector',guis)
control_sd15_scribble_detector = get_value('control_sd15_scribble_detector',guis)
control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse',guis)
control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source',guis)
control_sd15_shuffle_source = get_value('control_sd15_shuffle_source',guis)
control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source',guis)
controlnet_preprocess = get_value('controlnet_preprocess',guis)

detect_resolution  = get_value('detect_resolution',guis)
bg_threshold = get_value('bg_threshold',guis)
low_threshold = get_value('low_threshold',guis)
high_threshold = get_value('high_threshold',guis)
value_threshold = get_value('value_threshold',guis)
distance_threshold = get_value('distance_threshold',guis)
temporalnet_source = get_value('temporalnet_source',guis)
temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame',guis)
controlnet_multimodel_mode = get_value('controlnet_multimodel_mode',guis)
max_faces = get_value('max_faces',guis)

do_softcap = get_value('do_softcap',guis)
softcap_thresh = get_value('softcap_thresh',guis)
softcap_q = get_value('softcap_q',guis)

masked_guidance = get_value('masked_guidance',guis)
cc_masked_diffusion = get_value('cc_masked_diffusion',guis)
alpha_masked_diffusion = get_value('alpha_masked_diffusion',guis)
invert_alpha_masked_diffusion = get_value('invert_alpha_masked_diffusion',guis)

