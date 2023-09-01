import sys, os

import math
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image
import einops


import gc
import cv2
# import gradio as gr

import time
import skimage
import datetime

from opt_flw_plg.flow_utils import RAFT_estimate_flow, RAFT_clear_memory, compute_diff_map
from opt_flw_plg import utils

import ctrlnt_plg.hf_ctrl_plg as hfcnplg
from opt_flw_plg.sd_plg import FolderLoad, sd_model


class sdcn_anim_tmp:
  prepear_counter = 0
  process_counter = 0
  input_video = None
  output_video = None
  curr_frame = None
  prev_frame = None
  prev_frame_styled = None
  prev_frame_alpha_mask = None
  fps = None
  total_frames = None
  prepared_frames = None
  prepared_next_flows = None
  prepared_prev_flows = None
  frames_prepared = False

def read_frame_from_video():
  # Reading video file
  # print("sdcn_anim_tmp ", sdcn_anim_tmp.input_video.isOpened())
  if sdcn_anim_tmp.input_video.isOpened():
    ret, cur_frame = sdcn_anim_tmp.input_video.read()
    if cur_frame is not None: 
      cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB) 
  else:
    cur_frame = None
    sdcn_anim_tmp.input_video.release()
  
  return cur_frame

def get_cur_stat():
  stat =  f'Frames prepared: {sdcn_anim_tmp.prepear_counter + 1} / {sdcn_anim_tmp.total_frames}; '
  stat += f'Frames processed: {sdcn_anim_tmp.process_counter + 1} / {sdcn_anim_tmp.total_frames}; '
  return stat

def clear_memory_from_sd():
  if shared.sd_model is not None:
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    try:
      lowvram.send_everything_to_cpu()
    except Exception as e:
      ...
    del shared.sd_model
    shared.sd_model = None
  gc.collect()
  # devices.torch_gc()


def start_vid2vid(args):

  processing_start_time = time.time()
  args_dict = utils.args_to_dict(args)
  args_dict = utils.get_mode_args('v2v', args_dict)
  sdcn_anim_tmp.process_counter = 0
  sdcn_anim_tmp.prepear_counter = 0

  # Open the input video file
  sdcn_anim_tmp.input_video = cv2.VideoCapture(args_dict['file'])
  
  # Get useful info from the source video
  sdcn_anim_tmp.fps = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FPS))
  sdcn_anim_tmp.total_frames = int(sdcn_anim_tmp.input_video.get(cv2.CAP_PROP_FRAME_COUNT))

  # model_name_list = ["openpose"]
  frm_offset = args_dict['frm_offset']
  sd_dl = FolderLoad(args_dict["wrk_spc"], args_dict['model_name_list'], 
    args_dict['height'], mask_blur=1.0) # Assume width == height == 512
  sd_mdl = sd_model(args_dict['model_name_list'], 
    args_dict["prompt"], args_dict["n_prompt"], is_inpaint=False)
  loop_iterations = (sd_dl.__len__()-1-frm_offset) * 2


  # Create an output video file with the same fps, width, and height as the input video
  output_video_name = args_dict["wrk_spc"]+f'/v2v/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
  output_video_folder = os.path.splitext(output_video_name)[0]
  os.makedirs(os.path.dirname(output_video_name), exist_ok=True)
  
  if args_dict['save_frames_check']:
    os.makedirs(output_video_folder, exist_ok=True)

  def save_result_to_image(image, ind):
    if args_dict['save_frames_check']: 
      cv2.imwrite(os.path.join(output_video_folder, f'{ind:05d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

  # print("Arguments ", (args_dict['width'], args_dict['height']))
  sdcn_anim_tmp.output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 
    sdcn_anim_tmp.fps, (args_dict['width'], args_dict['height']))

  curr_frame = sd_dl.get_ref_img(frm_offset)
  # curr_frame = read_frame_from_video()
  # curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
  sdcn_anim_tmp.prepared_frames = np.zeros((11, args_dict['height'], args_dict['width'], 3), dtype=np.uint8)
  sdcn_anim_tmp.prepared_next_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
  sdcn_anim_tmp.prepared_prev_flows = np.zeros((10, args_dict['height'], args_dict['width'], 2))
  sdcn_anim_tmp.prepared_frames[0] = curr_frame

  args_dict['init_img'] = Image.fromarray(curr_frame)
  # utils.set_CNs_input_image(args_dict, Image.fromarray(curr_frame))
  # processed_frames, _, _, _ = utils.img2img(args_dict)
  # processed_frame = np.array(processed_frames[0])[...,:3]
  inpaint_msk, mask_pixel = sd_dl.get_inpaint_mask(frm_offset)
  img_lst = sd_dl.get_img_lst(frm_offset)
  # img_lst.append(inpaint_msk)
  processed_frame = sd_mdl.gen_img(img_lst, seed=args_dict['seed'])

  # processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)
  # processed_frame = mask_pixel*processed_frame + (1-mask_pixel)*curr_frame
  processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
  #print('Processed frame ', 0)

  sdcn_anim_tmp.curr_frame = curr_frame
  sdcn_anim_tmp.prev_frame = curr_frame.copy()
  sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()
  utils.shared.is_interrupted = False

  save_result_to_image(processed_frame, 1)
  stat = get_cur_stat() + utils.get_time_left(1, loop_iterations, processing_start_time)
  # yield stat, sdcn_anim_tmp.curr_frame, None, None, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)


  for step in range(loop_iterations):
    if utils.shared.is_interrupted: break
    
    args_dict = utils.args_to_dict(args)
    args_dict = utils.get_mode_args('v2v', args_dict)

    occlusion_mask = None
    prev_frame = None
    curr_frame = sdcn_anim_tmp.curr_frame
    # warped_styled_frame_ = gr.Image.update()
    # processed_frame = gr.Image.update()

    prepare_steps = 10

    if sdcn_anim_tmp.process_counter % prepare_steps == 0 and not sdcn_anim_tmp.frames_prepared: # prepare next 10 frames for processing
        #clear_memory_from_sd()
        # device = devices.get_optimal_device()
        device = "cuda" # KJN Hack

        # curr_frame = read_frame_from_video()
        curr_frame = sd_dl.get_ref_img(sdcn_anim_tmp.prepear_counter+1+frm_offset)
        
        if curr_frame is not None: 
            # curr_frame = cv2.resize(curr_frame, (args_dict['width'], args_dict['height']))
            prev_frame = sdcn_anim_tmp.prev_frame.copy()

            next_flow, prev_flow, occlusion_mask = RAFT_estimate_flow(prev_frame, curr_frame, device=device)
            occlusion_mask = np.clip(occlusion_mask * 0.1 * 255, 0, 255).astype(np.uint8)

            cn = sdcn_anim_tmp.prepear_counter % 10
            if sdcn_anim_tmp.prepear_counter % 10 == 0:
                sdcn_anim_tmp.prepared_frames[cn] = sdcn_anim_tmp.prev_frame
            sdcn_anim_tmp.prepared_frames[cn + 1] = curr_frame.copy()
            sdcn_anim_tmp.prepared_next_flows[cn] = next_flow.copy()
            sdcn_anim_tmp.prepared_prev_flows[cn] = prev_flow.copy()
            print('Prepared frame ', cn+1)

            sdcn_anim_tmp.prev_frame = curr_frame.copy()

        sdcn_anim_tmp.prepear_counter += 1
        if sdcn_anim_tmp.prepear_counter % prepare_steps == 0 or \
        sdcn_anim_tmp.prepear_counter >= sdcn_anim_tmp.total_frames - 1 or \
        curr_frame is None:
            # Remove RAFT from memory
            RAFT_clear_memory()
            sdcn_anim_tmp.frames_prepared = True
    else:
        # process frame
        sdcn_anim_tmp.frames_prepared = False

        cn = sdcn_anim_tmp.process_counter % 10 
        curr_frame = sdcn_anim_tmp.prepared_frames[cn+1][...,:3]
        prev_frame = sdcn_anim_tmp.prepared_frames[cn][...,:3]
        next_flow = sdcn_anim_tmp.prepared_next_flows[cn]
        prev_flow = sdcn_anim_tmp.prepared_prev_flows[cn]

        ### STEP 1
        alpha_mask, warped_styled_frame = compute_diff_map(next_flow, prev_flow, prev_frame, curr_frame, sdcn_anim_tmp.prev_frame_styled, args_dict)
        warped_styled_frame_ = warped_styled_frame.copy()

        #fl_w, fl_h = prev_flow.shape[:2]
        #prev_flow_n = prev_flow / np.array([fl_h,fl_w])
        #flow_mask = np.clip(1 - np.linalg.norm(prev_flow_n, axis=-1)[...,None] * 20, 0, 1)
        #alpha_mask = alpha_mask * flow_mask

        if sdcn_anim_tmp.process_counter > 0 and args_dict['occlusion_mask_trailing']:
            alpha_mask = alpha_mask + sdcn_anim_tmp.prev_frame_alpha_mask * 0.5
        sdcn_anim_tmp.prev_frame_alpha_mask = alpha_mask

        alpha_mask = np.clip(alpha_mask, 0, 1)
        occlusion_mask = np.clip(alpha_mask * 255, 0, 255).astype(np.uint8)

        # fix warped styled frame from duplicated that occures on the places where flow is zero, but only because there is no place to get the color from
        # warped_styled_frame = curr_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)
        # warped_styled_frame = prev_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)

        # process current frame
        # TODO: convert args_dict into separate dict that stores only params necessery for img2img processing
        img2img_args_dict = args_dict #copy.deepcopy(args_dict)
        # print('PROCESSING MODE:', args_dict['step_1_processing_mode'])

        if args_dict['step_1_processing_mode'] == 0: # Process full image then blend in occlusions
          img2img_args_dict['mode'] = 0
          img2img_args_dict['mask_img'] = None #Image.fromarray(occlusion_mask)
        elif args_dict['step_1_processing_mode'] == 1: # Inpaint occlusions
          img2img_args_dict['mode'] = 4
          img2img_args_dict['mask_img'] = Image.fromarray(occlusion_mask)
        else:
           raise Exception('Incorrect step 1 processing mode!')
        
        blend_alpha = args_dict['step_1_blend_alpha']
        init_img = warped_styled_frame * (1 - blend_alpha) + curr_frame * blend_alpha
        img2img_args_dict['init_img'] = Image.fromarray(np.clip(init_img, 0, 255).astype(np.uint8))
        img2img_args_dict['seed'] = args_dict['step_1_seed']
        # utils.set_CNs_input_image(img2img_args_dict, Image.fromarray(curr_frame))
        # processed_frames, _, _, _ = utils.img2img(img2img_args_dict)
        # processed_frame = np.array(processed_frames[0])[...,:3]

        # inpaint_msk = sd_dl.get_inpaint_mask(sdcn_anim_tmp.process_counter + 1)
        img_lst = sd_dl.get_img_lst(sdcn_anim_tmp.process_counter + frm_offset + 1)
        # print("Input mask alpha ", alpha_mask.shape, inpaint_msk.shape,
        #   init_img.shape)
        # print(np.max(alpha_mask), torch.max(inpaint_msk),
        #   np.max(init_img))
        inpaint_msk = np.clip(init_img/255.0, 0.0, 1.0)
        inpaint_msk[alpha_mask > 0.5] = -1.0
        inpaint_msk = torch.from_numpy(inpaint_msk)[None, ...]
        inpaint_msk = einops.rearrange(inpaint_msk, 'b h w c -> b c h w')
        # img_lst.append(inpaint_msk)
        processed_frame = sd_mdl.gen_img(img_lst, seed=args_dict['seed'])

        # normalizing the colors
        # processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)
        processed_frame = skimage.exposure.match_histograms(processed_frame, 
          sdcn_anim_tmp.prev_frame_styled, channel_axis=None)
        processed_frame = processed_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)
        
        #processed_frame = processed_frame * 0.94 + curr_frame * 0.06
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
        sdcn_anim_tmp.prev_frame_styled = processed_frame.copy()

        ### STEP 2
        if args_dict['fix_frame_strength'] > 0:
          img2img_args_dict = args_dict #copy.deepcopy(args_dict)
          img2img_args_dict['mode'] = 0
          img2img_args_dict['init_img'] = Image.fromarray(processed_frame)
          img2img_args_dict['mask_img'] = None
          img2img_args_dict['denoising_strength'] = args_dict['fix_frame_strength']
          img2img_args_dict['seed'] = args_dict['step_2_seed']
          # utils.set_CNs_input_image(img2img_args_dict, Image.fromarray(curr_frame))
          # processed_frames, _, _, _ = utils.img2img(img2img_args_dict)
          # processed_frame = np.array(processed_frames[0])
          # inpaint_msk = sd_dl.get_inpaint_mask(sdcn_anim_tmp.process_counter)
          # img_lst = sd_dl.get_img_lst(sdcn_anim_tmp.process_counter).append(inpaint_msk)
          processed_frame = sd_mdl.gen_img(img_lst, is_t2i=False,
              ref_img=img2img_args_dict['init_img'], seed=args_dict['seed'])

          processed_frame = skimage.exposure.match_histograms(processed_frame, curr_frame, channel_axis=None)

        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
        warped_styled_frame_ = np.clip(warped_styled_frame_, 0, 255).astype(np.uint8)
        
        # Write the frame to the output video
        frame_out = np.clip(processed_frame, 0, 255).astype(np.uint8)
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR) 
        sdcn_anim_tmp.output_video.write(frame_out)

        sdcn_anim_tmp.process_counter += 1
        #if sdcn_anim_tmp.process_counter >= sdcn_anim_tmp.total_frames - 1:
        #    sdcn_anim_tmp.input_video.release()
        #    sdcn_anim_tmp.output_video.release()
        #    sdcn_anim_tmp.prev_frame = None

        save_result_to_image(processed_frame, sdcn_anim_tmp.process_counter + 1)

    stat = get_cur_stat() + utils.get_time_left(step+2, loop_iterations+1, processing_start_time)
    # yield stat, curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, None, gr.Button.update(interactive=False), gr.Button.update(interactive=True)

  RAFT_clear_memory()

  sdcn_anim_tmp.input_video.release()
  sdcn_anim_tmp.output_video.release()

  # curr_frame = gr.Image.update()
  # occlusion_mask = gr.Image.update()
  # warped_styled_frame_ = gr.Image.update() 
  # processed_frame = gr.Image.update()

  # yield get_cur_stat(), curr_frame, occlusion_mask, warped_styled_frame_, processed_frame, output_video_name, gr.Button.update(interactive=True), gr.Button.update(interactive=False)

if __name__ == '__main__':


  args = {
    'glo_sdcn_process_mode': 0,
    'v2v_file': "../output/ttv/splt_vdo/output0016.mp4", # Not required for now
    'v2v_width': None, 
    'v2v_height': None, 
    'v2v_prompt': "Spiderman ", 
    'v2v_n_prompt': None, 
    'v2v_cfg_scale': None, 
    'v2v_seed': None, 
    'v2v_processing_strength': None, 
    'v2v_fix_frame_strength': None, 

    'v2v_sampler_index': None, 
    'v2v_steps': None, 
    'v2v_override_settings': None, #img2img

    'v2v_occlusion_mask_blur': 0.5, 
    'v2v_occlusion_mask_trailing': False, 
    'v2v_occlusion_mask_flow_multiplier': 0.0, # 10.0
    'v2v_occlusion_mask_difo_multiplier': 1.0, 
    'v2v_occlusion_mask_difs_multiplier': 0.0,
    'v2v_step_1_processing_mode': 1, # 0  is full image 1 is inpaint occlusion
    'v2v_step_1_blend_alpha': 0.0, # Blending current frame and warped frame
    'v2v_step_1_seed': -1, # Seed to img2img
    'v2v_step_2_seed': -1, # Seed for fix_frame_strength based img2img
    'glo_save_frames_check': True, # Save output video
    # 't2v_file', 't2v_width', 't2v_height', 't2v_prompt', 't2v_n_prompt', 't2v_cfg_scale', 't2v_seed', 't2v_processing_strength', 't2v_fix_frame_strength',
    # 't2v_sampler_index', 't2v_steps', 't2v_length', 't2v_fps',

    # # video to video params
    # 'v2v_mode': 0,
    # 'v2v_prompt': 'hello',
    # 'v2v_n_prompt': 'hallo',
    # 'v2v_prompt_styles': [],
    # 'v2v_init_video': None, # Always required

    # 'v2v_steps': 15,
    # 'v2v_sampler_index': 0, # 'Euler a'    
    # 'v2v_mask_blur': 0,

    # 'v2v_inpainting_fill': 1, # original
    # 'v2v_restore_faces': False,
    # 'v2v_tiling': False,
    # 'v2v_n_iter': 1,
    # 'v2v_batch_size': 1,
    # 'v2v_cfg_scale': 5.5,
    # 'v2v_image_cfg_scale': 1.5,
    # 'v2v_denoising_strength': 0.75,
    # 'v2v_fix_frame_strength': 0.15,
    # 'v2v_seed': -1,
    # 'v2v_subseed': -1,
    # 'v2v_subseed_strength': 0,
    # 'v2v_seed_resize_from_h': 512,
    # 'v2v_seed_resize_from_w': 512,
    # 'v2v_seed_enable_extras': False,
    # 'v2v_height': 512,
    # 'v2v_width': 512,
    # 'v2v_resize_mode': 1,
    # 'v2v_inpaint_full_res': True,
    # 'v2v_inpaint_full_res_padding': 0,
    # 'v2v_inpainting_mask_invert': False,

  }
  args["v2v_wrk_spc"] = "../output/ttv/"
  # args["v2v_prompt"] = "RAW photo, anime, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
  # args["v2v_prompt"] = "BTS boys doing acroyoga at the beach cosmic energy bright light ray cyberpunk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha"
  # args["v2v_prompt"] = "spiderman, intricate, elegant, highly detailed animal monster, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8 k"
  args["v2v_prompt"] = "scarlett johansson, beautiful body, fingers, smiling, symmetrical face, intricate, elegant, highly detailed, sharp focus, award - winning, masterpiece, trending on artstation, cinematic composition, beautiful lighting"
  # args["v2v_prompt"] = "Elsa, fantasy, intricate, elegant, highly detailed, digital painting, \
  #       artstation, concept art, matte, sharp focus, illustration, hearthstone, art by artgerm \
  #       and greg rutkowski and alphonse mucha, 8k"

  # args["v2v_n_prompt"] = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
  # args["v2v_n_prompt"] = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
  args["v2v_n_prompt"] = "disfigured, disproportionate , bad anatomy, bad proportions, \
        deformed iris, deformed nose, deformed pupils, ugly, out of frame, mangled, asymmetric, cross-eyed, depressed, \
        immature, mutated hands and fingers, stuffed animal, out of focus, high depth of field, cloned face, \
        cloned head, age spot, skin blemishes, collapsed eyeshadow, asymmetric ears, \
        imperfect eyes, unnatural, conjoined, missing limb, missing arm, \
        missing leg, poorly drawn face, poorly drawn feet, poorly drawn hands, \
        floating limb, disconnected limb, extra limb, malformed limbs, malformed hands, \
        poorly rendered face, poor facial details, poorly rendered hands, double face, \
        unbalanced body, unnatural body, lacking body, long body, cripple, old , fat, \
        cartoon, weird colors, unnatural skin tone, unnatural skin, stiff face, fused hand, \
        skewed eyes, surreal, cropped head, group of people"
  args["v2v_seed"]=1294216578
  args["v2v_step_1_seed"]=1294216578
  args["v2v_step_2_seed"]=1294216578
  args["v2v_model_name_list"] = ["zoe_dpth"] # "openpose" "zoe_dpth" "softedge_PIDI", "midas_dpth"
  args["v2v_frm_offset"] = 40

  start_vid2vid(args)
