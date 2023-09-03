if __name__ == "__main__":
    import sys
    # path = '/workspace/source_code/mdl_inf_jp3d'
    path = '/home/source_code/mdl_inf_jp3d/'
    sys.path.insert(0, path)

# from lib.hf.sd_ctrl import sd_model as sd_ctrl_mdl
# from lib.hf.sd_upscaler import sdx4upscaler
# from lib.hf.sdxl import create_controlnet, create_base_model, gen_one_img as gen_one_img_sdxl

from lib.sd_mdl.sd_api import sd_model

from lib.utils_plg import mem_plg, anno_plg
from PIL import Image, ImageOps
import numpy as np
import os
from controlnet_aux.processor import Processor
import ffmpegio, imageio
import cv2

def gen_inpaint_filler(theme, prompt, img_path, imgfmt_list, op_fld):
    # Handling images smaller than 512x512. Should resize to the smallest dimension
    # Add support fo gif with little zoom and pan effects
    # Add support fo whatsapp status and images
    # Tiktok and insta stories and status
    imgfmt_to_imgdim_mapper = {
        "insta_square" : (1080, 1080), # wxh, numpy hxw, PIL wxh
        "insta_potrait" : (1080, 1352), "insta_landscape" : (1080, 566),
        "fb_post":(1200, 632), #"fb_story":(1080, 1920), #"fb_cover":(820, 312), "fb_profile":(180, 180), 
        "twit_instream":(1200, 680), #"twit_header":(1500, 500), "twit_profile": (400, 400),
        "whatsapp_status":(1080, 1920),
    }
    format_to_vdo_dim_mapper = {
        ""
    }
    img = np.array(Image.open(img_path).convert("RGB"))/255.0
    inp_wth_x = img.shape[1]; inp_hgt_y = img.shape[0];
    out_img_info_lst = []
    for imgfmt in imgfmt_list:
        img_shape = imgfmt_to_imgdim_mapper[imgfmt]
        out_wth_x, out_hgt_y = img_shape
        inp_pos_x = int((out_wth_x - inp_wth_x)/2.0)
        inp_pos_y = int((out_hgt_y - inp_hgt_y)/2.0)
        msk_img = -1*np.ones((1, out_hgt_y, out_wth_x, 3))
        msk_img[0, inp_pos_y:inp_pos_y+inp_hgt_y, inp_pos_x:inp_pos_x+inp_wth_x, :] = img

        op_fmt_fld = os.path.join(op_fld, imgfmt)
        out_img_info = gen_img(theme, prompt, op_fmt_fld, control_type="inpaint", ctrl_img=msk_img, 
            n_prompt="", height=out_hgt_y, width=out_wth_x)
        out_img_info_lst.append(out_img_info[0])
        
    return out_img_info_lst


def vdo_ctrl_gen(theme, prompt, prompt_2, op_fld, ctrl_vdo_path, control_type="pidiedge",  
     n_prompt="", seed=-1, safety_checker=None, collect_cache=True):

    # Create SD model for the given theme and n_prompt
    # sd_model, processor = create_model(theme, n_prompt, control_type, safety_checker)

    sd_mdl = sd_model(theme=theme, control_type=control_type)
    reader = imageio.get_reader(ctrl_vdo_path)
    fps = reader.get_meta_data()['fps']
    print("Output array ", fps)
    vdo_path = os.path.join(op_fld, "vdo_ctrl")
    os.makedirs(vdo_path, exist_ok=True)
    vdo_filepath = os.path.join(vdo_path, "generated.mp4")

    fps = 6         
    writer = imageio.get_writer(vdo_filepath, fps=fps)

    try:
        i = 0
        for im in reader:
            if(i%10 == 0):
                vdo_frm = im[516:1404, :888, :]
                print("VDO Frame ", i, vdo_frm.shape)
                # vdo_frm = cv2.resize(vdo_frm, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                vdo_frm = cv2.resize(vdo_frm, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
                height, width, _ = vdo_frm.shape
                # output_info_list = gen_one_img(sd_model, processor, 
                #         control_type, None, vdo_frm, None, height, width, seed, num_images)
                
                # output_info_list = gen_one_img_sdxl(pipe, refiner, processor, prompt, prompt_2, None, vdo_frm, 
                #     op_fld=None, control_type=control_type, n_prompt=n_prompt, height=1024, width=1024,
                #     seed=seed, num_images=1, num_inf_steps=50,)
                # output_info_list[0]['image'].save(os.path.join(op_fld, "vdo_ctrl", str(i).zfill(5)+".png"))
                # writer.append_data(np.array(output_info_list[0]['image']))
                image = sd_mdl.gen_img(prompt=prompt, n_prompt=n_prompt, height=height, width=width, seed=seed, 
                    ctrl_img=vdo_frm, num_images=1) #init_image=None, mask_image=None, ctrl_img_path=None, 
                image[0].save(os.path.join(op_fld, "vdo_ctrl", str(i).zfill(5)+".png"))
                writer.append_data(np.array(image[0]))
                
            i += 1
            if(i >= 1000):
                break
    except RuntimeError:
        pass
    reader.close()
    writer.close()         
    # gen_frame_arr = []
    # for i in range(0, vdo_arr.shape[0], 10):
    #     vdo_frm = vdo_arr[i, 516:1404, :888, :]
    #     vdo_frm = cv2.resize(vdo_frm, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    #     height, width, _ = vdo_frm.shape
        
    #     output_info_list = gen_one_img(sd_model, processor, control_type, None, vdo_frm, None, height, width, seed, num_images)
    #     output_info_list[0]['image'].save(os.path.join(op_fld, "vdo_ctrl", str(i).zfill(5)+".png"))
    #     gen_frame_arr.append(np.array(output_info_list[0]['image']))

    # vdo_filepath = os.path.join(vdo_path, "generated.mp4")
    # gen_frame_arr = np.array(gen_frame_arr)
    # ffmpegio.video.write(vdo_filepath, fps, gen_frame_arr, overwrite=True) #, pix_fmt_in='yuv420p')

    if(collect_cache):
        del sd_mdl
        mem_plg.collect_cache()
    return [{'path':vdo_filepath}]



# Run this code from  /home/source_code/mdl_inf_jp3d/

if __name__ == "__main__":
    theme = "people" #"sdxl_base" #"people_lifelike"
    op_fld = "./share_vol/test/img3"
    # prompt = "Strawberry icecream with chocolate sauce placed on a wall in new york"
    # logo_path = "./share_vol/data_io/inp/logo_mealo.png" # edge_inv.png" # 
    logo_path = "./share_vol/data_io/inp/0.jpeg" # edge_inv.png" # 

    ctrl_vdo_path = "./share_vol/data_io/inp/ctrl.mp4" # edge_inv.png" # 
    control_type="midasdepth" #"depth"

    # prompt = "cyborg style, cyborg, 3d style,3d render,cg,beautiful, school girl, fully clothed, looking at viewer, long braid, sparkling eyes, cyborg , mechanical limbs, cyberpunk, \
    #     cute gloves 3d_render_style_xl this has good facial feature holding weapons and gadget in each hand"
    # prompt = "Portrait of cyborg style (( indian princess with robotic armor)) in futuristic city , beautiful face, d&d, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus, 8k, highly detailed, ((cinematic lighting)) ((scene from the end of the world)) dramatic, beautiful"
    prompt = "futubot, by issey miyake, (( indian princess with robotic armor)) cyberspace robot, pyramid, in the style of detailed hyperrealism, Egyptiancore, gold strong facial expression, richard bergh, bill gekas, benedick bana"
    # gen_img(theme, prompt, op_fld, control_type="midasdepth", ctrl_img_path=logo_path, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)

    # vdo_ctrl_gen(theme, prompt, op_fld, ctrl_vdo_path, control_type="pidiedge") 

    prompt = "cyborg_style_xl 1boy, science fiction, glowing, full body, humanoid robot, blue eyes, cable, mechanical parts, spine, armor, power armor, standing, robot, cyberpunk, scifi, , "
    prompt_2 = "cyborg_style_xl, high quality, high resolution, dslr, 8k, 4k, ultrarealistic, realistic,perfecteyes"
    n_prompt = "drawing, painting, illustration, rendered, low quality, low resolution"
    vdo_ctrl_gen(theme, prompt, prompt_2, op_fld, ctrl_vdo_path, control_type=control_type, n_prompt=n_prompt, seed=123456)
    
    logo_path = "./share_vol/data_io/inp/logo_mealo.png" # edge_inv.png" # 
    # gen_logo(logo_path, theme=theme, prompt=prompt, op_fld=op_fld, control_type="pidiedge")

    theme = "people"
    # prompt = "instagram photo, closeup face photo of 18 y.o swedish woman in dress, beautiful face, makeup, night city street, bokeh, motion blur"
    # prompt = "closeup face photo of caucasian man in black clothes, night city street, bokeh"
    prompt = "polaroid photo, night photo, photo of 24 y.o beautiful woman, pale skin, bokeh, motion blur"
    # gen_img(theme, prompt, op_fld, control_type=None, ctrl_img_path=None, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)


    # prompt = "Cyborg with multiple hands holding futuristic gadgets with beautiful face and sparkling eyes"
    # gen_img(theme, prompt, op_fld, control_type="pidiedge", 

    # prompt = "cyborg style, cyborg, 3d style,3d render,cg,beautiful, goddess Kaali, looking at viewer, long braid, sparkling eyes, cyborg , mechanical limbs, cyberpunk, \
    #     cute gloves 3d_render_style_xl this has good facial feature holding weapons and gadget in each hand"
    # gen_img(theme, prompt, op_fld, control_type="pidiedge", ctrl_img_path=logo_path, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)

    # lora_path = "./share_vol/models/lora/cyborg_style_xl-alpha.safetensors"
    # pipe, refiner, processor = create_controlnet(control_type=control_type, lora_path=None, lora_scale=0.8)
