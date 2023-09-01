if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/source_code/mdl_inf_jp3d/')

import os
import lib.utils_plg.linux_cmd as lc 
from lib.deforum import warping2d_plg #warp2d_mask_gen, gen_vdo
from lib import audio_plg
from lib.utils_plg.ffmpeg_plg import merge_vdo_ado, concat_vdo
from lib.utils_plg.mem_plg import collect_cache
from core_blks.ctrl_img_gen import gen_img, gen_logo


def gen_vdo(theme, prompt, img_path, motion_template, op_fld, num_sec=6, fps=60, bg_music_prompt=None, voice_text=None, 
    vdo_wth=1280, vdo_hgt=720, **gen_img_args):
    # Instead of two images we generate a single image with the necessary padding and 2D warping or movement
    # print("Motion template ", motion_template)
    msk_img = warping2d_plg.warp2d_mask_gen(img_path, motion_template=motion_template, num_sec=num_sec, fps=fps)

    # 
    img_fld = os.path.join(op_fld, "img")
    output_info_list = gen_img(theme, prompt, op_fld, control_type="inpaint", ctrl_img=msk_img, height=msk_img.shape[1], 
        width=msk_img.shape[0], **gen_img_args)

    # Generate videos with the single images
    vdo_img_path = output_info_list[0]['path']
    vdo_fld = os.path.join(op_fld, "vdo")
    gen_vdo_path = warping2d_plg.gen_vdo(vdo_img_path, motion_template, vdo_fld, num_sec=num_sec, fps=fps)

    if(bg_music_prompt or voice_text):
        audio_gen = audio_plg.audio_gen(musicgen_mdl='melody')
        ado_path = os.path.join(op_fld, "ado")
        mrg_path = os.path.join(op_fld, "mrg")

    if(bg_music_prompt):
        # Generate music for this clip
        gen_ado_path = audio_gen.gen_music([bg_music_prompt], num_sec, ado_path)
        # Merge generated audio and video
        merge_vdo_ado(gen_vdo_path['path'], gen_ado_path[0]['path'], mrg_path)

    if(voice_text):
        # Generate voice for logo
        gen_ado_path = audio_gen.gen_speech(voice_text, ado_path)
        # Merge generated audio and video
        merge_vdo_ado(gen_vdo_path['path'], gen_ado_path['path'], mrg_path)

    if(bg_music_prompt or voice_text):
        output = {'path':mrg_path}
    else:
        output = {'path':gen_vdo_path['path']}

    return output

def gen_logo_vdo(logo_path, theme, prompt, motion_template, op_fld, control_type="pidiedge", **gen_logo_vdo_args):

    logo_op_path = os.path.join(op_fld, "logo")
    output = gen_logo(logo_path, theme, prompt, op_fld=logo_op_path, control_type="pidiedge")

    return gen_vdo(theme, prompt, output[0]['img_path'], motion_template=motion_template,
        op_fld=op_fld, **gen_logo_vdo_args)

if __name__ == "__main__":
    theme = "food"
    # prompt = "Strawberry icecream with chocolate sauce placed on a wall in new york"
    prompt = "foodphoto, splashes, Hot piping coffee in a cup and croissant <lora:more_details:0.7>"
    # gen_img(theme, prompt, op_fld, control_type=None, ctrl_img=None, n_prompt="", height=512, width=512, 
    #     seed=-1, num_images=1, safety_checker=None, collect_cache=True)
    
    logo_path = "./share_vol/data_io/inp/edge_inv.png" # 
    # gen_logo(logo_path, theme=theme, prompt=prompt, op_fld=op_fld, control_type="pidiedge")

    img_path = "./share_vol/test/img/00000.png"
    op_fld = "./share_vol/test/stg1"
    motion_template = "pan_right"
    num_sec = 4
    fps = 24
    bg_music_prompt = None #"flute with guitar"
    voice_text = "flute with guitar and a penny of love" # None #
    # gen_vdo(theme, prompt, img_path, motion_template, op_fld, num_sec=num_sec, fps=fps, bg_music_prompt=bg_music_prompt, 
    #     voice_text=voice_text)
    gen_logo_vdo(logo_path, theme, prompt, motion_template, op_fld, control_type="pidiedge", num_sec=num_sec, fps=fps, bg_music_prompt=bg_music_prompt, 
        voice_text=voice_text)
