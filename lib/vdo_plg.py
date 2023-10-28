if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/source_code/mdl_inf_jp3d/')

from moviepy.editor import ImageClip, CompositeVideoClip, CompositeAudioClip, \
    AudioFileClip, VideoFileClip, concatenate_videoclips
import numpy as np
from PIL import Image
import os
from lib.img_plg import gen_one_img
from lib.audio_plg import gen_music, gen_speech


def generate_mask(img, inp_pos_x, inp_pos_y, inp_with_x, inp_hgt_y, op_wth, op_hgt):
    # print("Input image ", img.shape, np.max(img))

    msk_img = -1*np.ones((1, op_hgt, op_wth, 3))
    msk_img[0, inp_pos_y:inp_pos_y+inp_hgt_y, inp_pos_x:inp_pos_x+inp_with_x, :] = img
    return msk_img

def read_img(img_path):
    # Read input image
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    img = np.array(img)/255.0
    inp_wth_x = img.shape[1]; inp_hgt_y = img.shape[0] # Numpy HeightxWidth, PIL image WidthxHeight
    return (inp_wth_x, inp_hgt_y, img)

def gen_pan_video(img_path, theme, prompt, op_fld, template="bottom_right", 
    num_sec=4, fps=24, vdo_wth=1280, vdo_hgt=720, pan_speed=1, is_gif=False):
    # Read input image
    (inp_wth_x, inp_hgt_y, img) = read_img(img_path)

    # Find the parameter for pan using template
    pan_template_to_setting_map = { # start_right_x, start_bottom_y, pan_x, pan_y
        "left": (1,0,1,0), "right": (0,0,1,0), "top":(0,1,0,1), "bottom":(0,0,0,1), 
        "bottom_left":(1,0,1,1), "top_left":(1,1,1,1), "bottom_right":(0,0,1,1), "top_right":(0,1,1,1), 
    }
    (start_right_x, start_bottom_y, pan_x, pan_y) = pan_template_to_setting_map[template]

    # Generate mask for inpainting
    fps_pan_speed = fps*pan_speed
    inp_pos_x = start_right_x*num_sec*fps_pan_speed + round((vdo_wth - inp_wth_x)/2)
    inp_pos_y = start_bottom_y*num_sec*fps_pan_speed + round((vdo_hgt - inp_hgt_y)/2)
    canvas_wth_x = vdo_wth + num_sec*fps_pan_speed*pan_x
    canvas_hgt_y = vdo_hgt + num_sec*fps_pan_speed*pan_y
    img_mask = generate_mask(img, inp_pos_x, inp_pos_y, inp_wth_x, inp_hgt_y, canvas_wth_x, canvas_hgt_y)

    # Generate the image by inpainting
    img_fld = os.path.join(op_fld, "img")
    # output_info_list = gen_img(theme, prompt, op_fld, control_type="inpaint", ctrl_img=img_mask, height=img_mask.shape[1], 
    #     width=img_mask.shape[0])
    print("Image mask ", img_mask.shape)
    output_info_list = gen_one_img(theme, prompt, op_fld=op_fld, control_type="inpaint", ctrl_img=img_mask, 
        height=img_mask.shape[-3], width=img_mask.shape[-2])
    vdo_img_path = output_info_list[0]['path']

    # img_pth = os.path.join(img_fld, "00000.png")
    # Image.fromarray((img_mask[0,...]*255).astype(np.uint8)).save(img_pth)
    # vdo_img_path = img_pth

    vdo_path = os.path.join(op_fld, "vdo")
    os.makedirs(vdo_path, exist_ok=True)
    img_clip = ImageClip(vdo_img_path).set_fps(fps).set_duration(num_sec) \
    .set_position(lambda t:(-start_right_x*num_sec*fps_pan_speed + 2*(-0.5+start_right_x)*t*fps_pan_speed*pan_x, 
                   -start_bottom_y*num_sec*fps_pan_speed + 2*(-0.5+start_bottom_y)*t*fps_pan_speed*pan_y)) \
    # .resize(width=lambda t: round(vdo_wth*(1+0.5*np.sin(2*np.pi*t*fps_scale_speed/6))),
    #         height=lambda t: round(vdo_hgt*(1+0.5*np.sin(2*np.pi*t*fps_scale_speed/6))))
    composite = CompositeVideoClip([img_clip], size=(vdo_wth,vdo_hgt))
    if(is_gif):
        vdo_filepath = os.path.join(vdo_path, "generated.gif")
        composite.write_gif(vdo_filepath, fps=None, program='imageio', 
            opt='nq', fuzz=1, verbose=True, loop=0, dispose=False, colors=None, tempfiles=False, logger='bar')
    else:
        vdo_filepath = os.path.join(vdo_path, "generated.mp4")
        composite.write_videofile(vdo_filepath)
    return [{'path':vdo_filepath}]

def zoom_in_out(t):
    return 1 + 0.6*np.sin(2*np.pi*t/6)

def gen_zoom_video(img_path, theme, prompt, op_fld, zoom_in=True, num_sec=4, fps=24, vdo_wth=1280, vdo_hgt=720, 
    scale=1.6, pos_x='center', pos_y='center', is_gif=False): # [0, 0.5, 1.0]'center', 'left', 'right', 'top', 'bottom'
    # Read input image
    (inp_wth_x, inp_hgt_y, img) = read_img(img_path)
    # Generate mask for inpainting
    inp_pos_x = round((vdo_wth - inp_wth_x)/2)
    inp_pos_y = round((vdo_hgt - inp_hgt_y)/2)
    canvas_wth_x = round(vdo_wth*(1+(scale-1)/2))
    canvas_hgt_y = round(vdo_hgt*(1+(scale-1)/2))
    img_mask = generate_mask(img, inp_pos_x, inp_pos_y, inp_wth_x, inp_hgt_y, canvas_wth_x, canvas_hgt_y)

    # Generate the image by inpainting
    img_fld = os.path.join(op_fld, "img")
    
    # img_pth = os.path.join(img_fld, "00000.png")
    # Image.fromarray((img_mask[0,...]*255).astype(np.uint8)).save(img_pth)
    # vdo_img_path = img_pth

    # TBD: replace ctrl_img_path with ctrl_img 
    # output_info_list = gen_img(theme, prompt, op_fld, control_type="inpaint", ctrl_img=img_mask, height=img_mask.shape[1], 
    #     width=img_mask.shape[0])
    print("Image mask ", img_mask.shape)
    output_info_list = gen_one_img(theme, prompt, op_fld=op_fld, control_type="inpaint", ctrl_img=img_mask, 
        height=img_mask.shape[-3], width=img_mask.shape[-2])
    vdo_img_path = output_info_list[0]['path']

    vdo_path = os.path.join(op_fld, "vdo")
    os.makedirs(vdo_path, exist_ok=True)
    # fps_scale_speed = fps*speed
    if(zoom_in):
        multiplier = 1
    else:
        multiplier = -1
    # print("Output scale ", round(vdo_wth*(1+(0-num_sec/2)*(scale-1)/num_sec)), (0-num_sec/2)*(scale-1)/num_sec, scale, num_sec)
    img_clip = ImageClip(vdo_img_path).set_fps(fps).set_duration(num_sec) \
    .set_position((pos_x,pos_y)) \
    .resize(width=lambda t: round(vdo_wth*(1+multiplier*(t-num_sec/2)*(scale-1)/num_sec)),
            height=lambda t: round(vdo_hgt*(1+multiplier*(t-num_sec/2)*(scale-1)/num_sec)))
    # .resize(width=lambda t: round(vdo_wth*(1+0.5*np.sin(2*np.pi*t*fps_scale_speed/6))),
    #         height=lambda t: round(vdo_hgt*(1+0.5*np.sin(2*np.pi*t*fps_scale_speed/6))))
    composite = CompositeVideoClip([img_clip], size=(vdo_wth,vdo_hgt))

    if(is_gif):
        vdo_filepath = os.path.join(vdo_path, "generated.gif")
        composite.write_gif(vdo_filepath, fps=None, program='imageio', 
            opt='nq', fuzz=1, verbose=True, loop=0, dispose=False, colors=None, tempfiles=False, logger='bar')
    else:
        vdo_filepath = os.path.join(vdo_path, "generated.mp4")
        composite.write_videofile(vdo_filepath)
    return [{'path':vdo_filepath}]

def gen_vdo_ado_speech(theme, prompt, img_path, motion_template, op_fld, 
    bg_music_prompt=None, voice_text=None, num_sec=4, history_prompt="v2/en_speaker_1", melody_path=None, **gen_vdo_args):

    vdo_fld = os.path.join(op_fld, "vdo")
    if("pan" in motion_template):
        gen_vdo_path = gen_pan_video(img_path, theme, prompt, vdo_fld, 
            template=motion_template[4:], num_sec=num_sec, **gen_vdo_args)
    else:
        gen_vdo_path = gen_zoom_video(img_path, theme, prompt, vdo_fld, 
            zoom_in=("zoom_in" in motion_template), num_sec=num_sec, **gen_vdo_args)
        
    ado_clip_lst = []
    if(bg_music_prompt):
        # Generate music for this clip
        music_path = os.path.join(op_fld, "music")
        gen_ado_path = gen_music(bg_music_prompt, num_sec, music_path, melody_path=melody_path)
        ado_clip_lst.append(AudioFileClip(gen_ado_path[0]['path']))

    if(voice_text):
        # Generate voice for logo
        speech_path = os.path.join(op_fld, "speech")
        gen_ado_path = gen_speech(voice_text, speech_path, history_prompt=history_prompt)
        ado_clip_lst.append(AudioFileClip(gen_ado_path[0]['path']))
    
    if(bg_music_prompt or voice_text):
        mrg_path = os.path.join(op_fld, "mrg")
        os.makedirs(mrg_path, exist_ok=True)
        mrg_filepath = os.path.join(mrg_path, "generated.mp4")

        audioclip = CompositeAudioClip(ado_clip_lst)
        videoclip = VideoFileClip(gen_vdo_path[0]['path'])
        videoclip.audio = audioclip
        videoclip.write_videofile(mrg_filepath)

        output = [{'path':mrg_filepath}]
    else:
        output = [{'path':gen_vdo_path[0]['path']}]

    return output


def concat_vdo(vdo_file_lst, op_fld):
    clip_list = []
    for vdo_file in vdo_file_lst:
        clip_list.append(VideoFileClip(vdo_file))
    # Try replacing it with CompositeVideoClip
    composite = concatenate_videoclips(clip_list)

    vdo_path = os.path.join(op_fld, "vdo")
    os.makedirs(vdo_path, exist_ok=True)
    vdo_filepath = os.path.join(vdo_path, "generated.mp4")
    composite.write_videofile(vdo_filepath)
    return [{'path':vdo_filepath}]

def add_img_to_vdo(vdo_path, img_path, op_fld, pos_x="right", pos_y="bottom", scale=0.25):

    video = VideoFileClip(vdo_path)
    # print("Video info ", dir(video), video.w, video.h)
    hgt = int(video.h*scale)
    img = ImageClip(img_path).set_pos((pos_x,pos_y)).set_duration(video.duration).resize(height=hgt)
    qr_vdo = CompositeVideoClip([video, img])

    vdo_path = os.path.join(op_fld, "qrc")
    os.makedirs(vdo_path, exist_ok=True)
    vdo_filepath = os.path.join(vdo_path, "generated.mp4")
    qr_vdo.write_videofile(vdo_filepath)
    return [{'path':vdo_filepath}]



if __name__ == "__main__":
    img_path = "./share_vol/test/img/00001.png" # "./ext_lib/articulated_motion/SadTalker/examples/source_image/art_0.png"
    ref_vdo = "./ext_lib/articulated_motion/SadTalker/examples/ref_video/WDA_AlexandriaOcasioCortez_000.mp4"
    op_fld = "./share_vol/test/"
    theme = "sdxl_base" #"people"
    # gen_zoom_video(img_path, theme, "hello", op_fld, zoom_in=True, num_sec=4, fps=1, vdo_wth=1280, vdo_hgt=720, 
    #     scale=1.6, pos_x='right', pos_y='bottom', is_gif=False)
    gen_pan_video(img_path, theme, "cyber realistic girl ", op_fld, template="left",  num_sec=4, fps=1, vdo_wth=1920, vdo_hgt=1080, 
        pan_speed=50, is_gif=False)
    vdo_file_lst = ["./share_vol/test/stg1/mrg/generated.mp4", "./share_vol/test/vdo/mrg/generated.mp4"]
    # concat_vdo(vdo_file_lst, op_fld)

    # vdo_path = "./share_vol/test/stg1/mrg/generated.mp4"
    vdo_path = "./share_vol/test/vdo/mrg/generated.mp4"
    # add_img_to_vdo(vdo_path, img_path, op_fld, pos_x="right", pos_y="center", scale=0.5)

