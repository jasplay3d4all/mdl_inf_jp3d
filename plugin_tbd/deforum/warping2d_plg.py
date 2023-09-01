import numpy as np
from PIL import Image
import ffmpegio
import os

def generate_mask(img, inp_pos_x, inp_pos_y, inp_with_x, inp_hgt_y, op_wth, op_hgt):
    # print("Input image ", img.shape, np.max(img))

    msk_img = -1*np.ones((1, op_hgt, op_wth, 3))
    msk_img[0, inp_pos_y:inp_pos_y+inp_hgt_y, inp_pos_x:inp_pos_x+inp_with_x, :] = img
    return msk_img

def warp2d_mask_gen(img_path, motion_template="pan_right", vdo_wth=1280, vdo_hgt=720, num_sec=10, fps=24):
    off_x = 1; off_y = 1; 
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    img = np.array(img)/255.0
    inp_with_x = img.shape[1]; inp_hgt_y = img.shape[0]

    # Calculate offsets based on motion
    if(motion_template == "pan_right"):
        inp_pos_x = int((vdo_wth - inp_with_x)/2)
        inp_pos_y = int((vdo_hgt - inp_hgt_y)/2)
        op_wth = vdo_wth + num_sec*fps*off_x
        op_hgt = vdo_hgt #+ num_sec*fps 
    if(motion_template == "pan_left"):
        inp_pos_x = int((vdo_wth - inp_with_x)/2) + num_sec*fps
        inp_pos_y = int((vdo_hgt - inp_hgt_y)/2)
        op_wth = vdo_wth + num_sec*fps*off_x
        op_hgt = vdo_hgt #+ num_sec*fps 
    if(motion_template == "pan_bottom_right"):
        inp_pos_x = int((vdo_wth - inp_with_x)/2)
        inp_pos_y = int((vdo_hgt - inp_hgt_y)/2)
        op_wth = vdo_wth + num_sec*fps*off_x
        op_hgt = vdo_hgt + num_sec*fps*off_y
    if(motion_template == "pan_top_left"):
        inp_pos_x = num_sec*fps + int((vdo_wth - inp_with_x)/2)
        inp_pos_y = num_sec*fps + int((vdo_hgt - inp_hgt_y)/2)
        op_wth = vdo_wth + num_sec*fps*off_x 
        op_hgt = vdo_hgt + num_sec*fps*off_y 

    # Read input image
    # Generate the mask based on offsets
    img_mask = generate_mask(img, inp_pos_x, inp_pos_y, inp_with_x, inp_hgt_y, op_wth, op_hgt)
    return img_mask

def gen_vdo(img_path, motion_template, vdo_path, num_sec=1, fps=24):

    img = np.array(Image.open(img_path).convert("RGB"))
    if(motion_template == "pan_right"):
        start_pos_x = 0; start_pos_y = 0; off_x = 1; off_y = 0; vdo_wth=1280; vdo_hgt=720;
    if(motion_template == "pan_left"):
        off_x = -1; off_y = 0; vdo_wth=1280; vdo_hgt=720;
        start_pos_x = -1*num_sec*fps*off_x; start_pos_y = 0; 
    if(motion_template == "pan_bottom_right"):
        start_pos_x = 0; start_pos_y = 0; off_x = 1; off_y = 1; vdo_wth=1280; vdo_hgt=720;
    if(motion_template == "pan_top_left"):
        off_x = -1; off_y = -1; vdo_wth=1280; vdo_hgt=720;
        start_pos_x = -1*num_sec*fps*off_x; start_pos_y = -1*num_sec*fps*off_y;         

    # os.makedirs(img_lst_path, exist_ok=True)
    os.makedirs(vdo_path, exist_ok=True)
    vdo_filepath = os.path.join(vdo_path, "generated.mp4")

    img_lst = np.zeros((num_sec*fps, vdo_hgt, vdo_wth, 3)).astype(np.uint8)
    for i in range(num_sec*fps):
        img_lst[i,:,:,:] = img[start_pos_y+i*off_y:start_pos_y+vdo_hgt+i*off_y, 
                            start_pos_x+i*off_x:start_pos_x+vdo_wth+i*off_x, :]
        # img_extract = Image.fromarray(img_extract)
        # img_path = os.path.join(img_lst_path, "output"+str(i).zfill(5)+".png")
        # img_extract.save(img_path)
        # print("Op ", img_path, img_extract.size)
    print("Vdo min and max ", np.max(img_lst))
    ffmpegio.video.write(vdo_filepath, fps, img_lst, overwrite=True) #, pix_fmt_in='yuv420p')

    # with ffmpegio.open(vdo_filepath, 'wv', rate=fps) as fout:
    #     for i in range(num_sec*fps):
    #         img_extract = img[start_pos_y+i*off_y:start_pos_y+vdo_hgt+i*off_y, 
    #                           start_pos_x+i*off_x:start_pos_x+vdo_wth+i*off_x, :]
    #         print("Op ", vdo_path, img_extract.size)
    #         fout.write(img_extract)
    return {'path':vdo_filepath}

if __name__ == "__main__":

    audio_gen_mdl = audio_gen("melody")
    bg_music_prompt = ["Drum beats"]
    # audio_gen_mdl.gen_music(bg_music_prompt, 2, "./output")
    audio_gen_mdl.gen_speech("Mighty jungle in mighty jungle ", "./output")
