
import os
import sys
from PIL import Image
import  numpy as np
import time
import cv2

ctrlnet_path_for_plugin="./ext_lib/ControlNet-v1-1-nightly/"
# Add controlnet related path
sys.path.append(ctrlnet_path_for_plugin)  # Adds the parent directory to 

# from annotator.util import resize_image, HWC3
# from annotator.openpose import OpenposeDetector
# from annotator.midas import MidasDetector
# from annotator.zoe import ZoeDetector
# from annotator.hed import HEDdetector
# from annotator.pidinet import PidiNetDetector
# from annotator.canny import CannyDetector

from controlnet_aux import CannyDetector, ZoeDetector, HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector
# ContentShuffleDetector, , MediapipeFaceDetector, SamDetector, LeresDetector


def get_img_lst(in_fld_path):
    img_file_lst = os.listdir(in_fld_path)
    img_file_lst.sort()
    img_file_lst = [ x for x in img_file_lst if x.endswith(".png") ]

    return img_file_lst

anno_to_model = {
    "openpose": OpenposeDetector,
    "midasdepth": MidasDetector,
    "zoedepth":ZoeDetector,
    "hededge":HEDdetector,
    "pidiedge":PidiNetDetector,
    "canny":CannyDetector
}
def resize_image(input_image, resolution, is_min=True):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    if(is_min):
        k = float(resolution) / min(H, W)
    else:
        k = float(resolution) / max(H, W)
    
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)

    return img

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def gen_ctrl_img(img_pth, anno_type, safe=False, res=512):
        
    model = anno_to_model[anno_type]()
    img = np.asarray(Image.open(img_pth))

    if(img.dtype != np.uint8):
        print("Img input not supported ", img.dtype, img.shape)
        exit(-1)

    if(anno_type == "pidiedge" or anno_type == "hededge"):
        img = HWC3(model(HWC3(img), safe=safe))
    else:
        img = HWC3(model(HWC3(img)))
    img = resize_image(img, res, is_min=False)

    H, W, C = img.shape

    result = np.zeros((res,res,3))
    if(W == res):
        off_hgt = int((res - H)/2)
        result[off_hgt:off_hgt+H, :, :] = img
    else:
        off_wth = int((res - W)/2)
        result[:,off_wth:off_wth+W, :] = img
    return result/255.0

def execute_all_images(model, ip_fld, op_fld, res=512):

    os.makedirs(op_fld, exist_ok=True)

    for img_file in get_img_lst(ip_fld):
        # check if the image ends with png
        print("img ",img_file)
        img_pth = os.path.join(ip_fld, img_file)
        img = np.asarray(Image.open(img_pth))
        H, W, C = img.shape

        start = time.time()
        img = resize_image(HWC3(img), res)
        result = cv2.resize(HWC3(model(img)), (W, H), interpolation=cv2.INTER_LINEAR)
        # print("Total time ", time.time() - start, result.shape) 

        op_pth = os.path.join(op_fld, img_file)
        Image.fromarray(result).save(op_pth)

    return


def gen_pose(ip_fld, op_fld, res=512, hand_and_face=True):
    model_openpose = OpenposeDetector()
    os.makedirs(op_fld, exist_ok=True)

    for img_file in get_img_lst(ip_fld):
        # check if the image ends with png
        print("img ", img_file)
        img_pth = os.path.join(ip_fld, img_file)
        img = np.asarray(Image.open(img_pth))
        img = resize_image(HWC3(img), res)

        start = time.time()
        result = model_openpose(img, hand_and_face)
        # print("Total time ", time.time() - start) 
        op_pth = os.path.join(op_fld, img_file)
        Image.fromarray(result).save(op_pth)
    return 

def gen_depth(ip_fld, op_fld, res=512, is_midas=True):

    if(is_midas):
        model_depth = MidasDetector()
    else:
        model_depth = ZoeDetector()
    
    execute_all_images(model_depth, ip_fld, op_fld, res=512)
    return

def gen_softedge(ip_fld, op_fld, res=512, det='PIDI_safe'):
    if 'HED' in det:
        model = HEDdetector()
    if 'PIDI' in det:
        model = PidiNetDetector()
    def run_model(input):
        return model(input, safe='safe' in det)
    execute_all_images(run_model, ip_fld, op_fld, res=512)
    return
