import os
import sys
from PIL import Image
import  numpy as np
import time
import cv2

ctrlnet_path_for_plugin="../ControlNet-v1-1-nightly/"
# Add controlnet related path
sys.path.append(ctrlnet_path_for_plugin)  # Adds the parent directory to 

from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.zoe import ZoeDetector
from annotator.hed import HEDdetector
from annotator.pidinet import PidiNetDetector


def get_img_lst(in_fld_path):
    img_file_lst = os.listdir(in_fld_path)
    img_file_lst.sort()
    img_file_lst = [ x for x in img_file_lst if x.endswith(".png") ]

    return img_file_lst

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

    