# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
import torch
import os
import numpy
from PIL import Image
import numpy as np

mmdet_model_path="../models/MMDet"

def gen_pan_seg_img(
    in_fld_path,
    op_fld_path,
    device="cuda",
    config_path="configs/panoptic_fpn/panoptic-fpn_r101_fpn_ms-3x_coco.py", 
    model_path="panoptic_fpn_r101_fpn_mstrain_3x_coco_20210823_114712-9c99acc4.pth"):

    os.makedirs(op_fld_path, exist_ok=True)

    config_path = os.path.join(mmdet_model_path, config_path)
    model_path = os.path.join(mmdet_model_path, model_path)
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, model_path, device=device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    # video_reader = mmcv.VideoReader(video_path)
    # print("Video length ", len(video_reader),
    #     video_path)

    img_file_lst = os.listdir(in_fld_path)
    img_file_lst.sort()
    img_file_lst = [ x for x in img_file_lst if x.endswith(".png") ]



    # for idx, frame in enumerate((video_reader)):
    for idx, img_filename in enumerate(track_iter_progress(img_file_lst)):
        img_path = os.path.join(in_fld_path, img_filename)
        frame = mmcv.imread(img_path)
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        # print("Results ", result.gt_instances.keys())
        # print(result.pred_panoptic_seg.sem_seg.shape)
        # print(torch.min(result.pred_panoptic_seg.sem_seg))
        # print(torch.max(result.pred_panoptic_seg.sem_seg))
        # print(result.gt_instances.labels.shape)
        # print(result.gt_instances.bboxes.shape)
        sem_seg = result.pred_panoptic_seg.sem_seg.detach().cpu().squeeze()
        # print(sem_seg.shape)

        # Currently supporting only person segmentation
        sem_seg_np = np.zeros(sem_seg.shape, np.uint8)
        # pan_id = ins_id * INSTANCE_OFFSET + cat_id
        # ins_id = ??, cat_id = 0, INSTANCE_OFFSET = 1000
        # https://mmdetection.readthedocs.io/en/v2.15.1/_modules/mmdet/datasets/coco_panoptic.html
        # https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json 
        sem_seg_np[sem_seg % 1000 == 0] = 255
        # print(torch.sum(sem_seg % 1000 == 0))
        sem_seg_np = Image.fromarray(sem_seg_np)
        seg_file_path = os.path.join(op_fld_path, img_filename) #"output"+str(idx+1).zfill(4)+".png")
        sem_seg_np.save(seg_file_path)


    return 

