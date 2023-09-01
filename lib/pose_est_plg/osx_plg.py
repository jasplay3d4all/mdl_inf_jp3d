import os
# os.environ["PYOPENGL_PLATFORM"] = 'osmesa'

import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch

import cv2

# # Add paths using absolute 
# sys.path.insert(0, osp.join('../pose_est_3d/OSX/main'))
# sys.path.insert(0, osp.join('../pose_est_3d/OSX/data'))
# from config import cfg
# from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
# # from common.utils.vis import render_mesh, save_obj
# from common.utils.human_models import smpl_x

from .draw_pose_util import draw_pose
from PIL import Image

# sys.path.insert(0, osp.join('../'))
# from pose_est_3d.OSX import cfg, Demoer, load_img, process_bbox, generate_patch_image, render_mesh, save_obj, smpl_x


def get_img_lst(in_fld_path):
    img_file_lst = os.listdir(in_fld_path)
    img_file_lst.sort()
    img_file_lst = [ x for x in img_file_lst if x.endswith(".png") ]

    return img_file_lst

class osx_pose_est:
    def __init__(self):
        return
        gpu_ids = '0'
        encoder_setting = 'osx_l'
        decoder_setting = 'wo_decoder'
        pretrained_model_path = '../models/osx/osx_l_wo_decoder.pth.tar'
        # parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
        # parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
        # parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')

        cfg.set_args(gpu_ids)
        cudnn.benchmark = True
        cfg.set_additional_args(encoder_setting=encoder_setting, decoder_setting=decoder_setting, \
            pretrained_model_path=pretrained_model_path)

        from common.base import Demoer
        demoer = Demoer()
        demoer._make_model()
        assert osp.exists(pretrained_model_path), 'Cannot find model at ' + pretrained_model_path
        print('Load checkpoint from {}'.format(pretrained_model_path))

        demoer.model.eval()

        self.demoer = demoer# .model.eval()
        self.transform = transforms.ToTensor()

        # detect human bbox with yolov5s
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return

    def detect_smplx(self, img_path):
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]

        with torch.no_grad():
            results = self.detector(original_img)
        person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        class_ids, confidences, boxes = [], [], []
        for detection in person_results:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        out_list = []
        in_img_list = []
        for num, indice in enumerate(indices):
            bbox = boxes[indice]  # x,y,h,w
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = self.transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = self.demoer.model(inputs, targets, meta_info, 'test')
            # Pose details: 25 (Body Joints), 20+20 (left and right hand joints), 70 (face joints) - 135 joints
            # print("Output ", len(indices), out['img'].shape, out['joint_img'].shape, out['smplx_joint_proj'].shape)
            numpy_out = {}
            in_img = None
            for key in out.keys():
                # print("Output ", key, out[key].cpu().squeeze().detach().numpy().shape)
                if (key == 'img'):
                    in_img = out[key].cpu().squeeze().detach().numpy()
                else:
                    numpy_out[key] = out[key].cpu().squeeze().detach().numpy()
            # ['img', 'joint_img', 'smplx_joint_proj', 'smplx_mesh_cam', 'smplx_root_pose', 'smplx_body_pose', 
            # 'smplx_lhand_pose', 'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans', 
            # 'lhand_bbox', 'rhand_bbox', 'face_bbox']
            out_list.append(numpy_out)
            in_img_list.append(in_img)

        return out_list, in_img_list

    def render_mesh(self, original_img, out_list):
        vis_img = original_img.copy()
        for num, out in enumerate(out_list):
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
            mesh = mesh[0]

            # save mesh
            save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'person_{num}.obj'))

            # render mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
        return vis_img

    def gen_pose_lst(self, in_fld_path, op_fld_path):
        os.makedirs(op_fld_path, exist_ok=True)

        in_file_lst = get_img_lst(in_fld_path)
        for idx, img_file in enumerate(in_file_lst):
            img_file_path = os.path.join(in_fld_path, img_file)
            print("Img file ", img_file)
            out_list, in_img_list = self.detect_smplx(img_file_path)

            op_file = "output"+str(idx+1).zfill(4)+".npz"
            op_file_path = os.path.join(op_fld_path, op_file)
            np.savez(op_file_path,out_list)

            op_file = "in_img"+str(idx+1).zfill(4)+".npz"
            op_file_path = os.path.join(op_fld_path, op_file)
            np.savez(op_file_path,in_img_list)

    def draw_pose_lst(self, work_folder, op_fld_path, res=512):
        os.makedirs(op_fld_path, exist_ok=True)

        smplx_fld_path = os.path.join(work_folder, "smplxpose")
        smplx_file_lst = os.listdir(smplx_fld_path)
        smplx_file_lst.sort()
        smplx_file_lst = [ x for x in smplx_file_lst if x.startswith("output") ]

        for idx, smplx_file in enumerate(smplx_file_lst):
            smplx_file_path = os.path.join(smplx_fld_path, smplx_file)
            smplx_data_lst = np.load(smplx_file_path, allow_pickle=True)['arr_0'] # .files
            smplx_joint_proj_lst = []
            for smplx_data in smplx_data_lst:
                smplx_joint_proj_lst.append(smplx_data["smplx_joint_proj"])
            smplx_joint_proj_lst = np.array(smplx_joint_proj_lst)
            print("SMPLX file ", smplx_file, smplx_joint_proj_lst.shape, smplx_data['cam_trans'])
            canvas_np = draw_pose(smplx_joint_proj_lst, res, res, \
                draw_body=True, draw_hand=True, draw_face=True)
            

            op_file = "output"+str(idx+1).zfill(4)+".png"
            op_file_path = os.path.join(op_fld_path, op_file)
            print("Output img ", np.min(canvas_np), np.max(canvas_np))
            canvas = Image.fromarray(canvas_np)
            canvas.save(op_file_path)


        
# # load model
# model_path = args.pretrained_model_path

# original_img = load_img(args.img_path)
# original_img_height, original_img_width = original_img.shape[:2]
# os.makedirs(args.output_folder, exist_ok=True)

# # save rendered image
# cv2.imwrite(os.path.join(args.output_folder, f'render.jpg'), vis_img[:, :, ::-1])

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
#     parser.add_argument('--img_path', type=str, default='input.png')
#     parser.add_argument('--output_folder', type=str, default='output')
#     parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
#     parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
#     parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
#     args = parser.parse_args()

#     # test gpus
#     if not args.gpu_ids:
#         assert 0, print("Please set proper gpu ids")

#     if '-' in args.gpu_ids:
#         gpus = args.gpu_ids.split('-')
#         gpus[0] = int(gpus[0])
#         gpus[1] = int(gpus[1]) + 1
#         args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
#     return args

# args = parse_args()
# cfg.set_args(args.gpu_ids)
# cudnn.benchmark = True
