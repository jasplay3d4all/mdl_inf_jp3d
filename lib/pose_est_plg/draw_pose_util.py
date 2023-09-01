import math
import numpy as np
import matplotlib
import cv2


eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def draw_bodypose(canvas, smplx_joint_proj_lst):
    H, W, C = canvas.shape
    # Mapping from smplx to openpose
    # https://github.com/vchoutas/smplify-x/blob/3e11ff1daed20c88cd00239abf5b9fc7ba856bb6/smplifyx/utils.py#L96
    # Mapping from smplx to osx - human_models.py in OSX - common/utils/human_models.py
    # Image mapping for the same - https://github.com/vchoutas/smplify-x/issues/152#issuecomment-923715702
    # Extract the coco18 openpose joints from input pose
    # The smplx to 2D pose mapping by OSX: [0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55,]
    # Corresponding array idx              [0,1,2,3,4,5,6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,]
    # The smplx to coco19 joint mapping:  [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59,]
    # So mapping from OSX format to coco19
    #                                     [24,  7,  9, 11, 13,  8, 10, 12, 0, 2, 4, 6, 1, 3, 5, 23, 22, 21, 20,]
    # 
    # OSX format to coco25/openpose25: [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
    #                                  [24,  7,  9, 11, 13,  8, 10, 12, 0, 2, 4, 6, 1, 3, 5, 23, 22, 21, 20, 14, 15, 16, 17, 18, 19]
    #                                  [ 0,  1,  2,  3,  4,  5,  6,  7, 8, 9,10,11,12,13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # OSX format to coco18/openpose18: 
    # So mapping from OSX format to openpose https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1560
    # Mapping with and without neck https://stackoverflow.com/questions/56172454/conversion-between-keypoints-coco-and-open-pose
    # Finding the mapping using image: [ 0,  1,  2,  3,  4,  5,  6,  7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # Hand mapping :
    # [20, 37] - smplx to OSX is same as openpose mapping except it does not have 20
    # So the hand mapping is 20 -> 12 [12, 25-45]
    # Similarly for right hand [21, 52] and so 21 -> 13 [13, 45-65]

    # Face since it it contour 
    # [65-137] face contours - 72 points by OSX - 70 by openpose

    # How to deal with occlusion of parts?
    osx2op25_map = np.array([24,  7,  9, 11, 13,  8, 10, 12, 0, 2, 4, 6, 1, 3, 5, 23, 22, 21, 20, 14, 15, 16, 17, 18, 19])
    op25toop18_map = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    body_joints_lst = smplx_joint_proj_lst[:, osx2op25_map, :][:, op25toop18_map, :]
    # print("Input pose shape ", smplx_joint_proj_lst.shape, body_joints_lst.shape)
    # candidate = np.array(candidate)
    # subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for body_joints in body_joints_lst:
            # index = subset[n][np.array(limbSeq[i]) - 1]
            # if -1 in index:
            #     continue
            index = np.array(limbSeq[i]).astype(int) - 1
            Y = body_joints[index, 0] * float(W)
            X = body_joints[index, 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        # for n in range(len(subset)):
        for body_joints in body_joints_lst:
            # index = int(subset[n][i])
            # if index == -1:
            #     continue
            x, y = body_joints[i][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, smplx_joint_proj_lst):
    H, W, C = canvas.shape


    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    # So the hand mapping is 20 -> 12 [12, 25-45]
    # Similarly for right hand [21, 52] and so 21 -> 13 [13, 45-65]
    osx2lfthnd_map = np.array([12] + [i for i in range(25, 45)])
    os2rgthnd_map = np.array([13] + [i for i in range(45, 65)])
    osx2hnd_map_lst = [osx2lfthnd_map, os2rgthnd_map]
    for smplx_joint_proj in smplx_joint_proj_lst:
        # peaks = np.array(peaks)
        for osx2hnd_map in osx2hnd_map_lst:
            smpl_hnd_joint = smplx_joint_proj[osx2hnd_map, :]

            for ie, e in enumerate(edges):
                x1, y1 = smpl_hnd_joint[e[0]]
                x2, y2 = smpl_hnd_joint[e[1]]
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)
                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

            for i, keyponit in enumerate(smpl_hnd_joint):
                x, y = keyponit
                x = int(x * W)
                y = int(y * H)
                if x > eps and y > eps:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, smplx_joint_proj_lst):
    H, W, C = canvas.shape
    # [65-137] face contours - 72 points by OSX - 70 by openpose
    osx2face_map = np.array([i for i in range(65, 137)])
    smplx_joint_proj_lst = smplx_joint_proj_lst[:, osx2face_map, :]
    for smplx_joint_proj in smplx_joint_proj_lst:
        for lmk in smplx_joint_proj:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def draw_pose(smplx_joint_proj_lst, H, W, draw_body=True, draw_hand=True, draw_face=True):
    # bodies = pose['bodies']
    # faces = pose['faces']
    # hands = pose['hands']
    # candidate = bodies['candidate']
    # subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    smplx_joint_proj_lst[:,:,0] = (((smplx_joint_proj_lst[:,:,0] * 192/12) -  96) + 192)/384 #/5000
    smplx_joint_proj_lst[:,:,1] = (((smplx_joint_proj_lst[:,:,1] * 256/16) - 128) + 256)/512 #/5000
    print("Body joints ", smplx_joint_proj_lst)


    if draw_body:
        canvas = draw_bodypose(canvas, smplx_joint_proj_lst)

    if draw_hand:
        canvas = draw_handpose(canvas, smplx_joint_proj_lst)

    if draw_face:
        canvas = draw_facepose(canvas, smplx_joint_proj_lst)

    return canvas