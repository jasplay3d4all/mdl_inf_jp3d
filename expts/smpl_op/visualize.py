# from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
import torch
import numpy as np
import body_model as bm
import utils
import specs
import draw 


def load_result(res_path_dict):
    """
    load all saved results for a given iteration
    :param res_path_dict (dict) paths to relevant results
    returns dict of results
    """
    res_dict = {}
    for name, path in res_path_dict.items():
        res = np.load(path)
        res_dict[name] = to_torch({k: res[k] for k in res.files})
    return res_dict

def to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        return {k: to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_torch(x) for x in obj]
    return obj

def reproject(points3d, cam_R, cam_t, cam_f, cam_center):
    """
    reproject points3d into the scene cameras
    :param points3d (B, T, N, 3)
    :param cam_R (B, T, 3, 3)
    :param cam_t (B, T, 3)
    :param cam_f (T, 2)
    :param cam_center (T, 2)
    """
    B, T, N, _ = points3d.shape
    points3d = torch.einsum("btij,btnj->btni", cam_R, points3d)
    points3d = points3d + cam_t[..., None, :]  # (B, T, N, 3)
    points2d = points3d[..., :2] / points3d[..., 2:3]
    print(points2d.shape, cam_f.shape, cam_center.shape)
    points2d = cam_f * points2d + cam_center
    return points2d

path_to_pose = "../../slahmr/outputs/logs/video-val/2023-05-30/output0001-all-shot-0-0-180/motion_chunks/output0001_000400_world_results.npz"

res1 = np.load(path_to_pose)
print(dir(res1), res1.files)
res = {}
for x in res1.files:
    res[x] = res1[x]

# print(res['betas'].size, res['pose_body'].shape, isinstance(res, dict))
B, T, _ = res['trans'].shape

bdy_mdl = bm.BodyModel(bm_path="./models/smplh/neutral/model.npz", batch_size=B * T, num_betas=16, use_vtx_selector = True)
res = to_torch(res)
smpl2op_map = specs.smpl_to_openpose(
            bdy_mdl.model_type,
            use_hands=True,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco19",
        )
output = utils.run_smpl(bdy_mdl, res['trans'], res['root_orient'], res['pose_body'], betas=res['betas'])
joints3d = output['joints']
joints3d_op = joints3d[:, :, smpl2op_map, :]

print(output.keys(), output['joints'].shape, joints3d_op.shape, res['intrins'],
    res['cam_R'].shape, res['cam_t'].shape)
num_seq = joints3d_op.shape[1]
num_pep = joints3d_op.shape[0]
cam_f = res['intrins'][:2]
cam_center = res['intrins'][2:]
points2d = reproject(joints3d_op, res['cam_R'], res['cam_t'], cam_f, cam_center)
select = [i for i in range(61)]
select.remove(2)
points2d = points2d.detach().numpy()[:, :, select, :]
for i in range(num_seq):
    canvas = np.zeros((1920, 1080, 3))
    for j in range(num_pep):
        # points3d = joints3d_op[j,i]
        # points2d = points2d[]
        canvas = draw.draw_bodypose(canvas, points2d[j,i], range(18))






