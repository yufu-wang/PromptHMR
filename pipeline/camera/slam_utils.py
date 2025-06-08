import numpy as np
import torch
import cv2
from glob import glob
import argparse

# import evo
# from evo.core.trajectory import PoseTrajectory3D
# from evo.core import sync
# import evo.main_ape as main_ape
# from evo.core.metrics import PoseRelation
from torchvision.transforms import Resize

# Some default settings for DROID-SLAM
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="path to image directory")
parser.add_argument("--calib", type=str, help="path to calibration file")
parser.add_argument("--opt_intr", action="store_true", help="activate self-calibration")
parser.add_argument("--camera_model", default="focal", type=str, help="pinhole or mei or focal")
parser.add_argument("--t0", default=0, type=int, help="starting frame")
parser.add_argument("--stride", default=1, type=int, help="frame stride")

# parser.add_argument("--weights", default="data/pretrain/droid.pth")
parser.add_argument("--weights", default="data/pretrained-models/afv2_data/droidcalib.pth")
parser.add_argument("--buffer", type=int, default=512)
parser.add_argument("--image_size", default=[240, 320])
parser.add_argument("--image_size_target", default=[517, 384], help="image width and height; reduce to make inference faster.")
parser.add_argument("--disable_vis", action="store_true")

parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

parser.add_argument("--backend_thresh", type=float, default=22.0)
parser.add_argument("--backend_radius", type=int, default=2)
parser.add_argument("--backend_nms", type=int, default=3)
parser.add_argument("--upsample", action="store_true")
parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
slam_args = parser.parse_args([])
slam_args.stereo = False
slam_args.upsample = True
slam_args.disable_vis = True


def est_calib(images):
    """ Roughly estimate intrinsics by image dimensions """
    # imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = images[0] # cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def get_dimention(imagedir):
    """ Get proper image dimension for DROID """
    if isinstance(imagedir, str):
        imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
        image = cv2.imread(imgfiles[0])
    elif isinstance(imagedir, np.ndarray):
        image = imagedir

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    H, W, _ = image.shape
    return H, W


def image_stream(images, calib, stride=1, max_frame=None, depths=None, depth_keyframes=None):
    """ Image generator for DROID """
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # image_list = sorted(glob(f'{imagedir}/*.jpg'))
    images = images[::stride]
    if max_frame is not None:
        images = images[:max_frame]

    for t, img in enumerate(images):
        image = img[..., ::-1]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        
        if depths is None:
            depth = None
        else:
            if t in depth_keyframes:
                depth_idx = np.where(depth_keyframes == t)[0][0]
                depth = depths[depth_idx]
                depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_CUBIC)
                depth = depth[:h1-h1%8, :w1-w1%8]
                depth = torch.from_numpy(depth).float()
            else:
                depth = None

        # h1, w1 = (image.shape[1], image.shape[2])
        # size_factor = [(w1 / w0), (h1 / h0)]
        # intrinsics[0::2] *= size_factor[0]
        # intrinsics[1::2] *= size_factor[1]
        size_factor = [(w1 / w0), (h1 / h0)]
        
        yield t, image[None], intrinsics, depth, size_factor


def preprocess_masks(images, masks):
    """ Resize masks for masked droid """
    H, W = get_dimention(images[0])
    resize_1 = Resize((H, W), antialias=True)
    resize_2 = Resize((H//8, W//8), antialias=True)
    
    img_msks = []
    for i in range(0, len(masks), 500):
        m = resize_1(masks[i:i+500])
        img_msks.append(m)
    img_msks = torch.cat(img_msks)

    conf_msks = []
    for i in range(0, len(masks), 500):
        m = resize_2(masks[i:i+500])
        conf_msks.append(m)
    conf_msks = torch.cat(conf_msks)

    return img_msks, conf_msks

# def eval_slam(traj_est, cam_t, cam_q, return_traj=True, correct_scale=False, align=True, align_origin=False):
#     """ Evaluation for SLAM """
#     tstamps = np.array([i for i in range(len(traj_est))], dtype=np.float32)

#     traj_est = PoseTrajectory3D(
#         positions_xyz=traj_est[:,:3], 
#         orientations_quat_wxyz=traj_est[:,3:],
#         timestamps=tstamps)

#     traj_ref = PoseTrajectory3D(
#         positions_xyz=cam_t.copy(),
#         orientations_quat_wxyz=cam_q.copy(),
#         timestamps=tstamps)

#     traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
#     result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
#         pose_relation=PoseRelation.translation_part, align=align, align_origin=align_origin,
#         correct_scale=correct_scale)
    
#     stats = result.stats

#     if return_traj:
#         return stats, traj_ref, traj_est
    
#     return stats
