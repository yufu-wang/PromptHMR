import os
from pathlib import Path
import time
from tqdm import tqdm
from contextvars import ContextVar
tqdm_disabled = ContextVar("tqdm_disabled", default=False)

import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from torchvision.transforms import Resize

import sys
sys.path.insert(0, 'pipeline/droidcalib/droid_slam')
from droid import Droid
from .depth_utils import prep_metric3d, post_metric3d
from .slam_utils import slam_args, parser
from .slam_utils import get_dimention, est_calib, image_stream, preprocess_masks
from .est_scale import est_scale_hybrid
from ..image_folder import ImageFolder

from prompt_hmr.utils.rotation_conversions import quaternion_to_matrix
from pipeline.yvanyin_metric3d_main.hubconf import (
    metric3d_vit_large, 
    metric3d_vit_small, 
    metric3d_vit_giant2,
)

# torch.multiprocessing.set_start_method('spawn')
def run_metric_slam(
    images, 
    masks=None, 
    calib=None, 
    monodepth_method='zoedepth', 
    use_depth_inp=False,
    stride=1,
    opt_intr=False,
    save_depth=False,
    depth_stride=1,
    keyframes=None,
):
    '''
    Input:
        img_folder: directory that contain image files 
        masks: list or array of 2D masks for human. 
               If None, no masking applied during slam.
        calib: camera intrinsics [fx, fy, cx, cy]. 
               If None, will be naively estimated.
        depth_method: str, depth estimation method.
                options= ['zoedepth', 'metric3d_vit_small', 'metric3d_vit_large', 'metric3d_vit_giant2']
    Output:
    '''
    
    ##### Estimate Metric Depth #####
    if keyframes is None:
        keyframes = np.arange(0, images.shape[0], depth_stride)
    else:
        keyframes = np.unique(np.concatenate([keyframes, np.arange(0, images.shape[0], 10)]))
    
    if monodepth_method == 'zoedepth':
        repo = "isl-org/ZoeDepth"
        model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
        _ = model_zoe_n.eval()
        monodepth_model = model_zoe_n.to('cuda')
    elif 'metric3d' in monodepth_method:
        monodepth_model = eval(monodepth_method)(pretrain=True)
        monodepth_model = monodepth_model.cuda()
        # TODO: tensorrt, quantization, and compile
        monodepth_model = monodepth_model.half()
        monodepth_model.eval()
    
    if use_depth_inp:
        pred_depths = []
        H, W = get_dimention(images[0])
        
        # check frames with enough motion
        print(f"Keyframes for depth input: {keyframes}")
        
        if 'metric3d' in monodepth_method:
            from torch.utils.data import DataLoader
            from .depth_utils import prep_metric3d_img, post_metric3d_batch
            
            _, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(images[0], calib, monodepth_method)
            if monodepth_method == 'metric3d_vit_giant2':
                batch_size = 8
            elif monodepth_method == 'metric3d_vit_large':
                batch_size = 8
            elif monodepth_method == 'metric3d_vit_small':
                batch_size = 32
            
            dataloader = DataLoader(ImageFolder(images[keyframes], prep_metric3d_img), 
                                    batch_size=batch_size, shuffle=False, 
                                    num_workers=8 if os.cpu_count() > 8 else os.cpu_count())
            rgb_imgs_np = []
            # for batch in tqdm(dataloader, desc='Metric3D', disable=tqdm_disabled.get()):
            for batch in dataloader:
                rgb = batch['img'].cuda().half()
                with torch.inference_mode():
                    pred_depth, confidence, _ = monodepth_model.inference({'input': rgb})
                    
                pred_depth = post_metric3d_batch(pred_depth, confidence, pad_info, rgb_origin, intrinsic_prep)
                pred_depth = pred_depth.cpu().numpy()
                pred_depths += [x for x in pred_depth]
                if save_depth:
                    rgb_imgs_np += [x for x in batch['img'].cpu().numpy()]
                    
        else:
            # for imgf in tqdm(imgfiles, desc=f'Estimating depth using {monodepth_method}', disable=tqdm_disabled.get()):
            for img in images[keyframes]:

                if monodepth_method == 'zoedepth':
                    img = cv2.resize(img, (W, H))
                    img_pil = Image.fromarray(img)
                    pred_depth = model_zoe_n.infer_pil(img_pil)
                
                pred_depths.append(pred_depth)
        
    if opt_intr:
        print("Optimizing intrinsics...")
    
    ##### Masked droid slam #####
    depth_input = pred_depths if use_depth_inp else None
    droid, traj, intr_est, _ = run_slam(images, masks=masks, calib=calib, depths=depth_input, 
                                        opt_intr=opt_intr, depth_keyframes=keyframes, stride=stride)
    
    if opt_intr:
        print("Using optimized intrinsics...")
        calib = intr_est
    
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()
    
    ##### Estimate Metric Scale #####
    if use_depth_inp:
        scale = 1.0
    else:
        pred_depths = []
        H, W = get_dimention(images[0])
        # for t in tqdm(tstamp, desc=f'Estimating scale using {monodepth_method}', disable=tqdm_disabled.get()):
        if monodepth_method == 'zoedepth':
            for t in tstamp:
                img = images[t]
                img = cv2.resize(img, (W, H))
                if monodepth_method == 'zoedepth':
                    img_pil = Image.fromarray(img)
                    pred_depth = model_zoe_n.infer_pil(img_pil)
                pred_depths.append(pred_depth)
        elif 'metric3d' in monodepth_method:
            from torch.utils.data import DataLoader
            from .depth_utils import prep_metric3d_img, post_metric3d_batch
            
            pred_depths = []
            
            kf_imgs = [images[t] for t in tstamp]
            
            print(f"Number of keyframes: {len(kf_imgs)}, number of frames: {len(images)}")
            print(f"keyframe timestamps: {tstamp}")
            _, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(kf_imgs[0], calib, monodepth_method)
            if monodepth_method == 'metric3d_vit_giant2':
                batch_size = 8
            elif monodepth_method == 'metric3d_vit_large':
                batch_size = 8
            elif monodepth_method == 'metric3d_vit_small':
                batch_size = 32
            
            dataloader = DataLoader(ImageFolder(kf_imgs, prep_metric3d_img), 
                                    batch_size=batch_size, shuffle=False, 
                                    num_workers=8 if os.cpu_count() > 8 else os.cpu_count())
            # for batch in tqdm(dataloader, desc=f'Running {monodepth_method}'):
            for batch in dataloader:
                rgb = batch['img'].cuda().half()

                with torch.inference_mode():
                    pred_depth, confidence, _ = monodepth_model.inference({'input': rgb})
                    
                pred_depth = post_metric3d_batch(pred_depth, confidence, pad_info, rgb_origin, 
                                                 intrinsic_prep, resize_shape=(H, W))
                
                if rgb.shape[0] == 1:
                    pred_depth = pred_depth.unsqueeze(0)

                pred_depth = pred_depth.cpu().numpy()
                pred_depths += [x for x in pred_depth]
        else:
            raise ValueError(f"Unknown depth method: {monodepth_method}")
            
        scales_ = []
        n = len(tstamp)   # for each keyframe
        # for i in tqdm(range(n), disable=tqdm_disabled.get()):
        for i in range(n):
            t = tstamp[i]
            disp = disps[i]
            pred_depth = pred_depths[i]
            slam_depth = 1/disp
            
            if masks is None:
                msk = None
            else:
                msk = masks[t].numpy().astype(float)
            try:
                scale = est_scale_hybrid(slam_depth, pred_depth, msk=msk)
            except Exception as e:
                print(f"Error in estimating scale: {e}")
                raise e
                
            scales_.append(scale)
            
        scales_ = np.array(scales_)
        scales_ = scales_[~np.isnan(scales_)]
        scale = np.median(scales_)

    # convert to metric-scale camera extrinsics: R_wc, T_wc
    pred_cam_t = torch.tensor(traj[:, :3]) * scale
    pred_cam_q = torch.tensor(traj[:, 3:])
    pred_cam_r = quaternion_to_matrix(pred_cam_q[:,[3,0,1,2]])

    return pred_cam_r, pred_cam_t, calib
    
    
def run_slam(images, masks=None, calib=None, depths=None, opt_intr=False, stride=1, depth_keyframes=None):
    """ Maksed DROID-SLAM """
    droid = None
    if calib is None:
        calib = est_calib(images)

    if masks is not None:
        masks = masks[::stride]
        img_msks, conf_msks = preprocess_masks(images, masks)

    if opt_intr:
        print("Optimizing intrinsics...")
    
    slam_args.weights = 'data/pretrain/droidcalib.pth'
    slam_args.opt_intr = opt_intr
    for (t, image, intrinsics, depth, size_factor) in image_stream(images, calib, depths=depths, stride=stride, depth_keyframes=depth_keyframes):

        if droid is None:
            slam_args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(slam_args)
        
        if masks is not None:
            img_msk = img_msks[t]
            conf_msk = conf_msks[t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
        else:
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=None)  

    traj, _intr_est = droid.terminate(image_stream(images, calib))

    # rescale intrinsics
    intr_est = _intr_est.copy()
    intr_est[0:4:2] /= size_factor[0]
    intr_est[1:4:2] /= size_factor[1]
    
    print("Initial fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}".format(*calib))
    if slam_args.camera_model == "pinhole" or slam_args.camera_model == "focal":
        print("Optimized fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}".format(*intr_est))
    else:
        print("Optimized fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}, xi = {:.3f}".format(*intr_est))
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    print(f"Number of keyframes: {len(tstamp)}")
    print(f"keyframe timestamps: {tstamp}")
    return droid, traj, intr_est, tstamp


def search_focal_length(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # default estimate
    calib = np.array(est_calib(img_folder))
    best_focal = calib[0]
    best_err = test_slam(img_folder, input_msks, 
                         stride=stride, calib=calib, max_frame=max_frame)
    
    # search based on slam reprojection error
    for focal in range(low, high, step):
        calib[:2] = focal
        err = test_slam(img_folder, input_msks, 
                        stride=stride, calib=calib, max_frame=max_frame)

        if err < best_err:
            best_err = err
            best_focal = focal

    return best_focal


def calibrate_intrinsics(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # default estimate
    calib = np.array(est_calib(img_folder))
    
    best_focal = calib[0]
    best_err = test_slam(img_folder, input_msks, 
                         stride=stride, calib=calib, max_frame=max_frame)
    print(f"Initial focal length: {best_focal}, error: {best_err}")
    # search based on slam reprojection error
    for focal in np.arange(best_focal-(low/2.), best_focal+(high/2.), float(step)):
        calib[:2] = focal
        err = test_slam(img_folder, input_msks, 
                        stride=stride, calib=calib, max_frame=max_frame)
        print(f"Focal length: {focal}, error: {err}")
        if err < best_err:
            print(f"New best focal length: {focal}, error: {err}")
            best_err = err
            best_focal = focal
    
    calib[0] = calib[1] = best_focal
    
    return calib


def test_slam(imagedir, masks, calib, stride=10, max_frame=50):
    """ Shorter SLAM step to test reprojection error """
    args = parser.parse_args([])
    args.stereo = False
    args.upsample = False
    args.disable_vis = True
    args.frontend_window = 10
    args.frontend_thresh = 10
    droid = None

    for (t, image, intrinsics, _, _) in image_stream(imagedir, calib, stride, max_frame):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        if masks is not None:
            img_msk = masks[0][t]
            conf_msk = masks[1][t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, mask=conf_msk)  
        else:
            droid.track(t, image, intrinsics=intrinsics, mask=None)  

    reprojection_error = droid.compute_error()
    del droid

    return reprojection_error
