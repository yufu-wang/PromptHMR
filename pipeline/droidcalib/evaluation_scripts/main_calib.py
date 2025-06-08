# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
sys.path.append('droid_slam')
sys.path.append('evaluation_scripts')
from tqdm import tqdm
import numpy as np
import torch
import os
import glob 
import time

import torch.nn.functional as F
import numpy as np
import random
import torch
from droid import Droid
from types import SimpleNamespace
from geom import mapping_error as ma

from image_stream import image_stream
from trajectory_eval import compute_ape, compute_ape_tartan
from video_calib import rescale_intr


def main(args):
    # ======================================================================================== #
    # run droid slam on image stream
    # ======================================================================================== #

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    size_factor = [(args.rectified_size[0]+2*args.crop[0])/args.original_size[0], 
                   (args.rectified_size[1]+2*args.crop[1])/args.original_size[1]] 

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)
    time.sleep(5)

    start_time = time.time()
    tstamps = []

    for vals in tqdm(image_stream(**vars(args))):

        (t, image, intr, _) = vals
        droid.track(t, image, intrinsics=intr)


    if args.num_images is not None:
        args.num_images = args.stride * args.num_images

    # for evaluation we always use all images
    args.stride = 1 
    traj_est, intr_est = droid.terminate(image_stream(**vars(args)))

    runtime = (time.time() - start_time)
    
    # ======================================================================================== #
    # postprocessing
    # ======================================================================================== #

    intr = rescale_intr(intr_est, size_factor, args.crop)

    print('Estimated intrinsics: ', intr)
    if args.undistort:
        print('GT intrinsics: ', args.rectified_intr)
    else:
        print('GT intrinsics: ', args.original_intr[:6])
    print('Elapsed time: ', runtime)
    
    # ======================================================================================== #
    # trajectory error
    # ======================================================================================== #

    # get actual number of images
    num_images = len(sorted(glob.glob(os.path.join(args.datapath, '*.png')))[
                                            args.start_index::args.stride][:args.num_images])
    
    if len(args.trajectory_gt) > 0:

        if "Tartan" in args.trajectory_gt:
            ape, n_reg_images = compute_ape_tartan(traj_est, args.trajectory_gt, num_images, 
                                                   args.stride)
        else:
            ape, n_reg_images = compute_ape(traj_est, args.trajectory_gt, num_images, args.stride, 
                                            args.datapath)
    else:
        ape, n_reg_images = np.nan, np.nan

    # ======================================================================================== #
    # mapping error
    # ======================================================================================== #
    if args.undistort:
        m_error = ma.mapping_error(intr, args.rectified_intr, args.original_size)
    else:
        m_error = ma.mapping_error(intr, args.original_intr[:6], args.original_size)

    print("mapping error: ", m_error)
     
    return intr, m_error, runtime, ape, n_reg_images



def run_internal(seq, datapath, original_intr, rectified_intr, original_size, 
                 rectified_size, undistort, intr_error, crop, num_images, stride, 
                 camera_model, weights, opt_intr, cuda, trajectory_gt, droid_args, 
                 seed, start_index):
    
    args = SimpleNamespace()

    # fixed dataset specs
    args.seq = seq
    args.datapath = datapath
    args.original_intr = original_intr
    args.rectified_intr = rectified_intr
    args.original_size = original_size
    args.trajectory_gt = trajectory_gt

    # variable analysis params
    args.undistort = undistort
    args.rectified_size = rectified_size
    args.intr_error = intr_error
    args.crop = crop
    args.num_images = num_images
    args.stride = stride
    args.start_index = start_index
    args.camera_model = camera_model
    args.weights = weights
    args.opt_intr = opt_intr
    args.cuda = cuda
    
    # fixed for calib analyses
    args.depth = False
    args.stereo = False
    args.disable_vis = True
    args.plot_curve = False
    args.id = -1

    # untouched from droid slam
    args.buffer = droid_args['buffer']
    args.beta = droid_args['beta']
    args.filter_thresh = droid_args['filter_thresh']
    args.warmup = droid_args['warmup']
    args.keyframe_thresh = droid_args['keyframe_thresh']
    args.frontend_thresh = droid_args['frontend_thresh']
    args.frontend_window = droid_args['frontend_window']
    args.frontend_radius = droid_args['frontend_radius']
    args.frontend_nms = droid_args['frontend_nms']
    args.backend_thresh = droid_args['backend_thresh']
    args.backend_radius = droid_args['backend_radius']
    args.backend_nms = droid_args['backend_nms']
    args.upsample = False

    args.seed = seed
    args.image_size = [rectified_size[1], rectified_size[0]]  # droid slam needs order [h, w]

    return main(args)