# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

import torch.nn.functional as F
import numpy as np
import random
import torch
from droid import Droid
from types import SimpleNamespace


def image_stream(datapath, original_intr, rectified_intr, original_size, 
                rectified_size, crop=[0, 0], num_images=None, stride=1, start_index=0,
                camera_model='pinhole', intr_error=0, seed=0, undistort=True, **kwargs):

    """ image generator, using only one camera """

    fx, fy, cx, cy = original_intr[:4]
    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array(original_intr[4:])

    size_factor = [(rectified_size[0]+2*crop[0])/original_size[0], 
                   (rectified_size[1]+2*crop[1])/original_size[1]] # [w, h]

    # read all png images in folder
    images_list = sorted(glob.glob(os.path.join(datapath, '*.png')))[start_index::stride][:num_images]

    for t, imfile in enumerate(images_list):

        # rectify, resive and crop images
        image = cv2.imread(imfile)

        if undistort:
            image = cv2.undistort(image, K_l, d_l)

        image = cv2.resize(image, (rectified_size[0]+2*crop[0], 
                                   rectified_size[1]+2*crop[1]))
        image = torch.from_numpy(image).permute(2,0,1)
        image = image[:, crop[1]:-crop[1] or None, crop[0]:-crop[0] or None] # crop if crop != 0
        
        # adjust intrinsics
        fxa = fx * size_factor[0]
        fya = fy * size_factor[1]
        cxa = cx * size_factor[0]-crop[0]
        cya = cy * size_factor[1]-crop[1]

        if camera_model == 'pinhole':
            intr = np.array([fxa, fya, cxa, cya])
        elif camera_model == 'mei':
            intr = np.array([fxa, fya, cxa, cya, 0.0])
        elif camera_model == 'radial':
            intr = np.array([fxa, fya, cxa, cya, 0., 0.])
        else:
            raise Exception("Camera model not implemented!")

        # set un-informed initial values
        if intr_error is None:
            h = rectified_size[1]
            w = rectified_size[0]
            intr[:4] = np.array([(h+w)/2, (h+w)/2, w/2, h/2])

        # add intrinsics error
        else:
            np.random.seed(seed)
            errors = -intr_error + 2 * intr_error * \
                    np.random.uniform(size=intr.shape)
            intr += intr * errors

        intr = torch.as_tensor(intr).cuda()

        yield stride*t, image[None], intr, size_factor
