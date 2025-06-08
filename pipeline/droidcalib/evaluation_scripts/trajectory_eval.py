# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

import numpy as np
import glob
import os

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from evaluation.tartanair_evaluator import TartanAirEvaluator



def compute_ape(traj_est, gt_file, num_images, stride, datapath):
    
    n_reg_images = traj_est.shape[0]

    if n_reg_images == num_images:

        images_list = sorted(glob.glob(os.path.join(datapath, '*.png')))
        images_list = images_list[::stride][:num_images]
        tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]


        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:,3:],
            timestamps=np.array(tstamps))

        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                              pose_relation=PoseRelation.translation_part, 
                              align=True, correct_scale=True)
        print(result)
        ape = result.stats['rmse']


    else:
        ape = np.nan
        
    return ape, n_reg_images



def compute_ape_tartan(traj_est, gt_file, num_images, stride):

    n_reg_images = traj_est.shape[0]

    if n_reg_images == num_images:
        evaluator = TartanAirEvaluator()
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[::stride][
                            :num_images, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz

        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title='mono_sequence')

        print(results)
        ape = results['ate_score']

    else:
        ape = np.nan

    return ape, n_reg_images