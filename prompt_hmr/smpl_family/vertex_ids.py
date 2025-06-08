# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)

smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

smplx_to_openpose = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]   
smplx_lr_hand_inds = [20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

smplx42tohand42 = [5, 39, 38, 37, 6, 27, 26, 25, 7, 30, 29, 28, 8, 36, 35, 34, 9, 33, 32, 31, 41, 
                   0, 24, 23, 22, 1, 12, 11, 10, 2, 15, 14, 13, 3, 21, 20, 19, 4, 18, 17, 16, 40]

smplx_ids = {
    'nose':		    9120,
    'reye':		    9929,
    'leye':		    9448,
    'rear':		    616,
    'lear':		    6,
    'rthumb':		8079,
    'rindex':		7669,
    'rmiddle':		7794,
    'rring':		7905,
    'rpinky':		8022,
    'lthumb':		5361,
    'lindex':		4933,
    'lmiddle':		5058,
    'lring':		5169,
    'lpinky':		5286,
    'LBigToe':		5770,
    'LSmallToe':    5780,
    'LHeel':		8846,
    'RBigToe':		8463,
    'RSmallToe': 	8474,
    'RHeel':  		8635}


smplx42_lr_hand_ids = {'l_thumb4': 0, 'l_index4': 1, 'l_middle4': 2, 'l_ring4': 3, 'l_pinky4': 4, 
    'r_thumb4': 5, 'r_index4': 6, 'r_middle4': 7, 'r_ring4': 8, 'r_pinky4': 9, 
    'l_index1': 10, 'l_index2': 11, 'l_index3': 12, 'l_middle1': 13, 'l_middle2': 14, 'l_middle3': 15, 
    'l_pinky1': 16, 'l_pinky2': 17, 'l_pinky3': 18, 'l_ring1': 19, 'l_ring2': 20, 'l_ring3': 21, 
    'l_thumb1': 22, 'l_thumb2': 23, 'l_thumb3': 24, 'r_index1': 25, 'r_index2': 26, 'r_index3': 27, 
    'r_middle1': 28, 'r_middle2': 29, 'r_middle3': 30, 'r_pinky1': 31, 'r_pinky2': 32, 'r_pinky3': 33, 
    'r_ring1': 34, 'r_ring2': 35, 'r_ring3': 36, 'r_thumb1': 37, 'r_thumb2': 38, 'r_thumb3': 39, 
    'l_wrist': 40, 'r_wrist': 41}

hand42 = {'r_thumb4': 0, 'r_thumb3': 1, 'r_thumb2': 2, 'r_thumb1': 3, 
          'r_index4': 4, 'r_index3': 5, 'r_index2': 6, 'r_index1': 7, 
          'r_middle4': 8, 'r_middle3': 9, 'r_middle2': 10, 'r_middle1': 11, 
          'r_ring4': 12, 'r_ring3': 13, 'r_ring2': 14, 'r_ring1': 15, 
          'r_pinky4': 16, 'r_pinky3': 17, 'r_pinky2': 18, 'r_pinky1': 19,   'r_wrist': 20, 
          'l_thumb4': 21, 'l_thumb3': 22, 'l_thumb2': 23, 'l_thumb1': 24, 
          'l_index4': 25, 'l_index3': 26, 'l_index2': 27, 'l_index1': 28, 
          'l_middle4': 29, 'l_middle3': 30, 'l_middle2': 31, 'l_middle1': 32, 
          'l_ring4': 33, 'l_ring3': 34, 'l_ring2': 35, 'l_ring1': 36, 
          'l_pinky4': 37, 'l_pinky3': 38, 'l_pinky2': 39, 'l_pinky1': 40,   'l_wrist': 41}

def split2face_feet_hand_parts(ffh_joints):
    # ['nose', 'reye', 'leye', 'rear', 'lear']
    face_joints = ffh_joints[:, :5]
    # ['LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
    feet_joints = ffh_joints[:, 5:5+6]
    # 42 hand joints in InterHand2.6M's order
    hand_joints = ffh_joints[:, 11:][:, smplx42tohand42]
    return face_joints, feet_joints, hand_joints

def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class VertexJointSelector(nn.Module):
    def __init__(self, vertex_ids=None, **kwargs):
        super(VertexJointSelector, self).__init__()

        face_keyp_idxs = np.array([
                                    vertex_ids['nose'],
                                    vertex_ids['reye'],
                                    vertex_ids['leye'],
                                    vertex_ids['rear'],
                                    vertex_ids['lear']], dtype=np.int64)

        feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                    vertex_ids['LSmallToe'],
                                    vertex_ids['LHeel'],
                                    vertex_ids['RBigToe'],
                                    vertex_ids['RSmallToe'],
                                    vertex_ids['RHeel']], dtype=np.int32)

        self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        tips_idxs = []
        for hand_id in ['l', 'r']:
            for tip_name in self.tip_names:
                tips_idxs.append(vertex_ids[hand_id + tip_name])

        extra_joints_idxs = np.concatenate([face_keyp_idxs, feet_keyp_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(torch.long))

        return extra_joints