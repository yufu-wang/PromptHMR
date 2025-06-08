import torch
import numpy as np
import pickle
from typing import Optional
import contextlib
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput
from .vertex_ids import VertexJointSelector, smplx_ids, \
    smpl_to_openpose, smplx_to_openpose, smplx_lr_hand_inds, split2face_feet_hand_parts
from ..utils.rotation_conversions import axis_angle_to_matrix

SMPLX2SMPL_JOINTS = 'data/body_models/smplx2smpl_joints.npy'

class SMPL(smplx.SMPLLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(*args, **kwargs)
            
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        body_joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            body_joints[:,[9,12]] = body_joints[:,[9,12]] + \
                0.25*(body_joints[:,[9,12]]-body_joints[:,[12,9]]) + \
                0.5*(body_joints[:,[8]] - 0.5*(body_joints[:,[9,12]] + body_joints[:,[12,9]]))
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            body_joints = torch.cat([body_joints, extra_joints], dim=1)
        smpl_output.body_joints = body_joints
        return smpl_output
    
    def joints_from_vertices(self, vertices, update_hips=True):
        # Not exactly the same as smpl.foward. 
        # But gives a way: smplx-->smpl-->smpl_joints
        joints = self.J_regressor @ vertices
        joints = self.vertex_joint_selector(vertices, joints)
        body_joints = joints[:, self.joint_map]
    
        if update_hips:
            body_joints[:,[9,12]] = body_joints[:,[9,12]] + \
                0.25*(body_joints[:,[9,12]]-body_joints[:,[12,9]]) + \
                0.5*(body_joints[:,[8]] - 0.5*(body_joints[:,[9,12]] + body_joints[:,[12,9]]))
            
        return body_joints




class SMPLX(smplx.SMPLXLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = True, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLXLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPLX, self).__init__(*args, **kwargs)
        
        if joint_regressor_extra is not None:       
            self.register_buffer('joint_regressor_extra', torch.from_numpy(np.load(joint_regressor_extra)).float())
        self.register_buffer('joint_map', torch.tensor(smplx_to_openpose, dtype=torch.long))
        #self.register_buffer('hand_joint_map', torch.tensor(smplx_lr_hand_inds, dtype=torch.long))
        self.update_hips = update_hips
        self.ffh_joints_selector = VertexJointSelector(vertex_ids=smplx_ids, use_hands=True, use_feet_keypoints=True)

        # J regressor to get SMPL joints (compatible with SPIN)
        self.register_buffer('smpl_joints', torch.from_numpy(np.load(SMPLX2SMPL_JOINTS)).float())

        # Get mean hand pose
        smplx_data = dict(np.load(args[0] + '/SMPLX_NEUTRAL.npz', allow_pickle=True))
        hands_meanl = smplx_data['hands_meanl']
        hands_meanr = smplx_data['hands_meanr']
        hands_mean = np.concatenate([hands_meanl, hands_meanr])
        hands_mean = torch.tensor(hands_mean).reshape(1, 30, 3)
        hands_mean = axis_angle_to_matrix(hands_mean)
        self.register_buffer('hands_mean', hands_mean)
        
        
    def forward(self, *args, **kwargs):
        """
        Run forward pass. Same as SMPLX and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        body_joints = smplx_output.joints[:, self.joint_map, :]
        ffh_joints = torch.cat([self.ffh_joints_selector(smplx_output.vertices), \
                    smplx_output.joints[:, 25: 55, :], smplx_output.joints[:, 20: 22, :]], 1)
        face_joints, feet_joints, hand_joints = split2face_feet_hand_parts(ffh_joints)

        if self.update_hips:
            body_joints[:,[9,12]] = body_joints[:,[9,12]] + \
                0.25*(body_joints[:,[9,12]]-body_joints[:,[12,9]]) + \
                0.5*(body_joints[:,[8]] - 0.5*(body_joints[:,[9,12]] + body_joints[:,[12,9]]))
            
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smplx_output.vertices)
            body_joints = torch.cat([body_joints, extra_joints], dim=1)

        # Replace some SMPLX joints with SMPL joint for better alignment with datasets
        ### replace: elbows, knees and ankles
        smpl_joints = self.smpl_joints @ smplx_output.vertices
        body_joints[:, [3,6,10,13,11,14]] = smpl_joints[:, [19,18,5,4,8,7]].float()
        
        smplx_output.body_joints = body_joints
        smplx_output.face_joints = face_joints
        smplx_output.feet_joints = feet_joints
        smplx_output.hand_joints = hand_joints

        return smplx_output