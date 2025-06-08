import os
import torch
import numpy as np
import torch.nn.functional as F
from prompt_hmr.utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
    

def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def post_optimization(cfg, results, images, smplx, opt_contact=True, loss_kp_w=0.5):
    # optimize global translation, camera height, and camera orientation using reprojection error
    smplx = smplx.to('cuda')
    
    pred_cam = results['camera_world']
    img_focal = pred_cam['img_focal']
    img_center = pred_cam['img_center']
    Rwc = torch.tensor(pred_cam['Rwc']).float().cuda()
    Twc = torch.tensor(pred_cam['Twc']).float().cuda()
    
    K = np.eye(3)
    K[0, 0] = img_focal
    K[1, 1] = img_focal
    K[0, 2] = img_center[0]
    K[1, 2] = img_center[1]
    K = torch.tensor(K).float().cuda()
    
    # convert from dict to mperson
    seq_len = len(images)
    num_people = len(results['people'])
    people_ids = list(results['people'].keys())
    
    bboxes = np.zeros((num_people, seq_len, 4))
    pred_kp_2d = np.zeros((num_people, seq_len, 25, 3))

    transl_init = np.zeros((num_people, seq_len, 3))
    mask = np.zeros((num_people, seq_len))

    smplx_pose = np.zeros((num_people, seq_len, 55*3))
    smplx_betas = np.zeros((num_people, seq_len, 10))
    rcw = np.zeros((num_people, seq_len, 3, 3))
    tcw = np.zeros((num_people, seq_len, 3))

    num_contact_j = 6
    pred_contact = np.zeros((num_people, seq_len, num_contact_j))
    
    for pidx, pid in enumerate(people_ids):
        v = results['people'][pid]
        bboxes[pidx, v['frames']] = v['bboxes']
        pred_kp_2d[pidx, v['frames']] = v['keypoints_2d'][:, :25]
        transl_init[pidx, v['frames']] = v['smplx_world']['trans']
        mask[pidx, v['frames']] = 1
        smplx_pose[pidx, v['frames']] = v['smplx_world']['pose']
        smplx_betas[pidx, v['frames']] = v['smplx_world']['shape'][:, :10]
        pred_contact[pidx, v['frames']] = v['smplx_cam']['static_conf_logits']
        rcw[pidx, v['frames']] = pred_cam['Rcw'][v['frames']]
        tcw[pidx, v['frames']] = pred_cam['Tcw'][v['frames']]

    print(f"Postprocessing the results")
    bbox_height = bboxes[:, :, 3] - bboxes[:, :, 1]
    bbox_height = torch.from_numpy(bbox_height).float().cuda()
    bbox_height = bbox_height.clamp(min=1e-6)

    pred_kp_2d = torch.from_numpy(pred_kp_2d).float().cuda()
    pred_contact = torch.from_numpy(pred_contact).float().cuda()
    
    tcw = torch.from_numpy(tcw).float().cuda()
    rcw = torch.from_numpy(rcw).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()
    
    smplx_pose = torch.from_numpy(smplx_pose).float().cuda()
    smplx_betas = torch.from_numpy(smplx_betas).float().cuda()
    transl_init = torch.from_numpy(transl_init).float().cuda()
    transl_init.requires_grad = False
    
    transl = torch.zeros_like(transl_init)
    transl.requires_grad = True

    cam_height_offset = torch.zeros(1).float().cuda()
    cam_init_r6d = matrix_to_rotation_6d(torch.eye(3).float().cuda())

    if cfg.run_post_opt_cam:
        cam_height_offset.requires_grad = True
        cam_init_r6d.requires_grad = True
        optim = torch.optim.Adam([transl, cam_height_offset, cam_init_r6d], lr=cfg.postopt_lr)
    else:
        cam_height_offset.requires_grad = False
        cam_init_r6d.requires_grad = False
        optim = torch.optim.Adam([transl], lr=cfg.postopt_lr)


    contact_joint_ids = [7, 10, 8, 11] #, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    joint_mapping = np.array([
        55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
        8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65,
    ], dtype=np.int32)
    
    ignore_joints = [9, 8, 12]
    joint_idxs = [i for i in range(25) if i not in ignore_joints]

    with torch.no_grad():
        B = num_people * seq_len
        smplx_out = smplx(
            body_pose=smplx_pose[:, :, 3:66].reshape(-1, 21*3), 
            global_orient=smplx_pose[:, :, :3].reshape(-1, 3), 
            betas=smplx_betas.reshape(-1, 10), 
            left_hand_pose=smplx_pose[:, :, 75:120].reshape(-1, 15*3),
            right_hand_pose=smplx_pose[:, :, 120:165].reshape(-1, 15*3),
            transl=transl_init.reshape(-1, 3), 
            jaw_pose=torch.zeros(B, 3).to(smplx_pose),
            leye_pose=torch.zeros(B, 3).to(smplx_pose),
            reye_pose=torch.zeros(B, 3).to(smplx_pose),
            expression=torch.zeros(B, 10).to(smplx_pose),
            pose2rot=True
        )
        verts = smplx_out.vertices
        verts = verts.reshape(num_people, seq_len, -1, 3)
        joints = smplx_out.joints[:, joint_mapping]
        contact_joints = smplx_out.joints[:, contact_joint_ids]
        joints = torch.cat([joints, contact_joints], dim=1)
        joints = joints.reshape(num_people, seq_len, -1, 3)

    loss_dict = {}
    for i in range(1000):
        optim.zero_grad()
        
        j_world = joints + transl[:, :, None]

        cam_init_rotmat = rotation_6d_to_matrix(cam_init_r6d)
        rcw_mod = cam_init_rotmat[None, None, :, :] @ rcw

        tcw_mod = (cam_init_rotmat[None, None, :, :] @ tcw[:, :, :, None])[..., 0]
        tcw_mod[:, :, 1] += cam_height_offset
        
        j_cam = (rcw_mod[:, :, None] @ j_world[..., None])[..., 0] + tcw_mod[:, :, None]  # camera space
        
        # Correct perspective projection
        pj = (K[None, None, None, ...] @ j_cam[...,None])[...,0]  # apply intrinsics first
        pj = pj / (pj[..., 2:3] + 1e-6)  # then divide by z
        pj = pj[..., :2]  # get x,y coordinates

        # loss_kp_w = 0.5
        loss_smooth_w = 0.25
        loss_cont_vel_w = 1e3
        loss_cont_height_w = 10.0
        loss_below_floor_w = 1e2

        gt = pred_kp_2d[..., :2]
        
        loss_kp = gmof(pj[:, :, joint_idxs] - gt[:, :, joint_idxs]) / bbox_height[:, :, None, None]
        loss_kp = loss_kp * mask[:, :, None, None]
        loss_kp = ((pred_kp_2d[:, :, joint_idxs, 2] > 0.9) * loss_kp.mean(-1)).mean(-1).sum() / mask.sum()
        loss_kp = loss_kp_w * loss_kp
        
        f_transl = transl_init + transl
        loss_smooth = ((f_transl[:, 1:] - f_transl[:, :-1]) * cfg.fps).pow(2).mean(-1)
        loss_smooth = (loss_smooth * mask[:, 1:]).sum() / mask[:, 1:].sum()
        loss_smooth_vel = loss_smooth_w * loss_smooth

        loss_smooth = torch.linalg.norm((f_transl[:, 2:] + f_transl[:, :-2] - 2 * f_transl[:, 1:-1]) * cfg.fps, dim=-1)
        loss_smooth = (loss_smooth * mask[:, 1:-1]).sum() / mask[:, 1:-1].sum()
        loss_smooth_acc = loss_smooth_w * loss_smooth
        loss_smooth = loss_smooth_vel + loss_smooth_acc
        
        if opt_contact:
            # contact vel loss
            contacts_conf = torch.sigmoid(pred_contact)[..., :len(contact_joint_ids)]
            
            delta_pos = (j_world[:, 1:, 25:] - j_world[:, :-1, 25:])**2
            loss_contact_vel = ((delta_pos.sum(dim=-1) * contacts_conf[:, 1:]) * mask[:, 1:, None]).mean(-1).sum() / mask[:, 1:].sum()
            loss_contact_vel = loss_cont_vel_w * loss_contact_vel
            
            # contact height loss
            floor_diff = torch.abs(j_world[:, :, 25:, 1] - 0.08)
            loss_contact_height = ((floor_diff * contacts_conf) * mask[:, :, None]).mean(-1).sum() / mask.sum()
            loss_contact_height = loss_cont_height_w * loss_contact_height
        
        # no joints below the floor
        floor_diff = F.relu(-j_world[:, :, :, 1]) * mask[:, :, None]
        floor_diff = floor_diff.mean(-1).sum() / mask.sum()
        loss_below_floor = loss_below_floor_w * floor_diff
        
        loss = loss_kp + loss_smooth + loss_below_floor
        if opt_contact:
            loss += loss_contact_vel + loss_contact_height
        
        if 'loss' not in loss_dict.keys():
            loss_dict['loss'] = []
            loss_dict['loss_kp'] = []
            loss_dict['loss_smooth'] = []
            loss_dict['loss_below_floor'] = []
            if opt_contact:
                loss_dict['loss_contact_vel'] = []
                loss_dict['loss_contact_height'] = []
            
        loss_dict['loss'].append(loss.item())
        loss_dict['loss_kp'].append(loss_kp.item())
        loss_dict['loss_smooth'].append(loss_smooth.item())
        loss_dict['loss_below_floor'].append(loss_below_floor.item())
        if opt_contact: 
            loss_dict['loss_contact_vel'].append(loss_contact_vel.item())
            loss_dict['loss_contact_height'].append(loss_contact_height.item())
        
        loss.backward()
        optim.step()
    
    final_transl = transl_init + transl
    
    for pidx, person_id in enumerate(people_ids):
        transl_p = final_transl[pidx][mask[pidx].bool()].detach().cpu().numpy()
        results['people'][person_id][f'smplx_world']['trans'] = transl_p
    
    cam_init_rotmat = cam_init_rotmat.detach().cpu().numpy()
    Tcw = results['camera_world']['Tcw']
    Tcw = (cam_init_rotmat[None, :, :] @ Tcw[..., None])[..., 0]
    Tcw[:, 1] += cam_height_offset.clone().detach().cpu().numpy()
    Rcw = results['camera_world']['Rcw']
    Rcw = cam_init_rotmat[None, :, :] @ Rcw
    
    Rwc = Rcw.transpose(0, 2, 1)
    Twc = -(Rcw @ Tcw[..., None])[..., 0]  # Twc = -R_cw * t_cw
    
    results['camera_world']['Rcw'] = Rcw
    results['camera_world']['Tcw'] = Tcw
    results['camera_world']['Rwc'] = Rwc
    results['camera_world']['Twc'] = Twc

    return results