import torch
import numpy as np

from prompt_hmr.vis.traj import fit_floor_height
from prompt_hmr.utils.one_euro_filter import smooth_one_euro
from prompt_hmr.utils.rotation_conversions import (
    rotation_about_x, 
    rotation_about_y, 
    matrix_to_axis_angle,
)


def transform_smpl_params(root_orient, transl, R, t, smpl_t_pose_pelvis):
    '''
    Input:
        root_orient: [B, 3, 3]
        transl: [B, 3]
        R: [B, 3, 3]
        t: [B, 3]
        smpl_t_pose_pelvis: [3]
    
    Output: 
        root_orient: [B, 3, 3]
        transl: [B, 3]
    '''
    assert smpl_t_pose_pelvis.shape == (3,)
    smpl_t_pose_pelvis = smpl_t_pose_pelvis[None, : , None]
    transl = transl.unsqueeze(-1).float()
    t = t.unsqueeze(-1)

    transl = R @ (smpl_t_pose_pelvis + transl) + t - smpl_t_pose_pelvis
    root_orient = R @ root_orient
    transl = transl.squeeze()
    return root_orient, transl


def world_hps_estimation(cfg, results, smplx):

    colors = np.loadtxt('pipeline/colors.txt')/255
    colors = torch.from_numpy(colors).float()

    locations = []

    pred_cam = results['camera']
    img_focal = pred_cam['img_focal']
    img_center = pred_cam['img_center']
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])

    if cfg.smooth_cam:
        min_cutoff = 0.001
        beta = 0.1
        pred_cam_T = torch.from_numpy(smooth_one_euro(pred_cam_T.numpy(), min_cutoff, beta))
        pred_cam_R = torch.from_numpy(smooth_one_euro(pred_cam_R.numpy(), min_cutoff, beta, is_rot=True))
        
    cam_wc = torch.eye(4).repeat(len(pred_cam_R), 1, 1)
    cam_wc[:, :3, :3] = pred_cam_R
    cam_wc[:, :3, 3] = pred_cam_T

    # Tranform cam: from original to gravity rectify cam
    if cfg.use_spec_calib:
        spec_calib = results['spec_calib']['first_frame']
        Rpitch = rotation_about_x(-spec_calib['pitch'])[None, :3, :3]
        Rroll = rotation_about_y(spec_calib['roll'])[None, :3, :3]
        cam_wc[:, :3, :3] = Rpitch @ Rroll @ cam_wc[:, :3, :3]
        cam_wc[:, :3, 3] = (Rpitch @ Rroll @ cam_wc[:, :3, 3][..., None]).squeeze()
    elif cfg.use_floor_rectify:
        #TODO:
        pass
    Rwc = cam_wc[:, :3, :3]
    Twc = cam_wc[:, :3, 3]

    # Transform smpl: from camera to gravity-aligned camera
    for k, v in results['people'].items():

        pred_smpl = v['smplx_cam']
        for k_, v_ in pred_smpl.items():
            pred_smpl[k_] = torch.from_numpy(v_)
        
        pred_rotmat = pred_smpl['rotmat'].clone()
        pred_shape = pred_smpl['shape'].clone()
        pred_trans = pred_smpl['trans'].clone()
        
        frame = torch.from_numpy(v['frames'])
        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        cam_r = Rwc[frame]
        cam_t = Twc[frame]
        smpl_t_pose_pelvis = smplx(
            global_orient=torch.zeros(1, 3), 
            body_pose=torch.zeros(1, 21*3), 
            betas=mean_shape
        ).joints[0, 0]
        
        root_orient = pred_rotmat[:, 0]
        root_orient, pred_trans = transform_smpl_params(
            root_orient, pred_trans.squeeze(), cam_r, cam_t, smpl_t_pose_pelvis)
        pred_rotmat[:, 0] = root_orient
        pred_trans = pred_trans.squeeze()
        
        pred_pose_aa = matrix_to_axis_angle(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 55*3)
        B = pred_pose_aa.shape[0]
        pred = smplx(
            global_orient=pred_pose_aa[:, :3], 
            body_pose=pred_pose_aa[:, 3:66],
            left_hand_pose=pred_pose_aa[:, 75:120],
            right_hand_pose=pred_pose_aa[:, 120:],
            betas=pred_shape, 
            transl=pred_trans,
            jaw_pose=torch.zeros(B, 3).to(pred_pose_aa),
            leye_pose=torch.zeros(B, 3).to(pred_pose_aa),
            reye_pose=torch.zeros(B, 3).to(pred_pose_aa),
            expression=torch.zeros(B, 10).to(pred_pose_aa),
        )
        
        pred_vert_w = pred.vertices
        pred_j3d_w = pred.joints[:, :22]
        
        locations.append(pred_j3d_w[:, 0])
        
        results['people'][k]['smplx_world'] = {
            'verts': pred_vert_w,
            'joints': pred_j3d_w,
            'rotmat': pred_rotmat.clone(),
            'shape': pred_shape,
            'trans': pred_trans.clone(),
            't_pose_pelvis': smpl_t_pose_pelvis,    
        }

    # Transform: flip y-z 
    R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).float() @ torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
    all_pred_vert_w = torch.cat([v['smplx_world']['verts'] for v in results['people'].values()])
    pred_vert_gr = torch.einsum('ij,bnj->bni', R, all_pred_vert_w).detach()
    offset = fit_floor_height(pred_vert_gr, 'ransac', 'y')

    # if cfg.floor_fitting == 'lowest':
    #     offset = pred_vert_gr[..., 1].min()
    #     offset = torch.tensor([0, offset, 0])
      
    # elif cfg.floor_fitting == 'average':
    #     z, _ = pred_vert_gr[..., 1].min(dim=1)
    #     offset = z.mean()
    #     offset = torch.tensor([0, offset, 0])
       
    # elif cfg.floor_fitting == 'ransac':
    #     zs, _ = pred_vert_gr[..., 1].detach().min(dim=1)
    #     zs = np.sort(zs)

    #     # Get bottom x%
    #     alpha = 1.0
    #     min_z = np.min(zs)
    #     max_z = np.max(zs)
    #     zs = zs[zs <= min_z + (max_z - min_z) * alpha]

    #     inlier_thresh = 0.05 # 5cm
    #     best_inliers = 0
    #     best_z = 0.0
    #     for i in range(10_000):
    #         z = np.random.choice(zs)
    #         inliers_bool = np.abs(zs - z) < inlier_thresh
    #         inliers = np.sum(inliers_bool)
    #         if inliers > best_inliers:
    #             best_z = z
    #             best_inliers = inliers

    #     offset = float(np.median(zs[np.abs(zs - best_z) < inlier_thresh]))
    #     offset = torch.tensor([0, offset, 0])

    # get the locations of the two longest tracks
    locations = sorted(locations, key=len, reverse=True)[:2]
    locations = torch.cat(locations) 
    locations = torch.einsum('ij,bj->bi', R, locations) - offset

    for k,v in results['people'].items():
        del results['people'][k]['smplx_world']['verts']
        
    ##### Viewing Camera #####
    Rwc = torch.einsum('ij,bjk->bik', R, Rwc)
    Twc = torch.einsum('ij,bj->bi', R, Twc) - offset
    Rcw = Rwc.mT
    Tcw = -torch.einsum('bij,bj->bi', Rcw, Twc)

    locations = torch.cat([locations, Tcw])
    cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
    sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item())

    for k, v in results['people'].items():
        pose_rotmat = results['people'][k]['smplx_world']['rotmat']
        trans = results['people'][k]['smplx_world']['trans']
        shape = results['people'][k]['smplx_world']['shape']
        t_pose_pelvis = results['people'][k]['smplx_world']['t_pose_pelvis']
        root_orient, trans = transform_smpl_params(
            root_orient=pose_rotmat[:, 0],
            transl=trans,
            R=R,
            t=-offset,
            smpl_t_pose_pelvis=t_pose_pelvis,
        )
        pose_rotmat[:, 0] = root_orient
        trans = trans.squeeze()
        
        smplx_pose = matrix_to_axis_angle(pose_rotmat.reshape(-1, 3, 3)).reshape(-1, 55*3) 
        smplx_trans = trans
        smplx_betas = shape
        
        results['people'][k]['smplx_world'] = {
            'pose': smplx_pose,
            'shape': smplx_betas,
            'trans': smplx_trans,
        }
        

    results['camera_world'] = {
        'pred_cam_R': Rwc.numpy(),
        'pred_cam_T': Twc.numpy(),
        'Rwc': Rwc.numpy(),
        'Twc': Twc.numpy(),
        'Rcw': Rcw.numpy(),
        'Tcw': Tcw.numpy(),
        'img_focal': img_focal,
        'img_center': img_center,
        'viz_scale': scale,
        'viz_center': [cx.item(), 0, cz.item()]
    }

    return results
