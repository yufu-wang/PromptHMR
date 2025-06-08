import argparse
import os
import re
import cv2
import csv 
import sys
import smplx
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from renderer_pyrd import Renderer
from smplcodec.codec import SMPLCodec
from rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

# Suppress warnings
warnings.filterwarnings("ignore")

ANN_KEYS = [
    'imgnames', 'genders', 'betas', 'poses_cam', 'poses_world', 'trans_cam', 
    'trans_world', 'cam_ext', 'cam_int', 'joints2d', 'centers', 
    'scales', 'valid_mask', 'sub', 'bboxes'
]
    
MODEL_FOLDER = '/home/muhammed/projects/prompt_hmr/data/body_models/'
DEBUG_FOLDER = '.tmp/bedlam_prep_debug_images'
DEBUG = False
# LOG_FILE = '/mnt/data/datasets/BEDLAM2_0/problem_sequences.txt'
os.makedirs(DEBUG_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dbg_img_path(image_path, type):
    dbg_img_path = os.path.join(DEBUG_FOLDER, image_path.split('/')[-4], f'{type}_'+image_path.split('/')[-1])
    os.makedirs(os.path.dirname(dbg_img_path), exist_ok=True)
    
    # Check if the debug image path already exists
    base_path, ext = os.path.splitext(dbg_img_path)
    counter = 1
    while os.path.exists(dbg_img_path):
        # If the file exists, add _1, _2, etc. to the filename
        dbg_img_path = f"{base_path}_{counter}{ext}"
        counter += 1
    return dbg_img_path


def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    valid_j = []
    joints = np.copy(joints)
    for j in joints:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale
    return center, scale, len(valid_j), bbox

def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def toCamCoords(j3d, camPosWorld):
    # transform gt to camera coordinate frame
    j3d = j3d - camPosWorld
    return j3d

def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1.0, -1.0, 1.0])
    return points

def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1.0, -1.0, -1.0])
    return j3d


def get_cam_int(fl, sens_w, sens_h, cx, cy):
    flx = focalLength_mm2px(fl, sens_w, cx)
    fly = focalLength_mm2px(fl, sens_h, cy)

    cam_mat = np.eye(3)[None].repeat(len(fl), axis=0)
    cam_mat[:, 0, 0] = flx
    cam_mat[:, 1, 1] = fly
    cam_mat[:, 0, 2] = cx
    cam_mat[:, 1, 2] = cy
    return cam_mat

downsample_mat = pickle.load(open('downsample_mat_smplx.pkl', 'rb'))

smplx_model_neutral_betas16 = smplx.create(
    MODEL_FOLDER, 
    model_type='smplx',
    gender='neutral',
    ext='npz',
    flat_hand_mean=True,
    num_betas=16,
    use_pca=False
).to(device) 

smplx_model_neutral_betas11 = smplx.create(
    MODEL_FOLDER, 
    model_type='smplx',
    gender='neutral',
    ext='npz',
    flat_hand_mean=True,
    num_betas=11,
    use_pca=False
).to(device)  

@torch.no_grad()
def get_smplx_vertices(poses, betas, trans, gender='neutral'):

    np2th = lambda x: torch.from_numpy(x).float().to(device)
    
    if betas.shape[1] == 11:
        smplx_model = smplx_model_neutral_betas11
    else:
        smplx_model = smplx_model_neutral_betas16
        
    model_out = smplx_model(
        betas=np2th(betas),
        global_orient=np2th(poses[:, :3]),
        body_pose=np2th(poses[:, 3:66]),
        left_hand_pose=np2th(poses[:, 75:120]),
        right_hand_pose=np2th(poses[:, 120:165]),
        jaw_pose=np2th(poses[:, 66:69]),
        leye_pose=np2th(poses[:, 69:72]),
        reye_pose=np2th(poses[:, 72:75]),
        expression=smplx_model.expression.repeat(poses.shape[0], 1),
        transl=np2th(trans),
    )

    return model_out.vertices.cpu(), model_out.joints.cpu()


def project(points, cam_trans, cam_int):
    points = points + cam_trans.unsqueeze(1)
    cam_int = torch.tensor(cam_int).float()

    projected_points = points / points[...,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij, bkj->bki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()

def get_cam_trans(body_trans, cam_trans):
    cam_trans = np.array(cam_trans) / 100.
    cam_trans = unreal2cv2(cam_trans)

    body_trans = np.array(body_trans) / 100
    body_trans = unreal2cv2(np.reshape(body_trans, (1, 3)))

    trans = body_trans - cam_trans
    return trans

def get_cam_rotmat(body_yaw, pitch, yaw, roll):
    #Because bodies are rotation by 90
    body_rotaa = torch.zeros(3)
    body_rotaa[1] = ((body_yaw - 90) / 180) * np.pi
    body_rotmat = axis_angle_to_matrix(body_rotaa)
    rotaa_yaw = torch.zeros(yaw.shape[0], 3)
    rotaa_yaw[:, 1] = (torch.from_numpy(yaw) / 180) * np.pi
    rotmat_yaw = axis_angle_to_matrix(rotaa_yaw)
    rotaa_pitch = torch.zeros(pitch.shape[0], 3)
    rotaa_pitch[:, 0] = (torch.from_numpy(pitch) / 180) * np.pi
    rotmat_pitch = axis_angle_to_matrix(rotaa_pitch)
    rotaa_roll = torch.zeros(roll.shape[0], 3)
    rotaa_roll[:, 2] = (torch.from_numpy(roll) / 180) * np.pi
    rotmat_roll = axis_angle_to_matrix(rotaa_roll)
    final_rotmat = rotmat_roll @ (rotmat_pitch @ rotmat_yaw)
    return body_rotmat.numpy(), final_rotmat.numpy()


def visualize(image_path, verts,focal_length, smpl_faces):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h,w,c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                            faces=smpl_faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                    bg_img_rgb=img[:, :, ::-1].copy())
    dbg_img_path = get_dbg_img_path(image_path, 'full3d')
    cv2.imwrite(dbg_img_path, front_view[:, :, ::-1])


def visualize_2d(image_path, joints2d, bbox=None):
    from matplotlib import pyplot as plt

    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img[:, :, ::-1]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(len(joints2d)):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], s=0.2)
    if bbox is not None:
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.savefig(get_dbg_img_path(image_path, '2d'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point

    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    from skimage.transform import rotate, resize

    # resize image
    new_img = resize(new_img, res) # scipy.misc.imresize(new_img, res)
    return img,new_img


def visualize_crop(image_path, center, scale, verts,focal_length, smpl_faces):
    img = cv2.imread(image_path)
    if rotate_flag:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h,w,c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                            faces=smpl_faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                    bg_img_rgb=img[:, :, ::-1].copy())
  #  cv2.imwrite(image_path.split('/')[-1].replace('.png','_full.png'), front_view[:, :, ::-1])
    img,crop_img = crop(front_view[:, :, ::-1], center, scale, res=(224,224))
    
    cv2.imwrite(get_dbg_img_path(image_path, 'crop'), crop_img)
  #  cv2.imwrite(image_path.split('/')[-1], img)


def get_global_orient(pose, beta, transl, gender, body_yaw, cam_pitch, cam_yaw, cam_roll, cam_trans):
    # World coordinate transformation after assuming camera has 0 yaw and is at origin
    body_rot_aa = torch.zeros(body_yaw.shape[0], 3)
    # body_rot_aa[:, 1] = torch.from_numpy(np.array((body_yaw - 90 + cam_yaw) / 180 * np.pi)).float()
    body_rot_aa[:, 1] = torch.from_numpy(np.array((body_yaw - 90) / 180 * np.pi)).float()
    body_rotmat = axis_angle_to_matrix(body_rot_aa)
    yaw_rot_aa = torch.zeros(cam_yaw.shape[0], 3)
    yaw_rot_aa[:, 1] = (torch.from_numpy(cam_yaw) / 180) * np.pi
    yaw_rotmat = axis_angle_to_matrix(yaw_rot_aa)
    pitch_rot_aa = torch.zeros(cam_pitch.shape[0], 3)
    pitch_rot_aa[:, 0] = torch.from_numpy(np.array(cam_pitch / 180 * np.pi)).float()
    pitch_rotmat = axis_angle_to_matrix(pitch_rot_aa)
    roll_rot_aa = torch.zeros(cam_roll.shape[0], 3)
    roll_rot_aa[:, 2] = torch.from_numpy(np.array(cam_roll / 180 * np.pi)).float()
    roll_rotmat = axis_angle_to_matrix(roll_rot_aa)
    # final_rotmat = roll_rotmat @ pitch_rotmat
    final_rotmat = roll_rotmat @ (pitch_rotmat @ yaw_rotmat)
    transform_coordinate = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])[None]
    transform_body_rotmat = torch.matmul(body_rotmat, transform_coordinate)

    go_rotmat = axis_angle_to_matrix(torch.from_numpy(pose[:, :3])).float()
    
    w_global_orient = matrix_to_axis_angle(torch.matmul(transform_body_rotmat, go_rotmat))

    assert gender[0].item() == 'neutral', 'Gender is not neutral'
    
    _, joints_local = get_smplx_vertices(pose, beta, np.zeros((pose.shape[0], 3)), 'neutral')
    
    j0 = joints_local[:, 0].detach().cpu().numpy()
    rot_j0 = (transform_body_rotmat @ j0[..., None]).squeeze()
    l_translation_ = (transform_body_rotmat @ torch.from_numpy(transl)[..., None].float()).squeeze()
    l_translation = rot_j0 + l_translation_
    w_translation = l_translation - j0
    
    c_global_orient = matrix_to_axis_angle(final_rotmat @ axis_angle_to_matrix(w_global_orient))
    
    c_translation = (final_rotmat @ l_translation[..., None]).squeeze() - j0
    # convert to y-up direction
    rotation_x_180 = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]).to(w_global_orient)
    
    w_global_orient = matrix_to_axis_angle(rotation_x_180 @ axis_angle_to_matrix(w_global_orient))
    w_translation = (rotation_x_180 @ l_translation[..., None]).squeeze() - j0

    return w_global_orient, c_global_orient, c_translation, w_translation, final_rotmat


def get_params(image_folder, fl, start_frame, gender_sub, smplx_param_orig, trans_body, body_yaw_, cam_x, cam_y, cam_z, fps, person_id, cam_pitch_=0., cam_roll_=0., cam_yaw_=0., logging=False, subject_id=None):

    all_images = sorted(glob(os.path.join(image_folder, '*'+IMG_FORMAT)))
    # print(len(all_images))
    # every_fifth=-9
    # Initialize empty lists for each parameter
    
    if logging:
        import time
        start_time = time.time()
        
    img_inds = []
    cam_inds = []
    smplx_param_inds = []

    for img_ind, image_path in enumerate(all_images):
        # Saving every 5th frame
        # every_fifth += 4
        if fps == 6:
            if img_ind % 5 != 0:
                continue
            smplx_param_ind = img_ind+start_frame
            cam_ind = img_ind
        else:
            smplx_param_ind = img_ind+start_frame
            cam_ind = img_ind

        if smplx_param_ind > smplx_param_orig['poses'].shape[0]:
            break
        img_inds.append(img_ind)
        cam_inds.append(cam_ind)
        smplx_param_inds.append(smplx_param_ind)

    pose = smplx_param_orig['poses'][smplx_param_inds]
    transl = smplx_param_orig['trans'][smplx_param_inds] 
    # Warn MOYO has 300 betas but we are using 16
    beta = smplx_param_orig['betas'][:16][None].repeat(len(smplx_param_inds), axis=0)
    gender = smplx_param_orig['gender'][None].repeat(len(smplx_param_inds), axis=0)
    cam_pitch_ind = -np.array(cam_pitch_)[cam_inds]
    cam_yaw_ind = -np.array(cam_yaw_)[cam_inds]
    if rotate_flag:
        cam_roll_ind = -np.array(cam_roll_)[cam_inds] + 90
    else:
        cam_roll_ind = -np.array(cam_roll_)[cam_inds]

    CAM_INT = get_cam_int(np.array(fl)[cam_inds], SENSOR_W, SENSOR_H, IMG_W/2., IMG_H/2.)

    body_rotmat, cam_rotmat_for_trans = get_cam_rotmat(body_yaw_, cam_pitch_ind, cam_yaw_ind, cam_roll_ind)
    
    cam_t = np.stack(
        [
            np.array(cam_x)[cam_inds], 
            np.array(cam_y)[cam_inds], 
            np.array(cam_z)[cam_inds]
        ], 
        axis=1) 
    cam_trans = get_cam_trans(trans_body, cam_t)
    
    cam_trans = (cam_rotmat_for_trans @ cam_trans[..., None]).squeeze()
    body_yaw_ = np.array(body_yaw_)[None].repeat(len(img_inds), axis=0)
    
    params_dict = {
        'pose': pose,
        'beta': beta,
        'transl': transl,
        'gender': gender,
        'body_yaw': body_yaw_,
        'cam_pitch_ind': cam_pitch_ind,
        'cam_yaw_ind': cam_yaw_ind,
        'cam_roll_ind': cam_roll_ind,
        'cam_trans': cam_trans,
        'img_ind': img_inds
    }
    
    if logging:
        print(f"Time taken to get params: {time.time() - start_time}")
        start_time = time.time()
        
    w_global_orient, c_global_orient, c_trans, w_trans, cam_rotmat = get_global_orient(
        pose=params_dict['pose'],
        beta=params_dict['beta'],
        transl=params_dict['transl'],
        gender=params_dict['gender'],
        body_yaw=params_dict['body_yaw'],
        cam_pitch=params_dict['cam_pitch_ind'],
        cam_yaw=params_dict['cam_yaw_ind'],
        cam_roll=params_dict['cam_roll_ind'],
        cam_trans=params_dict['cam_trans']
    )
    
    if logging:
        print(f"Time taken to get global orient: {time.time() - start_time}")
        start_time = time.time()
        
    cam_trans = torch.from_numpy(params_dict['cam_trans'].squeeze())
    cam_ext_ = torch.zeros(cam_rotmat.shape[0], 4, 4)
    cam_ext_[:, :3, :3] = cam_rotmat
    cam_ext_[:, :3, 3] = cam_trans
    cam_ext_[:, 3, 3] = 1.
    
    pose_cam = torch.from_numpy(params_dict['pose'].copy())
    pose_cam[:, :3] = c_global_orient
    
    pose_world = torch.from_numpy(params_dict['pose'].copy())
    pose_world[:, :3] = w_global_orient
    
    vertices3d, joints3d = get_smplx_vertices(pose_cam.numpy(), params_dict['beta'], c_trans.numpy())
    cam_int_ = CAM_INT
    joints2d = project(joints3d, cam_trans, cam_int_)
    vertices3d_downsample = downsample_mat[None].matmul(vertices3d)
    proj_verts_ = project(torch.tensor(vertices3d_downsample).float(), cam_trans, cam_int_)
    x_min, y_min, x_max, y_max = 0., 0., IMG_W, IMG_H
    
    bbox_verts = [
        proj_verts_[:,:,0].min(1).clip(x_min, x_max),
        proj_verts_[:,:,1].min(1).clip(y_min, y_max),
        proj_verts_[:,:,0].max(1).clip(x_min, x_max),
        proj_verts_[:,:,1].max(1).clip(y_min, y_max)
    ]
    bbox_verts = np.stack(bbox_verts, axis=1)
    if logging:
        print(f"Time taken to get proj verts: {time.time() - start_time}")
    valid_mask = torch.ones(pose_cam.shape[0], dtype=torch.bool)
    if logging:
        start_time = time.time()
        
    center_list = []
    scale_list = []
    for i, j2d in enumerate(joints2d):
        center, scale, num_vis_joints, bbox = get_bbox_valid(j2d[:22], rescale=SCALE_FACTOR_BBOX, img_width=IMG_W, img_height=IMG_H)
        if center[0] < 0 or center[1] < 0 or scale <= 0:
            valid_mask[i] = False
        
        if num_vis_joints < 2:
            valid_mask[i] = False
        
        center_list.append(center)
        scale_list.append(scale)
        
    center = torch.tensor(center_list)
    scale = torch.tensor(scale_list)
    verts_cam2 = vertices3d + cam_trans.unsqueeze(1)
    if logging:
        print(f"Time taken to get verts cam2: {time.time() - start_time}")
        start_time = time.time()
    
    is_neg_vz = verts_cam2[:, 0, 2] < 0
    
    if (valid_mask & is_neg_vz).sum() > 0:
        seq_name = image_folder.split('/')[-3] + '/' + image_folder.split('/')[-1]
        # with open(LOG_FILE, 'a') as f:
        #     f.write(f'{seq_name},{person_id},num_neg_z_frames={is_neg_vz.sum()},total={len(is_neg_vz)}\n')
        if False:
            for idx in range(len(is_neg_vz)):
                if is_neg_vz[idx] and valid_mask[idx]:
                    print(f"Negative z frame: {idx}")
                    visualize_2d(all_images[params_dict['img_ind'][idx]], joints2d[idx])
    
    if valid_mask.sum() == 0:
        seq_name = image_folder.split('/')[-3] + '/' + image_folder.split('/')[-1]
        print(f'No valid frames: {seq_name}')
        return None
    
    valid_mask = valid_mask & ~is_neg_vz
    
    if valid_mask.sum() == 0:
        seq_name = image_folder.split('/')[-3] + '/' + image_folder.split('/')[-1]
        print(f'No valid frames: {seq_name}, {is_neg_vz.sum()}/{len(is_neg_vz)}')
        return None
    
    if DEBUG:
        idx = random.randint(0, len(params_dict['img_ind']) - 1)
        count = 0
        while valid_mask[idx] == False:
            idx = random.randint(0, len(params_dict['img_ind']) - 1)
            count += 1
            if count > 5:
                break
        # for idx in range(len(valid_mask)):
        if valid_mask[idx]:
            # visualize_crop(all_images[params_dict['img_ind'][idx]], center[idx], scale[idx], verts_cam2[idx], CAM_INT[idx,0,0], smplx_model_neutral_betas16.faces)
            # visualize_2d(all_images[params_dict['img_ind'][idx]], proj_verts_[idx], bbox_verts[idx])
            # visualize_2d(all_images[params_dict['img_ind'][idx]], joints2d[idx])
            visualize(all_images[params_dict['img_ind'][idx]], verts_cam2[idx], CAM_INT[idx,0,0], smplx_model_neutral_betas16.faces)
        
    if logging:
        print(f"Time taken to visualize crop: {time.time() - start_time}")
        start_time = time.time()
    
    data_dict = {k: [] for k in ANN_KEYS}
    imgnames = [os.path.join(all_images[x].split('/')[-2], all_images[x].split('/')[-1]) for i, x in enumerate(params_dict['img_ind'])]
    data_dict['imgnames'].append(np.array(imgnames))
    data_dict['genders'].append(np.array([gender_sub] * len(imgnames)))
    data_dict['betas'].append(params_dict['beta'])
    data_dict['poses_cam'].append(pose_cam)
    data_dict['poses_world'].append(pose_world)
    data_dict['trans_cam'].append(c_trans)
    data_dict['trans_world'].append(w_trans)
    data_dict['cam_ext'].append(cam_ext_)
    data_dict['cam_int'].append(cam_int_)
    data_dict['joints2d'].append(joints2d)
    data_dict['centers'].append(center)
    data_dict['scales'].append(scale)
    data_dict['bboxes'].append(bbox_verts)
    data_dict['sub'].append(np.array([person_id] * len(imgnames)))
    data_dict['valid_mask'].append(valid_mask)
    # data_dict['proj_verts'].append(proj_verts_)
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        data_dict[k] = np.concatenate(v, axis=0)
    
    if DEBUG:
        fname = all_images[params_dict['img_ind'][0]].split('/')[-4]
        seq = all_images[params_dict['img_ind'][0]].split('/')[-2]
        subject = '_'.join(subject_id.split('_')[:-1])
        
        smpl_file = f'.tmp/bedlam_smpl/{fname}/{seq}/{subject}_bedlam.smpl'
        os.makedirs(os.path.dirname(smpl_file), exist_ok=True)
        SMPLCodec(
            shape_parameters=params_dict['beta'][0, :10],
            body_pose=pose_world[:, :66].reshape(-1,22,3).numpy(), 
            body_translation=w_trans.numpy(),
            frame_count=pose_world.shape[0], 
            frame_rate=30.0,
        ).write(smpl_file)
        print(f"Saved {smpl_file}")
        
        pose = smplx_param_orig['poses'][smplx_param_inds]
        transl = smplx_param_orig['trans'][smplx_param_inds] 
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(transl[:, 1])
        # plt.savefig(f'.tmp/bedlam_smpl/{fname}/{seq}/{subject}_transl_y.png')
        # plt.close()
        smpl_file = f'.tmp/bedlam_smpl/{fname}/{seq}/{subject}_bedlam_orig.smpl'
        SMPLCodec(
            shape_parameters=beta[0, :10],
            body_pose=pose[:, :66].reshape(-1,22,3), 
            body_translation=transl,
            frame_count=pose.shape[0], 
            frame_rate=30.0,
        ).write(smpl_file)
        print(f"Saved {smpl_file}")
        
    if logging:
        print(f"Time taken to get data dict: {time.time() - start_time}")
    
    return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fps', type=int, required=True, help='FPS must be either 6 or 30')
    parser.add_argument('-v', '--version', type=int, default=1, help='BEDLAM version')
    args = parser.parse_args()
    
    fps = args.fps
    assert fps in [6, 30], f"FPS must be either 6 or 30"
    if args.version == 2:
        base_image_folder = '/mnt/data/datasets/BEDLAM2_0/images'
        output_folder = f'/home/muhammed/projects/GVHMR/inputs/BEDLAM2/training_labels_{fps}fps'
        src_folder = '/home/muhammed/projects/prompt_hmr/data/b2_src_data'
        scene_names = {
            '20240425_1_171_citysample_dolly': 'city-dolly-moyo1',
            '20240416_1_171_yogastudio_orbit_timeofday': 'yoga-orbit-moyo',
            '20240423_1_171_yogastudio_staticloc_timeofday': 'yoga-static-moyo',
            '20240424_1_171_citysample_orbit': 'city-orbit-moyo1',
            '20240425_1_171_hdri': 'hdri-moyo',
            '20240426_5_100_citysample_orbit': 'city-orbit-moyo2',
            '20240429_1_171_stadium': 'stadium-moyo',
            '20240502_5_200_citysample_dolly': 'city-dolly-moyo2',
            '20240506_10_200_hdri': 'hdri-moyo2',
            '20240506_5_200_citysample_orbit': 'city-orbit-moyo3',
            '20240507_5_200_citysample_dollyz': 'city-dollyz-moyo',
            '20240514_1_1001_citysample_tracking': 'city-tracking-b2v01',
            '20240604_5_500_citysample_tracking': 'city-tracking-b2v02',
            '20240605_3_500_busstation_tracking': 'bus-tracking-b2v01',
            '20240606_4_250_busstation_orbit': 'bus-orbit-b2v01',
            '20240606_1_500_stadium_closeup': 'stadium-closeup-b2v01',
            '20240611_5_250_archmodelsvol8_dolly': 'archmodel-dolly-b2v01',
            '20240611_5_200_citysamplenight_dolly': 'citynight-dolly-b2v01',
            '20240613_1_200_hdri': 'hdri-b2v01',
            '20240614_5_200_citysamplenight_tracking': 'citynight-tracking-b2v01',
            '20240614_1_300_hdri': 'hdri-b2v02',
            '20240617_10_500_hdri': 'hdri-b2v03',
            '20240618_1_500_ai0805_orbit': 'ai0805-orbit-b2v01',
            '20240619_2_250_ai1004_orbit': 'ai1004-orbit-b2v01',
            '20240619_1_250_ai1004_tracking': 'ai1004-tracking-b2v01',
            '20240620_5_250_archmodelsvol8_dollyz': 'archmodel-dollyz-b2v01',
            '20240621_1_250_archmodelsvol8_tracking': 'archmodel-tracking-b2v01',
            '20240625_1_2337_hdri': 'hdri-b2v11',
            '20240628_1_250_ai1004_tracking': 'ai1004-tracking-b2v11',
            '20240628_4_250_busstation_orbit': 'bus-tracking-b2v11',
            '20240701_1_250_ai0901_lookat': 'ai0901-lookat-b2v11',
            '20240703_1_250_ai0901_orbit_portrait': 'ai0901-orbit-portrait-b2v11',
            '20240708_1_250_ai0901_static_portrait': 'ai0901-static-portrait-b2v11',
            '20240709_5_250_archmodelsvol8_zoom': 'archmodel-zoom-b2v11',
            '20240710_1_250_ai0805_orbit_portrait': 'ai0805-orbit-portrait-b2v11',
            '20240711_5-10_250_busstation_orbit_zoom': 'bus-orbit-zoom-b2v11',
            '20240725_1_250_ai0805_vcam': 'ai0805-vcam-b2v11',
            '20240726_1_250_ai0805_vcam': 'ai0805-vcam-b2v12',
            '20240729_1_250_ai1004_vcam': 'ai1004-vcam-portrait-b2v11',
            '20240730_1_250_ai1101_vcam': 'ai1101-vcam-portrait-b2v11',
            '20240731_1_1827_hdri': 'hdri-b2v21',
            '20240805_5-10_250_busstation_orbit_zoom': 'bus-orbit-zoom-b2v21',
            '20240806_1_250_ai1101_vcam': 'ai1101-vcam-portrait-b2v21',
            '20240808_1_250_ai1105_vcam': 'ai1105-vcam-b2v21',
            '20240809_1_250_ai1102_vcam': 'ai1102-vcam-portrait-b2v21',
            '20240813_1_250_ai1004_tracking': 'ai1004-tracking-b2v21',
            '20240927_1_250_archmodelsvol8_tracking': 'archmodel-tracking-b2v0',
            '20241001_5-10_250_busstation_orbit_zoom': 'bus-orbit-zoom-b2v2',
        }
        scene_names = {
            '20240709_5_250_archmodelsvol8_zoom': 'archmodel-zoom-b2v11',
        }
    elif args.version == 1:
        base_image_folder = '/mnt/data/datasets/BEDLAM/data_30fps/images'
        output_folder = f'/home/muhammed/projects/GVHMR/inputs/BEDLAM1/training_labels_{fps}fps'
        src_folder = '/home/muhammed/datasets/BEDLAM/neutral_ground_truth_motioninfo'
        scene_names = {
            '20221010_3_1000_batch01hand': '',
            '20221010_3-10_500_batch01hand_zoom_suburb_d': '',
            '20221011_1_250_batch01hand_closeup_suburb_a': 'portrait',
            '20221011_1_250_batch01hand_closeup_suburb_b': 'portrait',
            '20221011_1_250_batch01hand_closeup_suburb_c': 'portrait',
            '20221011_1_250_batch01hand_closeup_suburb_d': 'portrait',
            '20221012_1_500_batch01hand_closeup_highSchoolGym': 'portrait',
            '20221012_3-10_500_batch01hand_zoom_highSchoolGym': '',
            '20221013_3-10_500_batch01hand_static_highSchoolGym': '',
            '20221013_3_250_batch01hand_orbit_bigOffice': '',
            '20221013_3_250_batch01hand_static_bigOffice': '',
            '20221014_3_250_batch01hand_orbit_archVizUI3_time15': '',
            '20221015_3_250_batch01hand_orbit_archVizUI3_time10': '',
            '20221015_3_250_batch01hand_orbit_archVizUI3_time12': '',
            '20221015_3_250_batch01hand_orbit_archVizUI3_time19': '',
            '20221017_3_1000_batch01hand': '',
            '20221018_1_250_batch01hand_zoom_suburb_b': '',
            '20221018_3_250_batch01hand_orbit_archVizUI3_time15': '',
            '20221018_3-8_250_batch01hand': '',
            '20221018_3-8_250_batch01hand_pitchDown52_stadium': '',
            '20221018_3-8_250_batch01hand_pitchUp52_stadium': '',
            '20221019_1_250_highbmihand_closeup_suburb_b': 'portrait',
            '20221019_1_250_highbmihand_closeup_suburb_c': 'portrait',
            '20221019_3_250_highbmihand': '',
            '20221019_3-8_1000_highbmihand_static_suburb_d': '',
            '20221019_3-8_250_highbmihand_orbit_stadium': '',
            '20221020_3-8_250_highbmihand_zoom_highSchoolGym_a': '',
            '20221022_3_250_batch01handhair_static_bigOffice': '',
            '20221024_10_100_batch01handhair_zoom_suburb_d': '',
            '20221024_3-10_100_batch01handhair_static_highSchoolGym': '',
        }
        
    os.makedirs(output_folder, exist_ok=True)
    
    for scene_folder, scene_info in tqdm(scene_names.items()):
        print(scene_folder, scene_info)
        
        outp_npz_file = os.path.join(output_folder, str(scene_folder)+'.npz')

        rotate_flag = False
        
        if 'portrait' in scene_info:
            print(f"Portrait scene: {scene_folder}")
            rotate_flag = True
            SENSOR_W = 20.25
            SENSOR_H = 36
            IMG_W = 720
            IMG_H = 1280
        else:
            rotate_flag = False
            SENSOR_W = 36
            SENSOR_H = 20.25
            IMG_W = 1280
            IMG_H = 720
                
        if args.version == 2:
            img_folder = os.path.join(base_image_folder, scene_folder, 'png')
            image_folder_base = img_folder
            base_folder = img_folder.replace('/png','')
        
            motion_seq_version = None
            
            if 'b2v0' in scene_names[scene_folder]:
                motion_seq_version = 'b2v0'
            elif 'b2v1' in scene_names[scene_folder]:
                motion_seq_version = 'b2v1'
            elif 'b2v2' in scene_names[scene_folder]:
                motion_seq_version = 'b2v2'
            elif 'moyo' in scene_names[scene_folder]:
                motion_seq_version = 'b2_moyo'
            else:
                raise ValueError(f"Motion source data version not found!!")
            IMG_FORMAT = '.png'
            
        elif args.version == 1:
            img_folder = os.path.join(base_image_folder, scene_folder, 'jpg')
            image_folder_base = img_folder
            base_folder = img_folder.replace('/jpg','')
            motion_seq_version = 'b1'
            IMG_FORMAT = '.jpg'
            
            
        outfile = outp_npz_file
        csv_path = os.path.join(base_folder, 'be_seq.csv')
        csv_data = pd.read_csv(csv_path).to_dict('list')
        
        seq_name = ''
        if args.version == 2:
            cam_csv_base = os.path.join(base_folder, 'ground_truth/meta_exr_csv')
        elif args.version == 1:
            cam_csv_base = os.path.join(base_folder, 'ground_truth/camera')


        SCALE_FACTOR_BBOX = 1.2

        seq_name = ''

        for idx, comment in enumerate(tqdm(csv_data['Comment'])):
                
            if 'sequence_name' in comment:
                #Get sequence name and corresponding camera details
                seq_name = comment.split(';')[0].split('=')[-1]
                cam_csv_data = pd.read_csv(os.path.join(cam_csv_base, seq_name+'_camera.csv'))
                cam_csv_data = cam_csv_data.to_dict('list')
                cam_x = cam_csv_data['x']
                cam_y = cam_csv_data['y']
                cam_z = cam_csv_data['z']
                cam_yaw_ = cam_csv_data['yaw']
                cam_pitch_ = cam_csv_data['pitch']
                cam_roll_ = cam_csv_data['roll']
                fl = cam_csv_data['focal_length']
                continue
            elif 'start_frame' in comment:
                # Get body details
                start_frame = int(comment.split(';')[0].split('=')[-1])
                body = csv_data['Body'][idx]
                texture_info = re.search(r'texture_body=([^;]+)', csv_data['Comment'][idx]).group(1)
                
                if 'moyo' in body:
                    gt_smplx_beta_base = os.path.join(src_folder, 'b2_moyo')
                    smplx_param_orig = np.load(os.path.join(gt_smplx_beta_base, body+'.npz'))
                else:
                    parts = body.rsplit('_', 1)
                    npz_file = os.path.join(parts[0], parts[1], 'motion_seq.npz')
                    
                    if args.version == 2:
                        gt_smplx_beta_base = os.path.join(src_folder, motion_seq_version)
                    elif args.version == 1:
                        gt_smplx_beta_base = src_folder
                        
                    smplx_param_orig = np.load(os.path.join(gt_smplx_beta_base, npz_file))
                
                smplx_param_orig = dict(smplx_param_orig)
                gender = smplx_param_orig['gender'].item()
                image_folder = os.path.join(image_folder_base, seq_name)
                person_id = body
                
                X = csv_data['X'][idx]
                Y = csv_data['Y'][idx]
                Z = csv_data['Z'][idx]
                trans_body = [X, Y, Z]
                body_yaw_ = csv_data['Yaw'][idx]
                data_dict = get_params(
                    image_folder, fl, start_frame, gender, smplx_param_orig, trans_body, body_yaw_, cam_x, cam_y, cam_z, fps, person_id, cam_pitch_=cam_pitch_, cam_roll_=cam_roll_, cam_yaw_=cam_yaw_, subject_id=person_id,
                )
                
                if not DEBUG:
                    subject = '_'.join(person_id.split('_')[:-1])
                    save_path = f'{output_folder}/{scene_folder}/{seq_name}-{subject}.pt'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if data_dict is None:
                        print(f'No data: {save_path}')
                    else:
                        torch.save(data_dict, save_path)
            else:
                continue
