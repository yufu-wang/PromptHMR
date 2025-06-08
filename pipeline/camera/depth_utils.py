import cv2
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def prep_metric3d(rgb_origin, intrinsic, model_version):
    # rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

    #### ajust input size to fit pretrained model
    # keep ratio resize
    if 'convnext' in model_version:
        input_size = (544, 1216)
    else:
        input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
                             cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    return rgb, intrinsic, pad_info, rgb_origin


def prep_metric3d_img(rgb_file):
    if isinstance(rgb_file, str):
        rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    elif isinstance(rgb_file, np.ndarray):
        rgb_origin = rgb_file

    #### ajust input size to fit pretrained model
    # keep ratio resize
    # if 'convnext' in model_version:
    #     input_size = (544, 1216)
    # else:
    input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic
    # intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, 
                             cv2.BORDER_CONSTANT, value=padding)
    # pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    # rgb = rgb[None, :, :, :]#.cuda()
    return rgb


def post_metric3d(pred_depth, confidence, pad_info, rgb_origin, intrinsic):
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], 
                            pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], 
        rgb_origin.shape[:2], mode='bilinear').squeeze()

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    return pred_depth


def post_metric3d_batch(pred_depth, confidence, pad_info, rgb_origin, intrinsic, resize_shape=None):
    # un pad
    pred_depth = pred_depth[:, 0]
    pred_depth = pred_depth[:, pad_info[0]:pred_depth.shape[-2]-pad_info[1], pad_info[2]:pred_depth.shape[-1]-pad_info[3]]
    
    # upsample to original size
    if not(resize_shape is None):
        pred_depth = torch.nn.functional.interpolate(pred_depth[:, None, :, :], resize_shape, mode='bilinear').squeeze()
    else:
        pred_depth = torch.nn.functional.interpolate(pred_depth[:, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    return pred_depth


def unproj_pcd(
        depth: torch.tensor,
        intrinsic: torch.tensor
    ):
    depth = depth.unsqueeze(0)  # [B, H, W]
    b, h, w = depth.size()
    v = torch.arange(0, h).view(1, h, 1).expand(b, h, w).type_as(depth)  # [B, H, W]
    u = torch.arange(0, w).view(1, 1, w).expand(b, h, w).type_as(depth)  # [B, H, W]
    x = (u - intrinsic[:, 0, 2]) / intrinsic[:, 0, 0] * depth # [B, H, W]
    y = (v - intrinsic[:, 1, 2]) / intrinsic[:, 0, 0] * depth # [B, H, W]
    pcd = torch.stack([x, y, depth], dim=0)
    return pcd


def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
      @pcd: Nx3 matrix, the XYZ coordinates
      @rgb: NX3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

         # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b)), fmt="%d %d %d %d %d %d", header=ply_head, comments='')