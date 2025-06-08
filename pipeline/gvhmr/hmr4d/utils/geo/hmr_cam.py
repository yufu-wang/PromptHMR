import torch
import numpy as np
from hmr4d.utils.geo_transform import project_p2d, convert_bbx_xys_to_lurb, cvt_to_bi01_p2d


def estimate_focal_length(img_w, img_h):
    return (img_w**2 + img_h**2) ** 0.5  # Diagonal FOV = 2*arctan(0.5) * 180/pi = 53


def estimate_K(img_w, img_h):
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def convert_K_to_K4(K):
    K4 = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()
    return K4


def convert_f_to_K(focal_length, img_w, img_h):
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def resize_K(K, f=0.5):
    K = K.clone() * f
    K[..., 2, 2] = 1.0
    return K


def create_camera_sensor(width=None, height=None, f_fullframe=None):
    if width is None or height is None:
        # The 4:3 aspect ratio is widely adopted by image sensors in mobile phones.
        if np.random.rand() < 0.5:
            width, height = 1200, 1600
        else:
            width, height = 1600, 1200

    # Sample FOV from common options:
    # 1. wide-angle lenses are common in mobile phones,
    # 2. telephoto lenses has less perspective effect, which should makes it easy to learn
    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)

    # We use diag to map focal-length: https://www.nikonians.org/reviews/fov-tables
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe

    K_fullimg = torch.eye(3)
    K_fullimg[0, 0] = focal_length
    K_fullimg[1, 1] = focal_length
    K_fullimg[0, 2] = width / 2
    K_fullimg[1, 2] = height / 2

    return width, height, K_fullimg


# ====== Compute cliffcam ===== #


def convert_xys_to_cliff_cam_wham(xys, res):
    """
    Args:
        xys: (N, 3) in pixel. Note s should not be touched by 200
        res: (2), e.g. [4112., 3008.]  (w,h)
    Returns:
        cliff_cam: (N, 3), normalized representation
    """

    def normalize_keypoints_to_image(x, res):
        """
        Args:
            x: (N, 2), centers
            res: (2), e.g. [4112., 3008.]
        Returns:
            x_normalized: (N, 2)
        """
        res = res.to(x.device)
        scale = res.max(-1)[0].reshape(-1)
        mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
        x = 2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - mean.reshape(
            *[1 for i in range(len(x.shape[1:-1]))], -1
        )
        return x

    centers = normalize_keypoints_to_image(xys[:, :2], res)  # (N, 2)
    scale = xys[:, 2:] / res.max()
    location = torch.cat((centers, scale), dim=-1)
    return location


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """impl as in BEDLAM
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg
        K_fullimg: ((B), (N), 3, 3)
    Returns:
        bbox_info: ((B), N, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info


# ====== Convert Prediction to Cam-t ===== #


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def compute_transl_full_cam_metric3d(pred_trans_in_pixel_space, bbox_center_scale, cam_int):
    '''
    pred_trans_in_pixel_space: (B, L, 3)
    bbox_center_scale: (B, L, 3)
    cam_int: (B, L, 3, 3)
    pz represents depth in a canonical camera with focal length 1000px.
    px and py are offsets of the human from the bbox center in pixel space.

    returns: pred_trans_in_metric_space (B, L, 3)
    '''

    # Extract camera intrinsics
    fx = cam_int[:, :, 0, 0]  # Focal length in x (B, L)
    fy = cam_int[:, :, 1, 1]  # Focal length in y (B, L)
    cx = cam_int[:, :, 0, 2]  # Principal point x-coordinate (B, L)
    cy = cam_int[:, :, 1, 2]  # Principal point y-coordinate (B, L)
    
    img_w, img_h = cx*2, cy*2

    # Extract bbox center and scale
    bbox_cx = bbox_center_scale[:, :, 0]  # Bbox center x-coordinate (B, L)
    bbox_cy = bbox_center_scale[:, :, 1]  # Bbox center y-coordinate (B, L)
    bbox_scale = bbox_center_scale[:, :, 2]  # Bbox scale (B, L)

    # Extract predicted translations in pixel space
    px = pred_trans_in_pixel_space[:, :, 0]  # Offset in x (B, L)
    py = pred_trans_in_pixel_space[:, :, 1]  # Offset in y (B, L)
    pz = pred_trans_in_pixel_space[:, :, 2]  # Depth in canonical camera (B, L)
    pz = 1 / (pz + 1e-6)
    # Convert normalized px, py to image space using image width and height
    px = px * img_w  # Scale to image width
    py = py * img_h  # Scale to image height
    
    # Compute the image coordinates of the human
    u = bbox_cx + px  # Image coordinate x (B, L)
    v = bbox_cy + py  # Image coordinate y (B, L)

    # Adjust depth from canonical camera to actual camera
    Z = (fx / 1000.0) * pz  # Depth in metric space (B, L)

    # Compute X and Y in the camera coordinate system
    X = (u - cx) * Z / fx  # Coordinate X in metric space (B, L)
    Y = (v - cy) * Z / fy  # Coordinate Y in metric space (B, L)

    # Stack the coordinates to form the translation vector
    pred_trans_in_metric_space = torch.stack([X, Y, Z], dim=-1)  # (B, L, 3)

    return pred_trans_in_metric_space


def inv_compute_transl_full_cam_metric3d(pred_trans_in_metric_space, bbox_center_scale, cam_int):
    '''
    pred_trans_in_metric_space: (B, L, 3) - Predicted translations [X, Y, Z] in metric space.
    bbox_center_scale: (B, L, 3) - Bounding box center coordinates and scale [bbox_cx, bbox_cy, bbox_scale].
    cam_int: (B, L, 3, 3) - Camera intrinsic matrices.

    Returns:
    pred_trans_in_pixel_space: (B, L, 3) - Predicted offsets [px, py, pz_in] in pixel space.
    '''
    import torch

    # Extract camera intrinsics
    fx = cam_int[:, :, 0, 0]  # Focal length in x (B, L)
    fy = cam_int[:, :, 1, 1]  # Focal length in y (B, L)
    cx = cam_int[:, :, 0, 2]  # Principal point x-coordinate (B, L)
    cy = cam_int[:, :, 1, 2]  # Principal point y-coordinate (B, L)

    img_w = cx * 2  # Image width (B, L)
    img_h = cy * 2  # Image height (B, L)
    
    # Extract bbox center and scale
    bbox_cx = bbox_center_scale[:, :, 0]  # Bounding box center x-coordinate (B, L)
    bbox_cy = bbox_center_scale[:, :, 1]  # Bounding box center y-coordinate (B, L)
    bbox_scale = bbox_center_scale[:, :, 2]  # Bounding box scale (B, L)

    # Extract predicted translations in metric space
    X = pred_trans_in_metric_space[:, :, 0]  # Coordinate X in metric space (B, L)
    Y = pred_trans_in_metric_space[:, :, 1]  # Coordinate Y in metric space (B, L)
    Z = pred_trans_in_metric_space[:, :, 2]  # Depth Z in metric space (B, L)

    # Avoid division by zero
    Z_safe = Z + 1e-6  # (B, L)

    # Adjust depth from actual camera to canonical camera
    f_c = 1000.0  # Canonical focal length
    s = fx / f_c  # Scaling factor (B, L)

    # Compute inverse depth in canonical camera space
    pz_in = (s / Z_safe) - 1e-6  # Predicted inverse depth (B, L)

    # Compute image coordinates u and v
    u = (X * fx) / Z_safe + cx  # Image coordinate x (B, L)
    v = (Y * fy) / Z_safe + cy  # Image coordinate y (B, L)

    # Compute offsets from bbox center
    px = u - bbox_cx  # Offset in x (B, L)
    py = v - bbox_cy  # Offset in y (B, L)
    
    px = px / img_w
    py = py / img_h

    # Stack the offsets to form the translation vector in pixel space
    pred_trans_in_pixel_space = torch.stack([px, py, pz_in], dim=-1)  # (B, L, 3)

    return pred_trans_in_pixel_space


def compute_transl_full_cam_prompthmr(transl, bbx_xys, K_fullimg):
    focal = K_fullimg[:,:,0,0]
    px, py, pz = transl.unbind(-1)

    pz = 1 / (pz + 1e-6)

    tx = px * pz
    ty = py * pz
    tz = pz * focal / 1000.
    t_full = torch.stack([tx, ty, tz], dim=-1)

    return t_full


def inv_compute_transl_full_cam_prompthmr(transl, bbx_xys, K_fullimg):
    focal = K_fullimg[:,:,0,0]
    tx, ty, tz = transl.unbind(-1)
    
    # Invert depth calculation
    pz = 1 / (tz * 1000. / focal + 1e-6)
    
    # Invert x,y calculations
    px = tx / tz
    py = ty / tz
    
    transl_out = torch.stack([px, py, pz], dim=-1)
    return transl_out


def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse operation of compute_transl_full_cam"""
    assert transl.ndim == bbx_xys.ndim  # (*, L, 3)
    assert K_fullimg.ndim == (bbx_xys.ndim + 1)  # (*, L, 3, 3)
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])  # (B, L)
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    gt_pred_cam = torch.stack([gt_s, gt_x, gt_y], dim=-1)
    return gt_pred_cam


# ====== 3D to 2D ===== #


def project_to_bi01(points, bbx_xys, K_fullimg):
    """
    points: (B, L, J, 3)
    bbx_xys: (B, L, 3)
    K_fullimg: (B, L, 3, 3)
    """
    # p2d = project_p2d(points, K_fullimg)
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    p2d_bi01 = cvt_to_bi01_p2d(p2d, bbx_lurb)
    return p2d_bi01


def perspective_projection(points, K):
    # points: (B, L, J, 3)
    # K: (B, L, 3, 3)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum("...ij,...kj->...ki", K, projected_points.float())
    return projected_points[..., :-1]


# ====== 2D (bbx from j2d) ===== #


def normalize_kp2d(obs_kp2d, bbx_xys, clamp_scale_min=False):
    """
    Args:
        obs_kp2d: (B, L, J, 3) [x, y, c]
        bbx_xys: (B, L, 3)
    Returns:
        obs: (B, L, J, 3)  [x, y, c]
    """
    obs_xy = obs_kp2d[..., :2]  # (B, L, J, 2)
    obs_conf = obs_kp2d[..., 2]  # (B, L, J)
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]

    # Mark keypoints outside the bounding box as invisible
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible_mask = (
        (obs_xy[..., 0] < xy_min[..., None, 0])
        + (obs_xy[..., 0] > xy_max[..., None, 0])
        + (obs_xy[..., 1] < xy_min[..., None, 1])
        + (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible_mask
    if clamp_scale_min:
        scale = scale.clamp(min=1e-5)
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)

    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)


def normalize_kp2d_fullimg(obs_kp2d, img_wh):
    """
    Args:
        obs_kp2d: (B, L, J, 3) [x,y,c]
        img_wh: (B, L, 2)
    Returns:
        obs: (B, L, J, 3) [x,y,c]
    """
    obs_conf = obs_kp2d[..., 2]
    obs_xy = obs_kp2d[..., :2]
    normalized_obs_xy = obs_xy / img_wh.unsqueeze(-2)
    # Mark keypoints outside the image bounds as invisible
    outside_img_mask = torch.logical_or(normalized_obs_xy < -1, normalized_obs_xy > 1)
    outside_img_mask = outside_img_mask.any(dim=-1)
    obs_conf = obs_conf * ~outside_img_mask
    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)



def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """Args: (B, L, J, 3) [x,y,c] -> Returns: (B, L, 3)"""
    # Center
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Size
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # fit w and h into aspect-ratio
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # apply a common factor to enlarge the bounding box
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)


def safely_render_x3d_K(x3d, K_fullimg, thr):
    """
    Args:
        x3d: (B, L, V, 3), should as least have a safe points (not examined here)
        K_fullimg: (B, L, 3, 3)
    Returns:
        bbx_xys: (B, L, 3)
        i_x2d: (B, L, V, 2)
    """
    #  For each frame, update unsafe z (<thr) to safe z (max)
    x3d = x3d.clone()  # (B, L, V, 3)
    x3d_unsafe_mask = x3d[..., 2] < thr  # (B, L, V)
    if (x3d_unsafe_mask).sum() > 0:
        x3d[..., 2][x3d_unsafe_mask] = thr
        if False:
            from hmr4d.utils.wis3d_utils import make_wis3d

            wis3d = make_wis3d(name="debug-update-z")
            bs, ls, vs = torch.where(x3d_unsafe_mask)
            bs = torch.unique(bs)
            for b in bs:
                for f in range(x3d.size(1)):
                    wis3d.set_scene_id(f)
                    wis3d.add_point_cloud(x3d[b, f], name="unsafe")
                pass

    # renfer
    i_x2d = perspective_projection(x3d, K_fullimg)  # (B, L, V, 2)
    return i_x2d


def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    Args:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
    Returns:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """

    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)  # (L, 2, 2)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys


def bbx_xyxy_from_x(p2d):
    """
    Args:
        p2d: (*, V, 2) - Tensor containing 2D points.

    Returns:
        bbx_xyxy: (*, 4) - Bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """
    # Compute the minimum and maximum coordinates for the bounding box
    xy_min = p2d.min(dim=-2).values  # (*, 2)
    xy_max = p2d.max(dim=-2).values  # (*, 2)

    # Concatenate min and max coordinates to form the bounding box
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (*, 4)

    return bbx_xyxy


def bbx_xyxy_from_masked_x(p2d, mask):
    """
    Args:
        p2d: (*, V, 2) - Tensor containing 2D points.
        mask: (*, V) - Boolean tensor indicating valid points.

    Returns:
        bbx_xyxy: (*, 4) - Bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """
    # Ensure the shapes of p2d and mask are compatible
    assert p2d.shape[:-1] == mask.shape, "The shape of p2d and mask are not compatible."

    # Flatten the input tensors for batch processing
    p2d_flat = p2d.view(-1, p2d.shape[-2], p2d.shape[-1])
    mask_flat = mask.view(-1, mask.shape[-1])

    # Set masked out values to a large positive and negative value respectively
    p2d_min = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("inf")).to(p2d_flat))
    p2d_max = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("-inf")).to(p2d_flat))

    # Compute the minimum and maximum coordinates for the bounding box
    xy_min = p2d_min.min(dim=1).values  # (BL, 2)
    xy_max = p2d_max.max(dim=1).values  # (BL, 2)

    # Concatenate min and max coordinates to form the bounding box
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (BL, 4)

    # Reshape back to the original shape prefix
    bbx_xyxy = bbx_xyxy.view(*p2d.shape[:-2], 4)

    return bbx_xyxy


def bbx_xyxy_ratio(xyxy1, xyxy2):
    """Designed for fov/unbounded
    Args:
        xyxy1: (*, 4)
        xyxy2: (*, 4)
    Return:
        ratio: (*), squared_area(xyxy1) / squared_area(xyxy2)
    """
    area1 = (xyxy1[..., 2] - xyxy1[..., 0]) * (xyxy1[..., 3] - xyxy1[..., 1])
    area2 = (xyxy2[..., 2] - xyxy2[..., 0]) * (xyxy2[..., 3] - xyxy2[..., 1])
    # Check
    area1[~torch.isfinite(area1)] = 0  # replace inf in area1 with 0
    assert (area2 > 0).all(), "area2 should be positive"
    return area1 / area2


def get_mesh_in_fov_category(mask):
    """mask: (L, V)
    The definition:
    1. FullyVisible: The mesh in every frame is entirely within the field of view (FOV).
    2. PartiallyVisible: In some frames, parts of the mesh are outside the FOV, while other parts are within the FOV.
    3. PartiallyOut: In some frames, the mesh is completely outside the FOV, while in others, it is visible.
    4. FullyOut: The mesh is completely outside the FOV in every frame.
    """
    mask = mask.clone().cpu()
    is_class1 = mask.all()  #  FullyVisible
    is_class2 = mask.any(1).all() * ~is_class1  # PartiallyVisible
    is_class4 = ~(mask.any())  # PartiallyOut
    is_class3 = ~is_class1 * ~is_class2 * ~is_class4  # FullyOut

    mask_frame_any_verts = mask.any(1)
    assert is_class1.int() + is_class2.int() + is_class3.int() + is_class4.int() == 1
    class_type = is_class1.int() + 2 * is_class2.int() + 3 * is_class3.int() + 4 * is_class4.int()
    return class_type.item(), mask_frame_any_verts


def get_infov_mask(p2d, w_real, h_real):
    """
    Args:
        p2d: (B, L, V, 2)
        w_real, h_real: (B, L) or int
    Returns:
        mask: (B, L, V)
    """
    x, y = p2d[..., 0], p2d[..., 1]
    if isinstance(w_real, int):
        mask = (x >= 0) * (x < w_real) * (y >= 0) * (y < h_real)
    else:
        mask = (x >= 0) * (x < w_real[..., None]) * (y >= 0) * (y < h_real[..., None])
    return mask
