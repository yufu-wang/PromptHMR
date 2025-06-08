import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from scipy.ndimage import gaussian_filter

from .tools import checkerboard_geometry
from prompt_hmr.utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_matrix


def traj_filter(pred_vert_w, pred_j3d_w, sigma=3):
    """ Smooth the root trajetory (xyz) """
    root = pred_j3d_w[:, 0]
    root_smooth = torch.from_numpy(gaussian_filter(root, sigma=sigma, axes=0))

    pred_vert_w = pred_vert_w + (root_smooth - root)[:, None]
    pred_j3d_w = pred_j3d_w + (root_smooth - root)[:, None]
    return pred_vert_w, pred_j3d_w


def cam_filter(cam_r, cam_t, r_sigma=3, t_sigma=15):
    """ Smooth camera trajetory (SO3) """
    cam_q = matrix_to_quaternion(cam_r)
    r_smooth = torch.from_numpy(gaussian_filter(cam_q, sigma=r_sigma, axes=0))
    t_smooth = torch.from_numpy(gaussian_filter(cam_t, sigma=t_sigma, axes=0))

    r_smooth = r_smooth / r_smooth.norm(dim=1, keepdim=True)
    r_smooth = quaternion_to_matrix(r_smooth)
    return r_smooth,  t_smooth


def rotate_meshes(verts, xrot=3.142/6, yrot=3.142/6):
    # verts: [B, V, 3]
    center = verts.mean(dim=[0,1], keepdim=True)
    verts_c = verts - center

    xrot = axis_angle_to_matrix(torch.tensor([xrot, 0, 0])).to(verts)
    yrot = axis_angle_to_matrix(torch.tensor([0, yrot, 0])).to(verts)
    rot = xrot @ yrot
    verts_rot = torch.einsum('ij, bvj->bvi', rot, verts_c) + center
    return verts_rot


def align_meshes_to_gravity(verts, gravity_cam, floor_scale=2, floor_color=None):
    # gravity_cam: R_gc = Rpitch @ Rroll; describe camera pose wrt gravity
    b = len(verts)
    if verts.shape[0] < 2:
        verts = verts.repeat(2,1,1)

    device = verts.device
    pred_verts = verts.clone()

    R_gc = gravity_cam.to(device)
    R_wg = torch.tensor([[1,0,0],
                         [0,-1,0],
                         [0,0,-1]]).float().to(device) # from gravity_cam to gravity_world (floor)
    R = R_wg @ R_gc # R_wc

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_verts)
    offset = - fit_floor_height(pred_vert_gr, 'ransac', 'y')

    verts_world = torch.einsum('ij,bnj->bni', R, pred_verts)
    verts_world += offset
    [gv, gf, gc] = get_floor_mesh(verts_world, scale=floor_scale, 
                                  floor_color=floor_color)
    gv = torch.tensor(gv).to(device)
    gf = torch.tensor(gf).to(device)
    gc = torch.tensor(gc).to(device)
    verts_world = verts_world.to(device)[:b]

    return verts_world, [gv, gf, gc], R, offset


def align_meshes_to_ground(verts, floor_scale=2, 
                           floor_color=None):
    b = len(verts)
    if verts.shape[0] < 2:
        verts = verts.repeat(2,1,1)

    device = verts.device
    pred_verts = verts.clone()

    # align meshes to ground
    _, indices = pred_verts[:, :, [1]].max(dim=1, keepdim=True)
    lowest = torch.gather(pred_verts, 1, indices.repeat(1,1,3))

    _, indices = pred_verts[:, :, [1]].min(dim=1, keepdim=True)
    highest = torch.gather(pred_verts, 1, indices.repeat(1,1,3))

    lowest = lowest.reshape(1, -1, 3)
    highest = highest.reshape(1, -1, 3)

    pl = fit_plane(lowest, idx=-1)
    normal = pl[0, :3]
    offset = pl[0, -1]
    person_up = (highest-lowest).mean(dim=1).squeeze()
    if (person_up @ normal).sign() < 0:
        normal = -normal
        offset = -offset

    yup = torch.tensor([0, 1., 0]).to(device)
    R = align_a2b(normal, yup)

    pred_vert_gr = torch.einsum('ij,bnj->bni', R, pred_verts)
    offset = pred_vert_gr[:, :, 1].min()
    offset = - torch.tensor([0, offset, 0]).to(device)

    verts_world = torch.einsum('ij,bnj->bni', R, pred_verts)
    verts_world += offset
    [gv, gf, gc] = get_floor_mesh(verts_world, scale=floor_scale, 
                                  floor_color=floor_color)
    gv = torch.tensor(gv).to(device)
    gf = torch.tensor(gf).to(device)
    gc = torch.tensor(gc).to(device)
    verts_world = verts_world.to(device)[:b]

    return verts_world, [gv, gf, gc], R, offset


def get_floor_mesh(pred_vert_gr, z_start=0, z_end=-1, scale=1.5, floor_color=None):
    """ Return the geometry of the floor mesh """
    verts = pred_vert_gr.clone()

    # Scale of the scene
    # sx, sz = (verts.mean(1).max(0)[0] - verts.mean(1).min(0)[0])[[0, 2]]
    sx, sz = (verts.max(0)[0].max(0)[0] - verts.min(0)[0].min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * scale

    # Center X
    cx = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[0]] / 2.0
    cx = cx.item()

    # Center Z: optionally using only a subsection
    verts = verts[z_start:z_end]
    cz = (verts.mean(1).max(0)[0] + verts.mean(1).min(0)[0])[[2]] / 2.0
    cz = cz.item()

    if floor_color is None:
        v, f, vc, fc = checkerboard_geometry(length=scale, c1=cx, c2=cz, up="y")
    else:
        v, f, vc, fc = checkerboard_geometry(length=scale, c1=cx, c2=cz, up="y",
                                            color0=floor_color[0],
                                            color1=floor_color[1],)
    vc = vc[:, :3] * 255
  
    return [v, f, vc]


def fit_floor_height(points, method='ransac', axis='y'):
    """
    Find height offset.
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    axis = {'x':0, 'y':1, 'z':2}[axis]

    if method == 'lowest':
        offset = points[..., axis].min()
      
    elif method == 'average':
        z, _ = points[..., axis].min(dim=1)
        offset = z.mean()
      
    elif method == 'ransac':
        zs, _ = points[..., axis].min(dim=1)
        zs = np.sort(zs)

        alpha = 1.0
        min_z = np.min(zs)
        max_z = np.max(zs)
        zs = zs[zs <= min_z + (max_z - min_z) * alpha]

        inlier_thresh = 0.05 # 5cm
        best_inliers = 0
        best_z = 0.0
        for i in range(10_000):
            z = np.random.choice(zs)
            inliers = np.abs(zs - z) < inlier_thresh
            inliers = np.sum(inliers)
            if inliers > best_inliers:
                best_z = z
                best_inliers = inliers

        offset = np.median(zs[np.abs(zs - best_z) < inlier_thresh])

    height_offset = torch.zeros([3])
    height_offset[axis] = offset.item()

    return height_offset


def fit_plane(points, idx=-1):
    """
    From SLAHMR
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    *dims, N, D = points.shape
    mean = points.mean(dim=-2, keepdim=True)
    # (*, N, D), (*, D), (*, D, D)
    U, S, Vh = torch.linalg.svd(points - mean)
    normal = Vh[..., idx, :]  # (*, D)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    offset = offset.mean(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)

    
def align_a2b(a, b):
    # Find a rotation that align a to b
    v = torch.cross(a, b)
    # s = v.norm()
    c = torch.dot(a, b)
    R = torch.eye(3).to(a) + skew(v) + skew(v)@skew(v) * (1/(1+c))
    return R


def skew(a):
    v1, v2, v3 = a
    m = torch.tensor([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]]).float().to(a)
    return m


def vis_traj(traj_1, traj_2, savefolder, grid=5):
    """ Plot & compare the trajetories in the xy plane """
    os.makedirs(savefolder, exist_ok=True)

    for seq in traj_1:
        traj_gt = traj_1[seq]['gt']
        traj_1 = traj_1[seq]['pred']
        traj_w = traj_2[seq]['pred']

        vis_center = traj_gt[0]
        traj_1 = traj_1 - vis_center
        traj_w = traj_w - vis_center
        traj_gt = traj_gt - vis_center
        
        length = len(traj_gt)
        step = int(length/100)

        a1 = np.linspace(0.3, 0.90, len(traj_gt[0::step,0]))
        a2 = np.linspace(0.3, 0.90, len(traj_w[0::step,0]))

        plt.rcParams['figure.figsize']=4,3
        fig, ax = plt.subplots()
        colors = ['tab:green', 'tab:blue', 'tab:orange']
        ax.scatter(traj_gt[0::step,0], traj_gt[0::step,2], s=10, c='tab:grey', alpha=a1, edgecolors='none')
        ax.scatter(traj_w[0::step,0], traj_w[0::step,2], s=10, c='tab:blue', alpha=a2, edgecolors='none')
        ax.scatter(traj_1[0::step,0], traj_1[0::step,2], s=10, c='tab:orange', alpha=a1, edgecolors='none')
        ax.set_box_aspect(1)
        ax.set_aspect(1, adjustable='datalim')
        ax.grid(linewidth=0.4, linestyle='--')

        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(grid)) 
        fig.savefig(f'{savefolder}/{seq}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


