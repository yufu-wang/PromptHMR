# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from scipy.optimize import least_squares
import numpy as np

N_GRIDPOINTS = 1000

def proj_pinhole(points, fx, fy, ppx, ppy):
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    points_normalized = points[0:2, :] / points[2, :]
    point_proj[0, :] = points_normalized[0, :] * fx + ppx
    point_proj[1, :] = points_normalized[1, :] * fy + ppy
    return point_proj

def iproj_pinhole(uv, fx, fy, ppx, ppy):
    npoints = uv.shape[1]
    rays = np.zeros((3, npoints))
    rays[0, :] = (uv[0, :] - ppx) / fx
    rays[1, :] = (uv[1, :] - ppy) / fy
    rays[2, :] = 1
    return rays

def proj_radial(points, fx, fy, ppx, ppy, k1, k2):
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    points_normalized = points[0:2, :] / points[2, :]
    n = np.sum(points_normalized ** 2, axis=0)
    r = 1 + k1 * n + k2 * n ** 2
    point_proj[0, :] = points_normalized[0, :] * fx * r + ppx
    point_proj[1, :] = points_normalized[1, :] * fy * r + ppy
    return point_proj



def proj_mei(points, fx, fy, ppx, ppy, xi, k1=0, k2=0, k3=0):
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    points_normalized = points[0:2, :] / (points[2, :] + xi * 
                        np.sqrt(np.sum(points ** 2, axis=0)))

    r2 = np.sum(points_normalized ** 2, axis=0)
    m = 1 + k1 * r2 + k2 * np.power(r2, 2) + k3 * np.power(r2, 3)

    points_distorted = np.zeros((2, npoints))
    points_distorted[0, :] = m * points_normalized[0, :]
    points_distorted[1, :] = m * points_normalized[1, :]

    point_proj[0, :] = points_distorted[0, :] * fx + ppx
    point_proj[1, :] = points_distorted[1, :] * fy + ppy

    return point_proj


def iproj_mei(uv, fx, fy, ppx, ppy, xi):
    npoints = uv.shape[1]
    nXY = np.zeros((3, npoints))
    nXY[0, :] = (uv[0, :] - ppx) / fx
    nXY[1, :] = (uv[1, :] - ppy) / fy

    rays = np.zeros((3, npoints))
    rays[2, :] = (xi + np.sqrt(1 + (1 - xi**2) * (nXY[0, :]**2 + 
                 nXY[1, :]**2))) / (1 + nXY[0, :]**2 + nXY[1, :]**2) - xi
    rays[0, :] = nXY[0, :] * (rays[2, :] + xi)
    rays[1, :] = nXY[1, :] * (rays[2, :] + xi)
    return rays


def rotate_rodrigues(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    if rot_vecs.ndim > 1:
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]

    else:
        theta = np.linalg.norm(rot_vecs)
        rot_vecs = np.array([rot_vecs for _ in range(points.shape[1])]).T

    if theta == 0:
        return points

    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.true_divide(rot_vecs, theta)
        v[v == np.inf] = 0
        v = np.nan_to_num(v)

    dot = np.sum(points * v, axis=0)[np.newaxis, :]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points, axisa=0, axisb=0).T + dot * (1 - cos_theta) * v


def proj_general(points, intr):
    if intr.shape[0] == 4:
        points_proj = proj_pinhole(points, *intr)
    elif intr.shape[0] == 5:
        points_proj = proj_mei(points, *intr)
    elif intr.shape[0] == 6:
        points_proj = proj_radial(points, *intr)
    return points_proj

def iproj_general(points, intr):
    if intr.shape[0] == 4:
        points_proj = iproj_pinhole(points, *intr)
    elif intr.shape[0] == 5:
        points_proj = iproj_mei(points, *intr)
    elif intr.shape[0] == 6:
        points_proj = iproj_radial(points, *intr)
    return points_proj


def mapping_error(intr_a, intr_b, image_size):
    """
     method to compute the difference between two camera models
     (two sets of parameters)
     simulate 2D image points that cover the entire image
     project these points with optimal params p0 to obtain rays.
     Then reproject with disturbed params pd to return to
     image space.
     compute the residual between both camera models
     Important: changes in the intrinsics can be compensated by changes in the
     extrinsics. Thus, we must compute the
     deviation in the camera model accounting for such a compensation.
     i.e. compute the smallest possible deviation
     after adjusting the extrinsics.
     See also Hagemann et al. IJCV 2021
    """

    def cost(r, intr_a, intr_b, image_size):

        nx = int(np.sqrt(N_GRIDPOINTS))
        ny = int(np.sqrt(N_GRIDPOINTS))
        x = np.linspace(0, image_size[0], nx+4)[2:-2]
        y = np.linspace(0, image_size[1], ny+4)[2:-2]
        points = [[xi, yi, 1] for xi in x for yi in y]
        points = np.array(points).T

        if len(intr_a) == len(intr_b) and (intr_a == intr_b).all() and np.sum(r) == 0:
            residuals = np.zeros(2 * nx * ny)
            return residuals


        rays = iproj_general(points, intr_a)
        X_cam_opt = rotate_rodrigues(rays, r)
        points_proj = proj_general(X_cam_opt, intr_b)

        residuals = np.zeros(2 * nx * ny)
        residuals[::2] = points[0, :] - points_proj[0, :]
        residuals[1::2] = points[1, :] - points_proj[1, :]

        return residuals
    
    compensating_rotation = np.zeros(3)
    res = least_squares(cost, compensating_rotation, 
                        args=(intr_a, intr_b, image_size))
    residuals = res.fun
    return np.sqrt(np.mean(residuals**2))

