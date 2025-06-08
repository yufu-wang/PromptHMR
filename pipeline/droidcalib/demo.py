# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import sys
# sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import time
import argparse
import time

from droid_slam.droid import Droid
import video_calib


def print_intrinsics(_intr_est, size_factor, args):
    
    # rescale intrinsics
    intr_est = _intr_est.copy()
    intr_est[0:4:2] /= size_factor[0]
    intr_est[1:4:2] /= size_factor[1]

    # recover initial intrinsics
    example_imfile = sorted(os.listdir(args.imagedir))[0]
    image = cv2.imread(os.path.join(args.imagedir, example_imfile))
    h0, w0, _ = image.shape
    intr_initial = np.array([(w0+h0)/2, (w0+h0)/2, w0/2, h0/2])

    # print self-calibration result
    print("#" * 32)
    print("Initial intrinsics:")
    print("fx = {}, fy = {}, ppx = {}, ppy = {}".format(*intr_initial))

    print("Estimated intrinsics:")

    if args.camera_model == "pinhole" or args.camera_model == "focal":
        print("fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}".format(*intr_est))
    else:
        print("fx = {:.2f}, fy = {:.2f}, ppx = {:.2f}, ppy = {:.2f}, xi = {:.3f}".format(*intr_est))


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, image_size, calib, stride, num_images, 
                 camera_model="pinhole"):
    """ image generator """

    wd, ht  = image_size

    if len(calib) > 0:
        calib = np.loadtxt(calib, delimiter=" ")
    else:
        example_imfile = sorted(os.listdir(imagedir))[0]
        image = cv2.imread(os.path.join(imagedir, example_imfile))
        h0, w0, _ = image.shape
        calib = np.array([(w0+h0)/2, (w0+h0)/2, w0/2, h0/2, 0, 0, 0, 0, 0])

    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[:num_images:stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((ht * wd) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((ht * wd) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        if camera_model == "pinhole" or camera_model == "focal":
            intrinsics = torch.as_tensor([fx, fy, cx, cy])
        elif camera_model == "mei":
            intrinsics = torch.as_tensor([fx, fy, cx, cy, 0])
        else:
            raise Exception("Camera model not implemented! Choose either \
                             pinhole or mei model.")

        h1, w1 = (image.shape[1], image.shape[2])
        size_factor = [(w1 / w0), (h1 / h0)]
        intrinsics[0::2] *= size_factor[0]
        intrinsics[1::2] *= size_factor[1]

        yield t, image[None], intrinsics, size_factor


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", default="", type=str, help="path to calibration file if available")
    parser.add_argument("--opt_intr", action="store_true", help="activate self-calibration")
    parser.add_argument("--camera_model", default="pinhole", type=str, help="pinhole or mei or focal")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=2, type=int, help="frame stride")
    parser.add_argument("--num_images", default=None, type=int, help="using only the first n images")
    parser.add_argument("--video_calib", action="store_true", help="create video with intrinsics overlay")

    parser.add_argument("--weights", default="droidcalib.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size_target", default=[517, 384], help="image width and height; reduce to make inference faster.")
    parser.add_argument("--visualize", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    args.disable_vis = not args.visualize
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    start_time = time.time()

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics, sf) in tqdm(image_stream(args.imagedir, args.image_size_target, 
                                                    args.calib, args.stride, args.num_images, 
                                                    args.camera_model)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

        if args.video_calib:
            intr_t = droid.video.intrinsics[0].clone()
            intr_t[:4] *= 8.0
            intr_t = video_calib.rescale_intr(intr_t, size_factor=sf)
            video_calib.save_image(t, image, intr_t, (time.time() - start_time))

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est, intr_est = droid.terminate(image_stream(args.imagedir, args.image_size_target, 
                                                      args.calib, args.stride, args.num_images))

    print_intrinsics(intr_est, sf, args)

    if args.video_calib:
        intr_t = droid.video.intrinsics[0].clone() 
        intr_t[:4] *= 8.0
        intr_t = video_calib.rescale_intr(intr_t, size_factor=sf)
        video_calib.save_image(9999, image, intr_t, (time.time() - start_time))
        video_calib.create_video()