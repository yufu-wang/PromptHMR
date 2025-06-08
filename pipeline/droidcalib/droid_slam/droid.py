import os
import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo, opt_intr=args.opt_intr, camera_model=args.camera_model)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            # from visualization import droid_visualization
            from vis_headless import droid_visualization
            print('Using headless ...')
            self.visualizer = Process(target=droid_visualization, args=(self.video, '.'))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """
        ckpt = torch.load(weights)
            
        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in ckpt.items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, mask)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()
            
    def has_motion(self, tstamp, image, thresh=None):
        with torch.no_grad():
            return self.filterx.has_motion(tstamp, image, thresh)

    def terminate(self, stream=None, backend=True):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        if backend:
            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(7)

            torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        intr = self.video.intrinsics[0, :].clone()
        intr[:4] *= 8.0
        return camera_trajectory.inv().data.cpu().numpy(), intr.data.cpu().numpy()
    
    def compute_error(self):
        """ compute slam reprojection error """

        del self.frontend

        torch.cuda.empty_cache()
        self.backend(12)

        return self.backend.errors[-1]
