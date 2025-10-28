import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import tyro
import torch
import numpy as np
from glob import glob
from torch.amp import autocast

from ultralytics import YOLO
from data_config import SMPL_PATH, SMPLX_PATH
from prompt_hmr import load_model_from_folder
from prompt_hmr.smpl_family import SMPLX, SMPL
from prompt_hmr.utils.visualizer import draw_boxes, save_ply

from prompt_hmr.vis.traj import align_meshes_to_ground, align_meshes_to_gravity
from prompt_hmr.models.inference import prepare_batch, predict_masks
from prompt_hmr.vis.viser import viser_vis_human

from pipeline.camcalib.model import CameraRegressorNetwork
from segment_anything import SamPredictor, sam_model_registry


def main(image='data/examples/example_1.jpg', gravity_align=False, detect_conf=0.3, render_overlap=False):
    savedir = os.path.basename(image)
    os.makedirs(savedir, exist_ok=True)

    smplx = SMPLX(SMPLX_PATH).cuda()
    yolo = YOLO("data/pretrain/yolov8x.pt")
    phmr = load_model_from_folder('data/pretrain/phmr')

    # Prompt HMR
    img = cv2.imread(image)[:,:,::-1]
    detection = yolo(image, verbose=False, conf=detect_conf, classes=0)
    boxes = detection[0].boxes.data.cpu()
    inputs = [{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}]

    # Inference
    with torch.no_grad() and autocast('cuda'):
        batch = prepare_batch(inputs, img_size=896, interaction=False)
        output = phmr(batch, use_mean_hands=True)[0]

    # Reconstruction
    keys = ['pose', 'betas', 'transl', 'rotmat', 'vertices', 'body_joints', 'cam_int']
    output = {k:output[k].detach().cpu() for k in keys}
    torch.save(output, f'{savedir}/output.pt')

    # Render
    verts = output['vertices']
    focal = batch[0]['cam_int_original'][0,0,0]
    if render_overlap:
        print("please install Pytorch3D to enable overlay rendering.")
        from prompt_hmr.vis.renderer import Renderer
        renderer = Renderer(img.shape[1], img.shape[0], focal, bin_size=0)
        img_rend = renderer.render_meshes(verts, smplx.faces, img)
        cv2.imwrite(f'{savedir}/output.jpg', img_rend[:,:,::-1])
        print(f'Rendered image saved to {os.path.abspath(savedir)}/output.jpg')

    # Align to gravity coordinate (optional)
    camera = np.eye(4)
    if gravity_align:
        # Use SPEC to estimate gravity direction
        spec = CameraRegressorNetwork()
        spec = spec.load_ckpt('data/pretrain/camcalib_sa_biased_l2.ckpt').to('cuda')
        with torch.no_grad():
            preds = spec(img, transform_data=True)
            pred_vfov, pred_pitch, pred_roll = preds
            pred_f_pix = img.shape[0] / 2. / np.tan(pred_vfov / 2.)
            gravity_cam = spec.to_gravity_cam(pred_pitch, pred_roll)

        verts, [gv, gf, gc], R, T = align_meshes_to_gravity(
                                                        verts, 
                                                        gravity_cam,
                                                        floor_scale=2, 
                                                        floor_color=[[0.73, 0.78, 0.82], 
                                                                    [0.61, 0.69, 0.72]]
                                                        )
        cam_r = R.mT
        cam_t = - cam_r @ T
        camera = np.eye(4)
        camera[:3,:3] = cam_r
        camera[:3, 3] = cam_t
        floor = [gv.numpy(), gf.numpy()]

    else:
        rot_180 = np.eye(3)
        rot_180[1, 1] = -1
        rot_180[2, 2] = -1  
        verts = verts @ rot_180

        camera = np.eye(4)
        camera[:3, :3] = rot_180 @ camera[:3, :3] 
        camera[:3, 3] = camera[:3, 3] @ rot_180
        floor = None

    # Webapp visualization
    viser_vis_human(verts, smplx.faces, cameras=[camera], floor=floor, image=img)


if __name__ == '__main__':
    tyro.cli(main)