# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import sys
import cv2
import torch
import joblib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from scipy.stats import norm
from skimage.io import imsave
import torch.nn.functional as F
from PIL import Image, ImageDraw
from collections import OrderedDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .resnet import resnet50, resnet101


# CKPT = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/logs/cam_reg/pano_scalenet_v3_softargmax_l2_lw10/22-02-2021_18-32-07_pano_scalenet_v3_softargmax_l2_lw10_train/tb_logs/0/checkpoints/epoch=26-step=337742.ckpt'


def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        # logger.warning(f'Removing "model." keyword from state_dict keys..')
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
                        # logger.warning(f'size mismatch for \"{pk}\": copying a param with shape {state_dict[pk].shape} '
                                    #    f'from checkpoint, the shape in current model is {model_state_dict[pk].shape}')

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            # logger.warning(f'Updated \"{pk}\" param to {updated_pretrained_state_dict[pk].shape} ')
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model


def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
    return images


def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)


def softargmax1d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, dim = heatmaps.shape
    points = torch.arange(0, dim, device=device, dtype=dtype).reshape(1, 1, dim).expand(batch_size, -1, -1)
    # y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW

    # points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (normalized_heatmap.reshape(batch_size, -1, dim) * points).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints = (keypoints / (dim - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, dim)
    
    
def get_bins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)

    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= (pdf.max())
    pdf *= alpha
    pdf = pdf.max()*beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size//2]

    return cumsum


pitch_bins = np.linspace(-0.6, 0.6, 255)
pitch_bins_centers = pitch_bins.copy()
pitch_bins_centers[:-1] += np.diff(pitch_bins_centers)/2
pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

horizon_bins = np.linspace(-0.5, 1.5, 255)
horizon_bins_centers = horizon_bins.copy()
horizon_bins_centers[:-1] += np.diff(horizon_bins_centers)/2
horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

roll_bins = get_bins(-np.pi/6, np.pi/6, 0.5, 0.04, 1.1, np.pi)
# roll_bins = get_bins(-np.pi/6, np.pi/6, 0.2, 0.04, 1.1, np.pi/3)
roll_bins_centers = roll_bins.copy()
roll_bins_centers[:-1] += np.diff(roll_bins_centers)/2
roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

vfov_bins = np.linspace(0.2617, 2.1, 255)
vfov_bins_centers = vfov_bins.copy()
vfov_bins_centers[:-1] += np.diff(vfov_bins_centers)/2
vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

roll_new_bins = np.linspace(-0.6, 0.6, 255)
roll_new_bins_centers = roll_new_bins.copy()
roll_new_bins_centers[:-1] += np.diff(roll_new_bins_centers)/2
roll_new_bins_centers = np.append(roll_new_bins_centers, roll_new_bins[-1])


def bins2horizon(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return horizon_bins_centers[idxes]


def bins2pitch(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return pitch_bins_centers[idxes]


def bins2roll(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return roll_bins_centers[idxes]


def bins2vfov(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return vfov_bins_centers[idxes]


def vfov2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(vfov_bins), max=np.max(vfov_bins))


def pitch2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(pitch_bins), max=np.max(pitch_bins))


def roll2soft_idx(angle):
    return angle_to_soft_idx(angle, min=-0.6, max=0.6)


def angle_to_soft_idx(angle, min, max):
    return 2 * ((angle - min) / (max - min)) - 1


def soft_idx_to_angle(soft_idx, min, max):
    return (max - min) * ((soft_idx + 1) / 2) + min


def get_softargmax(pred):
    pred = pred.unsqueeze(1)  # (N, 1, 256)
    pred_argmax, _ = softargmax1d(pred, normalize_keypoints=True)  # (N, 1, 1)
    pred_argmax = pred_argmax.reshape(-1)
    return pred_argmax


@torch.no_grad()
def convert_preds_to_angles(pred_vfov, pred_pitch, pred_roll, loss_type='kl', return_type='torch', legacy=False):
    if loss_type in ('kl', 'ce'):
        pred_vfov = bins2vfov(pred_vfov)
        pred_pitch = bins2pitch(pred_pitch)
        pred_roll = bins2roll(pred_roll)
    elif loss_type in ('softargmax_l2', 'softargmax_biased_l2'):
        pred_vfov = soft_idx_to_angle(get_softargmax(pred_vfov),
                                      min=np.min(vfov_bins), max=np.max(vfov_bins))
        pred_pitch = soft_idx_to_angle(get_softargmax(pred_pitch),
                                       min=np.min(pitch_bins), max=np.max(pitch_bins))
        if not legacy:
            pred_roll = soft_idx_to_angle(get_softargmax(pred_roll), min=-0.6, max=0.6)
        else:
            pred_roll = bins2roll(pred_roll)

    if return_type == 'np' and isinstance(pred_vfov, torch.Tensor):
        return pred_vfov.cpu().numpy(), \
               pred_pitch.cpu().numpy(), \
               pred_roll.cpu().numpy()

    if return_type == 'torch' and isinstance(pred_vfov, np.ndarray):
        return torch.from_numpy(pred_vfov), torch.from_numpy(pred_pitch), torch.from_numpy(pred_roll)

    return pred_vfov, pred_pitch, pred_roll


class ImageFolder(Dataset):
    def __init__(
            self,
            image_list,
            min_size=600,
            max_size=1000,
    ):
        self.images = image_list

        self.data_transform = transforms.Compose([
            transforms.Resize(min_size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = {}

        if isinstance(self.images[index], str):
            imgname = self.images[index]
            img = Image.open(imgname).convert('RGB')
            orig_img_shape = np.array(img).shape[:2]

        elif isinstance(self.images, np.ndarray):
            img = self.images[index]
            orig_img_shape = img.shape[:2]
            img = Image.fromarray(img)
            imgname = index

        norm_img = self.data_transform(img)

        item['img'] = norm_img
        item['imgname'] = imgname

        item['orig_shape'] = orig_img_shape

        return item
    

def show_horizon_line(
        image, vfov, pitch, roll, focal_length=-1,
        color=(0, 255, 0), width=5, debug=False, GT=False, text_size=16,
):
    """
    Angles should be in radians.
    """
    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = image.astype('uint8')

    if debug:
        if GT == False:
            image[0:text_size,:,:] = 0
        else:
            image[h-text_size:h,:,:] = 0

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    # text_size =  h // 25
    # fnt = ImageFont.truetype("/usr/share/fonts/truetype/gentium-basic/GenBasR.ttf", text_size)

    ctr = h * (0.5 - 0.5 * np.tan(pitch) / np.tan(vfov / 2))
    l = ctr - w * np.tan(roll) / 2
    r = ctr + w * np.tan(roll) / 2
    if debug:
        if GT == False:
            draw.text(
                (0, 0),
                "vfov:{0:.1f}, pitch:{1:.1f}, roll:{2:.1f}, f_pix:{3:.1f}".format(
                    np.degrees(vfov), np.degrees(pitch), np.degrees(roll), focal_length
                ),
                (255, 255, 255),
                # font=fnt,
            )
        else:
            draw.text(
                (0, h-text_size),
                "GT: vfov:{0:.1f}, pitch:{1:.1f}, roll:{2:.1f}, f_pix:{3:.1f}".format(
                    np.degrees(vfov), np.degrees(pitch), np.degrees(roll), focal_length
                ),
                (255, 255, 255),
                # font=fnt,
            )

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im), ctr/h


def get_backbone_info(backbone):
    info = {
        'resnet18': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet34': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet50': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_adf_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet101': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet152': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext50_32x4d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext101_32x8d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet50_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet101_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'mobilenet_v2': {'n_output_channels': 1280, 'downsample_rate': 4},
        'hrnet_w32': {'n_output_channels': 480, 'downsample_rate': 4},
        'hrnet_w48': {'n_output_channels': 720, 'downsample_rate': 4},
        # 'hrnet_w64': {'n_output_channels': 2048, 'downsample_rate': 4},
        'dla34': {'n_output_channels': 512, 'downsample_rate': 4},
    }
    return info[backbone]


class CameraRegressorNetwork(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            num_fc_layers=1,
            num_fc_channels=1024,
            num_out_channels=256,
    ):
        super(CameraRegressorNetwork, self).__init__()
        self.backbone = eval(backbone)(pretrained=False)

        self.num_out_channels = num_out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channels = get_backbone_info(backbone)['n_output_channels']

        assert num_fc_layers > 0, 'Number of FC layers should be more than 0'
        if num_fc_layers == 1:
            self.fc_vfov = nn.Linear(out_channels, num_out_channels)
            self.fc_pitch = nn.Linear(out_channels, num_out_channels)
            self.fc_roll = nn.Linear(out_channels, num_out_channels)

            nn.init.normal_(self.fc_vfov.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_vfov.bias, 0)

            nn.init.normal_(self.fc_pitch.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_pitch.bias, 0)

            nn.init.normal_(self.fc_roll.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_roll.bias, 0)

        else:
            self.fc_vfov = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_pitch = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_roll = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)

    def _get_fc_layers(self, num_layers, num_channels, inp_channels):
        modules = []

        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(inp_channels, num_channels))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_channels, self.num_out_channels))
            else:
                modules.append(nn.Linear(num_channels, num_channels))

        return nn.Sequential(*modules)

    def forward(self, images):
        x = self.backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        vfov = self.fc_vfov(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        return [vfov, pitch, roll]


def test_model():
    backbones = ['resnet50', 'resnet34']
    num_fc_layers = [1, 2, 3]
    num_fc_channels = [256, 512, 1024]
    img_size = [(224, 224), (480,640), (500, 450)]
    from itertools import product

    # print(list(product(backbones, num_fc_layers, num_fc_channels)))
    inp = torch.rand(1, 3, 128, 128)

    for (b, nl, nc, im_size) in list(product(backbones, num_fc_layers, num_fc_channels, img_size)):
        print('backbone', b, 'n_f_layer', nl, 'n_ch', nc, 'im_size', im_size)
        inp = torch.rand(1, 3, *im_size)
        model = CameraRegressorNetwork(backbone=b, num_fc_layers=nl, num_fc_channels=nc)
        out = model(inp)

        breakpoint()
        print('vfov', out[0].shape, 'pitch', out[1].shape, 'roll', out[2].shape)


@torch.no_grad()
def run_spec_calib(images, out_folder=None, loss_type='softargmax_l2', save_res=False, stride=1, first_frame_idx=0):
    # img_folder = args.img_folder
    # out_folder = args.out_folder
    # loss_type = args.loss
    images = images[first_frame_idx:]
    images = images[::stride]
    if isinstance(images, np.ndarray):
        imgsize = images[0].shape[:2]
    elif isinstance(images[0], str):
        imgsize = cv2.imread(images[0]).shape[:2]
        
    val_dataset = ImageFolder(images, min_size=min(imgsize))

    device = 'cuda'

    model = CameraRegressorNetwork(
        backbone='resnet50',
        num_fc_layers=1,
        num_fc_channels=1024,
    ).to(device)
    
    CKPT = 'data/pretrain/camcalib_sa_biased_l2.ckpt'
    
    if os.path.exists('/.dockerenv') or 'AWS_DEFAULT_REGION' in os.environ.keys():
        CKPT = os.path.abspath(CKPT.replace('data', '/code/data'))

    ckpt = torch.load(CKPT)
        
    model = load_pretrained_model(model, ckpt['state_dict'], remove_lightning=True, strict=True)

    # logger.info('Loaded pretrained model')

    model.eval()

    os.makedirs(out_folder, exist_ok=True)

    focal_length = []

    # logger.info('Running CamCalib')
    
    results = {}
    from contextvars import ContextVar
    tqdm_disabled = ContextVar("tqdm_disabled", default=False)
    
    # for idx, batch in enumerate(tqdm(val_dataset, disable=tqdm_disabled.get())):
    for idx, batch in enumerate(val_dataset):

        img_fname = batch['imgname']
        # results_file = os.path.join(output_path, os.path.basename(img_fname).split('.')[0] + '.pkl')

        img = batch['img'].unsqueeze(0).to(device).float()

        preds = model(img)

        pred_distributions = preds

        batch_img = img
        batch_img = denormalize_images(batch_img) * 255
        batch_img = np.transpose(batch_img.cpu().numpy(), (0, 2, 3, 1))

        extract = lambda x: x.detach().cpu().numpy().squeeze()
        img = batch_img[0].copy()

        if loss_type in ('kl', 'ce'):
            pred_vfov, pred_pitch, pred_roll = map(extract, preds)
            pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
                pred_vfov, pred_pitch, pred_roll, loss_type=loss_type,
                return_type='np',
            )
        else:
            preds = convert_preds_to_angles(
                *preds, loss_type=loss_type,
            )
            pred_vfov = extract(preds[0])
            pred_pitch = extract(preds[1])
            pred_roll = extract(preds[2])

        orig_img_w, orig_img_h = batch['orig_shape']

        pred_f_pix = orig_img_h / 2. / np.tan(pred_vfov / 2.)

        pitch = np.degrees(pred_pitch)
        roll = np.degrees(pred_roll)
        vfov = np.degrees(pred_vfov)

        results[img_fname] = {
            'vfov': pred_vfov.item(),
            'f_pix': pred_f_pix,
            'pitch': pred_pitch.item(),
            'roll': pred_roll.item(),
        }
        if idx == 0:
            results['first_frame'] = {
                'vfov': pred_vfov.item(),
                'f_pix': pred_f_pix,
                'pitch': pred_pitch.item(),
                'roll': pred_roll.item(),
            }
            img, _ = show_horizon_line(img.copy(), pred_vfov, pred_pitch, pred_roll, focal_length=-1,
                                    debug=True, color=(255, 0, 0), width=3, GT=False)
            imsave(os.path.join(out_folder, '0000.jpg'), img)

        focal_length.append(pred_f_pix)
    
    results['avg_focal_length'] = np.mean(focal_length)
    results['min_focal_length'] = np.min(focal_length)
    results['max_focal_length'] = np.max(focal_length)
    results['std_focal_length'] = np.std(focal_length)
    results['median_focal_length'] = np.median(focal_length)
    return results


@torch.no_grad()
def run_wildcam_calib(images=None, out_folder=None, save_res=False, stride=1, first_frame_idx=0):
    
    images = images[first_frame_idx:]
    images = images[::stride]
    
    results = {}
    
    model = torch.hub.load('mkocabas/PerspectiveFields', "perspective_fields", pretrained=True, verbose=False)
    if isinstance(images, np.ndarray):
        first_img_bgr = images[0][..., ::-1]
    elif isinstance(images[0], str):
        first_img_bgr = cv2.imread(images[0])
    pred = model.inference(first_img_bgr)
    pred_roll = -pred['pred_roll'].deg2rad().item() # degrees
    pred_pitch = -pred['pred_pitch'].deg2rad().item() # degrees
    pred_vfov = pred['pred_vfov'].deg2rad().item() # degrees
    del model
    
    results['first_frame'] = {
        'vfov': pred_vfov,
        'f_pix': None,
        'pitch': pred_pitch,
        'roll': pred_roll,
    }
    
    os.makedirs(out_folder, exist_ok=True)
    if isinstance(images, np.ndarray):
        img = images[0]
    elif isinstance(images[0], str):
        img = cv2.imread(images[0])[..., ::-1]
        
    img, _ = show_horizon_line(img.copy(), pred_vfov, pred_pitch, pred_roll, focal_length=-1,
                               debug=True, color=(255, 0, 0), width=3, GT=False)
    imsave(os.path.join(out_folder, '0000.jpg'), img)

    model = torch.hub.load('ShngJZ/WildCamera', "WildCamera", pretrained=True, verbose=False).to('cuda')
    model.eval()
    
    focal_length = []
    for idx, img in enumerate(images):
        if isinstance(img, np.ndarray):
            rgb = Image.fromarray(img)
        elif isinstance(img, str):
            rgb = Image.open(img)
        intrinsic, _ = model.inference(rgb, wtassumption=True)

        if idx == 0:
            results['first_frame']['f_pix'] = intrinsic[0, 0].item()

        results[f'{idx}'] = {'f_pix': intrinsic[0, 0].item()}
        focal_length.append(intrinsic[0, 0].item())
        
    results['avg_focal_length'] = np.mean(focal_length)
    results['min_focal_length'] = np.min(focal_length)
    results['max_focal_length'] = np.max(focal_length)
    results['std_focal_length'] = np.std(focal_length)
    results['median_focal_length'] = np.median(focal_length)
    return results


def run_cam_calib(
    images=None, img_folder=None, out_folder=None, 
    loss_type='softargmax_l2', save_res=False, stride=1,
    method='spec', first_frame_idx=0,
):
    
    if method == 'spec':
        return run_spec_calib(images, out_folder, loss_type, save_res, stride, first_frame_idx)
    elif method == 'wildcam':
        return run_wildcam_calib(images, out_folder, save_res, stride, first_frame_idx)


if __name__ == '__main__':
    run_cam_calib(sys.argv[1], sys.argv[2], save_res=True, stride=30)