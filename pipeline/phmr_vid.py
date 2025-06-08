import numpy as np
from PIL import Image, ImageOps
from omegaconf import OmegaConf
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d

import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose

import sys
sys.path.insert(0, 'pipeline/gvhmr')
from prompt_hmr import load_model as load_phmr
from pipeline.gvhmr.hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from pipeline.gvhmr.hmr4d.utils.geo_transform import compute_cam_angvel
from pipeline.gvhmr.hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, normalize_kp2d
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix


def load_video_head():
    phmr_vid_cfg =  OmegaConf.load('data/pretrain/phmr_vid/prhmr_release_002.yaml')
    phmr_vid_ckpt = 'data/pretrain/phmr_vid/prhmr_release_002.ckpt'
    vid_head = DemoPL(
        pipeline=phmr_vid_cfg.model.pipeline,
        smplx_path='data/body_models/smplx/SMPLX_NEUTRAL.npz',
    )
    vid_head = vid_head.eval().cuda()
    vid_head.load_pretrained_model(phmr_vid_ckpt)
    return vid_head


class PromptHMR_Video():
    def __init__(self,):
        super().__init__()
        self.model = load_phmr('data/pretrain/phmr/checkpoint.ckpt')
        self.vid_head = load_video_head()
    
    @torch.no_grad()
    def run(self, images, results, mask_prompt=True):
        tracks = results['people']
        camera = results['camera']
        
        img_focal = camera['img_focal']
        img_center = camera['img_center']
 
        cam_intrinsic = torch.eye(3)
        cam_intrinsic[0, 0] = img_focal.item()
        cam_intrinsic[1, 1] = img_focal.item()
        cam_intrinsic[0, 2] = img_center[0].item() if isinstance(img_center, np.ndarray) else img_center[0]
        cam_intrinsic[1, 2] = img_center[1].item() if isinstance(img_center, np.ndarray) else img_center[1]
        
        dataset = PromptHMRVideoDataset(images, tracks, cam_intrinsic)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda x: x)

        for k, v in tracks.items():
            tracks[k]['smplx_pose'] = []
            tracks[k]['smplx_transl'] = []
            tracks[k]['smplx_betas'] = []
            tracks[k]['prhmr_img_feats'] = []
        
        # Image model
        for batch in dataloader:
            with autocast('cuda'):
                output = self.model(batch, mask_prompt=mask_prompt)
            
            for bid in range(len(batch)):
                track_id = batch[bid]['track_ids']

                for tid in track_id:
                    btid = track_id.index(tid)
                    tracks[tid]['smplx_pose'].append(output[bid]['rotmat'][btid])
                    tracks[tid]['smplx_transl'].append(output[bid]['transl'][btid])
                    tracks[tid]['smplx_betas'].append(output[bid]['betas'][btid])
                    tracks[tid]['prhmr_img_feats'].append(output[bid]['features'][btid])
                
        for k, v in tracks.items():
            tracks[k]['smplx_pose'] = torch.stack(tracks[k]['smplx_pose']).float()
            tracks[k]['smplx_transl'] = torch.stack(tracks[k]['smplx_transl']).float()
            tracks[k]['smplx_betas'] = torch.stack(tracks[k]['smplx_betas']).float()
            tracks[k]['prhmr_img_feats'] = torch.stack(tracks[k]['prhmr_img_feats']).float()

        # Video model
        print(f"Running PRHMR-Vid for tracks")
        for idx, k in enumerate(list(tracks.keys())):
            seqlen = tracks[k]['prhmr_img_feats'].shape[0]
            R_w2c = torch.eye(3).repeat(seqlen, 1, 1)
            cam_angvel = compute_cam_angvel(R_w2c)
            bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(tracks[k]['bboxes'])).numpy()
            
            smoothed = np.array([signal.medfilt(param, 11) for param in bbx_xys.T]).T
            bbx_xys = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
            bbx_xys = torch.from_numpy(bbx_xys).float()
            
            vitpose = torch.from_numpy(tracks[k]['vitpose'])
            vitpose = normalize_kp2d(vitpose, bbx_xys).float()
            
            batch = {
                "length": torch.tensor([seqlen]),
                "obs": vitpose[None],
                "bbx_xys": bbx_xys[None],
                "K_fullimg": cam_intrinsic[None, None].repeat_interleave(seqlen, 1),
                "cam_angvel": cam_angvel[None],
                "f_imgseq": tracks[k]['prhmr_img_feats'][None],
            }
            
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                prhmr_vid_output_w_kpts = self.vid_head.pipeline.forward(batch, train=False, postproc=False, static_cam=True)
            
            batch['obs'] = torch.zeros_like(batch['obs'])
            with torch.amp.autocast('cuda'):
                prhmr_vid_output = self.vid_head.pipeline.forward(batch, train=False, postproc=False, static_cam=True)
            
            contact_label = prhmr_vid_output['model_output']['static_conf_logits'].sigmoid() > 0.8
            smplx_pose_aa = torch.cat(
                [
                    prhmr_vid_output['pred_smpl_params_incam']['global_orient'][0], 
                    prhmr_vid_output['pred_smpl_params_incam']['body_pose'][0], 
                    torch.zeros(seqlen, 9).cuda(),
                ], dim=-1)
            rotmat = axis_angle_to_matrix(smplx_pose_aa.reshape(-1, 3)).reshape(-1, 25, 3, 3)
            hand_pose_rotmat = torch.zeros(rotmat.shape[0], 30, 3, 3).to(rotmat)
            rotmat = torch.cat([rotmat, hand_pose_rotmat], dim=1)

            hps_results = {
                'rotmat': rotmat.cpu().float().numpy(),
                'pose': smplx_pose_aa.cpu().numpy(),
                'shape': prhmr_vid_output['pred_smpl_params_incam']['betas'][0].cpu().float().numpy(),
                'trans': prhmr_vid_output_w_kpts['pred_smpl_params_incam']['transl'][0].cpu().float().numpy(),
                'contact': contact_label[0].cpu().numpy(),
                'static_conf_logits': prhmr_vid_output['model_output']['static_conf_logits'][0].cpu().numpy(),
            }
            results['people'][k]['smplx_cam'] = hps_results

        return results
    

def pad_image(item, IMG_SIZE=896):
    img = item['image_cv']
    size = np.array([img.shape[1], img.shape[0]])
    scale = IMG_SIZE / max(size)
    offset = (IMG_SIZE - scale * size) / 2

    img_pil = Image.fromarray(img)
    img_pil = ImageOps.contain(img_pil, (IMG_SIZE,IMG_SIZE))
    img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE,IMG_SIZE))
    img = np.array(img_pil)

    item['image_cv'] = img
    item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
    item['cam_int'][:,:2] *= scale
    item['cam_int'][:,:2,-1] += offset
    item['boxes'] *= scale
    item['boxes'][:,:2] += offset
    item['boxes'][:,2:4] += offset
    kpts = item.get('kpts', None)
    if kpts is not None:
        kpts[:,:,:2] *= scale
        kpts[:,:,:2] += offset
        item['kpts'] = kpts
    
    return item
    
    
class PromptHMRVideoDataset(Dataset):
    def __init__(self, images, tracks, cam_int):
        self.images = images
        self.tracks = tracks
        
        frames = set([x for t in tracks.values() for x in t['frames'].tolist()])
        self.frames = sorted(list(frames))
        self.cam_int = cam_int
        self.normalization = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225])
        ])

        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        idx = self.frames[idx]
        image_cv = self.images[idx]
        
        boxes = []
        kpts = []
        masks = []
        track_ids = []
        for person_id, track in self.tracks.items():
            if idx in track['frames']:
                bbox = track['bboxes'][track['frames']==idx]
                bbox = np.concatenate([bbox, np.ones_like(bbox)[...,:1]], axis=-1)
                
                kpt = track['keypoints_2d'][track['frames']==idx][:, :25]
                obj_masks = track['masks'][track['frames']==idx]
                
                mm = []
                for mask in obj_masks:
                    msk_size = int(896/14 * 4)
                    mask = Image.fromarray(mask)
                    mask = ImageOps.contain(mask, (msk_size,msk_size))
                    mask = ImageOps.pad(mask, size=(msk_size,msk_size))
                    mm.append(np.array(mask))
                mm = np.array(mm)
                mm = torch.from_numpy(mm).float()
                
                track_ids.append(person_id)
                boxes.append(bbox)
                kpts.append(kpt)
                masks.append(mm)
                
        boxes = torch.from_numpy(np.concatenate(boxes)).float()
        kpts = torch.from_numpy(np.concatenate(kpts)).float()
        masks = torch.from_numpy(np.concatenate(masks)).float()[:, None]
        
        cam_int_batch = self.cam_int.float()[None].repeat(len(boxes), 1, 1)
        item = {
            'boxes': boxes,
            'cam_int': cam_int_batch,
            'image_cv': image_cv,
            'track_ids': track_ids,
            'kpts': kpts,
            'masks': masks,
        }
        item = pad_image(item, IMG_SIZE=896)
        item['image'] = self.normalization(item['image_cv'])
        item['image_cv'] = torch.tensor(item['image_cv'])
        return item
