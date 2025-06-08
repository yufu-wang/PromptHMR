import torch
import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple
import pickle as pkl
from torch.amp import autocast

from prompt_hmr.utils.rotation_utils import rotation_6d_to_matrix
from prompt_hmr.smpl_family import SMPLX, SMPL
from .components import ImageEncoder, PromptEncoder, SMPLDecoder

SMPLX_MODEL_DIR = 'data/body_models/smplx'
SMPL_MODEL_DIR = 'data/body_models/smpl'
SMPLX2SMPL = 'data/body_models/smplx2smpl.pkl'

class PHMR(pl.LightningModule):
    def __init__(
        self, 
        cfg,
        cam_encoder,
        image_encoder: ImageEncoder,
        prompt_encoder: PromptEncoder,
        smpl_decoder: SMPLDecoder,
        train: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMG_SIZE
        self.transl_type = cfg.MODEL.TRANSL
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.smpl_decoder = smpl_decoder
        self.smplx = SMPLX(SMPLX_MODEL_DIR)
        self.smpl = SMPL(SMPL_MODEL_DIR)
        self.cam_encoder = cam_encoder
        
        smplx2smpl = torch.from_numpy(pkl.load(open(SMPLX2SMPL, 'rb'))['matrix'])
        self.register_buffer('smplx2smpl', smplx2smpl.float())

        self.is_train = train
        self.mpjpe = []
        self.pa_mpjpe = []
        self.ca_mpjpe = []
        self.mpjpe_kpt = []
        self.pa_mpjpe_kpt = []
        self.pck10 = []
        self.pck05 = []

    def forward(
        self, 
        batch: List[Dict[str, Any]],
        box_prompt: bool = True,
        kpt_prompt: bool = True,
        text_prompt: bool = True,
        mask_prompt: bool = True,
        interaction_prompt: bool = True,
        use_mean_hands: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        
        device = self.prompt_encoder._get_device()
        K = torch.cat([b['cam_int'] for b in batch]).to(device)

        images = torch.stack([x["image"] for x in batch], dim=0)
        images = images.to(device)
        image_embeddings = self.image_encoder(images)
        image_embeddings = self.cam_encoder(image_embeddings, K)

        outputs = []
        for i in range(len(images)):
            cam_int =  batch[i]['cam_int'].to(device)
            boxes = batch[i].get('boxes', None) if box_prompt else None
            kpts = batch[i].get('kpts', None) if kpt_prompt else None
            text = batch[i].get('text', None) if text_prompt else None
            masks = batch[i].get('masks', None) if mask_prompt else None
            interact = batch[i].get('interaction', False) if interaction_prompt else False
            
            img_pe = self.prompt_encoder.get_dense_pe()
            img_embed = image_embeddings[[i]]
            prompt_embed, dense_embed = self.prompt_encoder(boxes, text, kpts, masks)

            predictions = self.smpl_decoder(cam_int, img_embed, img_pe, prompt_embed, dense_embed, interact)
            output = {'pose': predictions[0],
                      'betas': predictions[1],
                      'transl': predictions[2],
                      'transl_c': predictions[3],
                      'inv_depth_c': predictions[4],
                      'cam_int': cam_int.expand(len(prompt_embed),3,3),
                      'features': predictions[5],
                    }
            outputs.append(output)

        if self.is_train:
            outputs_ = {}
            for k in ['pose', 'betas', 'transl', 'transl_c', 'inv_depth_c', 'cam_int']:
                outputs_[k] = torch.cat([out[k] for out in outputs])
            outputs = self.process_output(outputs_, use_mean_hands)
        else:
            outputs = [self.process_output(v, use_mean_hands) for v in outputs]

        return outputs


    def process_output(self, output, use_mean_hands): 
        K = output['cam_int']
        transl = output['transl'].reshape(-1, 3) 
        shape = output['betas'].reshape(-1, 10)
        pose = output['pose'].reshape(-1, 22, 6)
        rotmat = rotation_6d_to_matrix(pose)
        B = len(rotmat)

        # smplx
        with autocast('cuda', enabled=False):
            if self.transl_type == 'root':
                root = self.smplx(betas=shape).joints[:,0]
                transl = transl - root.detach()
                output['transl'] = transl

            if use_mean_hands:
                smplx_out = self.smplx(global_orient=rotmat[:,:1],
                                    body_pose=rotmat[:,1:],
                                    betas=shape,
                                    transl=transl,
                                    left_hand_pose=self.smplx.hands_mean[:,:15].repeat(B,1,1,1),
                                    right_hand_pose=self.smplx.hands_mean[:,15:].repeat(B,1,1,1))
            else:
                smplx_out = self.smplx(global_orient=rotmat[:,:1],
                                    body_pose=rotmat[:,1:],
                                    betas=shape,
                                    transl=transl)
        
            vertices = smplx_out.vertices
            body_joints = smplx_out.body_joints

            output['rotmat'] = rotmat.reshape(B, -1, 3, 3)
            output['vertices'] = vertices.reshape(B, -1, 3)
            output['body_joints'] = body_joints.reshape(B, -1, 3) 

            pred_joints = output['body_joints'] 
            pred_joints = pred_joints/(pred_joints[...,[-1]] + 1e-5)
            pred_joints = torch.einsum('bij, bvj->bvi', K.reshape(-1,3,3), pred_joints)[...,:2]
            output['body_joints2d'] = pred_joints.clamp(-2e3, 2e3)

            # smpl
            smpl_verts = self.smplx2smpl @ vertices
            smpl_joints = self.smpl.joints_from_vertices(smpl_verts)

            smpl_joints2d = smpl_joints 
            smpl_joints2d = smpl_joints2d/(smpl_joints2d[...,[-1]] + 1e-5)
            smpl_joints2d = torch.einsum('bij, bvj->bvi', K.reshape(-1,3,3), smpl_joints2d)[...,:2]

            output['smpl_vertices'] = smpl_verts
            output['smpl_joints'] = smpl_joints
            output['smpl_joints2d'] = smpl_joints2d.clamp(-2e3, 2e3)
            output['smpl_j3d'] = self.smpl.J_regressor @ smpl_verts
            
        return output

