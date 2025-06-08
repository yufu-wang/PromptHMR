# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from prompt_hmr.smpl_family import SMPLX

SMPLX_MODEL_DIR = 'data/body_models/smplx/models/smplx'

class SMPLDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        smpl_head_depth: int = 3,
        smpl_head_hidden_dim: int = 256,
        inverse_depth: bool=False,
    ) -> None:
        """
        Predicts SMPL given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          smpl_head_depth (int): the depth of the MLP used to predict smpl
          smpl_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict smpl
        """
        super().__init__()
        self.inverse_depth = inverse_depth
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.smpl_token = nn.Embedding(2, transformer_dim) # smpl_loc
        
        self.pose_head = MLP(transformer_dim, smpl_head_hidden_dim, 22*6, smpl_head_depth)
        self.shape_head = MLP(transformer_dim, smpl_head_hidden_dim, 10, smpl_head_depth)
        self.transl_head = MLP(transformer_dim, smpl_head_hidden_dim, 2, smpl_head_depth)
        self.depth_head = MLP(transformer_dim, smpl_head_hidden_dim, 1, smpl_head_depth)
        self.initialize()


    def forward(
        self,
        cam_int: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        crossperson: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        """

        # Concatenate output tokens
        output_tokens = self.smpl_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        if dense_prompt_embeddings is not None:
            src = src + dense_prompt_embeddings

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, crossperson=crossperson)
        smpl_token, loc_token = hs[:,:2,:].permute(1,0,2)
        features = hs[:,:2,:]

        # Predictions
        pose = self.pose_head(smpl_token) + self.init_pose
        shape = self.shape_head(smpl_token) + self.init_betas
        depth_c = self.depth_head(loc_token) + self.init_depth
        transl_c = self.transl_head(loc_token) + self.init_transl

        transl = self.decode_transl(cam_int, transl_c, depth_c)

        return pose, shape, transl, transl_c, depth_c, features
    

    def decode_transl(self, cam_int, transl, depth):
        focal = cam_int.squeeze()[0,0]
        px, py = transl.unbind(-1)
        pz = depth.unbind(-1)[0]

        if self.inverse_depth:
            pz = 1 / (pz + 1e-6)

        tx = px * pz
        ty = py * pz
        tz = pz * focal / 1000
        t_full = torch.stack([tx, ty, tz], dim=-1)

        return t_full
    

    def initialize(self,):
        if self.inverse_depth:
            init_depth = torch.tensor([[1/10.]])
        else:
            init_depth = torch.tensor([[10.]])

        init_transl = torch.tensor([[0,0]])
        init_betas = torch.zeros([1,10])
        init_pose = torch.cat([torch.tensor([[1.,0,0,0,-1,0]]),
                               torch.tensor([[1.,0,0,0,1,0]]).tile(21)], dim=1)

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_transl', init_transl)
        self.register_buffer('init_depth', init_depth)

        nn.init.xavier_uniform_(self.pose_head.layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.shape_head.layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.transl_head.layers[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.depth_head.layers[-1].weight, gain=0.01)

        nn.init.constant_(self.pose_head.layers[-1].bias, 0)
        nn.init.constant_(self.shape_head.layers[-1].bias, 0)
        nn.init.constant_(self.transl_head.layers[-1].bias, 0)
        nn.init.constant_(self.depth_head.layers[-1].bias, 0)


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
