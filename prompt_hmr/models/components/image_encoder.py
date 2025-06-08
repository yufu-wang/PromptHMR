# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
import os
import math
import einops
import numpy as np
import torch
from torch import nn

from .common import LayerNorm2d
from ..dinov2.vision_transformer import vit_giant2, vit_large
    

class ImageEncoder(nn.Module):
    def __init__(
        self, 
        backbone: str = 'dinov2_vitb14',
        out_chans: int = 256,
    ):
        super().__init__()
        self.backbone = self.get_backbone(backbone)
        self.backbone.encoder.mask_token.requires_grad = False

        self.neck = nn.Sequential(
            nn.Conv2d(
                self.backbone.embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        b, s, d = x.shape
        h = w = np.sqrt(s).astype(int)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.neck(x)
        return x
    
    def get_backbone(self, backbone:str):
        # To allow for: DINOV2, SAM, RADIO, etc
        if backbone == 'dinov2_vitb14':
            net = Dinov2Backbone(name='dinov2_vitb14', pretrained=True)
        elif backbone == 'dinov2_vitl14':
            net = Dinov2Backbone(name='dinov2_vitl14', pretrained=True)
        elif backbone == 'dinov2_vitg14':
            net = Dinov2Backbone(name='dinov2_vitg14', pretrained=True)
        else:
            raise Exception('Backbone not implemented.')
        return net
    

class RadioEncoder(nn.Module):
    def __init__(
        self, 
        backbone: str = "radio_v2.5-l",
        out_chans: int = 1024,
    ):
        super().__init__()
        self.backbone = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                       version=backbone, 
                                       progress=True, 
                                       skip_validation=True)
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                self.backbone.model.patch_generator.embedder.out_features,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unnormalize(x)
        _, x = self.backbone(x, feature_fmt='NCHW') # return (summary, spatial_features)
        x = self.neck(x)
        return x
    
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H, W]
        # Unnormalize, as Radio normalizes inputs internally
        # Our encoder accepts normalized inputs to keep it consistent with DINOv2
        x = x * self.std + self.mean
        return x


class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=True, *args, **kwargs):
        super().__init__()
        
        if name == 'dinov2_vitb14':
            vit = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained)
        elif name == 'dinov2_vitl14':
            vit = vit_large(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
            ckpt = 'data/dinov2_vitl14_pretrain.pth'
            if os.path.exists(ckpt):
                ckpt = torch.load(ckpt, weights_only=True)
                _ = vit.load_state_dict(ckpt, strict=True)
            else:
                print('Not using DINOv2 weight')
        else:
            raise Exception('Backbone not implemented.')

        self.name = name
        self.encoder = vit
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim



    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        y = self.encoder.get_intermediate_layers(x)[0] 
        return y


# class Dinov2Backbone(nn.Module):
#     def __init__(self, name='dinov2_vitb14', pretrained=True, *args, **kwargs):
#         super().__init__()
#         self.name = name
#         self.encoder = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=pretrained)
#         self.patch_size = self.encoder.patch_size
#         self.embed_dim = self.encoder.embed_dim

#         # self.initialize_pos_embed()

#     def forward(self, x):
#         """
#         Encode a RGB image using a ViT-backbone
#         Args:
#             - x: torch.Tensor of shape [bs,3,w,h]
#         Return:
#             - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
#         """
#         assert len(x.shape) == 4
#         # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
#         y = self.encoder.get_intermediate_layers(x)[0] 
#         return y
    
#     def initialize_pos_embed(self, ):
#         pos_embed = self.encoder.pos_embed.detach()
#         class_pos_embed = pos_embed[:, 0]
#         patch_pos_embed = pos_embed[:, 1:]
#         N = pos_embed.shape[1] - 1
#         M = int(math.sqrt(N))

#         h, w = 896, 896  # hard-code this for now
#         dim = 1024
#         patch_size = self.encoder.patch_size
#         interpolate_offset = self.encoder.interpolate_offset
#         w0 = w // patch_size
#         h0 = h // patch_size

#         kwargs = {}
#         sx = float(w0 + interpolate_offset) / M
#         sy = float(h0 + interpolate_offset) / M
#         kwargs["scale_factor"] = (sx, sy)

#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
#             mode="bicubic",
#             antialias=self.encoder.interpolate_antialias,
#             **kwargs,
#         )
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

#         # re-initialize pos_embed for new resolution
#         self.encoder.pos_embed = nn.Parameter(pos_embed)
#         print('Re-initialize DINOv2 positional embedding')

#         return
        

