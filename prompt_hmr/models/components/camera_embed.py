# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import numpy as np
import einops
from .common import LayerNorm2d


class CameraEncoder(nn.Module):
    def __init__(self, patch_size=14, img_size=896):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.camera = FourierPositionEncoding(n=3, 
                                            num_bands=16, 
                                            max_resolution=64)
        
        self.conv = nn.Conv2d(1024+99, 1024, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(1024)

    def forward(self, img_embeddings, K):
        B, D, _h, _w = img_embeddings.shape
        device = img_embeddings.device

        with torch.no_grad():
            points = torch.stack([torch.arange(0,_h,1).reshape(-1,1).repeat(1,_w), 
                                torch.arange(0,_w,1).reshape(1,-1).repeat(_h,1)],-1).to(device).float() # [h,w,2]
            points = points * self.patch_size + self.patch_size // 2          # move to pixel space 
            points = points.expand(B, _h, _w, 2).reshape(B, -1, 2)            # (bs, N, 2): points

            rays = inverse_perspective_projection(points, K, distance=None)   # (bs, N, 3): rays
            rays_embeddings = self.camera(pos=rays)                           # (bs, N, 99): rays fourier embedding
            rays_embeddings = einops.rearrange(rays_embeddings, 'b (h w) c -> b c h w', h=_h, w=_w).contiguous()
            
        z = torch.concat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z
    

class CameraEncoder_focal(nn.Module):
    def __init__(self, patch_size=14, img_size=896):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        
        self.embed = nn.Linear(1, 32)
        self.conv = nn.Conv2d(1024+32, 1024, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(1024)

    def forward(self, img_embeddings, K):
        B, D, _h, _w = img_embeddings.shape

        focal = (K[:,0,0] + K[:,1,1]) / 2
        focal = focal / 1000 - 1
        focal_embeddings = self.embed(focal.reshape(-1, 1))
        focal_embeddings = einops.repeat(focal_embeddings, 'b d -> b d h w', h=_h, w=_w)

        z = torch.concat([img_embeddings, focal_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z
    

class CameraEncoder_pos(nn.Module):
    def __init__(self, patch_size=14, img_size=896):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.camera = FourierPositionEncoding(n=2, 
                                            num_bands=16, 
                                            max_resolution=64)
        
        self.conv = nn.Conv2d(1024+66, 1024, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(1024)

    def forward(self, img_embeddings, K):
        B, D, _h, _w = img_embeddings.shape
        device = img_embeddings.device

        with torch.no_grad():
            points = torch.stack([torch.arange(0,_h,1).reshape(-1,1).repeat(1,_w), 
                                torch.arange(0,_w,1).reshape(1,-1).repeat(_h,1)],-1).to(device).float() # [h,w,2]
            points = points * self.patch_size + self.patch_size // 2          # move to pixel space 
            points = points.expand(B, _h, _w, 2).reshape(B, -1, 2)            # (bs, N, 2): points

            pos_embeddings = self.camera(pos=points)                          # (bs, N, 66): rays fourier embedding
            pos_embeddings = einops.rearrange(pos_embeddings, 'b (h w) c -> b c h w', h=_h, w=_w).contiguous()
            
        z = torch.concat([img_embeddings, pos_embeddings], dim=1)
        z = self.norm(self.conv(z))

        return z
    
    

class FourierPositionEncoding(nn.Module):
    def __init__(self, n, num_bands, max_resolution):
        """
        Module that generate Fourier encoding - no learning involved
        """
        super().__init__()

        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n
    
    @property
    def channels(self):
        """
        Return the output dimension
        """        
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        encoding_size *= 2 # sin-cos
        encoding_size += num_dims # concat

        return encoding_size
    
    def forward(self, pos):
        """
        Forward pass that take rays as input and generate Fourier positional encodings
        """
        fourier_pos_enc = _generate_fourier_features(pos, num_bands=self.num_bands, max_resolution=self.max_resolution)
        return fourier_pos_enc
    

def _generate_fourier_features(pos, num_bands, max_resolution):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device

    # Linear frequency sampling
    min_freq = 1.0
    freq_bands = torch.stack([torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device) for res in max_resolution], dim=0)

    # Stacking
    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, n, -1)

    # Sin-Cos
    per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)

    # Concat with initial pos
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features


def inverse_perspective_projection(points, K, distance):
    """
    This function computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        K (bs,3,3): camera intrinsics params
        distance (bs, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """
    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum('bij,bkj->bki', torch.inverse(K), points)

    # Apply perspective distortion
    if distance == None:
        return points
    points = points * distance
    return points




