import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        clip_encoder,
        text_prompt: bool = True,
        kpt_prompt: bool = False,
        mask_prompt: bool = False,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # For Box prompt (mandatory)
        self.num_point_embeddings: int = 2  # 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_embeddings = nn.Embedding(1, embed_dim)

        # For Text prompt
        if text_prompt:
            self.clip_encoder = clip_encoder
            self.use_text_prompt = True
        else:
            self.use_text_prompt = False
        
        # For Keypoint prompt
        if kpt_prompt:
            self.num_kpts: int = 17   # 17 vitpose keypoint
            self.kpt_embeddings = nn.Embedding(self.num_kpts, embed_dim)
            self.not_kpt_embeddings = nn.Embedding(1, embed_dim)
            self.use_kpt_prompt = True
        else:
            self.use_kpt_prompt = False

        # For Mask prompt
        if mask_prompt:
            mask_in_chans = 16  # same as in SAM
            self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
            self.mask_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            )
            self.no_mask_embed = nn.Embedding(1, embed_dim)
            self.use_mask_prompt = True
        else:
            self.use_mask_prompt = False

        

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes.to(self._get_device())
        conf = boxes[..., [-1]].reshape(-1,1,1)
        boxes = boxes[..., :-1]

        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[0].weight
        corner_embedding[:, 1, :] += self.point_embeddings[1].weight
        corner_embedding = corner_embedding * conf + self.not_embeddings.weight * (1-conf)

        return corner_embedding
    
    
    def _embed_text(self, text: list[str]) -> torch.Tensor:
        """Embeds text prompts."""
        bn = len(text)
        text_features = self.clip_encoder.encode_text(text)
        text_features = text_features.reshape(bn, -1, self.embed_dim)
        null_features = self.not_embeddings.weight
        
        use_null = np.array(text) == 'NULL'
        use_null = torch.tensor(use_null).to(text_features).reshape(bn,1,1)

        text_features = use_null * null_features + (1-use_null) * text_features

        return text_features
    

    def _embed_kpts(self, kpts: torch.Tensor) -> torch.Tensor:
        """Embeds kpts prompts."""
        kpts = kpts.to(self._get_device())

        conf = kpts[..., 2:]
        loc = kpts[..., :2] + 0.5  # Shift to center of pixel

        kpt_embedding = self.pe_layer.forward_with_coords(loc, self.input_image_size)
        kpt_embedding = kpt_embedding + self.kpt_embeddings.weight
        kpt_embedding = kpt_embedding * conf + self.not_kpt_embeddings.weight * (1-conf)

        return kpt_embedding
    

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        masks = masks.to(self._get_device())
        conf = (masks.sum(dim=[-1,-2], keepdim=True) > 1e-3).float()
        mask_embedding = self.mask_downscaling(masks)
        mask_embedding = mask_embedding * conf + self.no_mask_embed.weight.reshape(1, -1, 1, 1) * (1-conf)

        return mask_embedding


    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device


    def forward(
        self,
        boxes: Optional[torch.Tensor],
        text: Optional[list]=None,
        kpts: Optional[torch.Tensor]=None,
        masks: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          boxes (torch.Tensor or none): boxes to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
        """
        if boxes is not None:
            bs = boxes.shape[0]
        elif text is not None:
            bs = len(text)
        elif masks is not None:
            bs = len(masks)
        
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        
        # BOX Prompt
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        else:
            box_embeddings = torch.zeros((bs, 2, self.embed_dim), device=self._get_device())
            box_embeddings += self.not_embeddings.weight
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # TEXT Prompt
        if text is not None and self.use_text_prompt:
            text_embeddings = self._embed_text(text)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
        elif self.use_text_prompt:
            text_embeddings = torch.zeros((bs, 1, self.embed_dim), device=self._get_device())
            text_embeddings += self.not_embeddings.weight
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        # MASK Prompt:
        if masks is not None and self.use_mask_prompt:
            dense_embeddings = self._embed_masks(masks)
        elif self.use_mask_prompt:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        else:
            dense_embeddings = None

    
        return sparse_embeddings, dense_embeddings
    

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
