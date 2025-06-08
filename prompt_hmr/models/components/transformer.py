from typing import Callable, Optional, Tuple
import torch
from einops import rearrange
from torch import Tensor, nn

from xformers.ops import memory_efficient_attention, unbind
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Linear(inner_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)


    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MemEffAttention(Attention):
    def forward(self, q, k, v, attn_bias=None) -> Tensor:
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        B, N, C = q.shape
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), [q, k, v])

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, self.inner_dim])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        skip_first_layer_pe: bool=False,
    ):
        super().__init__()

        self.skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = MemEffAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.cross_attn = MemEffAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout=dropout)

        # Post-Norm following SAM (for now)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        

    def forward(self, queries, keys, query_pe, key_pe):
        # Self attention
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        
        # Cross attention
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # Feed Forward
        mlp_out = self.feed_forward(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                AttentionBlock(
                    dim=embedding_dim,
                    heads=num_heads,
                    dim_head=head_dim,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        return queries, keys
    


