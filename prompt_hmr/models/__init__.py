import numpy as np
from .phmr import PHMR
from .components.image_encoder import ImageEncoder
from .components.smpl_decoder import SMPLDecoder
from .components.camera_embed import CameraEncoder
from .components.prompt_encoder import PromptEncoder
from .components.twoway_transformer import TwoWayTransformer
from .components.clip_encoder import ClipEncoder
from .inference import prepare_batch

def build_phmr(cfg):
    backbone = cfg.MODEL.BACKBONE
    image_size = cfg.MODEL.IMG_SIZE
    image_embedding_size = cfg.MODEL.IMG_SIZE // cfg.MODEL.PATCH_SIZE
    prompt_embed_dim = cfg.MODEL.EMBED_DIM
    decoder_depth = cfg.MODEL.DECODER_DEPTH
    regressor_depth = cfg.MODEL.REGRESSOR_DEPTH
    mlp_dim = cfg.MODEL.MLP_DIM
    inverse_depth = cfg.MODEL.INVERSE_DEPTH
    decoder_attention = cfg.MODEL.TRANSFORMER
    mask_prompt = cfg.MODEL.MASK_PROMPT
    prompt_type = cfg.MODEL.PROMPT_TYPE

    # Encoder
    image_encoder = ImageEncoder(backbone, prompt_embed_dim)
    cam_encoder = CameraEncoder(patch_size=cfg.MODEL.PATCH_SIZE,
                                img_size=cfg.MODEL.IMG_SIZE)
    
    # Prompot Encoder
    clip_encoder = ClipEncoder()
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        input_image_size=(image_size, image_size),
        image_embedding_size=(image_embedding_size, image_embedding_size),
        clip_encoder=clip_encoder,
        mask_prompt=mask_prompt,
    )

    # SMPL Decoder
    transformer=TwoWayTransformer(
        depth=decoder_depth,
        embedding_dim=prompt_embed_dim,
        mlp_dim=mlp_dim,
        num_heads=8,
        attention_block=decoder_attention
    )
    smpl_decoder = SMPLDecoder(
        transformer=transformer,
        transformer_dim=prompt_embed_dim,
        smpl_head_hidden_dim=prompt_embed_dim,
        smpl_head_depth=regressor_depth,
        inverse_depth=inverse_depth,
    )

    # PromptHMR
    phmr = PHMR(cfg,
                cam_encoder,
                image_encoder, 
                prompt_encoder, 
                smpl_decoder)
    
    return phmr


def num_params(model):
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])