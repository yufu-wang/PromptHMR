import os
import torch
from glob import glob
from .core.config import parse_args

def load_model(ckpt):
    from .models import build_phmr
    weight = torch.load(ckpt, map_location='cuda', weights_only=True)

    cfg = f'{os.path.dirname(ckpt)}/config.yaml'
    cfg = parse_args(['--cfg', cfg])
    model = build_phmr(cfg)
    model = model.cuda()
    _ = model.load_state_dict(weight['state_dict'], strict=True)
    _ = model.eval()
    model.is_train = False
    return model

def load_model_from_cfg(cfg, metrics='avg_mpjpe', folder='lightning_logs'):
    from .models import build_phmr
    cfg = parse_args(['--cfg', cfg])

    # Get the "best avg_mpjpe" checkpoint
    exp = cfg.EXP_NAME
    ckpts = glob(f'{folder}/{exp}/checkpoints/*.ckpt')
    valid = [i for i, ckpt in enumerate(ckpts) if metrics in ckpt][0]
    weight = torch.load(ckpts[valid], map_location='cuda', weights_only=True)
    print('Using checkpoint', ckpts[valid])

    # Get the model
    model = build_phmr(cfg)
    model = model.cuda()
    m = model.load_state_dict(weight['state_dict'], strict=False)
    _ = model.eval()
    print('Loaded ckpt:', m)

    return model

def load_model_from_folder(folder):
    from .models import build_phmr
    cfg = f'{folder}/config.yaml'
    cfg = parse_args(['--cfg', cfg])

    ckpt = f'{folder}/checkpoint.ckpt'
    weight = torch.load(ckpt, map_location='cuda', weights_only=True)

    model = build_phmr(cfg)
    model = model.cuda()
    _ = model.load_state_dict(weight['state_dict'], strict=True)
    _ = model.eval()
    model.is_train = False
    return model

