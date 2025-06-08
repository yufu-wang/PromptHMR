import sys
import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.cuda()
        batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size]

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
    
    
class PromptHMRExtractor:
    def __init__(self):
        sys.path.insert(0, '/home/muhammed/projects/prompt_hmr')
        from fhmr.models import build_fhmr, prepare_batch
        from fhmr.core.config import parse_args
        args = ['--cfg', '/home/muhammed/projects/prompt_hmr/lightning_logs/fhmr_94_yufu/config.yaml']
        ckpt = '/home/muhammed/projects/prompt_hmr/lightning_logs/fhmr_94_yufu/checkpoints/step=113694-pa_mpjpe=41.26.ckpt'
        weight = torch.load(ckpt, map_location='cpu')
        
        prhmr_cfg = parse_args(args)
        model = build_fhmr(prhmr_cfg)
        weight['state_dict']['prompt_encoder.kpt_embeddings.weight'] = torch.zeros_like(
            model.state_dict()['prompt_encoder.kpt_embeddings.weight']
        )
        model = model.cuda()
        
        _ = model.load_state_dict(weight['state_dict'], strict=False)
        _ = model.eval()
        _ = model.half()
        model.is_train = False
        self.model = model
        self.prepare_batch = prepare_batch
    
    def extract_video_features(self, video_path, bbx_xys, cam_int=None):
        cap = cv2.VideoCapture(video_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=nframes)
        
        # Convert bbx_xys (x,y,s) to xyxy format
        center_x = bbx_xys[..., 0]
        center_y = bbx_xys[..., 1] 
        size = bbx_xys[..., 2]
        half_size = size / 2
        
        # Calculate xyxy coordinates
        x1 = center_x - half_size
        y1 = center_y - half_size
        x2 = center_x + half_size 
        y2 = center_y + half_size
        
        bbx_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        frame_idx = 0
        features = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image = frame
                
                
                
                inputs = [
                    {
                        'image_cv': image[:,:,::-1],
                        'boxes': bbx_xyxy[frame_idx:frame_idx+1],
                    },
                ]
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'): 
                        batch = self.prepare_batch(inputs, img_size=896)
                        output = self.model(batch)

                features.append(output[0]['features'])
                pbar.update(1)
                frame_idx += 1
            else:
                break
        
        cap.release()
        pbar.close()
        features = torch.stack(features, dim=0)
        return features