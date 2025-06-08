import torch
from torch import nn
from torchvision.transforms.functional import normalize
import open_clip


class ClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = open_clip.create_model('ViT-L-14-quickgelu', 
                                              pretrained='metaclip_fullcc')
        # self.encoder = open_clip.create_model('ViT-B-32-quickgelu', 
        #                                     pretrained='metaclip_400m')
        self.encoder.visual = None
        self.encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad=False

        clip_dim = self.encoder.token_embedding.weight.shape[1]
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.ln_proj = nn.Linear(clip_dim, 1024)

    def forward(self, image: torch.Tensor, text: list[str], projection=True):
        image_features = self.encode_image(image, projection)
        text_features = self.encode_image(text, projection)
        
        return image_features, text_features

    def encode_image(self, image: torch.Tensor, projection=True):
        with torch.no_grad():
            device = self._get_device()
            img_tensor = image.permute(0,3,1,2)/ 255
            img_tensor = normalize(img_tensor, self.mean, self.std, inplace=False).to(device)
            img_features = self.encoder.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        
        if projection:
            img_features = self.ln_proj(img_features)
        
        return img_features

    def encode_text(self, text: list[str], projection=True):
        with torch.no_grad():
            device = self._get_device()
            text_token = open_clip.tokenize(text).to(device)
            text_features = self.encoder.encode_text(text_token).detach()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        if projection:
            text_features = self.ln_proj(text_features)

        return text_features
    
    def _get_device(self):
        return self.ln_proj.weight.device


    
