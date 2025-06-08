import torch, os, cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from torchvision import transforms
import torch.nn.functional as F

def preprocess_fn(img):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    # img_cv2 = cv2.cvtColor(cv2.imread(imgfname), cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return tf(img)

class ImageFolder(Dataset):
    def __init__(self, images, preprocess_fn=preprocess_fn) -> None:
        super().__init__()
        self.images = images
        self.preprocess_fn = preprocess_fn
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return {
            'img': self.preprocess_fn(self.images[index]),
        }


def segment_preprocess_fn(img, max_length=512):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    w, h = img.size[:2]
    scale = max(np.ceil(max(h, w) / max_length), 1)

    if scale >1:
        img = img.resize((int(w/ scale), int(h / scale)))

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return tf(img)

def segment_subjects(images, device='cuda', max_length=512):
    # segm_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
    #                             pretrained=True, verbose=False).to(device)
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
    )
    segm_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
                                weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
    segm_model.eval()

    #image_rescale = min(max(cv2.imread(images[0]).shape[:2]) // max_length, 1)
    if isinstance(images[0], str): 
        org_hw = cv2.imread(images[0]).shape[:2]
    else:
        org_hw = images[0].shape[:2]
    
    # segment_preprocess_fn downsample the image size to have max length as 512
    segm_dataloader = DataLoader(ImageFolder(images, segment_preprocess_fn), batch_size=16, shuffle=False, 
                                    num_workers=8 if os.cpu_count() > 8 else os.cpu_count())
    deeplab_masks = []
    # for batch in tqdm(segm_dataloader, desc='DeepLab segm', disable=tqdm_disabled.get()):
    for batch in segm_dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        with torch.no_grad():
            output = segm_model(batch['img'])['out']
        mask = (output.argmax(1) == 15).to(torch.float) # the max probability is the person class
        mask = mask.cpu()
        deeplab_masks.append(mask)
    
    deeplab_masks = torch.cat(deeplab_masks, dim=0)

    # interpolate back to the original image size back back.
    deeplab_masks = F.interpolate(deeplab_masks.unsqueeze(1), size=(org_hw[0],org_hw[1]), mode='bilinear', align_corners=True).squeeze(1)

    masks = (deeplab_masks > 0.1).cpu().numpy()
    return masks