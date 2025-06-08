import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
from torchvision import transforms
from PIL import Image


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
        