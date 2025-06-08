import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
from PIL import Image, ImageOps

from data_config import DATASET_FILES, DATASET_FOLDERS, SMPL_PATH
from prompt_hmr.smpl_family import SMPL
from prompt_hmr.utils.rotation_utils import axis_angle_to_matrix
from prompt_hmr.utils.imutils import crop, get_normalization, box_2_cs


def tt(ndarray):
    tensor = torch.from_numpy(ndarray).float()
    return tensor

class EMDBDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset='EMDB', img_size=896, validation_only=True):
        super(EMDBDataset, self).__init__()
        print(f'Loading: {dataset}')
        self.dataset = dataset
        self.img_size = img_size
        self.data_file = DATASET_FILES[dataset]
        self.image_dir = DATASET_FOLDERS[dataset]
        self.smpls = {g:SMPL(SMPL_PATH, gender=g) for g in ['neutral', 'male', 'female']}

        self.normalization = Compose([
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                            ])
        self.collate_fn = trivial_batch_collator
        self.worker_init_fn = worker_init_fn

        # Grab all data
        data = dict(np.load(self.data_file))
        valid = data['valid']
        self.imgnames = data['imgnames'][valid]
        self.genders = data['genders'][valid]
        self.cam_int = tt(data['cam_int'])[valid]
        self.cam_ext = tt(data['cam_ext'])[valid]
        self.bboxes  = tt(data['boxes'])[valid]
        self.smpl_betas = tt(data['smpl_betas'])[valid]
        self.smpl_poses = tt(data['smpl_body_poses'])[valid]
        self.smpl_grots = tt(data['smpl_grots_cam'])[valid]
        self.smpl_trans = tt(data['smpl_trans_cam'])[valid]

        # Valuation only
        self.indices = np.arange(len(self.imgnames))
        if validation_only:
            self.indices = self.indices[::15]


    def __getitem__(self, idx):
        index = self.indices[idx]
        item = {} 
        item['imgname'] = self.imgnames[index]
        item['gender'] = self.genders[index]
        item['boxes'] = self.bboxes[[index]]
        item['cam_int'] = self.cam_int[[index]]
        item['cam_ext'] = self.cam_ext[[index]]
        item['smpl_grot'] = self.smpl_grots[[index]]
        item['smpl_pose'] = self.smpl_poses[[index]]
        item['smpl_betas'] = self.smpl_betas[[index]]
        item['smpl_trans'] = self.smpl_trans[[index]]
        smpl = self.smpls[item['gender']]

        # IMAGE
        imgpath = os.path.join(self.image_dir, item['imgname'])
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop image (for pose)
        box = self.bboxes[index]
        center, scale = box_2_cs(box)
        img_crop = crop(img, center, scale, res=(256,256)).astype('uint8')
        img_crop_norm = self.normalization(img_crop)
        item['image_crop'] = img_crop
        item['image_crop_norm'] = img_crop_norm
        item['center'] = torch.tensor(center)
        item['scale'] = torch.tensor(scale)
        item['image_original'] = img

        # Pad image
        img, item = self.pad_image(img, item)

        # SMPLX
        item['has_smpl'] = 1.
        item['smpl_grot'] = axis_angle_to_matrix(item['smpl_grot'].reshape(-1, 1, 3))
        item['smpl_pose'] = axis_angle_to_matrix(item['smpl_pose'].reshape(-1, 23, 3))
        item['smpl_rotmat'] = torch.cat([item['smpl_grot'], 
                                         item['smpl_pose']], dim=1)
        
        smpl_out = smpl(global_orient = item['smpl_grot'],
                        body_pose = item['smpl_pose'],
                        betas = item['smpl_betas'],
                        transl = item['smpl_trans'])
        
        item['smpl_verts3d'] = smpl_out.vertices
        item['smpl_joints3d'] = smpl.J_regressor @ smpl_out.vertices
        item['smpl_joints2d'] = torch.einsum('bij, bvj->bvi', 
                                            item['cam_int'], 
                                            item['smpl_joints3d']/(item['smpl_joints3d'][...,[-1]]+1e-6))[...,:2]
        
        item['smpl_verts2d'] = torch.einsum('bij, bvj->bvi', 
                                            item['cam_int'], 
                                            item['smpl_verts3d']/(item['smpl_verts3d'][...,[-1]]+1e-6))[...,:2]
        
        item['smpl_joints3d'] = torch.cat([item['smpl_joints3d'],
                                 torch.ones_like(item['smpl_joints3d'])[...,:1]], dim=-1).float()
        item['smpl_joints2d'] = torch.cat([item['smpl_joints2d'],
                                 torch.ones_like(item['smpl_joints2d'])[...,:1]], dim=-1).float()

        item['image'] = self.normalization(img)
        item['image_cv'] = torch.tensor(img)

        # Keypoint prompts
        item['boxes'] = self.get_bbox_prompts(item)
        item['kpts'] = None
        item['text'] = ['NULL'] * len(item['smpl_grot'])
        item['interaction'] = False

        return item


    def __len__(self):
        return len(self.indices)


    def pad_image(self, img, item):
        img_size = self.img_size
        size = np.array([img.shape[1], img.shape[0]])
        scale = img_size / max(size)
        offset = (img_size - scale * size) / 2

        bound = np.array([0, 0, size[0], size[1]]) * scale 
        bound = bound + np.tile(offset, 2)

        img_pil = Image.fromarray(img)
        img_pil = ImageOps.contain(img_pil, (img_size,img_size))
        img_pil = ImageOps.pad(img_pil, size=(img_size,img_size))
        img = np.array(img_pil)

        item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
        item['cam_int'][:,:2] *= scale
        item['cam_int'][:,:2,-1] += offset
        item['bound'] = bound

        item['boxes'] = item['boxes'] * scale
        item['boxes'][:,:2] += offset
        item['boxes'][:,2:] += offset

        return img, item
    

    def get_bbox_prompts(self, item):
        #Get bbox using keypoints
        bbox = item['boxes']
        bbox = torch.cat([bbox, torch.ones_like(bbox)[...,:1]], dim=-1)

        return bbox

    
    def get_crop(self, img, item, index):
        # Crop image (for pose)
        box = self.bboxes[index]
        center, scale = box_2_cs(box)
        img_crop = crop(img, center, scale, res=(256,256)).astype('uint8')
        img_crop_norm = self.normalization(img_crop)
        item['image_crop'] = img_crop
        item['image_crop_norm'] = img_crop_norm
        item['center'] = torch.tensor(center)
        item['scale'] = torch.tensor(scale)
        item['image_original'] = img
        return item
    

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def worker_init_fn(worker_id):
    return
