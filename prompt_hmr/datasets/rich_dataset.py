import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
from PIL import Image, ImageOps

from data_config import DATASET_FILES, DATASET_FOLDERS, SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX
from prompt_hmr.utils.rotation_utils import axis_angle_to_matrix
from prompt_hmr.utils.imutils import read_img


np2th = lambda x: torch.from_numpy(x).float()


class RICHDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset, img_size=896, validation_only=False, augmentation=True, **kwargs):
        super(RICHDataset, self).__init__()
        print(f'Loading: {dataset}')
        self.smplxs = {g:SMPLX(SMPLX_PATH, gender=g) for g in ['neutral', 'male', 'female']}
        self.dataset = dataset
        self.img_size = img_size
        self.augmentation = augmentation
        self.normalization = Compose([
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                            ])
        self.collate_fn = trivial_batch_collator
        self.worker_init_fn = worker_init_fn
        
        # Data
        self.data = np.load(DATASET_FILES[dataset])
        
        self.img_dir = DATASET_FOLDERS[dataset]
        
        self.length = self.data['imgname'].shape[0]

        # Valuation only
        self.indices = np.arange(self.length)
        if validation_only:
            self.indices = self.indices[::15]
       
    def __getitem__(self, index):
        index = self.indices[index]
        item = {} 

        keys = ['cam_int', 'smplx_trans', 'smplx_grot', 'smplx_pose', 
                'smplx_betas', 'gender', 'imgname']

        for k in keys:
            if k in ['imgname', 'gender']:
                item[k] = self.data[k][index]
            else:
                item[k] = np2th(self.data[k][index])[None]
                
        item['cam_int'] = item['cam_int'][:, :3, :3]
        
        smplx = self.smplxs[item['gender']]
        
        # IMAGE
        imgpath = os.path.join(self.img_dir, item['imgname'])
        img = self.load_image(imgpath, item)
        img, item = self.pad_image(img, item)
        item = self.adjust_cam_center(item)

       
        # SMPLX
        item['smplx_betas'] = item['smplx_betas'][:,:10]
        item['smplx_grot'] = axis_angle_to_matrix(item['smplx_grot'].reshape(-1, 1, 3))
        item['smplx_pose'] = axis_angle_to_matrix(item['smplx_pose'].reshape(-1, 21, 3))
            
        item['smplx_rotmat'] = torch.cat([item['smplx_grot'], 
                                          item['smplx_pose']], dim=1)

        smpl_out = smplx(
            global_orient = item['smplx_grot'],
            body_pose = item['smplx_pose'],
            betas = item['smplx_betas'],
            transl = item['smplx_trans'],
        )
        
        item['smplx_joints3d'] = smpl_out.body_joints

        item['smplx_joints2d'] = torch.einsum(
            'bij, bvj->bvi', 
            item['cam_int'], 
            item['smplx_joints3d']/(item['smplx_joints3d'][...,[-1]]+1e-6))[...,:2]
        
        
        verts3d = smpl_out.vertices
        item['smplx_verts3d'] = verts3d
        item['smplx_verts2d'] = torch.einsum('bij, bvj->bvi', 
                                            item['cam_int'], 
                                            verts3d/(verts3d[...,[-1]]+1e-6))[...,:2]
        
        item['smplx_joints3d'] = torch.cat([item['smplx_joints3d'],
                                 torch.ones_like(item['smplx_joints3d'])[...,:1]], dim=-1).float()
        item['smplx_joints2d'] = torch.cat([item['smplx_joints2d'],
                                 torch.ones_like(item['smplx_joints2d'])[...,:1]], dim=-1).float()
      
        item['image'] = self.normalization(img)
        item['image_cv'] = torch.tensor(img)

        item['has_smplx'] = True
        item['has_smplx2d'] = True
        item['has_smpl'] = False
        item['has_smpl2d'] = False
        item['has_transl'] = True
        item['has_hands'] = True

        # Prompts
        item['boxes'] = self.get_bbox_prompts(item)
        item['kpts'] = None
        item['text'] = ['NULL'] * len(verts3d)
        
        return item


    def __len__(self):
        return len(self.indices)
    

    def load_image(self, imgpath, data):
        if not os.path.exists(imgpath):
            raise ValueError(f'Image does not exist: {imgpath}')

        img = read_img(imgpath)
        return img


    def pad_image(self, img, item):
        img_size = self.img_size
        size = np.array([img.shape[1], img.shape[0]])
        scale = img_size / max(size)
        offset = (img_size - scale * size) / 2

        bound = np.array([0, 0, size[0], size[1]]) * scale 
        bound = bound + np.tile(offset, 2)

        img_pil = Image.fromarray(img)
        img_pil = ImageOps.contain(img_pil, (img_size,img_size)) #, method=Image.LANCZOS)
        img_pil = ImageOps.pad(img_pil, size=(img_size,img_size))
        img = np.array(img_pil)

        item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
        item['cam_int'][:,:2] *= scale
        item['cam_int'][:,:2,-1] += offset
        item['bound'] = bound

        return img, item
    

    def adjust_cam_center(self, item):
        # For datasets where cam_center is not image center (CHI3D/H36M).
        # We adjust the cam_center to be at the image center,
        # and adjust smplx_transl to be consistent

        transl = item['smplx_trans']
        cam_int = item['cam_int']

        x_offset = cam_int[0,0,-1] - self.img_size/2
        y_offset = cam_int[0,1,-1] - self.img_size/2
        transl[:,0] += x_offset * transl[:,-1] / cam_int[0,0,0]
        transl[:,1] += y_offset * transl[:,-1] / cam_int[0,1,1]

        cam_int[0,:2,-1] = self.img_size / 2

        item['smplx_trans'] = transl
        item['cam_int'] = cam_int

        return item


    def reopen_files(self):
        self.data = np.load(DATASET_FILES[self.dataset], allow_pickle=True)


    def get_bbox_prompts(self, item):
        #Get bbox using keypoints
        img_size = self.img_size
        verts = item['smplx_verts2d']
        is_rotated = item.get('is_rotated', False)

        # Determine box bound
        if is_rotated:
            x_min, y_min, x_max, y_max = 0, 0, img_size, img_size
        else:
            x_min, y_min, x_max, y_max = item['bound']

        bbox = [
            verts[:,:,0].min(dim=1)[0].clip(x_min, x_max),
            verts[:,:,1].min(dim=1)[0].clip(y_min, y_max),
            verts[:,:,0].max(dim=1)[0].clip(x_min, x_max),
            verts[:,:,1].max(dim=1)[0].clip(y_min, y_max)
        ]
        bbox = torch.stack(bbox, dim=1)
        bbox = bbox.clip(0, img_size)
        bbox = torch.cat([bbox, torch.ones_like(bbox)[...,:1]], dim=-1)

        return bbox
            
        
def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    # open all files again in each worker
    dataset.reopen_files()