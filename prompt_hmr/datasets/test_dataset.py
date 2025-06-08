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


def tt(ndarray):
    tensor = torch.from_numpy(ndarray).float()
    return tensor

class TestDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset='3DPW_TEST', validation_only=True, img_size=896):
        super(TestDataset, self).__init__()
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

        # Data
        self.data = np.load(DATASET_FILES[dataset], allow_pickle=True)
        self.img_dir = DATASET_FOLDERS[dataset]
        
        # Meta data
        self.length = self.data['data_len']
        self.has_cam_int = self.data['has_cam_int']

        # Valuation only
        self.indices = np.arange(self.length)
        if validation_only:
            self.indices = self.indices[::15]

        # Joint regressor for evaluation purposes
        if dataset == '3DPW_TEST':
            self.j_regressor = tt(np.load('data/body_models/J_regressor_h36m.npy'))
        else:
            self.j_regressor = self.smpls['neutral'].J_regressor
                    

    def __getitem__(self, idx):
        index = self.indices[idx]
        data = self.data[str(index)].item()

        item = {} 
        keys = ['scale', 'center', 'cam_int', 'smpl_trans',
                'smpl_grot', 'smpl_pose', 'smpl_betas', 'gender', 'masks']
        for k in keys:
            item[k] = data.get(k, None)
        item['idx'] = idx
        item['index'] = index

        # IMAGE
        imgpath = os.path.join(self.img_dir, data['imgname'])
        img, item = self.load_and_preprocess(imgpath, item)
        item = self.adjust_cam_center(item)

        # SMPL
        item['smpl_grot'] = axis_angle_to_matrix(item['smpl_grot'].reshape(-1, 1, 3))
        item['smpl_pose'] = axis_angle_to_matrix(item['smpl_pose'].reshape(-1, 23, 3))
        item['smpl_rotmat'] = torch.cat([item['smpl_grot'], 
                                         item['smpl_pose']], dim=1)
        
        verts3d = []
        joints3d = []
        for i, gender in enumerate(item['gender']):
            smpl = self.smpls[gender]
            smpl_out = smpl(global_orient = item['smpl_grot'][[i]],
                            body_pose = item['smpl_pose'][[i]],
                            betas = item['smpl_betas'][[i]],
                            transl = item['smpl_trans'][[i]])
            verts3d.append(smpl_out.vertices)
            joints3d.append(smpl_out.joints)

        verts3d = torch.cat(verts3d)
        joints3d = torch.cat(joints3d)
        item['smpl_verts3d'] = verts3d
        item['smpl_verts2d'] = torch.einsum('bij, bvj->bvi', 
                                            item['cam_int'], 
                                            verts3d/(verts3d[...,[-1]]+1e-6))[...,:2]

        item['smpl_joints3d'] = self.j_regressor @ verts3d
        item['smpl_joints2d'] = torch.einsum('bij, bvj->bvi', 
                                            item['cam_int'], 
                                            item['smpl_joints3d']/(item['smpl_joints3d'][...,[-1]]+1e-6))[...,:2]
        
        item['smpl_joints3d'] = torch.cat([item['smpl_joints3d'],
                                 torch.ones_like(item['smpl_joints3d'])[...,:1]], dim=-1).float()
        item['smpl_joints2d'] = torch.cat([item['smpl_joints2d'],
                                 torch.ones_like(item['smpl_joints2d'])[...,:1]], dim=-1).float()

        item['image'] = self.normalization(img)
        item['image_cv'] = torch.tensor(img)

        # Box prompts
        item['boxes'] = self.get_bbox_prompts(item)
        item['masks'] = self.get_mask_prompts(item)
        item['kpts'] = None
        item['text'] = ['NULL'] * len(item['smpl_grot'])

        return item


    def __len__(self):
        return len(self.indices)


    def pad_image(self, img, item):
        IMG_SIZE = self.img_size
        size = np.array([img.shape[1], img.shape[0]])
        scale = IMG_SIZE / max(size)
        offset = (IMG_SIZE - scale * size) / 2

        bound = np.array([0, 0, size[0], size[1]]) * scale 
        bound = bound + np.tile(offset, 2)

        img_pil = Image.fromarray(img)
        img_pil = ImageOps.contain(img_pil, (IMG_SIZE,IMG_SIZE))
        img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE,IMG_SIZE))
        img = np.array(img_pil)

        item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
        item['cam_int'][:,:2] *= scale
        item['cam_int'][:,:2,-1] += offset
        item['bound'] = bound

        return img, item
    

    def load_and_preprocess(self, imgpath, item):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        item['image_original'] = img
        item['cam_int_original'] = item['cam_int']

        # Pad
        IMG_SIZE = self.img_size
        size = np.array([img.shape[1], img.shape[0]])
        scale = IMG_SIZE / max(size)
        offset = (IMG_SIZE - scale * size) / 2

        bound = np.array([0, 0, size[0], size[1]]) * scale 
        bound = bound + np.tile(offset, 2)

        img_pil = Image.fromarray(img)
        img_pil = ImageOps.contain(img_pil, (IMG_SIZE,IMG_SIZE))
        img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE,IMG_SIZE))
        img = np.array(img_pil)

        item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
        item['cam_int'][:,:2] *= scale
        item['cam_int'][:,:2,-1] += offset
        item['bound'] = bound

        # Load masks for HI4D
        if 'HI4D' in self.dataset:
            masks = []
            masks_vis = []
            seqname = os.path.dirname(imgpath)
            imgname = os.path.basename(imgpath)
            for i in range(2):
                maskpath = f'{seqname}/{i}/{imgname}'
                maskpath = maskpath.replace('images', 'seg/img_seg_mask').replace('.jpg', '.png')
                if os.path.isfile(maskpath):
                    msk = cv2.imread(maskpath)
                else:
                    msk = np.zeros_like(img)
                
                msk_size = int(self.img_size/14 * 4)
                mask = Image.fromarray(msk)
                mask = ImageOps.contain(mask, (msk_size,msk_size))
                mask = ImageOps.pad(mask, size=(msk_size,msk_size))
                mask = np.array(mask)[...,:3]
                masks.append(mask)

                msk_size = int(self.img_size)
                mask = Image.fromarray(msk)
                mask = ImageOps.contain(mask, (msk_size,msk_size))
                mask = ImageOps.pad(mask, size=(msk_size,msk_size))
                mask = np.array(mask)[...,:3]
                masks_vis.append(mask)

            item['masks'] = masks
            item['masks_vis'] = masks_vis
        
        return img, item
    

    def adjust_cam_center(self, item):
        # For datasets where cam_center is not image center (CHI3D/H36M/HI4D).
        # We adjust the cam_center to be at the image center,
        # and adjust smplx_transl to be consistent
        transl = item['smpl_trans']
        cam_int = item['cam_int']

        x_offset = cam_int[0,0,-1] - self.img_size/2
        y_offset = cam_int[0,1,-1] - self.img_size/2
        transl[:,0] += x_offset * transl[:,-1] / cam_int[0,0,0]
        transl[:,1] += y_offset * transl[:,-1] / cam_int[0,1,1]

        cam_int[0,:2,-1] = self.img_size / 2

        item['smpl_trans'] = transl
        item['cam_int'] = cam_int

        return item


    def get_mask_prompts(self, item):
        masks = item.get('masks', None)
        if masks is not None:
            masks = np.array(masks)
            masks = torch.tensor(masks).float() / 255.
            masks = masks.permute(0,3,1,2)[:,:1,:,:]
        return masks
    
    
    def get_bbox_prompts(self, item):
        #Get bbox using keypoints
        img_size = self.img_size
        verts = item['smpl_verts2d']
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
    

    def reopen_files(self):
        self.data = np.load(DATASET_FILES[self.dataset], allow_pickle=True)
    

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