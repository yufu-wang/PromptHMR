import os
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]="1"
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl
from torch.amp import autocast

from data_config import SMPL_PATH, SMPLX_PATH, SMPLX2SMPL
from prompt_hmr.smpl_family import SMPLX, SMPL
from prompt_hmr.datasets.emdb_dataset import EMDBDataset
from prompt_hmr.datasets.rich_dataset import RICHDataset
from prompt_hmr.datasets.test_dataset import TestDataset
from prompt_hmr.utils.eval_utils import batch_compute_similarity_transform_torch, batch_align_by_pelvis

def to_tensor(array):
	tensor = torch.from_numpy(array).float().cuda()
	return tensor

class Evaluator():
	def __init__(self, dataset, validation_only=True, batch_size=32, img_size=896, device='cuda'):

		self.ds_name = dataset
		self.smpl = SMPL(SMPL_PATH, gender='neutral').to(device)
		self.smplx =  SMPLX(SMPLX_PATH, gender='neutral').to(device)
		self.smplx2smpl = to_tensor(pkl.load(open(SMPLX2SMPL, 'rb'))['matrix'])

		# Joint regressors
		h36m_to_14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9][:14]
		self.j_h36m = to_tensor(np.load('data/body_models/J_regressor_h36m.npy'))
		self.j_lsp14 = self.j_h36m[h36m_to_14]
		self.j_smpl = self.smpl.J_regressor[:24]
		self.j_smplx = self.smplx.J_regressor[:24]

		# SMPL datasets
		if dataset in ['EMDB']:
			self.dataset = EMDBDataset(validation_only=validation_only, img_size=img_size)
			self.j_regressor = self.j_smpl
			self.pelvis_idx = [1, 2]          # as in wham
			self.type = 'smpl'

		elif dataset in ['3DPW_TEST']:
			self.dataset = TestDataset('3DPW_TEST', validation_only=validation_only, img_size=img_size)
			self.j_regressor = self.j_lsp14
			self.pelvis_idx = [2, 3]          # as in spin/wham
			self.type = 'smpl'

		elif dataset in ['HI4D_TEST']:
			self.dataset = TestDataset(dataset, validation_only=validation_only, img_size=img_size)
			self.j_regressor = self.j_lsp14
			self.pelvis_idx = [2, 3]         # as in buddi
			self.type = 'smpl'

		elif dataset in ['RICH_TEST']:
			self.dataset = RICHDataset('RICH_TEST', augmentation=False, validation_only=validation_only, img_size=img_size)
			self.j_regressor = self.smpl.J_regressor
			self.pelvis_idx = [1, 2] 
			self.type = 'smpl'
		
		else:
			print('Available testset: EMDB, 3DPW_TEST, HI4D_TEST, RICH_TEST')
			raise Exception("Dataset not available")
			
		self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, 
												shuffle=False, num_workers=16,
												collate_fn=self.dataset.collate_fn, 
												worker_init_fn=self.dataset.worker_init_fn,
												)

	def __call__(self, model, box_prompt=True, mask_prompt=False, interaction=False):

		results, accumulator = self.eval_3d(model, box_prompt=box_prompt, 
												mask_prompt=mask_prompt, 
												interaction=interaction)
		return results, accumulator


	def eval_3d(self, model, box_prompt=True, mask_prompt=False, interaction=False):
		accumulator = defaultdict(list)
		smpltype = self.type

		for batch in tqdm(self.loader):
			for item in batch:
				item['interaction'] = interaction
				
			# Model output
			with torch.no_grad():
				with autocast('cuda'):
					output = model(batch, box_prompt=box_prompt, mask_prompt=mask_prompt)
				
			pred_verts_cam = output['vertices'].float()
			if smpltype == 'smpl':
				pred_verts_cam = model.smplx2smpl @ pred_verts_cam
			pred_j3d_cam = self.j_regressor @ pred_verts_cam

			# Ground truth
			if self.ds_name == 'RICH_TEST':
				gt_verts_cam = torch.cat([b[f'smplx_verts3d'] for b in batch])[...,:3].to(pred_j3d_cam)
				gt_verts_cam = model.smplx2smpl @ gt_verts_cam
			else:
				gt_verts_cam = torch.cat([b[f'{smpltype}_verts3d'] for b in batch])[...,:3].to(pred_j3d_cam)

			gt_j3d_cam = self.j_regressor @ gt_verts_cam
			
			# Alignment
			m2mm = 1e3
			pred_j3d, gt_j3d, pred_verts, gt_verts = batch_align_by_pelvis(
				[pred_j3d_cam, gt_j3d_cam, pred_verts_cam, gt_verts_cam], pelvis_idxs=self.pelvis_idx
			)
			pred_j3d_pa = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d)

			pa_mpjpe = torch.sqrt(((pred_j3d_pa - gt_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
			mpjpe = torch.sqrt(((pred_j3d - gt_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
			pve = torch.sqrt(((pred_verts - gt_verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

			accumulator['pa_mpjpe'].append(pa_mpjpe)
			accumulator['mpjpe'].append(mpjpe)
			accumulator['pve'].append(pve)

			# For CHI3D/HI4D, compute joint-pa-mpjpe
			if self.ds_name in ['HI4D_TEST', 'CHI3D_VAL']:
				bn = len(pred_j3d_cam)
				pred_j3d_pair = pred_j3d_cam.reshape(bn//2, -1, 3)
				gt_j3d_pair = gt_j3d_cam.reshape(bn//2, -1, 3)

				pred_j3d_pair_pa = batch_compute_similarity_transform_torch(pred_j3d_pair, gt_j3d_pair)

				pair_pa_mpjpe = torch.sqrt(((pred_j3d_pair_pa - gt_j3d_pair) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
				accumulator['pair_pa_mpjpe'].append(pair_pa_mpjpe)


		# Final results
		results = {k: np.concatenate(accumulator[k]).mean() for k in accumulator}

		return results, accumulator


	