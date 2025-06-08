"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# Please change ROOT to the root directory of your datasets 
ROOT = './'
ANN_ROOT = 'data/annotations'

# Path to converted annotations
DATASET_FILES = {'EMDB': f'{ANN_ROOT}/emdb_mc_labels.npz',
                 '3DPW_TEST': f'{ANN_ROOT}/3dpw_test_mc_labels.npz',
                 'HI4D_TEST': f'{ANN_ROOT}/hi4d_test_mc_labels.npz',
                 'CHI3D_VAL': f'{ANN_ROOT}/chi3d_val_mc_labels.npz',
                 'COCO_VAL': f'{ANN_ROOT}/coco_val.npz',
                 'RICH_TEST': f'{ANN_ROOT}/rich_test_mc_labels.npz'
				}


# Path to image folders
DATASET_FOLDERS = {'EMDB': f'{ROOT}/EMDB',
                   '3DPW_TEST': f'{ROOT}/3DPW',
                   'HI4D_TEST': f'{ROOT}/HI4D',
                   'CHI3D_VAL': f'{ROOT}/CHI3D',
                   'COCO_VAL': f'{ROOT}/COCO',
                   'RICH_TEST': f'{ROOT}/RICH',
				  }

# Path to SMPL/SMPLX data
SMPLX_PATH = 'data/body_models/smplx'
SMPL_PATH = 'data/body_models/smpl'
SMPLX2SMPL = 'data/body_models/smplx2smpl.pkl'