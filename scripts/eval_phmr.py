import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import tyro

from prompt_hmr import load_model_from_folder
from prompt_hmr.evaluator import Evaluator

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_phmr(model_folder='data/phmr', dataset='EMDB', validation_only=False):
    # MODEL
    phmr = load_model_from_folder(model_folder)

    # Interaction dataset
    if dataset in ['HI4D_TEST', 'CHI3D_TEST']:
        interaction = True
        mask_prompt = True
    else:
        interaction = False
        mask_prompt = False

    ### EMDB dataset
    phmr.is_train = True
    evaluator = Evaluator(dataset, validation_only=validation_only)
    results, acc = evaluator(phmr, mask_prompt=mask_prompt, interaction=interaction)
    print(results)


if __name__ == '__main__':
    tyro.cli(evaluate_phmr)