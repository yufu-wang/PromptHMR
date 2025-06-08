# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import sys
import os
sys.path.append('droid_slam')
sys.path.append('evaluation_scripts')
import pandas as pd
import numpy as np
import yaml
import random

import main_calib


def run_and_store(args, config, dataset):
    """
    Run DroidCalib or DROID-SLAM with specified args and store result.
    """

    intr, mapping_error, runtime, ate, n_reg_images = \
                    main_calib.run_internal(**args, droid_args=config['droid_args'])

    df = pd.DataFrame.from_dict({'sequence': [args['seq']],
                                'num_images': [args['num_images']],
                                'intr_error': [args['intr_error']],
                                'seed': [args['seed']],
                                'undistort': [args['undistort']],

                                # config
                                'opt_intr': [args['opt_intr']] ,
                                'camera_model': [args['camera_model']],
                                'stride': [args['stride']],
                                'cuda': [args['cuda']],

                                # droid calib results
                                'fx': [intr[0]],
                                'fy': [intr[1]],
                                'ppx': [intr[2]],
                                'ppy': [intr[3]],
                                'xi': [intr[4] if args['camera_model']=='mei' else 0],

                                # main results
                                'mapping_error': [mapping_error],
                                'runtime': [runtime],
                                'n_reg_images': [n_reg_images],
                                'ate': [ate]})
    
    method = ('DroidCalib' if args['opt_intr'] else 'DROID-SLAM')
    output_path = 'figures/{}_{}.csv'.format(method, dataset)
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))



def eval_seqs(dataset='tartan', method='DroidCalib', intr_error=None, 
              undistort=True, camera_model='pinhole', seed=0):
    """
    Evaluate the method on all full sequences of the dataset.
    """

    with open("evaluation_scripts/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)[dataset]
        except yaml.YAMLError as exc:
            print(exc)

    args = dict()

    # fixed args from config
    args['original_intr'] = np.array(config['original_intr'])
    args['rectified_intr'] = np.array(config['rectified_intr'])
    args['original_size'] = config['original_size']
    args['rectified_size'] = config['rectified_size']
    args['crop'] = config['crop']

    # variable args
    args['weights'] = "droidcalib.pth" 
    args['stride'] = (1 if dataset == 'tartan' else 2)
    args['camera_model'] = camera_model
    args['opt_intr'] = (method == 'DroidCalib')
    args['cuda'] = True
    args['undistort'] = undistort
    args['intr_error'] = intr_error
    args['num_images'] = None
    args['start_index'] = 0
    args['seed'] = seed

    np.random.seed(args['seed'])
    random.seed(args['seed'])

    for seq in config['sequences']:
        args['seq'] = seq
        args['trajectory_gt'] = [f for f in config['trajectory_gts'] if seq in f][0]
        args['datapath'] = os.path.join(config['base_path'], seq, config['image_path'])

        try:
            run_and_store(args, config, dataset)

        except Exception as e:
            print(seq, e)
            continue



if __name__ == "__main__":

    # =============================================================== #
    # DroidCalib naive initial intrinsics
    # =============================================================== #

    eval_seqs(dataset='tartan', method='DroidCalib', intr_error=None, 
            undistort=True, camera_model='pinhole')

    eval_seqs(dataset='tum', method='DroidCalib', intr_error=None, 
            undistort=True, camera_model='pinhole')

    eval_seqs(dataset='euroc', method='DroidCalib', intr_error=None, 
            undistort=True, camera_model='pinhole')
    

    # =============================================================== #
    # Unified model
    # =============================================================== #
    eval_seqs(dataset='euroc', method='DroidCalib', intr_error=None, 
            undistort=False, camera_model='mei')


    # =============================================================== #
    # DROID-SLAM vs DroidCalib trajectory estimation
    # =============================================================== #
    for i in range(3):
        for intr_error in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            for method in ['DroidCalib']: #['DROID-SLAM', 'DroidCalib']:

                eval_seqs(dataset='euroc', method=method, intr_error=intr_error, 
                        undistort=True, camera_model='pinhole', seed=i)

                eval_seqs(dataset='tartan', method=method, intr_error=intr_error, 
                        undistort=True, camera_model='pinhole', seed=i)

                eval_seqs(dataset='tum', method=method, intr_error=intr_error, 
                        undistort=True, camera_model='pinhole', seed=i)

