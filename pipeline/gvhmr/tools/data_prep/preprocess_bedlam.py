import os
import torch
import numpy as np
import random
from smplcodec.codec import SMPLCodec
import glob

def main():
    gvhmr_data = torch.load("inputs/BEDLAM/hmr4d_support/smplpose_v2.pth")
    all_seqs = list(gvhmr_data.keys())
    files = sorted(glob.glob('/home/muhammed/projects/GVHMR/tools/data_prep/.tmp/bedlam_smpl/*/*/*_bedlam.smpl'))
    
    vis_keys = []
    for f in files:
        s = f.split('/')[-1].replace('_bedlam.smpl', '')
        k = f"inputs/bedlam/bedlam_download/{f.split('/')[-3]}/mp4/{f.split('/')[-2]}.mp4-{s}"
        vis_keys.append(k)
        if k not in all_seqs:
            print(f"Missing {k}")
        # assert k in all_seqs
        

    # random.shuffle(all_seqs)
    # all_seqs = 
    tid = 0
    for idx, seq in enumerate(vis_keys):
        print(seq)
        seq_name = seq.split("/")[-3]
        seq_id = seq.split('/')[-1].split('-')[0].split('/')[-1].split('.')[0]
        subject = seq.split("-")[-1]
        bedlam_data = np.load(f"/mnt/data/datasets/BEDLAM/data_30fps/bedlam_labels_30fps/{seq_name}.npz")
        bedlam_subj = bedlam_data["sub"]
        
        th2np = lambda x: x.cpu().numpy()
        gvhmr_pack = gvhmr_data[seq]
        gvhmr_pose = th2np(gvhmr_pack["pose"])
        gvhmr_trans = th2np(gvhmr_pack["trans"])
        gvhmr_betas = th2np(gvhmr_pack["beta"])
        gvhmr_trans_cam = th2np(gvhmr_pack["trans_incam"])
        gvhmr_go_cam = th2np(gvhmr_pack["global_orient_incam"])
        gvhmr_cam_ext = th2np(gvhmr_pack["cam_ext"])
        gvhmr_cam_int = th2np(gvhmr_pack["cam_int"])
        

        smpl_file_gvhmr = files[idx].replace('_bedlam.smpl', '_gvhmr.smpl')
        # smpl_file_gvhmr = f'.tmp/{tid:04d}_{seq_name}_gvhmr.smpl'
        SMPLCodec(
            shape_parameters=gvhmr_betas,
            body_pose=gvhmr_pose.reshape(-1,22,3), 
            body_translation=gvhmr_trans,
            frame_count=gvhmr_pose.shape[0], 
            frame_rate=30.0,
        ).write(smpl_file_gvhmr)
        print(f"Saved {smpl_file_gvhmr}")
        
        tid += 1
        

if __name__ == "__main__":
    main()