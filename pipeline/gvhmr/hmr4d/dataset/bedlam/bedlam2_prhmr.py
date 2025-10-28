import glob
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from hmr4d.utils.pylogger import Log
from hmr4d.utils.pytorch3d_transform import axis_angle_to_matrix, matrix_to_axis_angle
from time import time

from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video

import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.bedlam.utils import mid2featname, mid2vname
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

def get_longest_true_seq(valid_mask):
    longest_start = longest_end = 0
    current_start = current_length = longest_length = 0

    for i, value in enumerate(valid_mask):
        if value:
            if current_length == 0:  # Start of a new sequence
                current_start = i
            current_length += 1
        else:
            # End of a True sequence
            if current_length > longest_length:
                longest_length = current_length
                longest_start = current_start
                longest_end = i  # Set end as one past the last True
            current_length = 0

    # Check one last time in case the longest sequence is at the end
    if current_length > longest_length:
        longest_start = current_start
        longest_end = len(valid_mask)
    assert valid_mask[longest_start:longest_end].all(), "Longest sequence is not valid"
    return longest_start, longest_end


class Bedlam2PrhmrDataset(ImgfeatMotionDatasetBase):

    def __init__(
        self,
        lazy_load=True,  # Load from disk when needed
        random1024=False,  # Faster loading for debugging
        version="v1",
    ):
        self.root = Path("inputs/BEDLAM2")
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.lazy_load = lazy_load
        self.random1024 = random1024
        self.version = version
        
        self.smplx = make_smplx('smplx-bedlam2', gender='neutral').to('cuda')
        super().__init__()

    def _load_dataset(self):
        Log.info(f"[BEDLAM-2] Loading from {self.root}")
        tic = time()
        smplpose_file = f'smplpose_hand_v1.pt'
        
        if os.path.exists(self.root / smplpose_file):
            self.motion_files = torch.load(self.root / smplpose_file)
        else:
            # motion_files = glob.glob(f'{self.root}/training_labels_30fps/*/*.pt')
            img_feat_files = sorted(glob.glob(f'{self.root}/imgfeats/prhmr_{self.version}/*/*.pt'))
            print(f'Found {len(img_feat_files)} img_feat_files')
            motion_files = {}
            pbar = tqdm(img_feat_files, desc='Loading motion files')
            for imf in pbar:
                img_feats = torch.load(imf)
                mf = imf.replace(f'imgfeats/prhmr_{self.version}', 'training_labels_30fps')
                assert os.path.exists(mf), f'{mf} does not exist'
                motion_dict = torch.load(mf)
                # import ipdb; ipdb.set_trace()
                # print(len(img_feats), len(motion_dict['valid_mask']))
                assert len(img_feats) == len(motion_dict['valid_mask'])
                
                start, end = get_longest_true_seq(motion_dict['valid_mask'])
                if end - start < 30:
                    continue
                
                j3d = self.smplx(betas=torch.from_numpy(motion_dict['betas'][0:1]).to('cuda')).joints[0]
                n_motion_dict = {
                    'poses_world': motion_dict['poses_world'],
                    'trans_world': motion_dict['trans_world'],
                    'trans_cam': motion_dict['trans_cam'],
                    'global_orient_incam': motion_dict['poses_cam'][:, :3],
                    'betas': motion_dict['betas'][0],
                    'valid_mask': motion_dict['valid_mask'],
                    'bboxes': motion_dict['bboxes'],
                    'cam_int': motion_dict['cam_int'],
                    'cam_ext': motion_dict['cam_ext'],
                    'skeleton': j3d.cpu(),
                    'valid_start_end': (start, end),
                }
                for k, v in n_motion_dict.items():
                    if isinstance(v, np.ndarray):
                        n_motion_dict[k] = torch.from_numpy(v).float()
                        
                motion_files['/'.join(imf.split('/')[-4:])] = n_motion_dict
                pbar.set_postfix(valid_files=len(motion_files.keys()))
                
            print(f'Found {len(motion_files.keys())} motion files')
            torch.save(motion_files, self.root / smplpose_file)
            print(f'Saved to {self.root / smplpose_file}')
            self.motion_files = motion_files

        Log.info(f"[BEDLAM-2] Motion files loaded. Elapsed: {time() - tic:.2f}s")

    def _get_idx2meta(self):
        # sum_frame = sum([e-s for s, e in self.mid_to_valid_range.values()])
        self.idx2meta = list(self.motion_files.keys())
        Log.info(f"[BEDLAM-2] {len(self.idx2meta)} sequences. ")

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        # neutral smplx : "pose": (F, 63), "trans": (F, 3), "beta": (10),
        #           and : "skeleton": (J, 3)
        data = self.motion_files[mid].copy()
        mfname = self.root / mid.replace('imgfeats/prhmr_hand_v1', f'imgfeats/prhmr_{self.version}')
        f_imgseq = torch.load(mfname)[:, 0]
        
        # Random select a subset
        range1, range2 = data['valid_start_end']
        mlength = range2 - range1
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # the minimal mlength is 30 when generating data
            start = range1
            length = mlength
        else:
            effect_max_motion_len = min(max_motion_len, mlength)
            length = np.random.randint(min_motion_len, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
            
        end = start + length
        data["start_end"] = (start, end)
        data["length"] = length
        
        # Update data to a subset
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                data[k] = v[start:end]

        data["f_imgseq"] = f_imgseq[start:end].float()
        data["bbx_xyxy"] = data['bboxes']
        data['img_wh'] = ((data['cam_int'][:, :2, 2]) * 2).long()
        data["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)  # do not provide kp2d
        
        size = data['img_wh']
        img_size = 896
        scale = img_size / data['img_wh'].max(1).values
        offset = (img_size - (scale[:, None] * size)) / 2
        data['cam_int'][:, :2] *= scale[:, None, None]
        data['cam_int'][:, :2, -1] += offset
        boxes = data["bbx_xyxy"].reshape(-1, 2, 2)
        boxes *= scale[:, None, None]
        boxes += offset[:, None, :]
        data["bbx_xyxy"] = boxes.reshape(-1, 4)
        data["img_wh"] = ((data['cam_int'][0, :2, 2]) * 2).long()

        return data

    def _process_data(self, data, idx):
        length = data["length"]
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.float()
                
        # SMPL params in cam
        body_pose = data["poses_world"][:, 3:66]  # (F, 63)
        betas = data["betas"][:10].repeat(length, 1)  # (F, 10)
        global_orient = data["global_orient_incam"]  # (F, 3)
        transl = data["trans_cam"] + data["cam_ext"][:, :3, 3]  # (F, 3), bedlam convention
        smpl_params_c = {"body_pose": body_pose, "betas": betas, "transl": transl, "global_orient": global_orient}

        # SMPL params in world
        global_orient_w = data["poses_world"][:, :3]  # (F, 3)
        transl_w = data["trans_world"]  # (F, 3)
        smpl_params_w = {"body_pose": body_pose, "betas": betas, "transl": transl_w, "global_orient": global_orient_w}

        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        # import ipdb; ipdb.set_trace()
        T_w2c = get_T_w2c_from_wcparams(
            global_orient_w=global_orient_w,
            transl_w=transl_w,
            global_orient_c=global_orient,
            transl_c=transl,
            offset=data["skeleton"][0],
        )  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)
        data["bbx_xys"] = get_bbx_xys_from_xyxy(data['bbx_xyxy'], base_enlarge=1.0)
        # cam_angvel (slightly different from WHAM)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6)
        
        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.max_motion_frames
        return_data = {
            "meta": {"data_name": "bedlam2", "idx": idx},
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": data["cam_int"],  # (F, 3, 3)
            "f_imgseq": data["f_imgseq"],  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }

        if False:  # check transformation, wis3d: sampled motion (global, incam)
            wis3d = make_wis3d(name="debug-data-bedlam")
            smplx = make_smplx("supermotion")

            # global
            smplx_out = smplx(**smpl_params_w)
            w_gt_joints = smplx_out.joints
            add_motion_as_lines(w_gt_joints, wis3d, name="w-gt_joints")

            # incam
            smplx_out = smplx(**smpl_params_c)
            c_gt_joints = smplx_out.joints
            add_motion_as_lines(c_gt_joints, wis3d, name="c-gt_joints")

            # Check transformation works correctly
            print("T_w2c", (apply_T_on_points(w_gt_joints, T_w2c) - c_gt_joints).abs().max())
            R_c, t_c = get_c_rootparam(
                smpl_params_w["global_orient"], smpl_params_w["transl"], T_w2c, data["skeleton"][0]
            )
            print("transl_c", (t_c - smpl_params_c["transl"]).abs().max())
            R_diff = matrix_to_axis_angle(
                (axis_angle_to_matrix(R_c) @ axis_angle_to_matrix(smpl_params_c["global_orient"]).transpose(-1, -2))
            ).norm(dim=-1)
            print("global_orient_c", R_diff.abs().max())  # < 1e-6

            skeleton_beta = smplx.get_skeleton(smpl_params_c["betas"])
            print("Skeleton", (skeleton_beta[0] - data["skeleton"][:22]).abs().max())  # (1.2e-7)
            
            from smplcodec.codec import SMPLCodec
            smpl_file = f".tmp/bedlam_data_{idx:02d}.smpl"
            body_pose = torch.cat([return_data["smpl_params_w"]["global_orient"], 
                                   return_data["smpl_params_w"]["body_pose"]], dim=-1)
            SMPLCodec(
                shape_parameters=return_data["smpl_params_w"]["betas"][0, :10].numpy(),
                body_pose=body_pose.reshape(-1,22,3).numpy(), 
                body_translation=return_data["smpl_params_w"]["transl"].numpy(),
                frame_count=return_data["length"], 
                frame_rate=30.0,
            ).write(smpl_file)
            print(f"Saved {smpl_file}")
            

        if False:  # cam-overlay
            smplx = make_smplx("supermotion")

            # *. original bedlam param
            # mid = self.idx2meta[idx]
            # video_path = "-".join(mid.replace("bedlam_data/", "inputs/bedlam/").split("-")[:-1])
            # npz_file = "inputs/bedlam/processed_labels/20221024_3-10_100_batch01handhair_static_highSchoolGym.npz"
            # params = np.load(npz_file, allow_pickle=True)
            # mid2index = {}
            # for j in tqdm(range(len(params["video_name"]))):
            #     k = params["video_name"][j] + "-" + params["sub"][j]
            #     mid2index[k] = j
            # betas = params['shape'][mid2index[mid]][:length]
            # global_orient_incam = torch.from_numpy(params['pose_cam'][121][:, :3])
            # body_pose = torch.from_numpy(params['pose_cam'][121][:, 3:66])
            # transl_incam = torch.from_numpy(params["trans_cam"][121])

            smplx_out = smplx(**smpl_params_c)

            # ----- Render Overlay ----- #
            mid = Path(self.idx2meta[idx])
            video_path = self.root / "videos" / mid.parts[-2] / "mp4" / (mid.stem.split('-')[0] + '.mp4')
            images = read_video_np(video_path, data["start_end"][0], data["start_end"][1])
            render_dict = {
                "K": data["cam_int"][:1],  # only support batch-size 1
                "faces": smplx.faces,
                "verts": smplx_out.vertices,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            save_video(img_overlay, f".tmp/bedlam_data_{idx:02d}.mp4", crf=23)

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return return_data


# group_name = "train_datasets/imgfeat_bedlam"
# MainStore.store(name="v2", node=builds(BedlamDatasetV2), group=group_name)
# MainStore.store(name="v2_random1024", node=builds(BedlamDatasetV2, random1024=True), group=group_name)
