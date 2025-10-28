from pathlib import Path
import numpy as np
import torch
from torch.utils import data
from hmr4d.utils.pylogger import Log
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines

from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.pytorch3d_transform import quaternion_to_matrix
from hmr4d.utils.geo.hmr_cam import estimate_K, resize_K
from hmr4d.utils.geo.flip_utils import flip_kp2d_coco17

from .utils import EMDB1_NAMES, EMDB2_NAMES

VID_PRESETS = {1: EMDB1_NAMES, 2: EMDB2_NAMES}


from hmr4d.configs import MainStore, builds


class EmdbSmplFullSeqDataset(data.Dataset):
    def __init__(self, split=1, flip_test=False, version="v1"):
        """
        split: 1 for EMDB-1, 2 for EMDB-2
        flip_test: if True, extra flip data will be returned
        """
        super().__init__()
        self.dataset_name = "EMDB"
        self.split = split
        self.dataset_id = f"EMDB_{split}"
        Log.info(f"[{self.dataset_name}] Full sequence, split={split}")

        # Load evaluation protocol from WHAM labels
        tic = Log.time()
        self.emdb_dir = Path("inputs/EMDB/hmr4d_support")
        # 'name', 'gender', 'smpl_params', 'mask', 'K_fullimg', 'T_w2c', 'bbx_xys', 'kp2d', 'features'
        self.labels = torch.load(self.emdb_dir / "emdb_vit_v4.pt")
        
        for k, v in self.labels.items():
            img_feats_path = self.emdb_dir / f"imgfeats/prhmr_{version}" / f"{k}.pt"
            if not img_feats_path.exists():
                Log.warn(f"[{self.dataset_name}] {img_feats_path} does not exist")
            else:
                self.labels[k]["features"] = torch.load(img_feats_path)[:, 0]
                assert len(self.labels[k]["features"]) == len(self.labels[k]["mask"])
        
        self.cam_traj = torch.load(self.emdb_dir / "emdb_dpvo_traj.pt")  # estimated with DPVO

        # Setup dataset index
        self.idx2meta = []
        for vid in VID_PRESETS[split]:
            seq_length = len(self.labels[vid]["mask"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s")

        # If flip_test is enabled, we will return extra data for flipped test
        self.flip_test = flip_test
        if self.flip_test:
            Log.info(f"[{self.dataset_name}] Flip test enabled")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # [vid, start, end]
        vid, start, end = self.idx2meta[idx]
        length = end - start
        meta = {"dataset_id": self.dataset_id, "vid": vid, "vid-start-end": (start, end)}
        data.update({"meta": meta, "length": length})

        label = self.labels[vid]

        # smpl_params in world
        gender = label["gender"]
        smpl_params = label["smpl_params"]
        mask = label["mask"]
        data.update({"smpl_params": smpl_params, "gender": gender, "mask": mask})

        # camera
        K_fullimg = label["K_fullimg"]  # We use estimated K
        width_height = (1440, 1920) if vid != "P0_09_outdoor_walk" else (720, 960)
        # width_height = (720, 960)
        # K_fullimg = estimate_K(*width_height)
        T_w2c = label["T_w2c"]
        data.update({"K_fullimg": K_fullimg, "T_w2c": T_w2c, "img_wh": torch.tensor(width_height)[None].repeat(length, 1)})
        
        data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)
        
        # resize K_fullimg to PromptHMR input size 896x896
        size = torch.tensor(width_height)
        img_size = 896
        scale = img_size / max(width_height)
        offset = (img_size - (scale * size)) / 2
        data['K_fullimg'][:, :2] *= scale
        data['K_fullimg'][:, :2, -1] += offset
        
        # R_w2c -> cam_angvel
        use_DPVO = False
        if use_DPVO:
            traj = self.cam_traj[data["meta"]["vid"]]  # (L, 7)
            R_w2c = quaternion_to_matrix(traj[:, [6, 3, 4, 5]]).mT  # (L, 3, 3)
        else:  # GT
            R_w2c = data["T_w2c"][:, :3, :3]  # (L, 3, 3)
        data["cam_angvel"] = compute_cam_angvel(R_w2c)  # (L, 6)

        # image bbx, features
        bbx_xys = label["bbx_xys"]
        f_imgseq = label["features"]
        kp2d = label["kp2d"]
        data.update({"bbx_xys": bbx_xys, "f_imgseq": f_imgseq, "kp2d": kp2d})
        bbx_ul = bbx_xys[..., :2] - (bbx_xys[..., 2:] / 2)
        bbx_br = bbx_xys[..., :2] + (bbx_xys[..., 2:] / 2)
        bbx_xyxy = torch.cat([bbx_ul, bbx_br], dim=-1)
        data['bbx_xyxy'] = bbx_xyxy
        
        boxes = data["bbx_xyxy"].reshape(-1, 2, 2)
        boxes *= scale
        boxes += offset[None, None, :]
        data["bbx_xyxy"] = boxes.reshape(-1, 4)

        # to render a video
        video_path = self.emdb_dir / f"videos/{vid}.mp4"
        frame_id = torch.where(mask)[0].long()
        # resize_factor = 0.5
        # width_height_render = torch.tensor(width_height) # * resize_factor
        # K_render = resize_K(K_fullimg, resize_factor)
        # bbx_xys_render = bbx_xys * resize_factor
        data["meta_render"] = {
            "split": self.split,
            "name": vid,
            "video_path": str(video_path),
            "resize_factor": 1.0,
            "frame_id": frame_id,
            "width_height": torch.tensor(width_height).int(),
            "K": K_fullimg,
            "bbx_xys": bbx_xys,
            "R_cam_type": "DPVO" if use_DPVO else "GtGyro",
        }

        # if enable flip_test
        if self.flip_test:
            imgfeat_dir = self.emdb_dir / "imgfeats/emdb_flip"
            f_img_dict = torch.load(imgfeat_dir / f"{vid}.pt")

            flipped_bbx_xys = f_img_dict["bbx_xys"].float()  # (L, 3)
            flipped_features = f_img_dict["features"].float()  # (L, 1024)
            width = width_height[0]
            flipped_kp2d = flip_kp2d_coco17(kp2d, width)  # (L, 17, 3)

            R_flip_x = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
            flipped_R_w2c = R_flip_x @ R_w2c.clone()

            data_flip = {
                "bbx_xys": flipped_bbx_xys,
                "f_imgseq": flipped_features,
                "kp2d": flipped_kp2d,
                "cam_angvel": compute_cam_angvel(flipped_R_w2c),
            }
            data["flip_test"] = data_flip

        return data

    def _process_data(self, data):
        length = data["length"]
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


# EMDB-1 and EMDB-2
MainStore.store(
    name="v1",
    node=builds(EmdbSmplFullSeqDataset, populate_full_signature=True),
    group="test_datasets/emdb1",
)
MainStore.store(
    name="v1_fliptest",
    node=builds(EmdbSmplFullSeqDataset, flip_test=True, populate_full_signature=True),
    group="test_datasets/emdb1",
)
MainStore.store(
    name="v1",
    node=builds(EmdbSmplFullSeqDataset, split=2, populate_full_signature=True),
    group="test_datasets/emdb2",
)
MainStore.store(
    name="v1_fliptest",
    node=builds(EmdbSmplFullSeqDataset, split=2, flip_test=True, populate_full_signature=True),
    group="test_datasets/emdb2",
)

'''
P0_09_outdoor_walk torch.Size([2009]) tensor(2009)
P1_14_outdoor_climb torch.Size([1299]) tensor(1284)
P2_19_indoor_walk_off_mvs torch.Size([1299]) tensor(1299)
P2_20_outdoor_walk torch.Size([2724]) tensor(2713)
P2_23_outdoor_hug_tree torch.Size([1277]) tensor(1086)
P2_24_outdoor_long_walk torch.Size([3280]) tensor(3280)
P3_27_indoor_walk_off_mvs torch.Size([1448]) tensor(1448)
P3_28_outdoor_walk_lunges torch.Size([1836]) tensor(1836)
P3_29_outdoor_stairs_up torch.Size([1205]) tensor(1205)
P3_30_outdoor_stairs_down torch.Size([1170]) tensor(1137)
P3_31_outdoor_workout torch.Size([1216]) tensor(1216)
P3_32_outdoor_soccer_warmup_a torch.Size([1084]) tensor(1084)
P3_33_outdoor_soccer_warmup_b torch.Size([1433]) tensor(1433)
P4_35_indoor_walk torch.Size([1244]) tensor(1226)
P4_36_outdoor_long_walk torch.Size([2160]) tensor(2160)
P4_37_outdoor_run_circle torch.Size([881]) tensor(881)
P5_40_indoor_walk_big_circle torch.Size([2661]) tensor(2661)
P5_42_indoor_dancing torch.Size([1291]) tensor(1291)
P5_44_indoor_rom torch.Size([1381]) tensor(1381)
P6_48_outdoor_walk_downhill torch.Size([1973]) tensor(1959)
P6_49_outdoor_big_stairs_down torch.Size([1559]) tensor(1559)
P6_50_outdoor_workout torch.Size([1532]) tensor(1532)
P6_51_outdoor_dancing torch.Size([1427]) tensor(1427)
P7_55_outdoor_walk torch.Size([2179]) tensor(2179)
P7_56_outdoor_stairs_up_down torch.Size([1120]) tensor(1120)
P7_57_outdoor_rock_chair torch.Size([1558]) tensor(1558)
P7_58_outdoor_parcours torch.Size([1332]) tensor(1332)
P7_59_outdoor_rom torch.Size([1839]) tensor(1839)
P7_60_outdoor_workout torch.Size([1693]) tensor(1693)
P7_61_outdoor_sit_lie_walk torch.Size([1914]) tensor(1914)
P8_64_outdoor_skateboard torch.Size([1704]) tensor(1704)
P8_65_outdoor_walk_straight torch.Size([1981]) tensor(1981)
P8_68_outdoor_handstand torch.Size([1606]) tensor(1606)
P8_69_outdoor_cartwheel torch.Size([656]) tensor(656)
P9_76_outdoor_sitting torch.Size([1768]) tensor(1768)
P9_77_outdoor_stairs_up torch.Size([728]) tensor(728)
P9_78_outdoor_stairs_up_down torch.Size([1083]) tensor(1083)
P9_79_outdoor_walk_rectangle torch.Size([1917]) tensor(1917)
P9_80_outdoor_walk_big_circle torch.Size([2240]) tensor(2240)
'''