import os
import shutil
import cv2
import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

from smplx import SMPLX
from pipeline.detector import segment
from pipeline.detector.vitpose_estimator import load_vit_model, estimate_kp2ds_from_bbox_vitpose
from pipeline.kp_utils import convert_kps
from pipeline.utils import prepare_inputs, load_video_frames, interpolate_bboxes
from pipeline.tools import detect_track, detect_segment_track_sam, est_camera, est_calib
from pipeline.phmr_vid import PromptHMR_Video
from pipeline.camera import run_metric_slam, calibrate_intrinsics, run_slam
from pipeline.spec import run_cam_calib
from pipeline.world import world_hps_estimation
from pipeline.postprocessing import post_optimization
from pipeline.mcs_export_cam import export_scene_with_camera
from smplcodec import SMPLCodec


class Pipeline:
    def __init__(self, static_cam=False):
        self.images = None
        self.cfg = OmegaConf.load("pipeline/config.yaml")
        self.cfg.static_cam = static_cam
        
        checkpoint_dir = 'data/pretrain'
        self.data_dict = {
            'droid': os.path.join(checkpoint_dir, 'droid.pth'), 
            'sam': os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth"), 
            'sam2': os.path.join(checkpoint_dir, "sam2_ckpts"), 
            'yolo': os.path.join(checkpoint_dir, 'yolo11x.pt'), 
            'vitpose': os.path.join(checkpoint_dir, 'vitpose-h-coco_25.pth'), 
        }

        self.smplx = SMPLX(
            f'data/body_models/smplx/SMPLX_NEUTRAL.npz', 
            use_pca=False, 
            flat_hand_mean=True, 
            num_betas=10
        )


    def load_frames(self, input_video, output_folder, read_frames=True):
        if read_frames == True:
            images, seq_folder, fps = load_video_frames(
                input_video, 
                output_folder=output_folder, 
                max_height=896,
                max_fps=60,
            )
        else: 
            # this currently will cause issue with sam2 
            images, seq_folder, img_folder, fps = prepare_inputs(
                input_video, 
                output_folder=output_folder, 
                max_height=896,
                max_fps=60,
            )
        self.fps = fps
        return images, seq_folder


    def run_detect_track(self, ):
        if self.cfg.tracker == 'bytetrack':
            tracks = detect_track(self.images,
                                  bbox_interp=self.cfg.bbox_interp)
            masks = segment.segment_subjects(self.images)

        elif self.cfg.tracker == 'sam2':
            tracks, masks = detect_segment_track_sam(
                self.images, 
                self.seq_folder, 
                self.data_dict,
                debug_masks=False,
                sam2_type='tiny',
                detector_type='detectron2',
                num_max_people=self.cfg.num_max_people,
                det_thresh=self.cfg.det_thresh,
                score_thresh=self.cfg.det_score_thresh,
                height_thresh=self.cfg.det_height_thresh,
                bbox_interp=self.cfg.bbox_interp
            )
            
        self.results['masks'] = masks
        self.results['people'] = tracks
        self.results['has_tracks'] = True


    def estimate_2d_keypoints(self,):
        model = load_vit_model(model_path='data/pretrain/vitpose-h-coco_25.pth')
        for k, v in self.results['people'].items():
            kpts_2d = estimate_kp2ds_from_bbox_vitpose(model, self.images, v['bboxes'], k, v['frames'])
            kpts_2d = convert_kps(kpts_2d, 'vitpose25', 'openpose')
            self.results['people'][k]['keypoints_2d'] = kpts_2d
            coco_kp2d = convert_kps(kpts_2d, 'ophandface', 'cocoophf')
            self.results['people'][k]['vitpose'] = coco_kp2d
            
        self.results['has_2d_kpts'] = True
        del model
        return
    

    def hps_estimation(self,):
        if self.cfg.tracker == 'sam2':
            mask_prompt = True
        else:
            mask_prompt = False

        phmr = PromptHMR_Video()
        self.results = phmr.run(self.images, self.results, mask_prompt)
        self.results['contact_joint_ids'] = [7, 10, 8, 11, 20, 21]
        self.results['has_hps_cam'] = True
    
        return


    def camera_motion_estimation(self, static_cam = False):
        ##### Run Masked DROID-SLAM #####
        masks = self.results['masks']
        masks = torch.from_numpy(masks)
        
        assert masks.shape[0] == len(self.images), f"Masks and images should be same length {masks.shape[0]} != {len(self.images)}"
        
        opt_intr = False if self.cfg.use_depth else True
        keyframes = None
        if self.cfg.static_cam or static_cam:
            print("Using static camera assumption")
            static_cam = True
            if self.cfg.calib is None:
                cam_int = est_calib(self.images)
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False

        else:
            if self.cfg.calib is None:
                if self.cfg.focal is None and opt_intr == False:
                    try:
                        if self.cfg.calib_method == 'ba':
                            _, _, cam_int, keyframes = run_slam(
                                self.images, masks=masks, opt_intr=True, 
                                stride=self.cfg.calib_stride,
                            )
                        elif self.cfg.calib_method == 'iterative':    
                            cam_int = calibrate_intrinsics(self.cfg.img_folder, masks)
                    except ValueError as e:
                        static_cam = True
                        print(e)
                        print("Warning: probably there is not much camera motion in the video!!")
                        cam_int = est_calib(self.images)

                elif self.cfg.focal is not None:
                    cam_int = est_calib(self.images)
                    cam_int[0] = self.cfg.focal
                    cam_int[1] = self.cfg.focal
                    opt_intr = False
                else:
                    cam_int = est_calib(self.images)
            else:
                cam_int = np.loadtxt(self.cfg.calib)
                opt_intr = False
        
        if static_cam:
            cam_R = torch.eye(3)[None].repeat_interleave(len(masks), 0)
            cam_T = torch.zeros((len(masks), 3))
            print("Warning: probably there is not much camera motion in the video!!")
            print("Setting camera motion to zero")
        else:
            try:
                cam_R, cam_T, cam_int = run_metric_slam(
                    self.images, 
                    masks=masks, 
                    calib=cam_int, 
                    monodepth_method=self.cfg.depth_method, 
                    use_depth_inp=self.cfg.use_depth,
                    stride=self.cfg.stride,
                    opt_intr=opt_intr,
                    save_depth=self.cfg.save_depth,
                    keyframes=keyframes,
                )
            except ValueError as e:
                if str(e).startswith("not enough values to unpack"):
                    cam_R = torch.eye(3)[None].repeat_interleave(len(masks), 0)
                    cam_T = torch.zeros((len(masks), 3))
                    print("Warning: probably there is not much camera motion in the video!!")
                    print("Setting camera motion to zero")
                else:
                    raise e
                    
        print("Camera intrinsics:", cam_int)
        camera = {
            'pred_cam_R': cam_R.numpy(), 
            'pred_cam_T': cam_T.numpy(), 
            'img_focal': cam_int[0], 
            'img_center': cam_int[2:]
        }
        print("cam focal length: ", cam_int[0])
        self.results['camera'] = camera
        self.results['has_slam'] = True
        return


    def world_hps_estimation(self, ):
        self.results = world_hps_estimation(self.cfg, self.results, self.smplx)
        self.results['has_hps_world'] = True
        return
    

    def post_optimization(self):
        self.results = post_optimization(
            self.cfg, self.results, self.images, 
            self.smplx, opt_contact=True,
        )
        self.results['has_post_opt'] = True


    def get_K(self, ):
        camera = self.results['camera']
        K = np.eye(3)
        K[0,0] = camera['img_focal']
        K[1,1] = camera['img_focal']
        K[:2,-1] = camera['img_center']
        K = torch.tensor(K, dtype=torch.float)
        return K
    

    def create_world4d(self, results=None, total=None, step=1):
        if results is None:
            results = self.results
        if total is None:
            total = len(results['camera']['pred_cam_R'])
        else:
            total = min(total, len(results['camera']['pred_cam_R']))
            
        world4d = {}
        for i in range(0, total, step):
            pose = []
            shape = []
            transl = []
            track_id = []

            # People
            for pid in results['people']:
                people = results['people'][pid]
                frames = people['frames']
                in_frame = np.where(frames == i)[0]

                if len(in_frame) == 1:
                    smplx_w = people['smplx_world']
                    pose.append(smplx_w['pose'][in_frame])
                    shape.append(smplx_w['shape'][in_frame])
                    transl.append(smplx_w['trans'][in_frame])
                    track_id.append(people['track_id'])
            
            # Camera
            camera_w = results['camera_world']
            Rwc = camera_w['Rwc'][i]
            Twc = camera_w['Twc'][i]
            camera = np.eye(4)
            camera[:3,:3] = Rwc
            camera[:3, 3] = Twc

            if len(track_id) > 0:
                world4d[i] = {'pose': torch.tensor(np.concatenate(pose)).float().reshape(len(track_id),-1,3),
                            'shape': torch.tensor(np.concatenate(shape)).float(),
                            'trans': torch.tensor(np.concatenate(transl)).float(),
                            'track_id': torch.tensor(np.array(track_id)) - 1,
                            'camera': camera}
            else:
                world4d[i] = {'track_id': np.array([]),
                            'camera': camera}

        return world4d


    def __call__(self, input_video, output_folder, static_cam=False, 
                 save_only_essential=False, max_frame=None):

        def cvt_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    cvt_to_numpy(v)
                elif isinstance(v, torch.Tensor):
                    d[k] = v.detach().cpu().numpy()

        images, seq_folder = self.load_frames(input_video, 
                                              output_folder)

        self.images = images[:max_frame]
        self.seq_folder = seq_folder
        self.cfg.seq_folder = seq_folder

        if os.path.isfile(f'{seq_folder}/results.pkl'):
            print('Loading available results...')
            self.results = joblib.load(f'{seq_folder}/results.pkl')
            return self.results
        
        self.results = {
            'camera': {},
            'people': {},
            'timings': {},
            'masks': None,
            'has_tracks': False,
            'has_hps_cam': False,
            'has_hps_world': False,
            'has_slam': False,
            'has_hands': False,
            'has_2d_kpts': False,
            'has_post_opt': False,
        }

        ### naive camera
        if not self.results['has_slam']:
            self.results['camera'] = est_camera(images[0]) 

        ### spec camera
        if not self.results['has_slam']:
            stride = len(self.images)//30
            if stride == 0:
                stride = 1
            spec_calib = run_cam_calib(self.images, out_folder=seq_folder+'/spec_calib', 
                                        save_res=True, stride=stride, method='spec', 
                                        first_frame_idx=0)       
            self.results['spec_calib'] = spec_calib

        ### detect_segment_track 
        if not self.results['has_tracks']:
            print("Running detect, segment, and track pipeline...")
            self.run_detect_track()

        ### slam
        if not self.results['has_slam']:
            print("Running camera motion estimation...")
            self.camera_motion_estimation(static_cam)

        ### keypoints detection
        if not self.results['has_2d_kpts']:
            print("Estimating 2D keypoints...")
            self.estimate_2d_keypoints()
        
        ### hps
        if not self.results['has_hps_cam']:
            print("Running human mesh estimation...")
            self.hps_estimation()

        ### convert hps to world coordinate
        if not self.results['has_hps_world']:
            print("Running world coordinates estimation...")
            self.world_hps_estimation()

        cvt_to_numpy(self.results)

        # ### post optimization
        if self.cfg.run_post_opt and not self.results['has_post_opt']:
            print("Running post optimization...")
            self.post_optimization()

        if save_only_essential:
            _ = self.results.pop('masks', None)
            for tid, track in self.results['people'].items():
                _ = track.pop('masks', None)
                _ = track.pop('keypoints_2d', None)
                _ = track.pop('vitpose', None)
                _ = track.pop('prhmr_img_feats', None)
                
        joblib.dump(self.results, f'{seq_folder}/results.pkl')
        
        NUM_FRAMES = len(self.images)
        MCS_OUTPUT_PATH = f'{seq_folder}/world4d.mcs'
        smpl_paths = []
        per_body_frame_presence = []
        for k,v in self.results['people'].items():
            out_smpl_f = f'{os.path.abspath(self.cfg.seq_folder)}/subject-{k}.smpl'
            
            SMPLCodec(
                shape_parameters=v['smplx_world']['shape'].mean(0),
                body_pose=v['smplx_world']['pose'][:, :22*3].reshape(-1,22,3), 
                body_translation=v['smplx_world']['trans'],
                frame_count=v['frames'].shape[0], frame_rate=float(self.cfg.fps)
            ).write(out_smpl_f)
            smpl_paths.append(out_smpl_f)
            per_body_frame_presence.append([int(v['frames'][0]), int(v['frames'][-1])+1])
        
        export_scene_with_camera(
            smpl_buffers=[open(path, 'rb').read() for path in smpl_paths],
            frame_presences=per_body_frame_presence,
            num_frames=NUM_FRAMES,
            output_path=MCS_OUTPUT_PATH,
            rotation_matrices=self.results['camera_world']['Rcw'],
            translations=self.results['camera_world']['Tcw'],
            focal_length=self.results['camera_world']['img_focal'],
            principal_point=self.results['camera_world']['img_center'],
            frame_rate=float(self.cfg.fps),
            smplx_path='data/body_models/smplx/SMPLX_neutral_array_f32_slim.npz',
        )

        print("Usage:")
        print(f'\tYou can drag and drop the "world4d.mcs" file to https://me.meshcapade.com/editor to view the result')
        print(f'\tYou can import the "world4d.glb" file on Blender to view the result')
        
        return self.results