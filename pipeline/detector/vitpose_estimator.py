import time
import cv2
import numpy as np
import copy
from .ViTPose.easy_vitpose.inference import VitInference

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False

def load_vit_model(model_path=None, model_name= 'h',is_video=True, single_pose=False):
    # Initialize model
    model = VitInference(model_path, model_name,
                         det_class='human', dataset='coco_25', #["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]
                         is_video=is_video,
                         single_pose=single_pose)  # type: ignore
    return model


def estimate_kp2ds_from_bbox_vitpose(model, images, bboxes, track_id, frame_inds):
    all_kp2ds = []

    for ind, ith in enumerate(frame_inds):
        img, bbox = images[ind], bboxes[ind]

        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        keypoints, kps_img = model.inference_wbbox(img, bbox)
        kp2d_save = copy.deepcopy(keypoints)
        kp2d_save[:, :2] = kp2d_save[:, :2][:, ::-1] 
        all_kp2ds.append(kp2d_save)

    all_kp2ds = np.array(all_kp2ds)
    return all_kp2ds
