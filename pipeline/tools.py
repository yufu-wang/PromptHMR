import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from scipy.interpolate import make_interp_spline, interp1d

from .image_folder import ImageFolder


def est_camera(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    h, w = image.shape[:2]
    camera = {
        'img_focal': np.max([h, w]),
        'img_center': np.array([w/2, h/2])
    }
    return camera

def est_calib(images):
    """ Roughly estimate intrinsics by image dimensions """
    # imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = images[0] # cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def detect_track(images, savedir=None, visualization=False,
                 yolo_thresh=0.10, 
                 bytetrack_thresh=0.25, 
                 bytetrack_match=0.8,
                 bbox_interp=False):

    # Yolo + Bytetrack
    yolo = YOLO("data/yolo11x.pt")
    box_annotator = sv.BoxAnnotator(thickness=5)
    tracker = sv.ByteTrack(track_activation_threshold=bytetrack_thresh,
                            minimum_matching_threshold=bytetrack_match)
    tracks = defaultdict(lambda: defaultdict(list))

    for i in range(len(images)):
        img = images[i]
        if isinstance(img, str):
            img = cv2.imread(images[i])[:,:,::-1]
            
        results = yolo(img, verbose=False, classes=0, conf=yolo_thresh)[0] 
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for k in range(len(detections)):
            d = detections[k]
            tid = d.tracker_id.item()
            tracks[tid]['masks'].append(np.full(img.shape[:2], False))
            tracks[tid]['bboxes'].append(d.xyxy[0])
            tracks[tid]['frames'].append(i)

        # Save visualization
        if visualization:
            annotated_frame = box_annotator.annotate(img[:,:,::-1].copy(), 
                                                    detections=detections,
                                                    custom_color_lookup=detections.tracker_id
                                                    )
            height, width, layers = annotated_frame.shape
            size = (width, height)

            try:
                out.write(annotated_frame)
            except Exception:
                print('Start writing video ...')
                video_path = f'{savedir}/vis.mp4'
                fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), 30
                out = cv2.VideoWriter(video_path, fourcc, fps, size)

    for k in tracks:
        masks = np.stack(tracks[k]['masks'])
        bboxes = np.stack(tracks[k]['bboxes'])
        frames = np.array(tracks[k]['frames'])

        if bbox_interp:
            interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(bboxes, frames, masks, fn='linear')
        else:
            interp_bboxes, interp_frames, interp_masks = bboxes, frames, masks
        
        tracks[k]['track_id'] = k
        tracks[k]['frames'] = interp_frames
        tracks[k]['bboxes'] = interp_bboxes
        tracks[k]['masks'] = interp_masks
        tracks[k]['detected'] = np.sum(interp_masks, axis=(1, 2)) > 1
        
    tracks = recursive_to_dict(tracks)

    if visualization:
        out.release()
        del out

    return tracks


def detect_segment_track_sam(images, out_path, paths_dict, debug_masks, sam2_type, 
                             detector_type='detectron2', filter_ng_points=False, kp_thres=0.1, 
                             num_max_people=10, height_thresh=0.3, score_thresh=0.4, det_thresh=0.5, 
                             bbox_interp=False):
    from torch.utils.data import DataLoader
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    from detectron2.config import get_cfg
    from .detector.sam2_video_predictor import build_sam2_video_predictor
    from .utils_detectron2 import DefaultPredictor
    
    device = 'cuda'
    segm_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
                                weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
    segm_model.eval()
    
    segm_dataloader = DataLoader(ImageFolder(images), batch_size=8, shuffle=False, 
                                    num_workers=4 if os.cpu_count() > 4 else os.cpu_count())

    deeplab_masks = []
    for batch in segm_dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        with torch.no_grad():
            output = segm_model(batch['img'])['out']
        mask = (output.argmax(1) == 15).to(torch.float) # the max probability is the person class
        mask = mask.cpu()
        deeplab_masks.append(mask > 0.5)
    deeplab_masks = torch.cat(deeplab_masks, dim=0)
    masks = deeplab_masks.cpu().numpy()
    del deeplab_masks
    
    sam2_registry = {
        'tiny': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_tiny.pt',
            'config': "pipeline/sam2/sam2_hiera_t.yaml",
        },
        'small': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_small.pt',
            'config': "pipeline/sam2/sam2_hiera_s.yaml",
        },
        'base_plus': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_base_plus.pt',
            'config': "pipeline/sam2/sam2_hiera_b+.yaml",
        },
        'large': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_large.pt',
            'config': "pipeline/sam2/sam2_hiera_l.yaml"
        },
    }

    checkpoint = sam2_registry[sam2_type]['checkpoint']
    model_cfg = sam2_registry[sam2_type]['config']
    if not os.path.exists(model_cfg):
        model_cfg = '/code/' + model_cfg
    model_cfg = '/' + os.path.abspath(model_cfg)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    if detector_type == 'detectron2':
        # ViTDet
        cfg_path = 'pipeline/detectron2/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
        if not os.path.exists(cfg_path):
            cfg_path = f"/code/{cfg_path}"
        
        import logging
        logging.getLogger().setLevel(logging.WARNING)

        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
        cfg.MODEL.WEIGHTS = f"{paths_dict['sam2']}/keypoint_rcnn_5ad38f.pkl"
        detector = DefaultPredictor(cfg)
        kp_thres = 0.1
     
    start_frame = -1
    for i in range(len(images)):
        if isinstance(images[i], str):
            img_cv2 = cv2.imread(images[i])
        elif isinstance(images[i], np.ndarray):
            img_cv2 = images[i][..., ::-1]

        if detector_type == 'detectron2':
            det_out = detector(img_cv2)
            keypoints = det_out['instances'].pred_keypoints.cpu().numpy()
            scores = det_out['instances'].scores.cpu().numpy()
            boxes = det_out['instances'].pred_boxes.tensor.cpu().numpy()

        if len(scores) > 0:
            keep = nms_oks(keypoints.transpose(0, 2, 1), boxes, 0.5)
            keypoints = keypoints[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            conf = scores > det_thresh
            keypoints = keypoints[conf]
            scores = scores[conf]
            boxes = boxes[conf]
            
            # Calculate areas of bounding boxes
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            bbox_height = boxes[:, 3] - boxes[:, 1]
            
            # Normalize areas and scores
            normalized_areas = areas / np.max(areas)
            normalized_scores = scores / np.max(scores)
            
            # Combine area and score with equal weight
            combined_score = 0.5 * normalized_areas + 0.5 * normalized_scores
            print(f"Combined score: {combined_score}")
            best_idxs = [x for x in np.argsort(combined_score)[::-1] if combined_score[x] > score_thresh]
            
            if len(best_idxs) > 1:
                largest_person_idx = np.argmax(bbox_height)
                largest_person_height = bbox_height[largest_person_idx]
                print(f'number of people before filtering (height < {height_thresh} max): {len(best_idxs)}')

                best_idxs = [x for x in best_idxs if bbox_height[x] > largest_person_height * height_thresh]
                print(f'number of people after filtering (height < {height_thresh} max): {len(best_idxs)}')

            # Get the details of the best person
            best_boxes = boxes[best_idxs]
            best_scores = scores[best_idxs]
            best_keypoints = keypoints[best_idxs]
            best_combined_scores = combined_score[best_idxs]
            start_frame = i
            print(f"Best person scores: {best_scores}")
            print(f"Best person combined scores: {best_combined_scores}")
            break
        else:
            print("No persons detected in the image.")
            continue
    
    if start_frame == -1:
        raise ValueError("No persons detected in any of the images.")
            
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if isinstance(images[0], str):
            state = predictor.init_state(
                video_path=os.path.dirname(images[0]),
                async_loading_frames=False,
                offload_video_to_cpu=True,
            )
        elif isinstance(images[0], np.ndarray):
            state = predictor.init_state(
                video_frames=images,
                async_loading_frames=False,
                offload_video_to_cpu=True,
            )

        if debug_masks:
            img_debug = images[start_frame][..., ::-1].copy()
            
        for idx in range(len(best_boxes)):
            pos_kp = best_keypoints[idx][best_keypoints[idx, :, 2] > kp_thres]
            neg_kp = []
            if filter_ng_points:
                for pid in range(len(best_boxes)):
                    if pid != idx:
                        neg_p_kp = best_keypoints[pid][best_keypoints[pid, :, 2] > kp_thres]
                        # check if the keypoints are within the bounding box
                        x1, y1, x2, y2 = best_boxes[idx]
                        mask = (neg_p_kp[:, 0] >= x1) & (neg_p_kp[:, 0] <= x2) & \
                            (neg_p_kp[:, 1] >= y1) & (neg_p_kp[:, 1] <= y2)
                        if mask.any():
                            neg_kp.append(neg_p_kp[mask])
                        
            if len(neg_kp) > 0:
                neg_kp = np.concatenate(neg_kp, axis=0)
                box_p = best_boxes[idx].reshape(-1, 2)
                box_labels = np.array([2, 3])
                points = np.concatenate([pos_kp[:, :2], neg_kp[:, :2], box_p], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.zeros(len(neg_kp)), box_labels]).astype(np.int64)
            else:
                points = np.concatenate([pos_kp[:, :2], best_boxes[idx].reshape(-1, 2)], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.array([2, 3])]).astype(np.int64)
            
            _, out_id, msk = predictor.add_new_points(
                state,
                frame_idx=start_frame,
                obj_id=idx+1,
                labels=point_labels, 
                points=points,
            )
            
            if debug_masks:                
                mask_colors = [
                    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), 
                    (0, 255, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0),
                ]
                c = mask_colors[idx % len(mask_colors)]
                
                for pt, l in zip(points, point_labels):
                    pc = c if l != 0 else (0, 0, 0)
                    img_debug = cv2.circle(img_debug, (int(pt[0]), int(pt[1])), 3, pc, -1)
                
                # draw text object id in the middle of the bounding box
                cx, cy = (best_boxes[idx, :2] + best_boxes[idx, 2:]) / 2
                img_debug = cv2.putText(img_debug, f"{idx+1}", (int(cx), int(cy)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                img_debug = cv2.putText(img_debug, f"{idx+1}", (int(cx), int(cy)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                color_mask = np.zeros_like(img_debug)
                msk = (msk[idx, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                color_mask[msk > 0] = c
                img_debug = cv2.rectangle(img_debug, 
                                      (int(best_boxes[idx, 0]), int(best_boxes[idx, 1])), 
                                      (int(best_boxes[idx, 2]), int(best_boxes[idx, 3])), 
                                      c, 2)
                img_v = img_debug.copy() * 0.7 + color_mask * 0.3
                img_debug[msk > 0] = img_v[msk > 0]
                cv2.imwrite(f"{out_path}/init_masks.jpg", img_debug)
        
        video_segments = {}
        track_results = {}
        # propagate the prompts to get masklets throughout the video
        for t, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if debug_masks:
                video_segments[t] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            boxes = batched_mask_to_box(out_mask_logits > 0.0)
            for obj_id, box, obj_msk in zip(out_obj_ids, boxes, out_mask_logits):
                if obj_id not in track_results:
                    track_results[obj_id] = {
                        'track_id': obj_id,
                        'bbox_format': 'x1y1x2y2',
                        'frames': [],
                        'bboxes': [],
                        'masks': [],
                    }
                track_results[obj_id]['frames'].append(t)
                obj_msk = (obj_msk[0] > 0.0).cpu().numpy().astype(np.bool_)
                track_results[obj_id]['bboxes'].append(box[0].cpu().numpy())
                track_results[obj_id]['masks'].append(obj_msk)
                
        for k in track_results.keys():
            track_results[k]['frames'] = np.array(track_results[k]['frames'])
            track_results[k]['bboxes'] = np.array(track_results[k]['bboxes'])
            track_results[k]['masks'] = np.array(track_results[k]['masks'])
            bboxes = track_results[k]['bboxes']
            frames = track_results[k]['frames']
            obj_masks = track_results[k]['masks']
            
            bb_heights = bboxes[:, 3] - bboxes[:, 1]
            bb_widths = bboxes[:, 2] - bboxes[:, 0]
            
            # Remove the 0s at the beginning and end of the bb_heights
            non_zero_indices = np.where(np.logical_and(bb_heights > 0, bb_widths > 0))[0]
            if len(non_zero_indices) > 0:
                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1] + 1
                
                frames = frames[start_index:end_index]
                bboxes = bboxes[start_index:end_index]
                obj_masks = obj_masks[start_index:end_index]
                bb_heights = bb_heights[start_index:end_index]
                bb_widths = bb_widths[start_index:end_index]
                
            # for the remaining interpolated frames, interpolate the bounding boxes
            bboxes = bboxes[np.logical_and(bb_heights > 10, bb_widths > 10)]
            frames = frames[np.logical_and(bb_heights > 10, bb_widths > 10)]
            obj_masks = obj_masks[np.logical_and(bb_heights > 10, bb_widths > 10)]
            
            if len(bboxes) == 0 or len(frames) == 0:
                print(f"No valid bboxes or frames for track {k}")
                continue
            
            if bbox_interp:
                interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(bboxes, frames, obj_masks, fn='linear')
            else:
                interp_bboxes, interp_frames, interp_masks = bboxes, frames, obj_masks
            
            track_results[k]['frames'] = interp_frames
            track_results[k]['bboxes'] = interp_bboxes
            track_results[k]['masks'] = interp_masks
            track_results[k]['detected'] = np.sum(interp_masks, axis=(1, 2)) > 1
            
    # sort the track_results by the number of frames and then take the num_max_people 
    track_results = dict(sorted(track_results.items(), key=lambda x: len(x[1]['frames']), reverse=True))
    keep_track_ids = list(track_results.keys())[:num_max_people]
    track_results = {k: v for k, v in track_results.items() if k in keep_track_ids}
    
    del detector
    del predictor
    del segm_model

    return track_results, masks


def recursive_to_dict(obj):
    """Convert a nested structure of defaultdicts (and any normal dicts) into plain dicts."""
    if isinstance(obj, defaultdict):          # or `collections.abc.Mapping`
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):               # keep lists intact, but descend into them
        return [recursive_to_dict(v) for v in obj]
    else:
        return obj 
    
    
def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e


def interpolate_bboxes(bboxes, frames, masks, fn='linear'):
    '''
    bboxes: numpy array of shape (len(frames), 4) representing the bounding boxes in x, y, x, y format
    frames: example [0, 1, 2, 8, 9, 10] -> here the frames 3-7 are missing and should be interpolated
    '''
    
    # Create a continuous range of frames
    all_frames = np.arange(frames[0], frames[-1] + 1)
    
    # Interpolate bounding boxes
    if fn == 'spline':
        interp_bboxes = make_interp_spline(frames, bboxes, k=3)(all_frames)
    elif fn == 'linear':
        # Use scipy's interp1d for linear interpolation
        interp_func = interp1d(frames, bboxes, axis=0, kind='linear')
        interp_bboxes = interp_func(all_frames)
    else:
        raise ValueError("Invalid interpolation function. Choose 'spline' or 'linear'.")   
    
    all_masks = np.full([len(all_frames), masks.shape[1], masks.shape[2]], False, dtype=bool)
    indices = frames - all_frames[0]
    all_masks[indices] = masks

    return interp_bboxes, all_frames, all_masks







