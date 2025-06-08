import torch
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam, 
    compute_transl_full_cam, 
    compute_transl_full_cam_prompthmr, 
    inv_compute_transl_full_cam_prompthmr,
    compute_transl_full_cam_metric3d,
    inv_compute_transl_full_cam_metric3d,
    get_a_pred_cam, 
    project_to_bi01,
)

bedlam1 = torch.load('/home/muhammed/projects/GVHMR/inputs/BEDLAM/hmr4d_support/smplpose_v2.pth')
bedlam2_smplpose = torch.load('/home/muhammed/projects/GVHMR/inputs/BEDLAM2/smplpose.pt')

transl_list = []
for k, v in bedlam2_smplpose.items():
    transl = v["trans_cam"] + v["cam_ext"][:, :3, 3]
    transl_list.append(transl)
    inv_compute_transl_full_cam_metric3d(transl, bbox_center_scale, cam_int)
import ipdb; ipdb.set_trace()