import numpy as np


def keypoint_hflip(kp, img_width):
    # Flip a keypoint horizontally around the y-axis
    # kp N,2
    if len(kp.shape) == 2:
        kp[:,0] = (img_width - 1.) - kp[:,0]
    elif len(kp.shape) == 3:
        kp[:, :, 0] = (img_width - 1.) - kp[:, :, 0]
    return kp


def convert_mmpose_body_to_openpose_body(kpts):
    kp_op = convert_kps(kpts, 'mmpose', 'openpose')
    left_shoulder = kp_op[:, 5].copy()
    right_shoulder = kp_op[:, 2].copy()
    left_hip = kp_op[:, 12].copy()
    right_hip = kp_op[:, 9].copy()
    
    neck = left_shoulder + (right_shoulder - left_shoulder) / 2.
    neck[:, 2] = (left_shoulder[:, 2] + right_shoulder[:, 2]) / 2.
    midhip = left_hip + (right_hip - left_hip) / 2.
    midhip[:, 2] = (left_hip[:, 2] + right_hip[:, 2]) / 2.
    
    kp_op[:, 1] = neck
    kp_op[:, 8] = midhip
    return kp_op


def convert_coco_to_ophandface(kpts):
    kp_op = convert_kps(kpts, 'cocoophf', 'ophandface')
    left_shoulder = kp_op[:, 5].copy()
    right_shoulder = kp_op[:, 2].copy()
    left_hip = kp_op[:, 12].copy()
    right_hip = kp_op[:, 9].copy()
    
    neck = left_shoulder + (right_shoulder - left_shoulder) / 2.
    neck[:, 2] = (left_shoulder[:, 2] + right_shoulder[:, 2]) / 2.
    midhip = left_hip + (right_hip - left_hip) / 2.
    midhip[:, 2] = (left_hip[:, 2] + right_hip[:, 2]) / 2.
    kp_op[:, 1] = neck
    kp_op[:, 8] = midhip
    return kp_op


def convert_wholebody_to_ophandface(kpts, ret_json=False):
    kp_op = convert_kps(kpts, 'wholebody', 'ophandface')
    left_shoulder = kp_op[:, 5].copy()
    right_shoulder = kp_op[:, 2].copy()
    left_hip = kp_op[:, 12].copy()
    right_hip = kp_op[:, 9].copy()
    
    neck = left_shoulder + (right_shoulder - left_shoulder) / 2.
    neck[:, 2] = (left_shoulder[:, 2] + right_shoulder[:, 2]) / 2.
    midhip = left_hip + (right_hip - left_hip) / 2.
    midhip[:, 2] = (left_hip[:, 2] + right_hip[:, 2]) / 2.
    
    kp_op[:, 1] = neck
    kp_op[:, 8] = midhip
    
    if ret_json:
        json_list = []
        
        for idx in range(kp_op.shape[0]):
            kp_dict = {
                "version":1.3,
                "people":[
                    {
                        "person_id":[-1],
                        "pose_keypoints_2d": kp_op[idx, 0:25].reshape(-1).tolist(),
                        "face_keypoints_2d": kp_op[idx, 25:95].reshape(-1).tolist(),
                        "hand_left_keypoints_2d": kp_op[idx, 95:116].reshape(-1).tolist(),
                        "hand_right_keypoints_2d": kp_op[idx, 116:137].reshape(-1).tolist(),
                        "pose_keypoints_3d":[],
                        "face_keypoints_3d":[],
                        "hand_left_keypoints_3d":[],
                        "hand_right_keypoints_3d":[]
                    }]
                }
            json_list.append(kp_dict)
        
        return kp_op, json_list
    
    return kp_op


def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), joints2d.shape[-1]))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs


def get_mpii3d_test_joint_names():
    return [
        'headtop', # 'head_top',
        'neck',
        'rshoulder',# 'right_shoulder',
        'relbow',# 'right_elbow',
        'rwrist',# 'right_wrist',
        'lshoulder',# 'left_shoulder',
        'lelbow', # 'left_elbow',
        'lwrist', # 'left_wrist',
        'rhip', # 'right_hip',
        'rknee', # 'right_knee',
        'rankle',# 'right_ankle',
        'lhip',# 'left_hip',
        'lknee',# 'left_knee',
        'lankle',# 'left_ankle'
        'hip',# 'pelvis',
        'Spine (H36M)',# 'spine',
        'Head (H36M)',# 'head'
    ]


def get_mpii3d_joint_names():
    return [
        'spine3', # 0,
        'spine4', # 1,
        'spine2', # 2,
        'Spine (H36M)', #'spine', # 3,
        'hip', # 'pelvis', # 4,
        'neck', # 5,
        'Head (H36M)', # 'head', # 6,
        "headtop", # 'head_top', # 7,
        'left_clavicle', # 8,
        "lshoulder", # 'left_shoulder', # 9,
        "lelbow", # 'left_elbow',# 10,
        "lwrist", # 'left_wrist',# 11,
        'left_hand',# 12,
        'right_clavicle',# 13,
        'rshoulder',# 'right_shoulder',# 14,
        'relbow',# 'right_elbow',# 15,
        'rwrist',# 'right_wrist',# 16,
        'right_hand',# 17,
        'lhip', # left_hip',# 18,
        'lknee', # 'left_knee',# 19,
        'lankle', #left ankle # 20
        'left_foot', # 21
        'left_toe', # 22
        "rhip", # 'right_hip',# 23
        "rknee", # 'right_knee',# 24
        "rankle", #'right_ankle', # 25
        'right_foot',# 26
        'right_toe' # 27
    ]


# def get_insta_joint_names():
#     return [
#         'rheel'            ,   # 0
#         'rknee'            ,   # 1
#         'rhip'             ,   # 2
#         'lhip'             ,   # 3
#         'lknee'            ,   # 4
#         'lheel'            ,   # 5
#         'rwrist'           ,   # 6
#         'relbow'           ,   # 7
#         'rshoulder'        ,   # 8
#         'lshoulder'        ,   # 9
#         'lelbow'           ,   # 10
#         'lwrist'           ,   # 11
#         'neck'             ,   # 12
#         'headtop'          ,   # 13
#         'nose'             ,   # 14
#         'leye'             ,   # 15
#         'reye'             ,   # 16
#         'lear'             ,   # 17
#         'rear'             ,   # 18
#         'lbigtoe'          ,   # 19
#         'rbigtoe'          ,   # 20
#         'lsmalltoe'        ,   # 21
#         'rsmalltoe'        ,   # 22
#         'lankle'           ,   # 23
#         'rankle'           ,   # 24
#     ]


def get_insta_joint_names():
    return [
        'OP RHeel',
        'OP RKnee',
        'OP RHip',
        'OP LHip',
        'OP LKnee',
        'OP LHeel',
        'OP RWrist',
        'OP RElbow',
        'OP RShoulder',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP Neck',
        'headtop',
        'OP Nose',
        'OP LEye',
        'OP REye',
        'OP LEar',
        'OP REar',
        'OP LBigToe',
        'OP RBigToe',
        'OP LSmallToe',
        'OP RSmallToe',
        'OP LAnkle',
        'OP RAnkle',
    ]


def get_mmpose_joint_names():
    # this naming is for the first 23 joints of MMPose
    # does not include hands and face
    return [
        'OP Nose', # 1
        'OP LEye', # 2
        'OP REye', # 3
        'OP LEar', # 4
        'OP REar', # 5
        'OP LShoulder', # 6
        'OP RShoulder', # 7
        'OP LElbow', # 8
        'OP RElbow', # 9
        'OP LWrist', # 10
        'OP RWrist', # 11
        'OP LHip', # 12
        'OP RHip', # 13
        'OP LKnee', # 14
        'OP RKnee', # 15
        'OP LAnkle', # 16
        'OP RAnkle', # 17
        'OP LBigToe', # 18
        'OP LSmallToe', # 19
        'OP LHeel', # 20
        'OP RBigToe', # 21
        'OP RSmallToe', # 22
        'OP RHeel', # 23
    ]
    
    
def get_mmposehands_joint_names():
    return [
        'OP Nose', # 1
        'OP LEye', # 2
        'OP REye', # 3
        'OP LEar', # 4
        'OP REar', # 5
        'OP LShoulder', # 6
        'OP RShoulder', # 7
        'OP LElbow', # 8
        'OP RElbow', # 9
        'OP LWrist', # 10
        'OP RWrist', # 11
        'OP LHip', # 12
        'OP RHip', # 13
        'OP LKnee', # 14
        'OP RKnee', # 15
        'OP LAnkle', # 16
        'OP RAnkle', # 17
        'OP LBigToe', # 18
        'OP LSmallToe', # 19
        'OP LHeel', # 20
        'OP RBigToe', # 21
        'OP RSmallToe', # 22
        'OP RHeel', # 23
    ]


def get_insta_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [4 , 5],
            [6 , 7],
            [7 , 8],
            [8 , 9],
            [9 ,10],
            [2 , 8],
            [3 , 9],
            [10,11],
            [8 ,12],
            [9 ,12],
            [12,13],
            [12,14],
            [14,15],
            [14,16],
            [15,17],
            [16,18],
            [0 ,20],
            [20,22],
            [5 ,19],
            [19,21],
            [5 ,23],
            [0 ,24],
        ])


def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )


def get_staf_joint_names():
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]


def get_spin_op_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
    ]


def get_openpose_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
    ]
    

def get_vitpose25_joint_names():
    return [
        'OP Nose', 
        'OP LEye', 
        'OP REye', 
        'OP LEar', 
        'OP REar', 
        'OP Neck', 
        'OP LShoulder', 
        'OP RShoulder', 
        'OP LElbow', 
        'OP RElbow', 
        'OP LWrist', 
        'OP RWrist', 
        'OP LHip', 
        'OP RHip', 
        'OP MidHip', 
        'OP LKnee', 
        'OP RKnee', 
        'OP LAnkle', 
        'OP RAnkle', 
        'OP LBigToe', 
        'OP LSmallToe', 
        'OP LHeel', 
        'OP RBigToe', 
        'OP RSmallToe', 
        'OP RHeel'
    ]


def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

def get_muco3dhp_joint_names():
    return [
        'headtop',
        'thorax',
        'rshoulder',
        'relbow',
        'rwrist',
        'lshoulder',
        'lelbow',
        'lwrist',
        'rhip',
        'rknee',
        'rankle',
        'lhip',
        'lknee',
        'lankle',
        'hip',
        'Spine (H36M)',
        'Head (H36M)',
        'R_Hand',
        'L_Hand',
        'R_Toe',
        'L_Toe'
    ]

def get_h36m_joint_names():
    return [
        'hip',  # 0
        'lhip',  # 1
        'lknee',  # 2
        'lankle',  # 3
        'rhip',  # 4
        'rknee',  # 5
        'rankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]


def get_spin_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
            [0 ,38],
        ]
    )


def get_spin_op_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
        ]
    )


def get_openpose_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
        ]
    )


def get_posetrack_joint_names():
    return [
        "nose",
        "neck",
        "headtop",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]


def get_posetrack_original_kp_names():
    return [
        'nose',
        'head_bottom',
        'head_top',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]


def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]


def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]


def get_common_paper_joint_names():
    return [
        "Right Ankle",    # 0  "lankle",    # 0
        "Right Knee",     # 1  "lknee",     # 1
        "Right Hip",      # 2  "lhip",      # 2
        "Left Hip",      # 3  "rhip",      # 3
        "Left Knee",     # 4  "rknee",     # 4
        "Left Ankle",    # 5  "rankle",    # 5
        "Right Wrist",    # 6  "lwrist",    # 6
        "Right Elbow",    # 7  "lelbow",    # 7
        "Right Shoulder", # 8  "lshoulder", # 8
        "Left Shoulder", # 9  "rshoulder", # 9
        "Left Elbow",    # 10  "relbow",    # 10
        "Left Wrist",    # 11  "rwrist",    # 11
        "Neck",      # 12  "neck",      # 12
        "Head",   # 13  "headtop",   # 13
    ]


def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [12, 13],
        ]
    )


def get_coco_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]



def get_cocoop_joint_names():
    return [
        "OP Nose",      # 0
        "OP LEye",      # 1
        "OP REye",      # 2
        "OP LEar",      # 3
        "OP REar",      # 4
        "OP LShoulder", # 5
        "OP RShoulder", # 6
        "OP LElbow",    # 7
        "OP RElbow",    # 8
        "OP LWrist",    # 9
        "OP RWrist",    # 10
        "OP LHip",      # 11
        "OP RHip",      # 12
        "OP LKnee",     # 13
        "OP RKnee",     # 14
        "OP LAnkle",    # 15
        "OP RAnkle",    # 16
    ]
    
def get_cocoophf_joint_names():
    return [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
    ]

def get_ochuman_joint_names():
    return [
        'rshoulder',
        'relbow',
        'rwrist',
        'lshoulder',
        'lelbow',
        'lwrist',
        'rhip',
        'rknee',
        'rankle',
        'lhip',
        'lknee',
        'lankle',
        'headtop',
        'neck',
        'rear',
        'lear',
        'nose',
        'reye',
        'leye'
    ]


def get_crowdpose_joint_names():
    return [
        'lshoulder',
        'rshoulder',
        'lelbow',
        'relbow',
        'lwrist',
        'rwrist',
        'lhip',
        'rhip',
        'lknee',
        'rknee',
        'lankle',
        'rankle',
        'headtop',
        'neck'
    ]

def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )


def get_mpii_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "hip",       # 6
        "thorax",    # 7
        "neck",      # 8
        "headtop",   # 9
        "rwrist",    # 10
        "relbow",    # 11
        "rshoulder", # 12
        "lshoulder", # 13
        "lelbow",    # 14
        "lwrist",    # 15
    ]


def get_mpii_skeleton():
    # 0  - rankle,
    # 1  - rknee,
    # 2  - rhip,
    # 3  - lhip,
    # 4  - lknee,
    # 5  - lankle,
    # 6  - hip,
    # 7  - thorax,
    # 8  - neck,
    # 9  - headtop,
    # 10 - rwrist,
    # 11 - relbow,
    # 12 - rshoulder,
    # 13 - lshoulder,
    # 14 - lelbow,
    # 15 - lwrist,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 6 ],
            [ 6, 3 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 9 ],
            [ 7, 12],
            [12, 11],
            [11, 10],
            [ 7, 13],
            [13, 14],
            [14, 15]
        ]
    )


def get_aich_joint_names():
    return [
        "rshoulder", # 0
        "relbow",    # 1
        "rwrist",    # 2
        "lshoulder", # 3
        "lelbow",    # 4
        "lwrist",    # 5
        "rhip",      # 6
        "rknee",     # 7
        "rankle",    # 8
        "lhip",      # 9
        "lknee",     # 10
        "lankle",    # 11
        "headtop",   # 12
        "neck",      # 13
    ]


def get_aich_skeleton():
    # 0  - rshoulder,
    # 1  - relbow,
    # 2  - rwrist,
    # 3  - lshoulder,
    # 4  - lelbow,
    # 5  - lwrist,
    # 6  - rhip,
    # 7  - rknee,
    # 8  - rankle,
    # 9  - lhip,
    # 10 - lknee,
    # 11 - lankle,
    # 12 - headtop,
    # 13 - neck,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [13, 0 ],
            [13, 3 ],
            [ 0, 6 ],
            [ 3, 9 ]
        ]
    )


def get_3dpw_joint_names():
    return [
        "nose",      # 0
        "thorax",    # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "rhip",      # 8
        "rknee",     # 9
        "rankle",    # 10
        "lhip",      # 11
        "lknee",     # 12
        "lankle",    # 13
    ]


def get_3dpw_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 3 ],
            [ 3, 4 ],
            [ 1, 5 ],
            [ 5, 6 ],
            [ 6, 7 ],
            [ 2, 8 ],
            [ 5, 11],
            [ 8, 11],
            [ 8, 9 ],
            [ 9, 10],
            [11, 12],
            [12, 13]
        ]
    )


def get_smplcoco_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "rwrist",    # 6
        "relbow",    # 7
        "rshoulder", # 8
        "lshoulder", # 9
        "lelbow",    # 10
        "lwrist",    # 11
        "neck",      # 12
        "headtop",   # 13
        "nose",      # 14
        "leye",      # 15
        "reye",      # 16
        "lear",      # 17
        "rear",      # 18
    ]


def get_smplcoco_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 12],
            [12, 9 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [ 8, 2 ],
            [ 9, 3 ],
            [ 2, 3 ],
        ]
    )


def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]


def get_smpl_paper_joint_names():
    return [
        'Hips',            # 0
        'Left Hip',       # 1
        'Right Hip',      # 2
        'Spine',           # 3
        'Left Knee',         # 4
        'Right Knee',        # 5
        'Spine_1',          # 6
        'Left Ankle',        # 7
        'Right Ankle',       # 8
        'Spine_2',          # 9
        'Left Toe',     # 10
        'Right Toe',    # 11
        'Neck',            # 12
        'Left Shoulder',    # 13
        'Right Shoulder',   # 14
        'Head',            # 15
        'Left Arm',         # 16
        'Right Arm',        # 17
        'Left Elbow',     # 18
        'Right Elbow',    # 19
        'Left Hand',        # 20
        'Right Hand',       # 21
        'Left Thumb',  # 22
        'Right Thumb', # 23
    ]


def get_smpl_neighbor_triplets():
    return [
        [ 0,  1, 2 ],  # 0
        [ 1,  4, 0 ],  # 1
        [ 2,  0, 5 ],  # 2
        [ 3,  0, 6 ],  # 3
        [ 4,  7, 1 ],  # 4
        [ 5,  2, 8 ],  # 5
        [ 6,  3, 9 ],  # 6
        [ 7, 10, 4 ],  # 7
        [ 8,  5, 11],  # 8
        [ 9, 13, 14],  # 9
        [10,  7, 4 ],  # 10
        [11,  8, 5 ],  # 11
        [12,  9, 15],  # 12
        [13, 16, 9 ],  # 13
        [14,  9, 17],  # 14
        [15,  9, 12],  # 15
        [16, 18, 13],  # 16
        [17, 14, 19],  # 17
        [18, 20, 16],  # 18
        [19, 17, 21],  # 19
        [20, 22, 18],  # 20
        [21, 19, 23],  # 21
        [22, 20, 18],  # 22
        [23, 19, 21],  # 23
    ]


def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def map_spin_joints_to_smpl():
    # this function primarily will be used to copy 2D keypoint
    # confidences to pose parameters
    return [
        [(39, 27, 28), 0],  # hip,lhip,rhip->hips
        [(28,), 1],  # lhip->leftUpLeg
        [(27,), 2],  # rhip->rightUpLeg
        [(41, 27, 28, 39), 3],  # Spine->spine
        [(29,), 4],  # lknee->leftLeg
        [(26,), 5],  # rknee->rightLeg
        [(41, 40, 33, 34,), 6],  # spine, thorax ->spine1
        [(30,), 7],  # lankle->leftFoot
        [(25,), 8],  # rankle->rightFoot
        [(40, 33, 34), 9],  # thorax,shoulders->spine2
        [(30,), 10],  # lankle -> leftToe
        [(25,), 11],  # rankle -> rightToe
        [(37, 42, 33, 34), 12],  # neck, shoulders -> neck
        [(34,), 13],  # lshoulder->leftShoulder
        [(33,), 14],  # rshoulder->rightShoulder
        [(33, 34, 38, 43, 44, 45, 46, 47, 48,), 15],  # nose, eyes, ears, headtop, shoulders->head
        [(34,), 16],  # lshoulder->leftArm
        [(33,), 17],  # rshoulder->rightArm
        [(35,), 18],  # lelbow->leftForeArm
        [(32,), 19],  # relbow->rightForeArm
        [(36,), 20],  # lwrist->leftHand
        [(31,), 21],  # rwrist->rightHand
        [(36,), 22],  # lhand -> leftHandIndex
        [(31,), 23],  # rhand -> rightHandIndex
    ]


def map_smpl_to_common():
    return [
        [(11, 8), 0], # rightToe, rightFoot -> rankle
        [(5,), 1], # rightleg -> rknee,
        [(2,), 2], # rhip
        [(1,), 3], # lhip
        [(4,), 4], # leftLeg -> lknee
        [(10, 7), 5], # lefttoe, leftfoot -> lankle
        [(21, 23), 6], # rwrist
        [(18,), 7], # relbow
        [(17, 14), 8],  # rshoulder
        [(16, 13), 9],  # lshoulder
        [(19,), 10],  # lelbow
        [(20, 22), 11],  # lwrist
        [(0, 3, 6, 9, 12), 12],  # neck
        [(15,), 13],  # headtop
    ]


def relation_among_spin_joints():
    # this function primarily will be used to copy 2D keypoint
    # confidences to 3D joints
    return [
        [(), 25],
        [(), 26],
        [(39,), 27],
        [(39,), 28],
        [(), 29],
        [(), 30],
        [(), 31],
        [(), 32],
        [(), 33],
        [(), 34],
        [(), 35],
        [(), 36],
        [(40,42,44,43,38,33,34,), 37],
        [(43,44,45,46,47,48,33,34,), 38],
        [(27,28,), 39],
        [(27,28,37,41,42,), 40],
        [(27,28,39,40,), 41],
        [(37,38,44,45,46,47,48,), 42],
        [(44,45,46,47,48,38,42,37,33,34,), 43],
        [(44,45,46,47,48,38,42,37,33,34), 44],
        [(44,45,46,47,48,38,42,37,33,34), 45],
        [(44,45,46,47,48,38,42,37,33,34), 46],
        [(44,45,46,47,48,38,42,37,33,34), 47],
        [(44,45,46,47,48,38,42,37,33,34), 48],
    ]
    

def get_mmpose_joint_names():
    # this naming is for the first 23 joints of MMPose
    # does not include hands and face
    return [
        'OP Nose', # 1
        'OP LEye', # 2
        'OP REye', # 3
        'OP LEar', # 4
        'OP REar', # 5
        'OP LShoulder', # 6
        'OP RShoulder', # 7
        'OP LElbow', # 8
        'OP RElbow', # 9
        'OP LWrist', # 10
        'OP RWrist', # 11
        'OP LHip', # 12
        'OP RHip', # 13
        'OP LKnee', # 14
        'OP RKnee', # 15
        'OP LAnkle', # 16
        'OP RAnkle', # 17
        'OP LBigToe', # 18
        'OP LSmallToe', # 19
        'OP LHeel', # 20
        'OP RBigToe', # 21
        'OP RSmallToe', # 22
        'OP RHeel', # 23
    ]


def get_wholebody_joint_names():
    kp_id2name = {
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle',
        17: 'left_big_toe',
        18: 'left_small_toe',
        19: 'left_heel',
        20: 'right_big_toe',
        21: 'right_small_toe',
        22: 'right_heel',
        23: 'face-0',
        24: 'face-1',
        25: 'face-2',
        26: 'face-3',
        27: 'face-4',
        28: 'face-5',
        29: 'face-6',
        30: 'face-7',
        31: 'face-8',
        32: 'face-9',
        33: 'face-10',
        34: 'face-11',
        35: 'face-12',
        36: 'face-13',
        37: 'face-14',
        38: 'face-15',
        39: 'face-16',
        40: 'face-17',
        41: 'face-18',
        42: 'face-19',
        43: 'face-20',
        44: 'face-21',
        45: 'face-22',
        46: 'face-23',
        47: 'face-24',
        48: 'face-25',
        49: 'face-26',
        50: 'face-27',
        51: 'face-28',
        52: 'face-29',
        53: 'face-30',
        54: 'face-31',
        55: 'face-32',
        56: 'face-33',
        57: 'face-34',
        58: 'face-35',
        59: 'face-36',
        60: 'face-37',
        61: 'face-38',
        62: 'face-39',
        63: 'face-40',
        64: 'face-41',
        65: 'face-42',
        66: 'face-43',
        67: 'face-44',
        68: 'face-45',
        69: 'face-46',
        70: 'face-47',
        71: 'face-48',
        72: 'face-49',
        73: 'face-50',
        74: 'face-51',
        75: 'face-52',
        76: 'face-53',
        77: 'face-54',
        78: 'face-55',
        79: 'face-56',
        80: 'face-57',
        81: 'face-58',
        82: 'face-59',
        83: 'face-60',
        84: 'face-61',
        85: 'face-62',
        86: 'face-63',
        87: 'face-64',
        88: 'face-65',
        89: 'face-66',
        90: 'face-67',
        91: 'left_hand_root',
        92: 'left_thumb1',
        93: 'left_thumb2',
        94: 'left_thumb3',
        95: 'left_thumb4',
        96: 'left_forefinger1',
        97: 'left_forefinger2',
        98: 'left_forefinger3',
        99: 'left_forefinger4',
        100: 'left_middle_finger1',
        101: 'left_middle_finger2',
        102: 'left_middle_finger3',
        103: 'left_middle_finger4',
        104: 'left_ring_finger1',
        105: 'left_ring_finger2',
        106: 'left_ring_finger3',
        107: 'left_ring_finger4',
        108: 'left_pinky_finger1',
        109: 'left_pinky_finger2',
        110: 'left_pinky_finger3',
        111: 'left_pinky_finger4',
        112: 'right_hand_root',
        113: 'right_thumb1',
        114: 'right_thumb2',
        115: 'right_thumb3',
        116: 'right_thumb4',
        117: 'right_forefinger1',
        118: 'right_forefinger2',
        119: 'right_forefinger3',
        120: 'right_forefinger4',
        121: 'right_middle_finger1',
        122: 'right_middle_finger2',
        123: 'right_middle_finger3',
        124: 'right_middle_finger4',
        125: 'right_ring_finger1',
        126: 'right_ring_finger2',
        127: 'right_ring_finger3',
        128: 'right_ring_finger4',
        129: 'right_pinky_finger1',
        130: 'right_pinky_finger2',
        131: 'right_pinky_finger3',
        132: 'right_pinky_finger4'
    }
    
    kp_name2id = {v: k for k, v in kp_id2name.items()}
    
    return list(kp_name2id.keys())


def get_ophandface_joint_names():
    kp_id2name = {
        0: 'nose',
        1: 'neck',
        2: 'right_shoulder',
        3: 'right_elbow',
        4: 'right_wrist',
        5: 'left_shoulder',
        6: 'left_elbow',
        7: 'left_wrist',
        8: 'mid_hip',
        9: 'right_hip',
        10: 'right_knee',
        11: 'right_ankle',
        12: 'left_hip',
        13: 'left_knee',
        14: 'left_ankle',
        15: 'right_eye',
        16: 'left_eye',
        17: 'right_ear',
        18: 'left_ear',
        19: 'left_big_toe',
        20: 'left_small_toe',
        21: 'left_heel',
        22: 'right_big_toe',
        23: 'right_small_toe',
        24: 'right_heel',
        25: 'face-0',
        26: 'face-1',
        27: 'face-2',
        28: 'face-3',
        29: 'face-4',
        30: 'face-5',
        31: 'face-6',
        32: 'face-7',
        33: 'face-8',
        34: 'face-9',
        35: 'face-10',
        36: 'face-11',
        37: 'face-12',
        38: 'face-13',
        39: 'face-14',
        40: 'face-15',
        41: 'face-16',
        42: 'face-17',
        43: 'face-18',
        44: 'face-19',
        45: 'face-20',
        46: 'face-21',
        47: 'face-22',
        48: 'face-23',
        49: 'face-24',
        50: 'face-25',
        51: 'face-26',
        52: 'face-27',
        53: 'face-28',
        54: 'face-29',
        55: 'face-30',
        56: 'face-31',
        57: 'face-32',
        58: 'face-33',
        59: 'face-34',
        60: 'face-35',
        61: 'face-36',
        62: 'face-37',
        63: 'face-38',
        64: 'face-39',
        65: 'face-40',
        66: 'face-41',
        67: 'face-42',
        68: 'face-43',
        69: 'face-44',
        70: 'face-45',
        71: 'face-46',
        72: 'face-47',
        73: 'face-48',
        74: 'face-49',
        75: 'face-50',
        76: 'face-51',
        77: 'face-52',
        78: 'face-53',
        79: 'face-54',
        80: 'face-55',
        81: 'face-56',
        82: 'face-57',
        83: 'face-58',
        84: 'face-59',
        85: 'face-60',
        86: 'face-61',
        87: 'face-62',
        88: 'face-63',
        89: 'face-64',
        90: 'face-65',
        91: 'face-66',
        92: 'face-67',
        93: 'face-68', # extra points eye pupil left
        94: 'face-69', # extra points eye pupil right
        95: 'left_hand_root',
        96: 'left_thumb1',
        97: 'left_thumb2',
        98: 'left_thumb3',
        99: 'left_thumb4',
        100: 'left_forefinger1',
        101: 'left_forefinger2',
        102: 'left_forefinger3',
        103: 'left_forefinger4',
        104: 'left_middle_finger1',
        105: 'left_middle_finger2',
        106: 'left_middle_finger3',
        107: 'left_middle_finger4',
        108: 'left_ring_finger1',
        109: 'left_ring_finger2',
        110: 'left_ring_finger3',
        111: 'left_ring_finger4',
        112: 'left_pinky_finger1',
        113: 'left_pinky_finger2',
        114: 'left_pinky_finger3',
        115: 'left_pinky_finger4',
        116: 'right_hand_root',
        117: 'right_thumb1',
        118: 'right_thumb2',
        119: 'right_thumb3',
        120: 'right_thumb4',
        121: 'right_forefinger1',
        122: 'right_forefinger2',
        123: 'right_forefinger3',
        124: 'right_forefinger4',
        125: 'right_middle_finger1',
        126: 'right_middle_finger2',
        127: 'right_middle_finger3',
        128: 'right_middle_finger4',
        129: 'right_ring_finger1',
        130: 'right_ring_finger2',
        131: 'right_ring_finger3',
        132: 'right_ring_finger4',
        133: 'right_pinky_finger1',
        134: 'right_pinky_finger2',
        135: 'right_pinky_finger3',
        136: 'right_pinky_finger4'
    }
    
    kp_name2id = {v: k for k, v in kp_id2name.items()}
    
    return list(kp_name2id.keys())
