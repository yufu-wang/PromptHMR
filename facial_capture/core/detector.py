"""
面部检测和关键点提取模块
支持真人和动漫风格的面部
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import mediapipe as mp


class FacialLandmarkDetector:
    """面部关键点检测器"""

    def __init__(self, use_anime_detector: bool = False):
        """
        初始化检测器

        Args:
            use_anime_detector: 是否使用动漫面部检测器
        """
        self.use_anime_detector = use_anime_detector

        # MediaPipe Face Mesh (478 个关键点)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # 包含虹膜关键点
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 关键点索引定义（MediaPipe 478 点模型）
        self._define_landmark_indices()

    def _define_landmark_indices(self):
        """定义重要的面部关键点索引"""
        # 这些索引对应 MediaPipe Face Mesh 的 478 个关键点
        self.landmark_indices = {
            # 眼睛轮廓
            'left_eye': [33, 160, 158, 133, 153, 144, 145, 159],
            'right_eye': [362, 385, 387, 263, 373, 380, 374, 386],

            # 眼睛中心
            'left_eye_center': [468],  # 虹膜中心
            'right_eye_center': [473],

            # 眼睑
            'left_upper_eyelid': [159, 145, 144],
            'left_lower_eyelid': [33, 160, 158],
            'right_upper_eyelid': [386, 374, 380],
            'right_lower_eyelid': [362, 385, 387],

            # 眉毛
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],

            # 嘴部
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191],
            'upper_lip': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            'lower_lip': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],

            # 嘴角
            'mouth_left_corner': [61],
            'mouth_right_corner': [291],

            # 脸颊
            'left_cheek': [205, 50, 117],
            'right_cheek': [425, 280, 346],

            # 下巴和下颌
            'chin': [152],
            'jaw_left': [234, 227, 137],
            'jaw_right': [454, 447, 366],

            # 鼻子
            'nose_tip': [1],
            'nose_bridge': [168, 6, 197],
            'nose_left': [98],
            'nose_right': [327],

            # 面部轮廓
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                          288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                          150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }

    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测面部关键点

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            包含关键点和元数据的字典，如果检测失败则返回 None
        """
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 检测关键点
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        # 获取第一个面部的关键点
        face_landmarks = results.multi_face_landmarks[0]

        # 转换为 numpy 数组 (478, 3) - (x, y, z)
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ])

        # 归一化关键点 (相对于面部边界框)
        bbox = self._get_face_bbox(landmarks)
        normalized_landmarks = self._normalize_landmarks(landmarks, bbox)

        return {
            'landmarks': landmarks,  # 原始关键点 (像素坐标)
            'normalized_landmarks': normalized_landmarks,  # 归一化关键点
            'bbox': bbox,  # 边界框 [x, y, w, h]
            'image_shape': (h, w),
            'num_landmarks': len(landmarks)
        }

    def _get_face_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """计算面部边界框"""
        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])

        width = x_max - x_min
        height = y_max - y_min

        # 添加 10% 边距
        margin = 0.1
        x_min -= width * margin
        y_min -= height * margin
        width *= (1 + 2 * margin)
        height *= (1 + 2 * margin)

        return (int(x_min), int(y_min), int(width), int(height))

    def _normalize_landmarks(self, landmarks: np.ndarray,
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """归一化关键点到 [0, 1] 范围"""
        x, y, w, h = bbox
        normalized = landmarks.copy()
        normalized[:, 0] = (landmarks[:, 0] - x) / w
        normalized[:, 1] = (landmarks[:, 1] - y) / h
        normalized[:, 2] = landmarks[:, 2] / w  # z 坐标也归一化

        return normalized

    def get_landmark_group(self, landmarks: np.ndarray, group_name: str) -> np.ndarray:
        """
        获取一组关键点

        Args:
            landmarks: 关键点数组 (478, 3)
            group_name: 关键点组名称

        Returns:
            指定组的关键点
        """
        if group_name not in self.landmark_indices:
            raise ValueError(f"Unknown landmark group: {group_name}")

        indices = self.landmark_indices[group_name]
        return landmarks[indices]

    def visualize(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            detection_result: detect() 返回的结果

        Returns:
            带有关键点标记的图像
        """
        vis_image = image.copy()
        landmarks = detection_result['landmarks']
        bbox = detection_result['bbox']

        # 绘制边界框
        x, y, w, h = bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制关键点
        for idx, (lx, ly, lz) in enumerate(landmarks):
            # 根据深度改变颜色
            depth_color = int(255 * (lz + 0.5))  # 归一化深度
            color = (depth_color, 255 - depth_color, 128)
            cv2.circle(vis_image, (int(lx), int(ly)), 1, color, -1)

        # 绘制重要特征
        self._draw_facial_features(vis_image, landmarks)

        return vis_image

    def _draw_facial_features(self, image: np.ndarray, landmarks: np.ndarray):
        """绘制重要的面部特征"""
        # 眼睛
        for eye_name in ['left_eye', 'right_eye']:
            pts = self.get_landmark_group(landmarks, eye_name)
            pts = pts[:, :2].astype(np.int32)
            cv2.polylines(image, [pts], True, (255, 0, 0), 1)

        # 嘴部
        mouth_pts = self.get_landmark_group(landmarks, 'mouth_outer')
        mouth_pts = mouth_pts[:, :2].astype(np.int32)
        cv2.polylines(image, [mouth_pts], True, (0, 255, 255), 1)

        # 眉毛
        for brow_name in ['left_eyebrow', 'right_eyebrow']:
            pts = self.get_landmark_group(landmarks, brow_name)
            pts = pts[:, :2].astype(np.int32)
            cv2.polylines(image, [pts], False, (255, 255, 0), 1)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
