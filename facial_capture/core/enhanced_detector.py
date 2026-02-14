"""
增强的面部检测器
专注于精确检测舌头位置和瞳孔位置
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import mediapipe as mp
from .detector import FacialLandmarkDetector


class EnhancedFacialDetector(FacialLandmarkDetector):
    """
    增强的面部检测器
    添加了舌头和瞳孔的精确检测
    """

    def __init__(self, use_anime_detector: bool = False):
        super().__init__(use_anime_detector)

        # 定义舌头关键点（MediaPipe Face Mesh 中的舌头点）
        # MediaPipe 在 refine_landmarks=True 时包含舌头关键点
        self.tongue_indices = {
            'tongue_tip': 10,      # 舌尖
            'tongue_center': 13,   # 舌头中心
            'tongue_left': 12,     # 舌头左侧
            'tongue_right': 14,    # 舌头右侧
        }

        # 虹膜关键点（MediaPipe 虹膜模型）
        self.iris_indices = {
            # 左眼虹膜（5个点：中心 + 4个边缘）
            'left_iris_center': 468,
            'left_iris_top': 469,
            'left_iris_bottom': 470,
            'left_iris_left': 471,
            'left_iris_right': 472,

            # 右眼虹膜
            'right_iris_center': 473,
            'right_iris_top': 474,
            'right_iris_bottom': 475,
            'right_iris_left': 476,
            'right_iris_right': 477,
        }

        # 眼白区域（用于计算瞳孔相对位置）
        self.eye_white_indices = {
            'left_eye_corners': [33, 133],  # 左眼内外角
            'right_eye_corners': [362, 263],  # 右眼内外角
        }

    def detect_with_details(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测面部并返回详细信息（包括舌头和瞳孔）

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            包含舌头和瞳孔详细信息的字典
        """
        # 基础检测
        base_detection = self.detect(image)
        if base_detection is None:
            return None

        landmarks = base_detection['landmarks']
        normalized_landmarks = base_detection['normalized_landmarks']

        # 分析舌头
        tongue_info = self._analyze_tongue(landmarks, normalized_landmarks)

        # 分析瞳孔
        pupil_info = self._analyze_pupils(landmarks, normalized_landmarks)

        # 合并结果
        result = base_detection.copy()
        result['tongue'] = tongue_info
        result['pupils'] = pupil_info

        return result

    def _analyze_tongue(self, landmarks: np.ndarray,
                       normalized_landmarks: np.ndarray) -> Dict:
        """
        分析舌头位置和姿态

        Returns:
            舌头信息字典
        """
        tongue_info = {
            'detected': False,
            'position': None,
            'extension': 0.0,  # 伸出程度 (0-1)
            'direction': None,  # 方向 (x, y)
            'visibility': 0.0,  # 可见度
        }

        # 检查舌头关键点是否可用
        if len(landmarks) <= max(self.tongue_indices.values()):
            return tongue_info

        # 获取舌头关键点
        tongue_tip = normalized_landmarks[10]
        tongue_center = normalized_landmarks[13]

        # 获取嘴部参考点
        upper_lip = normalized_landmarks[13]
        lower_lip = normalized_landmarks[14]
        mouth_center = (upper_lip + lower_lip) / 2

        # 计算舌头伸出程度
        # 检测舌头是否超出嘴部边界
        tongue_y = tongue_tip[1]
        lower_lip_y = lower_lip[1]
        upper_lip_y = upper_lip[1]

        mouth_height = abs(lower_lip_y - upper_lip_y)

        # 舌头相对于嘴部的位置
        if tongue_y > lower_lip_y:
            # 舌头伸出嘴外（阿黑颜典型特征）
            extension = (tongue_y - lower_lip_y) / mouth_height
            tongue_info['detected'] = True
            tongue_info['extension'] = min(1.0, extension * 0.5)  # 归一化
            tongue_info['visibility'] = 1.0
        elif tongue_y > upper_lip_y and tongue_y < lower_lip_y:
            # 舌头在嘴内但可见
            extension = (tongue_y - upper_lip_y) / mouth_height
            tongue_info['detected'] = True
            tongue_info['extension'] = extension * 0.3
            tongue_info['visibility'] = 0.5

        # 计算舌头方向
        if tongue_info['detected']:
            direction = tongue_tip - mouth_center
            # 归一化方向向量
            norm = np.linalg.norm(direction[:2])
            if norm > 0:
                tongue_info['direction'] = (direction[:2] / norm).tolist()

            # 舌头位置（归一化坐标）
            tongue_info['position'] = {
                'x': float(tongue_tip[0]),
                'y': float(tongue_tip[1]),
                'z': float(tongue_tip[2])
            }

        return tongue_info

    def _analyze_pupils(self, landmarks: np.ndarray,
                       normalized_landmarks: np.ndarray) -> Dict:
        """
        精确分析瞳孔位置

        Returns:
            瞳孔信息字典
        """
        pupil_info = {
            'left': {
                'center': None,
                'position_in_eye': None,  # 在眼睛中的相对位置 (-1到1)
                'gaze_direction': None,   # 注视方向 (x, y)
                'iris_size': 0.0,
            },
            'right': {
                'center': None,
                'position_in_eye': None,
                'gaze_direction': None,
                'iris_size': 0.0,
            }
        }

        # 检查虹膜关键点是否可用
        if len(landmarks) < 478:
            return pupil_info

        # 左眼分析
        left_iris = self._analyze_single_pupil(
            normalized_landmarks,
            self.iris_indices['left_iris_center'],
            [self.iris_indices['left_iris_top'],
             self.iris_indices['left_iris_bottom'],
             self.iris_indices['left_iris_left'],
             self.iris_indices['left_iris_right']],
            self.landmark_indices['left_eye']
        )
        pupil_info['left'] = left_iris

        # 右眼分析
        right_iris = self._analyze_single_pupil(
            normalized_landmarks,
            self.iris_indices['right_iris_center'],
            [self.iris_indices['right_iris_top'],
             self.iris_indices['right_iris_bottom'],
             self.iris_indices['right_iris_left'],
             self.iris_indices['right_iris_right']],
            self.landmark_indices['right_eye']
        )
        pupil_info['right'] = right_iris

        return pupil_info

    def _analyze_single_pupil(self, landmarks: np.ndarray,
                             center_idx: int, edge_indices: list,
                             eye_outline_indices: list) -> Dict:
        """
        分析单个瞳孔
        """
        result = {
            'center': None,
            'position_in_eye': None,
            'gaze_direction': None,
            'iris_size': 0.0,
        }

        # 虹膜中心
        iris_center = landmarks[center_idx]
        result['center'] = {
            'x': float(iris_center[0]),
            'y': float(iris_center[1]),
            'z': float(iris_center[2])
        }

        # 计算虹膜大小（通过边缘点）
        edge_points = landmarks[edge_indices]
        iris_width = np.linalg.norm(edge_points[2] - edge_points[3])  # 左右宽度
        iris_height = np.linalg.norm(edge_points[0] - edge_points[1])  # 上下高度
        result['iris_size'] = float((iris_width + iris_height) / 2)

        # 眼睛轮廓中心
        eye_outline = landmarks[eye_outline_indices]
        eye_center = np.mean(eye_outline, axis=0)

        # 计算瞳孔在眼睛中的相对位置
        # 获取眼睛的边界
        eye_left = np.min(eye_outline[:, 0])
        eye_right = np.max(eye_outline[:, 0])
        eye_top = np.min(eye_outline[:, 1])
        eye_bottom = np.max(eye_outline[:, 1])

        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top

        # 归一化位置 (-1 到 1)
        if eye_width > 0 and eye_height > 0:
            pos_x = (iris_center[0] - eye_center[0]) / (eye_width / 2)
            pos_y = (iris_center[1] - eye_center[1]) / (eye_height / 2)

            result['position_in_eye'] = {
                'x': float(np.clip(pos_x, -1, 1)),
                'y': float(np.clip(pos_y, -1, 1))
            }

            # 注视方向（基于瞳孔偏移）
            gaze_vector = iris_center - eye_center
            norm = np.linalg.norm(gaze_vector[:2])
            if norm > 0:
                result['gaze_direction'] = {
                    'x': float(gaze_vector[0] / norm),
                    'y': float(gaze_vector[1] / norm)
                }

        return result

    def visualize_enhanced(self, image: np.ndarray,
                          detection_result: Dict) -> np.ndarray:
        """
        增强的可视化（包括舌头和瞳孔标记）
        """
        # 基础可视化
        vis_image = self.visualize(image, detection_result)

        landmarks = detection_result['landmarks']

        # 可视化舌头
        if detection_result.get('tongue', {}).get('detected'):
            tongue_info = detection_result['tongue']
            if tongue_info['position']:
                # 舌头位置
                tx = int(tongue_info['position']['x'] * image.shape[1])
                ty = int(tongue_info['position']['y'] * image.shape[0])

                # 绘制舌头标记（红色圆圈）
                cv2.circle(vis_image, (tx, ty), 8, (0, 0, 255), -1)
                cv2.circle(vis_image, (tx, ty), 10, (0, 255, 255), 2)

                # 显示伸出程度
                ext_text = f"Tongue: {tongue_info['extension']:.2f}"
                cv2.putText(vis_image, ext_text, (tx + 15, ty),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 可视化瞳孔
        if 'pupils' in detection_result:
            pupils = detection_result['pupils']

            for eye_name, eye_data in [('left', pupils['left']),
                                       ('right', pupils['right'])]:
                if eye_data['center']:
                    # 瞳孔中心
                    cx = int(eye_data['center']['x'] * image.shape[1])
                    cy = int(eye_data['center']['y'] * image.shape[0])

                    # 绘制瞳孔（蓝色圆圈）
                    cv2.circle(vis_image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.circle(vis_image, (cx, cy), 7, (255, 255, 0), 2)

                    # 绘制注视方向
                    if eye_data['gaze_direction']:
                        gaze = eye_data['gaze_direction']
                        # 箭头指示注视方向
                        end_x = int(cx + gaze['x'] * 30)
                        end_y = int(cy + gaze['y'] * 30)
                        cv2.arrowedLine(vis_image, (cx, cy), (end_x, end_y),
                                      (0, 255, 0), 2, tipLength=0.3)

                    # 显示位置信息
                    if eye_data['position_in_eye']:
                        pos = eye_data['position_in_eye']
                        pos_text = f"({pos['x']:.2f}, {pos['y']:.2f})"
                        offset_y = -15 if eye_name == 'left' else 25
                        cv2.putText(vis_image, pos_text, (cx - 30, cy + offset_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return vis_image

    def get_ahegao_features(self, detection_result: Dict) -> Dict:
        """
        提取阿黑颜（Ahegao）表情的关键特征

        Returns:
            阿黑颜特征字典
        """
        features = {
            'is_ahegao': False,
            'confidence': 0.0,
            'tongue_out': 0.0,      # 舌头伸出程度
            'eyes_rolled': 0.0,      # 眼睛翻白程度
            'mouth_open': 0.0,       # 嘴巴张开程度
            'drool': 0.0,            # 流口水效果（基于嘴角）
        }

        if not detection_result:
            return features

        # 检查舌头伸出
        tongue = detection_result.get('tongue', {})
        if tongue.get('detected'):
            features['tongue_out'] = tongue.get('extension', 0.0)

        # 检查眼睛翻白（瞳孔向上或斜上）
        pupils = detection_result.get('pupils', {})

        left_eye = pupils.get('left', {})
        right_eye = pupils.get('right', {})

        # 计算眼睛翻转程度
        eye_roll_score = 0.0

        if left_eye.get('position_in_eye'):
            left_pos_y = left_eye['position_in_eye']['y']
            # 负值表示瞳孔向上
            if left_pos_y < -0.3:
                eye_roll_score += abs(left_pos_y + 0.3)

        if right_eye.get('position_in_eye'):
            right_pos_y = right_eye['position_in_eye']['y']
            if right_pos_y < -0.3:
                eye_roll_score += abs(right_pos_y + 0.3)

        features['eyes_rolled'] = min(1.0, eye_roll_score)

        # 检查嘴巴张开（从基础检测结果获取）
        landmarks = detection_result.get('normalized_landmarks')
        if landmarks is not None:
            # 上下唇距离
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_height = abs(lower_lip[1] - upper_lip[1])
            features['mouth_open'] = min(1.0, mouth_height / 0.05)

        # 综合判断是否为阿黑颜表情
        ahegao_score = (
            features['tongue_out'] * 0.4 +
            features['eyes_rolled'] * 0.3 +
            features['mouth_open'] * 0.3
        )

        features['confidence'] = ahegao_score
        features['is_ahegao'] = ahegao_score > 0.5

        return features
