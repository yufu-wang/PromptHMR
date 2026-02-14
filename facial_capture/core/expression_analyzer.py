"""
表情分析器 - 将面部关键点转换为 ARKit Blendshapes
"""

import numpy as np
from typing import Dict, Optional
from .arkit_blendshapes import ARKitBlendshapeData


class ExpressionAnalyzer:
    """表情分析器 - 将 MediaPipe 关键点映射到 ARKit Blendshapes"""

    def __init__(self):
        """初始化表情分析器"""
        self.baseline_ratios = None  # 中性表情的基准比例

    def analyze(self, landmarks: np.ndarray, normalized_landmarks: np.ndarray) -> ARKitBlendshapeData:
        """
        分析面部关键点并生成 ARKit Blendshapes

        Args:
            landmarks: 原始关键点 (478, 3)
            normalized_landmarks: 归一化关键点 (478, 3)

        Returns:
            ARKitBlendshapeData 对象
        """
        blendshapes = ARKitBlendshapeData()

        # 使用归一化关键点进行分析
        pts = normalized_landmarks

        # 分析各个面部区域
        self._analyze_eyes(pts, blendshapes)
        self._analyze_brows(pts, blendshapes)
        self._analyze_mouth(pts, blendshapes)
        self._analyze_cheeks(pts, blendshapes)
        self._analyze_jaw(pts, blendshapes)
        self._analyze_nose(pts, blendshapes)

        return blendshapes

    def _analyze_eyes(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析眼睛相关的 blendshapes"""

        # 左眼
        left_eye_top = pts[159, 1]  # 上眼睑
        left_eye_bottom = pts[145, 1]  # 下眼睑
        left_eye_height = abs(left_eye_top - left_eye_bottom)

        # 右眼
        right_eye_top = pts[386, 1]
        right_eye_bottom = pts[374, 1]
        right_eye_height = abs(right_eye_top - right_eye_bottom)

        # 眼睛高度的基准值（完全睁开时）
        baseline_eye_height = 0.03  # 归一化坐标系中的典型值

        # eyeBlink - 闭眼程度 (高度减小)
        left_blink = 1.0 - min(left_eye_height / baseline_eye_height, 1.0)
        right_blink = 1.0 - min(right_eye_height / baseline_eye_height, 1.0)
        blendshapes.set('eyeBlinkLeft', max(0, left_blink))
        blendshapes.set('eyeBlinkRight', max(0, right_blink))

        # eyeWide - 睁大眼睛 (高度增加)
        left_wide = max(0, (left_eye_height / baseline_eye_height - 1.0) * 2.0)
        right_wide = max(0, (right_eye_height / baseline_eye_height - 1.0) * 2.0)
        blendshapes.set('eyeWideLeft', min(1.0, left_wide))
        blendshapes.set('eyeWideRight', min(1.0, right_wide))

        # eyeSquint - 眯眼
        # 当眼睛高度适中减小但不完全闭合时
        left_squint = max(0, (0.7 - left_eye_height / baseline_eye_height) * 2.0)
        right_squint = max(0, (0.7 - right_eye_height / baseline_eye_height) * 2.0)
        if left_blink < 0.5:  # 不是完全闭眼时才算眯眼
            blendshapes.set('eyeSquintLeft', min(1.0, left_squint))
        if right_blink < 0.5:
            blendshapes.set('eyeSquintRight', min(1.0, right_squint))

        # 眼睛注视方向
        # 使用虹膜中心相对于眼睛中心的位置
        if len(pts) > 468:  # 如果有虹膜关键点
            # 左眼虹膜
            left_iris_center = pts[468]
            left_eye_center = np.mean(pts[[33, 133, 160, 159, 158, 145, 144, 153]], axis=0)

            # 右眼虹膜
            right_iris_center = pts[473]
            right_eye_center = np.mean(pts[[362, 263, 385, 386, 387, 374, 380, 373]], axis=0)

            # 计算注视方向
            left_gaze = left_iris_center - left_eye_center
            right_gaze = right_iris_center - right_eye_center

            # eyeLookUp/Down (Y 轴)
            gaze_threshold = 0.005
            if left_gaze[1] < -gaze_threshold:
                blendshapes.set('eyeLookUpLeft', min(1.0, abs(left_gaze[1]) / 0.02))
            elif left_gaze[1] > gaze_threshold:
                blendshapes.set('eyeLookDownLeft', min(1.0, left_gaze[1] / 0.02))

            if right_gaze[1] < -gaze_threshold:
                blendshapes.set('eyeLookUpRight', min(1.0, abs(right_gaze[1]) / 0.02))
            elif right_gaze[1] > gaze_threshold:
                blendshapes.set('eyeLookDownRight', min(1.0, right_gaze[1] / 0.02))

            # eyeLookIn/Out (X 轴)
            if left_gaze[0] > gaze_threshold:  # 左眼向内
                blendshapes.set('eyeLookInLeft', min(1.0, left_gaze[0] / 0.02))
            elif left_gaze[0] < -gaze_threshold:  # 左眼向外
                blendshapes.set('eyeLookOutLeft', min(1.0, abs(left_gaze[0]) / 0.02))

            if right_gaze[0] < -gaze_threshold:  # 右眼向内
                blendshapes.set('eyeLookInRight', min(1.0, abs(right_gaze[0]) / 0.02))
            elif right_gaze[0] > gaze_threshold:  # 右眼向外
                blendshapes.set('eyeLookOutRight', min(1.0, right_gaze[0] / 0.02))

    def _analyze_brows(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析眉毛相关的 blendshapes"""

        # 眉毛关键点
        left_brow_inner = pts[70]  # 左眉内侧
        left_brow_outer = pts[105]  # 左眉外侧
        right_brow_inner = pts[300]  # 右眉内侧
        right_brow_outer = pts[334]  # 右眉外侧

        # 眼睛上方参考点
        left_eye_top = pts[159]
        right_eye_top = pts[386]

        # 眉毛到眼睛的距离（中性位置的基准）
        baseline_brow_distance = 0.04

        # 左眉距离
        left_brow_distance = left_brow_inner[1] - left_eye_top[1]
        left_brow_outer_distance = left_brow_outer[1] - left_eye_top[1]

        # 右眉距离
        right_brow_distance = right_brow_inner[1] - right_eye_top[1]
        right_brow_outer_distance = right_brow_outer[1] - right_eye_top[1]

        # browDown - 眉毛下压（皱眉）
        left_down = max(0, (baseline_brow_distance - left_brow_distance) / baseline_brow_distance * 2.0)
        right_down = max(0, (baseline_brow_distance - right_brow_distance) / baseline_brow_distance * 2.0)
        blendshapes.set('browDownLeft', min(1.0, left_down))
        blendshapes.set('browDownRight', min(1.0, right_down))

        # browInnerUp - 内眉上扬（惊讶/担忧）
        inner_up = max(0, ((left_brow_distance + right_brow_distance) / 2 - baseline_brow_distance) / baseline_brow_distance * 2.0)
        blendshapes.set('browInnerUp', min(1.0, inner_up))

        # browOuterUp - 外眉上扬
        left_outer_up = max(0, (left_brow_outer_distance - baseline_brow_distance) / baseline_brow_distance * 2.0)
        right_outer_up = max(0, (right_brow_outer_distance - baseline_brow_distance) / baseline_brow_distance * 2.0)
        blendshapes.set('browOuterUpLeft', min(1.0, left_outer_up))
        blendshapes.set('browOuterUpRight', min(1.0, right_outer_up))

    def _analyze_mouth(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析嘴部相关的 blendshapes"""

        # 嘴角
        mouth_left = pts[61]
        mouth_right = pts[291]

        # 上下唇中心
        upper_lip_top = pts[13]
        lower_lip_bottom = pts[14]
        upper_lip_center = pts[0]
        lower_lip_center = pts[17]

        # 嘴巴宽度和高度
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_height = abs(lower_lip_bottom[1] - upper_lip_top[1])

        # 基准值
        baseline_mouth_width = 0.15
        baseline_mouth_height = 0.01

        # jawOpen - 张嘴
        jaw_open = max(0, (mouth_height - baseline_mouth_height) / baseline_mouth_height)
        blendshapes.set('jawOpen', min(1.0, jaw_open))

        # mouthClose - 闭嘴（抿嘴）
        mouth_close = max(0, 1.0 - mouth_height / baseline_mouth_height)
        if jaw_open < 0.1:  # 只在嘴巴不是张开状态时
            blendshapes.set('mouthClose', min(1.0, mouth_close))

        # mouthSmile - 微笑（嘴角上扬）
        # 检测嘴角相对于嘴巴中心的高度
        mouth_center_y = (upper_lip_center[1] + lower_lip_center[1]) / 2
        left_corner_lift = max(0, mouth_center_y - mouth_left[1])
        right_corner_lift = max(0, mouth_center_y - mouth_right[1])

        smile_threshold = 0.01
        left_smile = max(0, (left_corner_lift - smile_threshold) / smile_threshold * 2.0)
        right_smile = max(0, (right_corner_lift - smile_threshold) / smile_threshold * 2.0)
        blendshapes.set('mouthSmileLeft', min(1.0, left_smile))
        blendshapes.set('mouthSmileRight', min(1.0, right_smile))

        # mouthFrown - 皱眉（嘴角下垂）
        left_frown = max(0, (mouth_left[1] - mouth_center_y - smile_threshold) / smile_threshold * 2.0)
        right_frown = max(0, (mouth_right[1] - mouth_center_y - smile_threshold) / smile_threshold * 2.0)
        blendshapes.set('mouthFrownLeft', min(1.0, left_frown))
        blendshapes.set('mouthFrownRight', min(1.0, right_frown))

        # mouthPucker - 撅嘴（嘴巴前伸）
        # 检测嘴部的 Z 深度
        mouth_depth = (mouth_left[2] + mouth_right[2]) / 2
        pucker = max(0, mouth_depth * 10.0)  # Z 坐标越大越前伸
        blendshapes.set('mouthPucker', min(1.0, pucker))

        # mouthFunnel - 嘟嘴（嘴巴呈 O 形）
        # 当嘴巴高度增加但宽度减小时
        width_ratio = mouth_width / baseline_mouth_width
        height_ratio = mouth_height / baseline_mouth_height
        funnel = max(0, height_ratio - width_ratio)
        blendshapes.set('mouthFunnel', min(1.0, funnel * 0.5))

        # mouthStretch - 嘴巴横向拉伸
        stretch = max(0, (mouth_width - baseline_mouth_width) / baseline_mouth_width)
        left_stretch = stretch * (1.0 if mouth_left[0] < 0.5 else 0.5)
        right_stretch = stretch * (1.0 if mouth_right[0] > 0.5 else 0.5)
        blendshapes.set('mouthStretchLeft', min(1.0, left_stretch))
        blendshapes.set('mouthStretchRight', min(1.0, right_stretch))

        # mouthLeft/Right - 嘴巴左右移动
        face_center_x = 0.5
        mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
        mouth_shift = mouth_center_x - face_center_x

        shift_threshold = 0.02
        if mouth_shift < -shift_threshold:
            blendshapes.set('mouthLeft', min(1.0, abs(mouth_shift) / shift_threshold))
        elif mouth_shift > shift_threshold:
            blendshapes.set('mouthRight', min(1.0, mouth_shift / shift_threshold))

        # 上下唇运动
        upper_lip_lift = max(0, (upper_lip_center[1] - upper_lip_top[1]) * 20.0)
        lower_lip_down = max(0, (lower_lip_bottom[1] - lower_lip_center[1]) * 20.0)

        blendshapes.set('mouthUpperUpLeft', min(1.0, upper_lip_lift))
        blendshapes.set('mouthUpperUpRight', min(1.0, upper_lip_lift))
        blendshapes.set('mouthLowerDownLeft', min(1.0, lower_lip_down))
        blendshapes.set('mouthLowerDownRight', min(1.0, lower_lip_down))

    def _analyze_cheeks(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析脸颊相关的 blendshapes"""

        # 脸颊关键点
        left_cheek = pts[205]
        right_cheek = pts[425]

        # 脸部中心线
        nose_tip = pts[1]

        # cheekPuff - 鼓腮（脸颊向外凸出）
        # 使用 Z 深度检测
        left_puff = max(0, left_cheek[2] * 10.0)
        right_puff = max(0, right_cheek[2] * 10.0)
        avg_puff = (left_puff + right_puff) / 2
        blendshapes.set('cheekPuff', min(1.0, avg_puff))

        # cheekSquint - 脸颊上提（微笑时）
        # 检测脸颊高度相对于基准位置
        baseline_cheek_y = 0.6  # 归一化坐标
        left_squint = max(0, (baseline_cheek_y - left_cheek[1]) / 0.05)
        right_squint = max(0, (baseline_cheek_y - right_cheek[1]) / 0.05)
        blendshapes.set('cheekSquintLeft', min(1.0, left_squint))
        blendshapes.set('cheekSquintRight', min(1.0, right_squint))

    def _analyze_jaw(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析下颌相关的 blendshapes"""

        # 下巴和下颌点
        chin = pts[152]
        jaw_left = pts[234]
        jaw_right = pts[454]

        # 面部中心和宽度
        face_center_x = 0.5
        face_width = 0.3  # 基准宽度

        # jawForward - 下巴前伸
        jaw_forward = max(0, chin[2] * 10.0)  # Z 深度
        blendshapes.set('jawForward', min(1.0, jaw_forward))

        # jawLeft/Right - 下巴左右移动
        chin_shift = chin[0] - face_center_x
        shift_threshold = 0.02

        if chin_shift < -shift_threshold:
            blendshapes.set('jawLeft', min(1.0, abs(chin_shift) / shift_threshold))
        elif chin_shift > shift_threshold:
            blendshapes.set('jawRight', min(1.0, chin_shift / shift_threshold))

    def _analyze_nose(self, pts: np.ndarray, blendshapes: ARKitBlendshapeData):
        """分析鼻子相关的 blendshapes"""

        # 鼻孔关键点
        nose_left = pts[98]
        nose_right = pts[327]
        nose_tip = pts[1]

        # noseSneer - 鼻子皱起（鼻孔上提）
        # 检测鼻孔相对于鼻尖的高度
        baseline_nostril_y = 0.02  # 相对距离

        left_nostril_lift = max(0, (nose_tip[1] - nose_left[1] - baseline_nostril_y) / baseline_nostril_y)
        right_nostril_lift = max(0, (nose_tip[1] - nose_right[1] - baseline_nostril_y) / baseline_nostril_y)

        blendshapes.set('noseSneerLeft', min(1.0, left_nostril_lift))
        blendshapes.set('noseSneerRight', min(1.0, right_nostril_lift))

    def calibrate_neutral(self, landmarks: np.ndarray, normalized_landmarks: np.ndarray):
        """
        使用中性表情进行校准

        Args:
            landmarks: 中性表情的关键点
            normalized_landmarks: 归一化的中性表情关键点
        """
        # 存储中性表情的基准比例
        # 这可以用于个性化调整
        self.baseline_ratios = {
            'eye_height': self._calculate_eye_height(normalized_landmarks),
            'mouth_width': self._calculate_mouth_width(normalized_landmarks),
            'brow_distance': self._calculate_brow_distance(normalized_landmarks)
        }

    def _calculate_eye_height(self, pts: np.ndarray) -> float:
        """计算眼睛高度"""
        left_height = abs(pts[159, 1] - pts[145, 1])
        right_height = abs(pts[386, 1] - pts[374, 1])
        return (left_height + right_height) / 2

    def _calculate_mouth_width(self, pts: np.ndarray) -> float:
        """计算嘴巴宽度"""
        return np.linalg.norm(pts[291] - pts[61])

    def _calculate_brow_distance(self, pts: np.ndarray) -> float:
        """计算眉毛到眼睛的距离"""
        left_dist = pts[70, 1] - pts[159, 1]
        right_dist = pts[300, 1] - pts[386, 1]
        return (left_dist + right_dist) / 2
