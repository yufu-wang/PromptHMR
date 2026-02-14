"""
扩展的 Blendshape 系统
在 ARKit 52 个标准 blendshapes 基础上，添加舌头和瞳孔精确控制
"""

from typing import Dict
import numpy as np
from .arkit_blendshapes import ARKitBlendshapeData, ARKIT_BLENDSHAPES


# 扩展的 blendshape 参数（舌头和瞳孔）
EXTENDED_BLENDSHAPES = [
    # 舌头控制 (6个)
    'tongueOutExtension',      # 舌头伸出程度 (0-1)
    'tongueUpDown',            # 舌头上下位置 (-1到1, 0为中性)
    'tongueLeftRight',         # 舌头左右位置 (-1到1)
    'tongueCurl',              # 舌头卷曲
    'tongueWidth',             # 舌头宽度
    'tongueThickness',         # 舌头厚度

    # 左眼瞳孔精确控制 (4个)
    'leftPupilPositionX',      # 左瞳孔 X 位置 (-1到1)
    'leftPupilPositionY',      # 左瞳孔 Y 位置 (-1到1)
    'leftPupilDilation',       # 左瞳孔扩张
    'leftIrisRotation',        # 左虹膜旋转

    # 右眼瞳孔精确控制 (4个)
    'rightPupilPositionX',     # 右瞳孔 X 位置
    'rightPupilPositionY',     # 右瞳孔 Y 位置
    'rightPupilDilation',      # 右瞳孔扩张
    'rightIrisRotation',       # 右虹膜旋转

    # 阿黑颜特效 (4个)
    'ahegaoIntensity',         # 阿黑颜强度
    'eyeRollIntensity',        # 眼睛翻白强度
    'droolEffect',             # 流口水效果
    'heartPupils',             # 心形瞳孔效果
]

# 所有 blendshape（ARKit + 扩展）
ALL_BLENDSHAPES = ARKIT_BLENDSHAPES + EXTENDED_BLENDSHAPES


class ExtendedBlendshapeData(ARKitBlendshapeData):
    """
    扩展的 Blendshape 数据类
    包含 ARKit 52 个 + 扩展 18 个 = 70 个参数
    """

    def __init__(self):
        """初始化"""
        super().__init__()

        # 添加扩展参数
        self.extended_values = {name: 0.0 for name in EXTENDED_BLENDSHAPES}

    def set_extended(self, name: str, value: float):
        """设置扩展 blendshape 值"""
        if name in EXTENDED_BLENDSHAPES:
            # 舌头位置和瞳孔位置可以是负值
            if name in ['tongueUpDown', 'tongueLeftRight',
                       'leftPupilPositionX', 'leftPupilPositionY',
                       'rightPupilPositionX', 'rightPupilPositionY']:
                self.extended_values[name] = max(-1.0, min(1.0, value))
            else:
                self.extended_values[name] = max(0.0, min(1.0, value))
        else:
            raise ValueError(f"Invalid extended blendshape name: {name}")

    def get_extended(self, name: str) -> float:
        """获取扩展 blendshape 值"""
        return self.extended_values.get(name, 0.0)

    def to_dict_extended(self) -> Dict:
        """转换为完整字典（ARKit + 扩展）"""
        full_dict = self.to_dict()
        full_dict.update(self.extended_values)
        return full_dict

    def to_array_extended(self) -> list:
        """转换为完整数组（70个值）"""
        arkit_array = self.to_array()
        extended_array = [self.extended_values[name] for name in EXTENDED_BLENDSHAPES]
        return arkit_array + extended_array

    def from_dict_extended(self, data: Dict):
        """从完整字典加载"""
        # 加载 ARKit 参数
        arkit_data = {k: v for k, v in data.items() if k in ARKIT_BLENDSHAPES}
        self.from_dict(arkit_data)

        # 加载扩展参数
        for name in EXTENDED_BLENDSHAPES:
            if name in data:
                self.set_extended(name, data[name])

    def set_tongue_position(self, extension: float, x: float = 0.0, y: float = 0.0):
        """
        设置舌头位置

        Args:
            extension: 伸出程度 (0-1)
            x: 左右位置 (-1到1)
            y: 上下位置 (-1到1)
        """
        self.set_extended('tongueOutExtension', extension)
        self.set_extended('tongueLeftRight', x)
        self.set_extended('tongueUpDown', y)

    def set_pupil_position(self, eye: str, x: float, y: float):
        """
        设置瞳孔位置

        Args:
            eye: 'left' 或 'right'
            x: 水平位置 (-1到1, -1=向内, 1=向外)
            y: 垂直位置 (-1到1, -1=向上, 1=向下)
        """
        if eye == 'left':
            self.set_extended('leftPupilPositionX', x)
            self.set_extended('leftPupilPositionY', y)
        elif eye == 'right':
            self.set_extended('rightPupilPositionX', x)
            self.set_extended('rightPupilPositionY', y)

    def set_ahegao_expression(self, intensity: float = 1.0):
        """
        设置阿黑颜表情

        Args:
            intensity: 强度 (0-1)
        """
        self.set_extended('ahegaoIntensity', intensity)

        # 设置典型的阿黑颜特征
        # 舌头伸出
        self.set_tongue_position(
            extension=0.8 * intensity,
            x=0.0,
            y=0.3 * intensity  # 稍微向下
        )

        # 眼睛翻白（瞳孔向上）
        self.set_pupil_position('left', x=-0.2, y=-0.8 * intensity)
        self.set_pupil_position('right', x=0.2, y=-0.8 * intensity)
        self.set_extended('eyeRollIntensity', intensity)

        # 嘴巴张开
        self.set('jawOpen', 0.8 * intensity)

        # 流口水效果
        self.set_extended('droolEffect', 0.6 * intensity)

        # 脸颊红晕（通过 cheekSquint）
        self.set('cheekSquintLeft', 0.5 * intensity)
        self.set('cheekSquintRight', 0.5 * intensity)

    def get_tongue_info(self) -> Dict:
        """获取舌头信息"""
        return {
            'extension': self.get_extended('tongueOutExtension'),
            'x': self.get_extended('tongueLeftRight'),
            'y': self.get_extended('tongueUpDown'),
            'curl': self.get_extended('tongueCurl'),
            'width': self.get_extended('tongueWidth'),
            'thickness': self.get_extended('tongueThickness'),
        }

    def get_pupil_info(self, eye: str) -> Dict:
        """获取瞳孔信息"""
        if eye == 'left':
            return {
                'x': self.get_extended('leftPupilPositionX'),
                'y': self.get_extended('leftPupilPositionY'),
                'dilation': self.get_extended('leftPupilDilation'),
                'rotation': self.get_extended('leftIrisRotation'),
            }
        elif eye == 'right':
            return {
                'x': self.get_extended('rightPupilPositionX'),
                'y': self.get_extended('rightPupilPositionY'),
                'dilation': self.get_extended('rightPupilDilation'),
                'rotation': self.get_extended('rightIrisRotation'),
            }
        return {}

    def __repr__(self):
        arkit_count = sum(1 for v in self.values.values() if v > 0.01)
        extended_count = sum(1 for v in self.extended_values.values() if abs(v) > 0.01)
        return f"ExtendedBlendshapeData(ARKit: {arkit_count}, Extended: {extended_count})"


def convert_detection_to_extended_blendshapes(detection_result: Dict) -> ExtendedBlendshapeData:
    """
    从增强检测结果转换为扩展 blendshapes

    Args:
        detection_result: EnhancedFacialDetector.detect_with_details() 的结果

    Returns:
        ExtendedBlendshapeData 对象
    """
    blendshapes = ExtendedBlendshapeData()

    # 处理舌头
    tongue_info = detection_result.get('tongue', {})
    if tongue_info.get('detected'):
        extension = tongue_info.get('extension', 0.0)
        direction = tongue_info.get('direction', [0.0, 0.0])

        blendshapes.set_tongue_position(
            extension=extension,
            x=direction[0] if len(direction) > 0 else 0.0,
            y=direction[1] if len(direction) > 1 else 0.0
        )

        # 设置标准的 tongueOut blendshape
        blendshapes.set('tongueOut', extension)

    # 处理瞳孔
    pupils_info = detection_result.get('pupils', {})

    # 左眼
    left_eye = pupils_info.get('left', {})
    if left_eye.get('position_in_eye'):
        pos = left_eye['position_in_eye']
        blendshapes.set_pupil_position('left', pos['x'], pos['y'])

        # 同时更新 ARKit 的眼球注视 blendshapes
        if pos['y'] < -0.3:
            blendshapes.set('eyeLookUpLeft', abs(pos['y']) * 0.5)
        elif pos['y'] > 0.3:
            blendshapes.set('eyeLookDownLeft', pos['y'] * 0.5)

        if pos['x'] < -0.3:
            blendshapes.set('eyeLookInLeft', abs(pos['x']) * 0.5)
        elif pos['x'] > 0.3:
            blendshapes.set('eyeLookOutLeft', pos['x'] * 0.5)

    # 右眼
    right_eye = pupils_info.get('right', {})
    if right_eye.get('position_in_eye'):
        pos = right_eye['position_in_eye']
        blendshapes.set_pupil_position('right', pos['x'], pos['y'])

        # 同时更新 ARKit 的眼球注视 blendshapes
        if pos['y'] < -0.3:
            blendshapes.set('eyeLookUpRight', abs(pos['y']) * 0.5)
        elif pos['y'] > 0.3:
            blendshapes.set('eyeLookDownRight', pos['y'] * 0.5)

        if pos['x'] > 0.3:  # 注意右眼方向相反
            blendshapes.set('eyeLookInRight', pos['x'] * 0.5)
        elif pos['x'] < -0.3:
            blendshapes.set('eyeLookOutRight', abs(pos['x']) * 0.5)

    return blendshapes
