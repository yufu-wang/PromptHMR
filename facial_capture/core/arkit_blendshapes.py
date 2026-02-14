"""
ARKit Blendshape 定义
52 个标准 ARKit 面部表情单元
"""

# ARKit 标准 52 个 blendshape 名称
ARKIT_BLENDSHAPES = [
    # 眼部 (16个)
    'eyeBlinkLeft',
    'eyeBlinkRight',
    'eyeLookDownLeft',
    'eyeLookDownRight',
    'eyeLookInLeft',
    'eyeLookInRight',
    'eyeLookOutLeft',
    'eyeLookOutRight',
    'eyeLookUpLeft',
    'eyeLookUpRight',
    'eyeSquintLeft',
    'eyeSquintRight',
    'eyeWideLeft',
    'eyeWideRight',

    # 眉毛 (8个)
    'browDownLeft',
    'browDownRight',
    'browInnerUp',
    'browOuterUpLeft',
    'browOuterUpRight',

    # 嘴部 (28个)
    'mouthClose',
    'mouthFunnel',
    'mouthPucker',
    'mouthLeft',
    'mouthRight',
    'mouthSmileLeft',
    'mouthSmileRight',
    'mouthFrownLeft',
    'mouthFrownRight',
    'mouthDimpleLeft',
    'mouthDimpleRight',
    'mouthStretchLeft',
    'mouthStretchRight',
    'mouthRollLower',
    'mouthRollUpper',
    'mouthShrugLower',
    'mouthShrugUpper',
    'mouthPressLeft',
    'mouthPressRight',
    'mouthLowerDownLeft',
    'mouthLowerDownRight',
    'mouthUpperUpLeft',
    'mouthUpperUpRight',

    # 脸颊和下巴 (6个)
    'cheekPuff',
    'cheekSquintLeft',
    'cheekSquintRight',
    'jawOpen',
    'jawForward',
    'jawLeft',
    'jawRight',

    # 鼻子 (2个)
    'noseSneerLeft',
    'noseSneerRight',

    # 舌头 (1个)
    'tongueOut'
]

# Blendshape 分组
BLENDSHAPE_GROUPS = {
    'eyes': [
        'eyeBlinkLeft', 'eyeBlinkRight',
        'eyeLookDownLeft', 'eyeLookDownRight',
        'eyeLookInLeft', 'eyeLookInRight',
        'eyeLookOutLeft', 'eyeLookOutRight',
        'eyeLookUpLeft', 'eyeLookUpRight',
        'eyeSquintLeft', 'eyeSquintRight',
        'eyeWideLeft', 'eyeWideRight'
    ],
    'brows': [
        'browDownLeft', 'browDownRight',
        'browInnerUp',
        'browOuterUpLeft', 'browOuterUpRight'
    ],
    'mouth': [
        'mouthClose', 'mouthFunnel', 'mouthPucker',
        'mouthLeft', 'mouthRight',
        'mouthSmileLeft', 'mouthSmileRight',
        'mouthFrownLeft', 'mouthFrownRight',
        'mouthDimpleLeft', 'mouthDimpleRight',
        'mouthStretchLeft', 'mouthStretchRight',
        'mouthRollLower', 'mouthRollUpper',
        'mouthShrugLower', 'mouthShrugUpper',
        'mouthPressLeft', 'mouthPressRight',
        'mouthLowerDownLeft', 'mouthLowerDownRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight'
    ],
    'cheeks': [
        'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight'
    ],
    'jaw': [
        'jawOpen', 'jawForward', 'jawLeft', 'jawRight'
    ],
    'nose': [
        'noseSneerLeft', 'noseSneerRight'
    ],
    'tongue': [
        'tongueOut'
    ]
}


class ARKitBlendshapeData:
    """ARKit Blendshape 数据类"""

    def __init__(self):
        """初始化所有 blendshape 值为 0"""
        self.values = {name: 0.0 for name in ARKIT_BLENDSHAPES}

    def set(self, name: str, value: float):
        """设置 blendshape 值 (0-1 范围)"""
        if name in ARKIT_BLENDSHAPES:
            self.values[name] = max(0.0, min(1.0, value))
        else:
            raise ValueError(f"Invalid blendshape name: {name}")

    def get(self, name: str) -> float:
        """获取 blendshape 值"""
        return self.values.get(name, 0.0)

    def get_group(self, group: str) -> dict:
        """获取一组 blendshape 值"""
        if group not in BLENDSHAPE_GROUPS:
            raise ValueError(f"Invalid group: {group}")
        return {name: self.values[name] for name in BLENDSHAPE_GROUPS[group]}

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.values.copy()

    def from_dict(self, data: dict):
        """从字典加载"""
        for name, value in data.items():
            if name in ARKIT_BLENDSHAPES:
                self.set(name, value)

    def to_array(self) -> list:
        """转换为数组 (按标准顺序)"""
        return [self.values[name] for name in ARKIT_BLENDSHAPES]

    def from_array(self, arr: list):
        """从数组加载 (按标准顺序)"""
        if len(arr) != len(ARKIT_BLENDSHAPES):
            raise ValueError(f"Array length must be {len(ARKIT_BLENDSHAPES)}")
        for name, value in zip(ARKIT_BLENDSHAPES, arr):
            self.set(name, value)

    def __repr__(self):
        non_zero = {k: v for k, v in self.values.items() if v > 0.01}
        return f"ARKitBlendshapeData({len(non_zero)} active blendshapes)"
