"""
程序化表情动画生成器
支持在两个表情之间生成平滑的循环动画
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from enum import Enum
import math


class EasingType(Enum):
    """缓动类型"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    ELASTIC = "elastic"
    BOUNCE = "bounce"
    SPRING = "spring"
    OVERSHOOT = "overshoot"
    SINE = "sine"
    CUBIC = "cubic"
    EXPO = "expo"


class LoopMode(Enum):
    """循环模式"""
    PING_PONG = "ping_pong"      # A→B→A 来回
    LOOP = "loop"                  # A→B→A (直接跳回)
    SEAMLESS = "seamless"          # A→B 无缝循环
    HOLD = "hold"                  # A→B 停止


class EasingFunctions:
    """缓动函数集合"""

    @staticmethod
    def linear(t: float) -> float:
        """线性插值"""
        return t

    @staticmethod
    def ease_in_quad(t: float) -> float:
        """二次缓入"""
        return t * t

    @staticmethod
    def ease_out_quad(t: float) -> float:
        """二次缓出"""
        return t * (2 - t)

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """二次缓入缓出"""
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """三次缓入"""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """三次缓出"""
        return (--t) * t * t + 1

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """三次缓入缓出"""
        return 4 * t * t * t if t < 0.5 else (t - 1) * (2 * t - 2) * (2 * t - 2) + 1

    @staticmethod
    def ease_in_expo(t: float) -> float:
        """指数缓入"""
        return 0 if t == 0 else math.pow(2, 10 * (t - 1))

    @staticmethod
    def ease_out_expo(t: float) -> float:
        """指数缓出"""
        return 1 if t == 1 else 1 - math.pow(2, -10 * t)

    @staticmethod
    def ease_in_out_expo(t: float) -> float:
        """指数缓入缓出"""
        if t == 0 or t == 1:
            return t
        if t < 0.5:
            return 0.5 * math.pow(2, 20 * t - 10)
        return 1 - 0.5 * math.pow(2, -20 * t + 10)

    @staticmethod
    def ease_in_sine(t: float) -> float:
        """正弦缓入"""
        return 1 - math.cos(t * math.pi / 2)

    @staticmethod
    def ease_out_sine(t: float) -> float:
        """正弦缓出"""
        return math.sin(t * math.pi / 2)

    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        """正弦缓入缓出"""
        return 0.5 * (1 - math.cos(t * math.pi))

    @staticmethod
    def elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
        """弹性缓动（回弹效果）"""
        if t == 0 or t == 1:
            return t
        s = period / 4
        return amplitude * math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / period) + 1

    @staticmethod
    def bounce(t: float) -> float:
        """弹跳效果"""
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    @staticmethod
    def spring(t: float, damping: float = 0.3) -> float:
        """弹簧效果"""
        return 1 - (math.exp(-damping * t) * math.cos(10 * t))

    @staticmethod
    def overshoot(t: float, amount: float = 1.70158) -> float:
        """过冲效果（超过目标再回来）"""
        return t * t * ((amount + 1) * t - amount)

    @staticmethod
    def get_easing_function(easing_type: EasingType) -> Callable[[float], float]:
        """获取缓动函数"""
        mapping = {
            EasingType.LINEAR: EasingFunctions.linear,
            EasingType.EASE_IN: EasingFunctions.ease_in_cubic,
            EasingType.EASE_OUT: EasingFunctions.ease_out_cubic,
            EasingType.EASE_IN_OUT: EasingFunctions.ease_in_out_cubic,
            EasingType.ELASTIC: EasingFunctions.elastic,
            EasingType.BOUNCE: EasingFunctions.bounce,
            EasingType.SPRING: EasingFunctions.spring,
            EasingType.OVERSHOOT: EasingFunctions.overshoot,
            EasingType.SINE: EasingFunctions.ease_in_out_sine,
            EasingType.CUBIC: EasingFunctions.ease_in_out_cubic,
            EasingType.EXPO: EasingFunctions.ease_in_out_expo,
        }
        return mapping.get(easing_type, EasingFunctions.linear)


class ExpressionKeyframe:
    """表情关键帧"""

    def __init__(self, time: float, blendshapes: Dict[str, float],
                 easing: EasingType = EasingType.EASE_IN_OUT):
        """
        Args:
            time: 时间点（0-1）
            blendshapes: blendshape 值字典
            easing: 到下一个关键帧的缓动类型
        """
        self.time = time
        self.blendshapes = blendshapes
        self.easing = easing


class ProceduralAnimationGenerator:
    """程序化动画生成器"""

    def __init__(self):
        """初始化"""
        self.keyframes: List[ExpressionKeyframe] = []

    def add_keyframe(self, time: float, blendshapes: Dict[str, float],
                    easing: EasingType = EasingType.EASE_IN_OUT):
        """添加关键帧"""
        keyframe = ExpressionKeyframe(time, blendshapes, easing)
        self.keyframes.append(keyframe)
        # 按时间排序
        self.keyframes.sort(key=lambda k: k.time)

    def interpolate_blendshapes(self, bs_a: Dict[str, float], bs_b: Dict[str, float],
                               t: float, easing_func: Callable[[float], float]) -> Dict[str, float]:
        """
        在两个 blendshape 状态之间插值

        Args:
            bs_a: 起始 blendshapes
            bs_b: 结束 blendshapes
            t: 插值系数 (0-1)
            easing_func: 缓动函数

        Returns:
            插值后的 blendshapes
        """
        # 应用缓动
        eased_t = easing_func(t)

        # 获取所有 blendshape 名称
        all_names = set(bs_a.keys()) | set(bs_b.keys())

        result = {}
        for name in all_names:
            value_a = bs_a.get(name, 0.0)
            value_b = bs_b.get(name, 0.0)
            # 线性插值
            result[name] = value_a + (value_b - value_a) * eased_t

        return result

    def generate_animation(self, duration: float, fps: float = 30.0,
                          loop_mode: LoopMode = LoopMode.PING_PONG) -> List[Dict[str, float]]:
        """
        生成动画序列

        Args:
            duration: 动画时长（秒）
            fps: 帧率
            loop_mode: 循环模式

        Returns:
            blendshape 序列
        """
        if len(self.keyframes) < 2:
            raise ValueError("至少需要 2 个关键帧")

        num_frames = int(duration * fps)
        sequence = []

        for frame_idx in range(num_frames):
            # 计算当前时间 (0-1)
            t = frame_idx / (num_frames - 1) if num_frames > 1 else 0

            # 根据循环模式调整时间
            t = self._apply_loop_mode(t, loop_mode)

            # 找到当前时间所在的关键帧区间
            blendshapes = self._evaluate_at_time(t)
            sequence.append(blendshapes)

        return sequence

    def _apply_loop_mode(self, t: float, loop_mode: LoopMode) -> float:
        """应用循环模式"""
        if loop_mode == LoopMode.PING_PONG:
            # 来回循环：0→1→0
            if t > 0.5:
                t = 1.0 - t
            t *= 2  # 映射到 0-1
        elif loop_mode == LoopMode.SEAMLESS:
            # 无缝循环：使用正弦函数使首尾平滑
            t = (math.sin((t - 0.25) * 2 * math.pi) + 1) / 2
        # LOOP 和 HOLD 保持原样

        return t

    def _evaluate_at_time(self, t: float) -> Dict[str, float]:
        """在指定时间点计算 blendshapes"""
        # 找到 t 所在的关键帧区间
        if t <= self.keyframes[0].time:
            return self.keyframes[0].blendshapes.copy()

        if t >= self.keyframes[-1].time:
            return self.keyframes[-1].blendshapes.copy()

        # 找到相邻的两个关键帧
        for i in range(len(self.keyframes) - 1):
            kf_a = self.keyframes[i]
            kf_b = self.keyframes[i + 1]

            if kf_a.time <= t <= kf_b.time:
                # 计算局部插值系数
                local_t = (t - kf_a.time) / (kf_b.time - kf_a.time)

                # 获取缓动函数
                easing_func = EasingFunctions.get_easing_function(kf_a.easing)

                # 插值
                return self.interpolate_blendshapes(
                    kf_a.blendshapes,
                    kf_b.blendshapes,
                    local_t,
                    easing_func
                )

        return self.keyframes[-1].blendshapes.copy()

    def create_simple_transition(self, start_bs: Dict[str, float],
                                end_bs: Dict[str, float],
                                duration: float = 2.0,
                                fps: float = 30.0,
                                easing: EasingType = EasingType.EASE_IN_OUT,
                                loop_mode: LoopMode = LoopMode.PING_PONG) -> List[Dict[str, float]]:
        """
        创建简单的两个表情之间的过渡动画

        Args:
            start_bs: 起始表情
            end_bs: 结束表情
            duration: 持续时间（秒）
            fps: 帧率
            easing: 缓动类型
            loop_mode: 循环模式

        Returns:
            动画序列
        """
        # 清空现有关键帧
        self.keyframes = []

        # 添加起始和结束关键帧
        self.add_keyframe(0.0, start_bs, easing)
        self.add_keyframe(1.0, end_bs, easing)

        # 生成动画
        return self.generate_animation(duration, fps, loop_mode)

    def create_multi_stage_animation(self, keyframe_data: List[Tuple[float, Dict[str, float], EasingType]],
                                    duration: float = 4.0,
                                    fps: float = 30.0,
                                    loop_mode: LoopMode = LoopMode.LOOP) -> List[Dict[str, float]]:
        """
        创建多阶段动画

        Args:
            keyframe_data: [(time, blendshapes, easing), ...]
            duration: 持续时间
            fps: 帧率
            loop_mode: 循环模式

        Returns:
            动画序列
        """
        self.keyframes = []

        for time, blendshapes, easing in keyframe_data:
            self.add_keyframe(time, blendshapes, easing)

        return self.generate_animation(duration, fps, loop_mode)

    def add_procedural_variations(self, base_sequence: List[Dict[str, float]],
                                 noise_amount: float = 0.05,
                                 target_blendshapes: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        添加程序化的微小变化（使动画更自然）

        Args:
            base_sequence: 基础动画序列
            noise_amount: 噪声强度 (0-1)
            target_blendshapes: 要添加变化的 blendshape 列表，None 表示所有

        Returns:
            添加变化后的序列
        """
        result = []

        for frame_bs in base_sequence:
            new_bs = frame_bs.copy()

            for name, value in frame_bs.items():
                # 检查是否需要添加变化
                if target_blendshapes and name not in target_blendshapes:
                    continue

                # 添加柏林噪声或随机变化
                noise = (np.random.random() - 0.5) * 2 * noise_amount
                new_bs[name] = max(0.0, min(1.0, value + noise))

            result.append(new_bs)

        return result

    def blend_animations(self, anim_a: List[Dict[str, float]],
                        anim_b: List[Dict[str, float]],
                        blend_factor: float = 0.5) -> List[Dict[str, float]]:
        """
        混合两个动画

        Args:
            anim_a: 动画 A
            anim_b: 动画 B
            blend_factor: 混合系数 (0=全A, 1=全B)

        Returns:
            混合后的动画
        """
        # 确保长度一致
        min_len = min(len(anim_a), len(anim_b))
        result = []

        for i in range(min_len):
            blended = self.interpolate_blendshapes(
                anim_a[i],
                anim_b[i],
                blend_factor,
                EasingFunctions.linear
            )
            result.append(blended)

        return result


class R18AnimationPresets:
    """R18 动画预设"""

    @staticmethod
    def ahegao_transition(intensity: float = 0.8) -> List[Tuple[float, Dict[str, float], EasingType]]:
        """
        阿黑颜渐进动画预设

        Returns:
            关键帧数据列表
        """
        keyframes = [
            # 中性表情
            (0.0, {
                'tongueOut': 0.0,
                'jawOpen': 0.0,
                'eyeLookUpLeft': 0.0,
                'eyeLookUpRight': 0.0,
            }, EasingType.EASE_IN),

            # 开始反应
            (0.3, {
                'jawOpen': 0.3 * intensity,
                'eyeWideLeft': 0.5 * intensity,
                'eyeWideRight': 0.5 * intensity,
                'mouthSmileLeft': 0.2,
                'mouthSmileRight': 0.2,
            }, EasingType.EASE_OUT),

            # 高潮阶段
            (0.7, {
                'tongueOut': 0.85 * intensity,
                'jawOpen': 0.8 * intensity,
                'eyeLookUpLeft': 0.9 * intensity,
                'eyeLookUpRight': 0.9 * intensity,
                'eyeSquintLeft': 0.6 * intensity,
                'eyeSquintRight': 0.6 * intensity,
                'cheekSquintLeft': 0.7,
                'cheekSquintRight': 0.7,
            }, EasingType.ELASTIC),

            # 持续高潮（微小震颤）
            (1.0, {
                'tongueOut': 0.9 * intensity,
                'jawOpen': 0.85 * intensity,
                'eyeLookUpLeft': 0.95 * intensity,
                'eyeLookUpRight': 0.95 * intensity,
                'eyeSquintLeft': 0.7 * intensity,
                'eyeSquintRight': 0.7 * intensity,
            }, EasingType.SPRING),
        ]

        return keyframes

    @staticmethod
    def breathing_cycle() -> List[Tuple[float, Dict[str, float], EasingType]]:
        """呼吸循环动画"""
        keyframes = [
            (0.0, {
                'jawOpen': 0.1,
                'mouthFunnel': 0.05,
            }, EasingType.SINE),

            (0.5, {
                'jawOpen': 0.15,
                'mouthFunnel': 0.12,
                'cheekPuff': 0.08,
            }, EasingType.SINE),

            (1.0, {
                'jawOpen': 0.1,
                'mouthFunnel': 0.05,
            }, EasingType.SINE),
        ]

        return keyframes
