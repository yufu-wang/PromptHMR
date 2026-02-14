"""
从视频中提取表情序列
支持 MMD 视频和普通视频
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class VideoExpressionExtractor:
    """视频表情提取器"""

    def __init__(self, detector=None):
        """
        Args:
            detector: 面部检测器（EnhancedFacialDetector）
        """
        if detector is None:
            from .enhanced_detector import EnhancedFacialDetector
            self.detector = EnhancedFacialDetector()
        else:
            self.detector = detector

    def extract_from_video(self, video_path: str,
                          fps: Optional[float] = None,
                          start_time: float = 0.0,
                          end_time: Optional[float] = None,
                          skip_frames: int = 1,
                          min_face_confidence: float = 0.5) -> Tuple[List[Dict], float]:
        """
        从视频提取表情序列

        Args:
            video_path: 视频路径
            fps: 输出帧率，None 表示使用原始帧率
            start_time: 开始时间（秒）
            end_time: 结束时间（秒），None 表示到视频结尾
            skip_frames: 跳帧数（提高处理速度）
            min_face_confidence: 最小面部置信度

        Returns:
            (blendshapes 序列, 实际 fps)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        if fps is None:
            fps = original_fps

        # 计算起止帧
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps) if end_time else total_frames

        print(f"视频信息:")
        print(f"  FPS: {original_fps}")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {duration:.2f}s")
        print(f"  提取范围: {start_time:.2f}s - {(end_frame/original_fps):.2f}s")
        print(f"  跳帧: {skip_frames}")

        # 设置起始位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sequence = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # 跳帧
            if (frame_idx - start_frame) % skip_frames != 0:
                frame_idx += 1
                continue

            # 检测表情
            detection = self.detector.detect_with_details(frame)

            if detection:
                # 转换为 blendshapes
                from .extended_blendshapes import convert_detection_to_extended_blendshapes
                blendshapes = convert_detection_to_extended_blendshapes(detection)
                sequence.append(blendshapes.to_dict_extended())

                # 显示进度
                if len(sequence) % 10 == 0:
                    progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                    print(f"  进度: {progress:.1f}% ({len(sequence)} 帧)")

            frame_idx += 1

        cap.release()

        actual_fps = fps / skip_frames

        print(f"\n✓ 提取完成: {len(sequence)} 帧 @ {actual_fps:.2f} FPS")

        return sequence, actual_fps

    def extract_keyframes(self, video_path: str,
                         num_keyframes: int = 5,
                         method: str = "uniform") -> List[Tuple[float, Dict]]:
        """
        提取关键帧

        Args:
            video_path: 视频路径
            num_keyframes: 关键帧数量
            method: 提取方法 ("uniform", "peaks", "variance")

        Returns:
            [(time, blendshapes), ...]
        """
        # 先提取完整序列
        sequence, fps = self.extract_from_video(video_path, skip_frames=2)

        if len(sequence) == 0:
            return []

        keyframes = []

        if method == "uniform":
            # 均匀采样
            indices = np.linspace(0, len(sequence) - 1, num_keyframes, dtype=int)

            for idx in indices:
                time = idx / fps
                keyframes.append((time, sequence[idx]))

        elif method == "peaks":
            # 找表情强度峰值
            # 计算每帧的表情强度（所有 blendshape 值的和）
            intensities = []
            for frame_bs in sequence:
                intensity = sum(abs(v) for v in frame_bs.values())
                intensities.append(intensity)

            # 找峰值
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(intensities, distance=len(sequence) // num_keyframes)

            # 如果峰值不够，补充均匀采样
            if len(peaks) < num_keyframes:
                uniform_indices = np.linspace(0, len(sequence) - 1, num_keyframes - len(peaks), dtype=int)
                peaks = np.concatenate([peaks, uniform_indices])
                peaks = np.unique(peaks)
                peaks.sort()

            # 取前 num_keyframes 个
            peaks = peaks[:num_keyframes]

            for idx in peaks:
                time = idx / fps
                keyframes.append((time, sequence[idx]))

        elif method == "variance":
            # 基于变化率选择关键帧
            variances = []

            for i in range(1, len(sequence)):
                # 计算相邻帧的差异
                diff = sum(abs(sequence[i].get(k, 0) - sequence[i-1].get(k, 0))
                          for k in sequence[i].keys())
                variances.append(diff)

            # 找变化最大的帧
            top_indices = np.argsort(variances)[-num_keyframes:]
            top_indices.sort()

            for idx in top_indices:
                time = idx / fps
                keyframes.append((time, sequence[idx]))

        # 按时间排序
        keyframes.sort(key=lambda x: x[0])

        return keyframes

    def analyze_expression_patterns(self, sequence: List[Dict],
                                   fps: float) -> Dict:
        """
        分析表情模式

        Args:
            sequence: blendshape 序列
            fps: 帧率

        Returns:
            分析结果
        """
        if len(sequence) == 0:
            return {}

        # 统计每个 blendshape 的使用情况
        stats = {}

        for bs_name in sequence[0].keys():
            values = [frame.get(bs_name, 0.0) for frame in sequence]

            stats[bs_name] = {
                'mean': float(np.mean(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
                'std': float(np.std(values)),
                'active_ratio': float(np.sum(np.abs(values) > 0.05) / len(values)),
            }

        # 找出最活跃的 blendshapes
        active_bs = sorted(
            stats.items(),
            key=lambda x: x[1]['active_ratio'] * x[1]['max'],
            reverse=True
        )

        return {
            'stats': stats,
            'top_active': [(name, data) for name, data in active_bs[:10]],
            'duration': len(sequence) / fps,
            'num_frames': len(sequence),
            'fps': fps,
        }

    def create_loop_from_video(self, video_path: str,
                              loop_duration: float = 2.0,
                              method: str = "seamless") -> List[Dict]:
        """
        从视频创建循环动画

        Args:
            video_path: 视频路径
            loop_duration: 循环时长（秒）
            method: 循环方法 ("seamless", "ping_pong", "direct")

        Returns:
            循环动画序列
        """
        # 提取序列
        sequence, fps = self.extract_from_video(video_path)

        if len(sequence) == 0:
            return []

        # 根据方法创建循环
        if method == "seamless":
            # 找到最相似的起止帧
            start_bs = sequence[0]
            best_end_idx = 0
            min_diff = float('inf')

            for i in range(len(sequence) // 2, len(sequence)):
                diff = sum(abs(sequence[i].get(k, 0) - start_bs.get(k, 0))
                          for k in start_bs.keys())
                if diff < min_diff:
                    min_diff = diff
                    best_end_idx = i

            # 截取到最佳结束点
            loop_sequence = sequence[:best_end_idx + 1]

        elif method == "ping_pong":
            # 正向 + 反向
            loop_sequence = sequence + sequence[::-1]

        else:  # direct
            loop_sequence = sequence

        # 重采样到目标时长
        target_frames = int(loop_duration * fps)

        if len(loop_sequence) != target_frames:
            # 重采样
            resampled = self._resample_sequence(loop_sequence, target_frames)
            return resampled

        return loop_sequence

    def _resample_sequence(self, sequence: List[Dict], target_frames: int) -> List[Dict]:
        """重采样序列到目标帧数"""
        if len(sequence) == target_frames:
            return sequence

        from .animation_generator import ProceduralAnimationGenerator, EasingType

        generator = ProceduralAnimationGenerator()

        # 创建关键帧
        for i, frame_bs in enumerate(sequence):
            time = i / (len(sequence) - 1) if len(sequence) > 1 else 0
            generator.add_keyframe(time, frame_bs, EasingType.LINEAR)

        # 生成目标帧数
        duration = target_frames / 30.0  # 假设 30fps
        resampled = generator.generate_animation(duration, fps=30.0)

        return resampled[:target_frames]


class MMDExpressionAnalyzer:
    """MMD 表情分析器（针对 MMD 视频优化）"""

    def __init__(self):
        """初始化"""
        self.extractor = VideoExpressionExtractor()

    def extract_mmd_patterns(self, video_path: str,
                            focus_region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        从 MMD 视频提取表情模式

        Args:
            video_path: MMD 视频路径
            focus_region: 聚焦区域 (x, y, w, h)，用于裁剪面部

        Returns:
            提取的表情模式
        """
        # 提取表情序列
        sequence, fps = self.extractor.extract_from_video(video_path, skip_frames=1)

        # 分析模式
        analysis = self.extractor.analyze_expression_patterns(sequence, fps)

        # MMD 特有的表情检测
        mmd_patterns = self._detect_mmd_patterns(sequence, fps)

        return {
            'sequence': sequence,
            'fps': fps,
            'analysis': analysis,
            'mmd_patterns': mmd_patterns,
        }

    def _detect_mmd_patterns(self, sequence: List[Dict], fps: float) -> Dict:
        """检测 MMD 特有的表情模式"""
        patterns = {
            'blinks': [],           # 眨眼时刻
            'mouth_opens': [],      # 张嘴时刻
            'smile_peaks': [],      # 微笑峰值
            'expression_changes': [], # 表情变化点
        }

        # 检测眨眼
        for i in range(1, len(sequence)):
            blink_left = sequence[i].get('eyeBlinkLeft', 0)
            blink_right = sequence[i].get('eyeBlinkRight', 0)

            if (blink_left + blink_right) / 2 > 0.7:
                time = i / fps
                patterns['blinks'].append(time)

        # 检测张嘴
        for i in range(1, len(sequence)):
            jaw_open = sequence[i].get('jawOpen', 0)

            if jaw_open > 0.5:
                time = i / fps
                patterns['mouth_opens'].append(time)

        # 检测微笑峰值
        smile_values = [
            (sequence[i].get('mouthSmileLeft', 0) + sequence[i].get('mouthSmileRight', 0)) / 2
            for i in range(len(sequence))
        ]

        from scipy.signal import find_peaks
        smile_peaks, _ = find_peaks(smile_values, height=0.3)

        for idx in smile_peaks:
            time = idx / fps
            patterns['smile_peaks'].append(time)

        return patterns
