"""
表情数据导出器
支持多种格式：JSON, FBX, Alembic
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from .arkit_blendshapes import ARKitBlendshapeData, ARKIT_BLENDSHAPES


class ExpressionExporter:
    """表情数据导出器"""

    def __init__(self):
        """初始化导出器"""
        pass

    def export_json(self, blendshapes: ARKitBlendshapeData, output_path: str,
                   metadata: Optional[Dict] = None):
        """
        导出为 JSON 格式

        Args:
            blendshapes: ARKit blendshape 数据
            output_path: 输出文件路径
            metadata: 可选的元数据
        """
        data = {
            'version': '1.0',
            'format': 'ARKit',
            'blendshapes': blendshapes.to_dict(),
            'metadata': metadata or {}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_json_sequence(self, blendshape_sequence: List[ARKitBlendshapeData],
                            output_path: str, fps: float = 30.0,
                            metadata: Optional[Dict] = None):
        """
        导出表情序列为 JSON 格式

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
            metadata: 可选的元数据
        """
        frames = []
        for idx, blendshapes in enumerate(blendshape_sequence):
            frame_data = {
                'frame': idx,
                'time': idx / fps,
                'blendshapes': blendshapes.to_dict()
            }
            frames.append(frame_data)

        data = {
            'version': '1.0',
            'format': 'ARKit',
            'fps': fps,
            'num_frames': len(frames),
            'duration': len(frames) / fps,
            'frames': frames,
            'metadata': metadata or {}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_csv(self, blendshape_sequence: List[ARKitBlendshapeData],
                   output_path: str, fps: float = 30.0):
        """
        导出为 CSV 格式（适合 Excel 或数据分析）

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ['Frame', 'Time'] + ARKIT_BLENDSHAPES
            writer.writerow(header)

            # 写入数据
            for idx, blendshapes in enumerate(blendshape_sequence):
                row = [idx, idx / fps] + blendshapes.to_array()
                writer.writerow(row)

    def export_numpy(self, blendshape_sequence: List[ARKitBlendshapeData],
                    output_path: str, fps: float = 30.0):
        """
        导出为 NumPy 格式

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
        """
        # 转换为 numpy 数组 (num_frames, 52)
        data = np.array([bs.to_array() for bs in blendshape_sequence])

        # 保存
        np.savez(output_path,
                blendshapes=data,
                fps=fps,
                num_frames=len(blendshape_sequence),
                blendshape_names=ARKIT_BLENDSHAPES)

    def export_fbx(self, blendshape_sequence: List[ARKitBlendshapeData],
                   output_path: str, fps: float = 30.0):
        """
        导出为 FBX 格式（需要额外的 FBX SDK 或转换工具）

        注意：这个方法需要额外的依赖，暂时导出为中间格式

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
        """
        # 导出为 JSON 格式，可以使用 Blender/Unity 脚本导入
        json_path = Path(output_path).with_suffix('.json')
        self.export_json_sequence(blendshape_sequence, str(json_path), fps,
                                 metadata={'target_format': 'fbx'})

        print(f"FBX 导出需要额外工具，已导出为 JSON: {json_path}")
        print("请使用 Blender 或 Unity 插件导入此 JSON 文件")

    def load_json(self, input_path: str) -> ARKitBlendshapeData:
        """
        从 JSON 加载单帧数据

        Args:
            input_path: 输入文件路径

        Returns:
            ARKitBlendshapeData 对象
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        blendshapes = ARKitBlendshapeData()
        blendshapes.from_dict(data['blendshapes'])
        return blendshapes

    def load_json_sequence(self, input_path: str) -> tuple:
        """
        从 JSON 加载序列数据

        Args:
            input_path: 输入文件路径

        Returns:
            (blendshape_sequence, fps, metadata) 元组
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sequence = []
        for frame_data in data['frames']:
            blendshapes = ARKitBlendshapeData()
            blendshapes.from_dict(frame_data['blendshapes'])
            sequence.append(blendshapes)

        return sequence, data['fps'], data.get('metadata', {})

    def export_unity_animation(self, blendshape_sequence: List[ARKitBlendshapeData],
                              output_path: str, fps: float = 30.0):
        """
        导出为 Unity 动画曲线格式

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
        """
        # Unity AnimationClip 格式（YAML/JSON）
        curves = {}

        for blendshape_name in ARKIT_BLENDSHAPES:
            keyframes = []
            for idx, blendshapes in enumerate(blendshape_sequence):
                time = idx / fps
                value = blendshapes.get(blendshape_name) * 100.0  # Unity 使用 0-100
                keyframes.append({
                    'time': time,
                    'value': value,
                    'inTangent': 0,
                    'outTangent': 0
                })
            curves[blendshape_name] = keyframes

        data = {
            'version': '1.0',
            'format': 'Unity AnimationClip',
            'fps': fps,
            'length': len(blendshape_sequence) / fps,
            'curves': curves
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def export_blender_action(self, blendshape_sequence: List[ARKitBlendshapeData],
                             output_path: str, fps: float = 30.0):
        """
        导出为 Blender Action 格式

        Args:
            blendshape_sequence: ARKit blendshape 序列
            output_path: 输出文件路径
            fps: 帧率
        """
        # Blender Action 格式
        fcurves = {}

        for blendshape_name in ARKIT_BLENDSHAPES:
            keyframes = []
            for idx, blendshapes in enumerate(blendshape_sequence):
                frame = idx
                value = blendshapes.get(blendshape_name)
                keyframes.append({
                    'frame': frame,
                    'value': value,
                    'interpolation': 'BEZIER'
                })
            fcurves[blendshape_name] = keyframes

        data = {
            'version': '1.0',
            'format': 'Blender Action',
            'fps': fps,
            'frame_start': 0,
            'frame_end': len(blendshape_sequence) - 1,
            'fcurves': fcurves
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


class LiveLinkExporter:
    """
    Unreal Engine Live Link 格式导出
    用于实时面部动画流式传输
    """

    def __init__(self, port: int = 54321):
        """
        初始化 Live Link 导出器

        Args:
            port: 网络端口
        """
        self.port = port

    def export_frame(self, blendshapes: ARKitBlendshapeData) -> bytes:
        """
        将单帧导出为 Live Link 格式的字节流

        Args:
            blendshapes: ARKit blendshape 数据

        Returns:
            二进制数据
        """
        # Live Link 格式（简化版）
        import struct

        data = bytearray()

        # 魔数
        data.extend(struct.pack('I', 0x4C4C4652))  # 'LLFR'

        # 版本
        data.extend(struct.pack('I', 1))

        # Blendshape 数量
        data.extend(struct.pack('I', 52))

        # Blendshape 值
        for name in ARKIT_BLENDSHAPES:
            value = blendshapes.get(name)
            data.extend(struct.pack('f', value))

        return bytes(data)
