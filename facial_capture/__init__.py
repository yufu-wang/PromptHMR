"""
Facial Expression Capture Plugin

从 2D 图像（包括漫画、插图）中提取面部表情并映射到 ARKit Blendshapes
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict

from .core import (
    FacialLandmarkDetector,
    ExpressionAnalyzer,
    ARKitBlendshapeData,
    ExpressionExporter,
)


class FacialExpressionCapture:
    """面部表情捕捉主类"""

    def __init__(self, use_anime_detector: bool = False):
        """
        初始化表情捕捉器

        Args:
            use_anime_detector: 是否使用动漫风格检测器
        """
        self.detector = FacialLandmarkDetector(use_anime_detector=use_anime_detector)
        self.analyzer = ExpressionAnalyzer()
        self.exporter = ExpressionExporter()

    def extract_from_image(self, image_path: Union[str, Path, np.ndarray],
                          visualize: bool = False) -> Optional[ARKitBlendshapeData]:
        """
        从图像中提取面部表情

        Args:
            image_path: 图像路径或 numpy 数组
            visualize: 是否可视化检测结果

        Returns:
            ARKitBlendshapeData 对象，如果检测失败则返回 None
        """
        # 加载图像
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法加载图像: {image_path}")
                return None
        else:
            image = image_path

        # 检测面部关键点
        detection = self.detector.detect(image)
        if detection is None:
            print("未检测到面部")
            return None

        # 分析表情
        blendshapes = self.analyzer.analyze(
            detection['landmarks'],
            detection['normalized_landmarks']
        )

        # 可视化
        if visualize:
            vis_image = self.detector.visualize(image, detection)
            cv2.imshow('Facial Detection', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return blendshapes

    def extract_from_video(self, video_path: Union[str, Path],
                          output_dir: Optional[Union[str, Path]] = None,
                          fps: Optional[float] = None,
                          visualize: bool = False) -> List[ARKitBlendshapeData]:
        """
        从视频中提取面部表情序列

        Args:
            video_path: 视频路径
            output_dir: 输出目录（如果为 None，不保存）
            fps: 输出帧率（如果为 None，使用原始帧率）
            visualize: 是否可视化检测结果

        Returns:
            ARKitBlendshapeData 列表
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None:
            fps = original_fps
            frame_skip = 1
        else:
            frame_skip = max(1, int(original_fps / fps))

        print(f"处理视频: {video_path}")
        print(f"原始 FPS: {original_fps}, 目标 FPS: {fps}, 跳帧: {frame_skip}")
        print(f"总帧数: {total_frames}")

        blendshape_sequence = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # 提取表情
            blendshapes = self.extract_from_image(frame, visualize=False)

            if blendshapes is not None:
                blendshape_sequence.append(blendshapes)

                # 显示进度
                if frame_idx % 30 == 0:
                    print(f"已处理: {frame_idx}/{total_frames} 帧")

            # 可视化
            if visualize and blendshapes is not None:
                detection = self.detector.detect(frame)
                if detection:
                    vis_frame = self.detector.visualize(frame, detection)
                    cv2.imshow('Video Processing', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"提取完成: {len(blendshape_sequence)} 帧")

        # 保存结果
        if output_dir and blendshape_sequence:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存为多种格式
            video_name = Path(video_path).stem
            self.exporter.export_json_sequence(
                blendshape_sequence,
                str(output_dir / f"{video_name}_arkit.json"),
                fps=fps
            )
            self.exporter.export_numpy(
                blendshape_sequence,
                str(output_dir / f"{video_name}_arkit.npz"),
                fps=fps
            )
            print(f"结果已保存到: {output_dir}")

        return blendshape_sequence

    def extract_from_images(self, image_paths: List[Union[str, Path]],
                           output_path: Optional[Union[str, Path]] = None,
                           fps: float = 30.0) -> List[ARKitBlendshapeData]:
        """
        从图像序列中提取面部表情

        Args:
            image_paths: 图像路径列表
            output_path: 输出文件路径（如果为 None，不保存）
            fps: 帧率

        Returns:
            ARKitBlendshapeData 列表
        """
        blendshape_sequence = []

        print(f"处理 {len(image_paths)} 张图像...")

        for idx, image_path in enumerate(image_paths):
            blendshapes = self.extract_from_image(image_path)

            if blendshapes is not None:
                blendshape_sequence.append(blendshapes)

            if (idx + 1) % 10 == 0:
                print(f"已处理: {idx + 1}/{len(image_paths)}")

        print(f"提取完成: {len(blendshape_sequence)} 帧")

        # 保存结果
        if output_path and blendshape_sequence:
            self.exporter.export_json_sequence(blendshape_sequence, str(output_path), fps=fps)
            print(f"结果已保存到: {output_path}")

        return blendshape_sequence

    def export_arkit(self, output_path: Union[str, Path],
                    blendshapes: Union[ARKitBlendshapeData, List[ARKitBlendshapeData]],
                    fps: float = 30.0, format: str = 'json'):
        """
        导出 ARKit 数据

        Args:
            output_path: 输出路径
            blendshapes: 单帧或序列数据
            fps: 帧率（序列数据）
            format: 导出格式 ('json', 'csv', 'numpy', 'unity', 'blender')
        """
        output_path = Path(output_path)

        if isinstance(blendshapes, list):
            # 序列数据
            if format == 'json':
                self.exporter.export_json_sequence(blendshapes, str(output_path), fps=fps)
            elif format == 'csv':
                self.exporter.export_csv(blendshapes, str(output_path), fps=fps)
            elif format == 'numpy':
                self.exporter.export_numpy(blendshapes, str(output_path), fps=fps)
            elif format == 'unity':
                self.exporter.export_unity_animation(blendshapes, str(output_path), fps=fps)
            elif format == 'blender':
                self.exporter.export_blender_action(blendshapes, str(output_path), fps=fps)
            else:
                raise ValueError(f"不支持的格式: {format}")
        else:
            # 单帧数据
            if format == 'json':
                self.exporter.export_json(blendshapes, str(output_path))
            else:
                raise ValueError(f"单帧数据只支持 JSON 格式")

        print(f"已导出为 {format} 格式: {output_path}")

    def calibrate_neutral_expression(self, image_path: Union[str, Path, np.ndarray]):
        """
        使用中性表情校准分析器

        Args:
            image_path: 中性表情图像路径
        """
        # 加载图像
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
        else:
            image = image_path

        # 检测关键点
        detection = self.detector.detect(image)
        if detection is None:
            print("校准失败: 未检测到面部")
            return

        # 校准
        self.analyzer.calibrate_neutral(
            detection['landmarks'],
            detection['normalized_landmarks']
        )

        print("校准完成")


__all__ = ['FacialExpressionCapture']
