#!/usr/bin/env python3
"""
示例：批量处理图像序列
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture import FacialExpressionCapture


def main():
    # 图像序列目录
    image_dir = Path("image_sequence")  # 替换为你的图像目录
    output_path = "sequence_expression.json"

    # 查找所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(sorted(image_dir.glob(f"*{ext}")))

    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")

    # 创建表情捕捉器
    capturer = FacialExpressionCapture()

    # 处理图像序列
    sequence = capturer.extract_from_images(
        image_files,
        output_path=output_path,
        fps=30.0
    )

    print(f"\n成功处理 {len(sequence)} 张图像")
    print(f"结果已保存到: {output_path}")

    # 导出为其他格式
    print("\n导出为其他格式...")

    # CSV 格式（用于 Excel）
    csv_path = Path(output_path).with_suffix('.csv')
    capturer.export_arkit(csv_path, sequence, fps=30.0, format='csv')
    print(f"- CSV: {csv_path}")

    # NumPy 格式（用于数据分析）
    npz_path = Path(output_path).with_suffix('.npz')
    capturer.export_arkit(npz_path, sequence, fps=30.0, format='numpy')
    print(f"- NumPy: {npz_path}")

    # Unity 格式
    unity_path = Path(output_path).with_suffix('.unity.json')
    capturer.export_arkit(unity_path, sequence, fps=30.0, format='unity')
    print(f"- Unity: {unity_path}")

    # Blender 格式
    blender_path = Path(output_path).with_suffix('.blender.json')
    capturer.export_arkit(blender_path, sequence, fps=30.0, format='blender')
    print(f"- Blender: {blender_path}")


if __name__ == "__main__":
    main()
