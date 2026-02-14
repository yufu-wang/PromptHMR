#!/usr/bin/env python3
"""
示例：从单张图像提取面部表情
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture import FacialExpressionCapture


def main():
    # 输入图像路径
    image_path = "input.jpg"  # 替换为你的图像路径
    output_path = "expression.json"

    # 创建表情捕捉器
    capturer = FacialExpressionCapture()

    print(f"正在处理图像: {image_path}")

    # 提取表情（可视化）
    blendshapes = capturer.extract_from_image(image_path, visualize=True)

    if blendshapes is None:
        print("错误: 未检测到面部")
        return

    # 显示非零的 blendshape 值
    print("\n检测到的表情:")
    print("-" * 50)

    for name, value in blendshapes.to_dict().items():
        if value > 0.01:  # 只显示明显的表情
            print(f"{name:25s}: {value:.3f}")

    # 导出为 JSON
    capturer.export_arkit(output_path, blendshapes, format='json')
    print(f"\n表情已保存到: {output_path}")


if __name__ == "__main__":
    main()
