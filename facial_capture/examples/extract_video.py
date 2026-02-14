#!/usr/bin/env python3
"""
示例：从视频提取面部表情序列
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture import FacialExpressionCapture


def main():
    # 输入视频路径
    video_path = "input.mp4"  # 替换为你的视频路径
    output_dir = "output"

    # 创建表情捕捉器
    capturer = FacialExpressionCapture()

    print(f"正在处理视频: {video_path}")

    # 提取表情序列
    sequence = capturer.extract_from_video(
        video_path,
        output_dir=output_dir,
        fps=30.0,  # 输出帧率
        visualize=False  # 设置为 True 可以实时查看
    )

    if not sequence:
        print("错误: 未能提取表情")
        return

    print(f"\n成功提取 {len(sequence)} 帧")

    # 统计表情使用情况
    print("\n表情统计:")
    print("-" * 50)

    from collections import defaultdict
    stats = defaultdict(float)

    for blendshapes in sequence:
        for name, value in blendshapes.to_dict().items():
            stats[name] = max(stats[name], value)

    # 显示主要表情
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    for name, max_value in sorted_stats[:10]:
        if max_value > 0.1:
            print(f"{name:25s}: 最大值 {max_value:.3f}")

    print(f"\n结果已保存到: {output_dir}/")
    print("- JSON 格式: *_arkit.json")
    print("- NumPy 格式: *_arkit.npz")


if __name__ == "__main__":
    main()
