#!/usr/bin/env python3
"""
从 MMD 视频提取表情并生成循环动画
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture.core.video_expression_extractor import (
    VideoExpressionExtractor,
    MMDExpressionAnalyzer
)
from facial_capture.core.animation_generator import ProceduralAnimationGenerator, LoopMode
from facial_capture.core import ExpressionExporter


def main():
    print("=" * 70)
    print("MMD 视频表情提取器")
    print("=" * 70)

    print("\n选择功能:")
    print("1. 从 MMD 视频提取完整表情序列")
    print("2. 提取关键帧并生成循环动画")
    print("3. 分析 MMD 表情模式")
    print("4. 创建基于 MMD 的循环动画")

    mode = input("\n选择 (1-4): ").strip()

    if mode == "1":
        extract_full_sequence()
    elif mode == "2":
        extract_keyframes_and_loop()
    elif mode == "3":
        analyze_mmd_patterns()
    elif mode == "4":
        create_mmd_loop()
    else:
        print("无效选择")


def extract_full_sequence():
    """提取完整表情序列"""
    print("\n=== 提取完整表情序列 ===\n")

    video_path = input("MMD 视频路径: ").strip()
    start_time = float(input("开始时间（秒，默认 0）: ") or "0")
    duration_input = input("持续时间（秒，留空=全部）: ").strip()
    end_time = float(duration_input) + start_time if duration_input else None

    skip_frames = int(input("跳帧（1=全部，2=每隔一帧，默认 1）: ") or "1")

    print("\n开始提取...")
    extractor = VideoExpressionExtractor()

    try:
        sequence, fps = extractor.extract_from_video(
            video_path,
            start_time=start_time,
            end_time=end_time,
            skip_frames=skip_frames
        )

        # 保存
        output_path = "mmd_expression_sequence.json"

        # 转换为可序列化格式
        json_data = {
            'version': '2.0',
            'format': 'Expression Sequence',
            'fps': fps,
            'num_frames': len(sequence),
            'duration': len(sequence) / fps,
            'frames': [
                {'frame': i, 'time': i / fps, 'blendshapes': frame}
                for i, frame in enumerate(sequence)
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 已保存到: {output_path}")
        print(f"  总帧数: {len(sequence)}")
        print(f"  时长: {len(sequence) / fps:.2f}s")
        print(f"  FPS: {fps}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def extract_keyframes_and_loop():
    """提取关键帧并生成循环"""
    print("\n=== 提取关键帧并生成循环 ===\n")

    video_path = input("MMD 视频路径: ").strip()
    num_keyframes = int(input("提取关键帧数量（默认 5）: ") or "5")

    print("\n选择关键帧提取方法:")
    print("1. uniform - 均匀采样")
    print("2. peaks - 表情强度峰值")
    print("3. variance - 变化率最大处")

    method_idx = int(input("选择 (1-3): ") or "2") - 1
    methods = ["uniform", "peaks", "variance"]
    method = methods[method_idx]

    print(f"\n使用 {method} 方法提取关键帧...")
    extractor = VideoExpressionExtractor()

    try:
        keyframes = extractor.extract_keyframes(
            video_path,
            num_keyframes=num_keyframes,
            method=method
        )

        if not keyframes:
            print("未能提取到关键帧")
            return

        # 显示关键帧
        print(f"\n提取到 {len(keyframes)} 个关键帧:")
        for i, (time, bs) in enumerate(keyframes):
            active_count = sum(1 for v in bs.values() if abs(v) > 0.05)
            print(f"  {i+1}. 时间 {time:.2f}s - {active_count} 个活跃参数")

        # 生成循环动画
        print("\n生成循环动画...")
        loop_duration = float(input("循环时长（秒，默认 2）: ") or "2")
        fps = float(input("帧率（默认 30）: ") or "30")

        # 创建动画
        from facial_capture.core.animation_generator import EasingType

        generator = ProceduralAnimationGenerator()

        # 添加关键帧
        for time, bs in keyframes:
            normalized_time = time / keyframes[-1][0] if keyframes[-1][0] > 0 else 0
            generator.add_keyframe(normalized_time, bs, EasingType.EASE_IN_OUT)

        # 生成
        sequence = generator.generate_animation(loop_duration, fps, LoopMode.SEAMLESS)

        # 保存
        output_path = "mmd_loop_animation.json"
        exporter = ExpressionExporter()

        from facial_capture.core import ExtendedBlendshapeData
        sequence_bs = []
        for frame in sequence:
            bs = ExtendedBlendshapeData()
            bs.from_dict_extended(frame)
            sequence_bs.append(bs)

        exporter.export_json_sequence(sequence_bs, output_path, fps=fps)

        print(f"\n✓ 已保存到: {output_path}")
        print(f"  帧数: {len(sequence)}")
        print(f"  时长: {len(sequence) / fps:.2f}s")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def analyze_mmd_patterns():
    """分析 MMD 表情模式"""
    print("\n=== 分析 MMD 表情模式 ===\n")

    video_path = input("MMD 视频路径: ").strip()

    print("\n分析中...")
    analyzer = MMDExpressionAnalyzer()

    try:
        result = analyzer.extract_mmd_patterns(video_path)

        # 显示分析结果
        analysis = result['analysis']
        mmd_patterns = result['mmd_patterns']

        print(f"\n视频信息:")
        print(f"  时长: {analysis['duration']:.2f}s")
        print(f"  帧数: {analysis['num_frames']}")
        print(f"  FPS: {analysis['fps']}")

        print(f"\n最活跃的表情参数:")
        for i, (name, data) in enumerate(analysis['top_active'][:10], 1):
            print(f"  {i}. {name:25s}: 最大值={data['max']:.3f}, "
                  f"活跃率={data['active_ratio']*100:.1f}%")

        print(f"\nMMD 特有模式:")
        print(f"  眨眼次数: {len(mmd_patterns['blinks'])}")
        print(f"  张嘴次数: {len(mmd_patterns['mouth_opens'])}")
        print(f"  微笑峰值: {len(mmd_patterns['smile_peaks'])}")

        if mmd_patterns['blinks']:
            print(f"  眨眼时刻: {', '.join(f'{t:.2f}s' for t in mmd_patterns['blinks'][:5])}")

        # 保存分析结果
        output_path = "mmd_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis': analysis,
                'mmd_patterns': mmd_patterns,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 分析结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def create_mmd_loop():
    """创建基于 MMD 的循环动画"""
    print("\n=== 创建 MMD 循环动画 ===\n")

    video_path = input("MMD 视频路径: ").strip()
    loop_duration = float(input("循环时长（秒，默认 2）: ") or "2")

    print("\n选择循环方法:")
    print("1. seamless - 找相似起止点，无缝循环")
    print("2. ping_pong - 正向+反向")
    print("3. direct - 直接循环")

    method_idx = int(input("选择 (1-3): ") or "1") - 1
    methods = ["seamless", "ping_pong", "direct"]
    method = methods[method_idx]

    print(f"\n使用 {method} 方法创建循环...")
    extractor = VideoExpressionExtractor()

    try:
        sequence = extractor.create_loop_from_video(
            video_path,
            loop_duration=loop_duration,
            method=method
        )

        if not sequence:
            print("未能创建循环")
            return

        # 保存
        output_path = "mmd_seamless_loop.json"
        fps = 30.0  # 固定 30fps

        from facial_capture.core import ExtendedBlendshapeData, ExpressionExporter

        sequence_bs = []
        for frame in sequence:
            bs = ExtendedBlendshapeData()
            bs.from_dict_extended(frame)
            sequence_bs.append(bs)

        exporter = ExpressionExporter()
        exporter.export_json_sequence(sequence_bs, output_path, fps=fps)

        print(f"\n✓ 已保存到: {output_path}")
        print(f"  帧数: {len(sequence)}")
        print(f"  时长: {len(sequence) / fps:.2f}s")
        print(f"\n这个动画可以无缝循环播放！")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
