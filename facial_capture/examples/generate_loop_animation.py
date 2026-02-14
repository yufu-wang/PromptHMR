#!/usr/bin/env python3
"""
生成两个 R18 表情之间的循环动画
支持多种缓动效果和循环模式
"""

import sys
from pathlib import Path
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture.core import EnhancedFacialDetector, convert_detection_to_extended_blendshapes
from facial_capture.core.animation_generator import (
    ProceduralAnimationGenerator,
    EasingType,
    LoopMode,
    R18AnimationPresets
)
from facial_capture.core import ExpressionExporter


def main():
    print("=" * 70)
    print("R18 表情循环动画生成器")
    print("=" * 70)

    # 选择模式
    print("\n选择模式:")
    print("1. 从两张图片生成循环动画")
    print("2. 使用预设的阿黑颜动画")
    print("3. 自定义多阶段动画")

    mode = input("\n输入选择 (1-3): ").strip()

    if mode == "1":
        generate_from_two_images()
    elif mode == "2":
        generate_ahegao_preset()
    elif mode == "3":
        generate_custom_animation()
    else:
        print("无效选择")


def generate_from_two_images():
    """从两张图片生成循环动画"""
    print("\n=== 从两张图片生成循环动画 ===\n")

    # 输入
    image_a_path = input("第一张图片路径（起始表情）: ").strip()
    image_b_path = input("第二张图片路径（结束表情）: ").strip()

    # 参数
    duration = float(input("动画时长（秒，默认 2.0）: ") or "2.0")
    fps = float(input("帧率（默认 30）: ") or "30")

    # 缓动类型
    print("\n选择缓动类型:")
    for i, easing in enumerate(EasingType, 1):
        print(f"{i}. {easing.value}")

    easing_idx = int(input("选择 (1-11): ") or "4") - 1
    easing_type = list(EasingType)[easing_idx]

    # 循环模式
    print("\n选择循环模式:")
    print("1. PING_PONG - 来回循环 (A→B→A)")
    print("2. SEAMLESS - 无缝循环 (使用正弦平滑)")
    print("3. LOOP - 直接循环 (A→B→A 瞬间跳回)")
    print("4. HOLD - 单次过渡 (A→B 停止)")

    loop_idx = int(input("选择 (1-4): ") or "1") - 1
    loop_mode = list(LoopMode)[loop_idx]

    # 检测表情
    print("\n检测表情...")
    detector = EnhancedFacialDetector()

    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)

    if image_a is None or image_b is None:
        print("错误: 无法加载图像")
        return

    detection_a = detector.detect_with_details(image_a)
    detection_b = detector.detect_with_details(image_b)

    if not detection_a or not detection_b:
        print("错误: 未检测到面部")
        return

    # 转换为 blendshapes
    bs_a = convert_detection_to_extended_blendshapes(detection_a)
    bs_b = convert_detection_to_extended_blendshapes(detection_b)

    print(f"✓ 表情 A: {sum(1 for v in bs_a.to_dict_extended().values() if abs(v) > 0.05)} 个活跃参数")
    print(f"✓ 表情 B: {sum(1 for v in bs_b.to_dict_extended().values() if abs(v) > 0.05)} 个活跃参数")

    # 生成动画
    print(f"\n生成动画... (时长: {duration}s, FPS: {fps}, 缓动: {easing_type.value}, 循环: {loop_mode.value})")

    generator = ProceduralAnimationGenerator()
    sequence = generator.create_simple_transition(
        bs_a.to_dict_extended(),
        bs_b.to_dict_extended(),
        duration=duration,
        fps=fps,
        easing=easing_type,
        loop_mode=loop_mode
    )

    print(f"✓ 生成了 {len(sequence)} 帧")

    # 导出
    output_path = "loop_animation.json"
    exporter = ExpressionExporter()
    exporter.export_json_sequence(
        [convert_dict_to_blendshape_data(frame) for frame in sequence],
        output_path,
        fps=fps
    )

    print(f"\n✓ 已保存到: {output_path}")

    # 额外格式
    print("\n导出其他格式...")
    exporter.export_blender_action(
        [convert_dict_to_blendshape_data(frame) for frame in sequence],
        "loop_animation_blender.json",
        fps=fps
    )
    print("✓ Blender: loop_animation_blender.json")

    exporter.export_unity_animation(
        [convert_dict_to_blendshape_data(frame) for frame in sequence],
        "loop_animation_unity.json",
        fps=fps
    )
    print("✓ Unity: loop_animation_unity.json")

    print("\n完成！可以在 Blender/Unity 中导入这些文件。")


def generate_ahegao_preset():
    """生成预设的阿黑颜动画"""
    print("\n=== 生成阿黑颜渐进动画 ===\n")

    intensity = float(input("强度（0-1，默认 0.8）: ") or "0.8")
    duration = float(input("动画时长（秒，默认 4.0）: ") or "4.0")
    fps = float(input("帧率（默认 30）: ") or "30")

    # 获取预设关键帧
    keyframes = R18AnimationPresets.ahegao_transition(intensity)

    print(f"\n使用 {len(keyframes)} 个关键帧:")
    for i, (time, bs, easing) in enumerate(keyframes):
        active_count = sum(1 for v in bs.values() if abs(v) > 0.05)
        print(f"  {i+1}. 时间 {time:.2f} - {active_count} 个活跃参数 - 缓动: {easing.value}")

    # 生成动画
    print(f"\n生成动画...")
    generator = ProceduralAnimationGenerator()
    sequence = generator.create_multi_stage_animation(
        keyframes,
        duration=duration,
        fps=fps,
        loop_mode=LoopMode.LOOP
    )

    print(f"✓ 生成了 {len(sequence)} 帧")

    # 添加微小变化（更自然）
    print("添加程序化变化...")
    sequence = generator.add_procedural_variations(
        sequence,
        noise_amount=0.02,
        target_blendshapes=['eyeLookUpLeft', 'eyeLookUpRight', 'tongueOut']
    )

    # 导出
    output_path = "ahegao_animation.json"
    exporter = ExpressionExporter()

    from facial_capture.core import ExtendedBlendshapeData
    sequence_bs = []
    for frame in sequence:
        bs = ExtendedBlendshapeData()
        bs.from_dict_extended(frame)
        sequence_bs.append(bs)

    exporter.export_json_sequence(sequence_bs, output_path, fps=fps)

    print(f"\n✓ 已保存到: {output_path}")
    print("\n提示: 这个动画包含舌头伸出、眼睛翻白等阿黑颜特征")


def generate_custom_animation():
    """生成自定义多阶段动画"""
    print("\n=== 自定义多阶段动画 ===\n")

    print("输入关键帧（输入 'done' 结束）")
    print("格式: 时间(0-1) 图片路径 缓动类型(1-11)")
    print("示例: 0.0 neutral.jpg 4")
    print()

    keyframes = []
    detector = EnhancedFacialDetector()

    while True:
        line = input(f"关键帧 {len(keyframes)+1}: ").strip()

        if line.lower() == 'done':
            break

        try:
            parts = line.split()
            time = float(parts[0])
            image_path = parts[1]
            easing_idx = int(parts[2]) - 1

            # 检测表情
            image = cv2.imread(image_path)
            if image is None:
                print(f"  错误: 无法加载 {image_path}")
                continue

            detection = detector.detect_with_details(image)
            if not detection:
                print(f"  错误: 未检测到面部")
                continue

            bs = convert_detection_to_extended_blendshapes(detection)
            easing = list(EasingType)[easing_idx]

            keyframes.append((time, bs.to_dict_extended(), easing))
            print(f"  ✓ 添加关键帧: 时间={time}, 缓动={easing.value}")

        except Exception as e:
            print(f"  错误: {e}")

    if len(keyframes) < 2:
        print("至少需要 2 个关键帧")
        return

    # 生成
    duration = float(input("\n动画时长（秒）: "))
    fps = float(input("帧率: ") or "30")

    generator = ProceduralAnimationGenerator()
    sequence = generator.create_multi_stage_animation(
        keyframes,
        duration=duration,
        fps=fps,
        loop_mode=LoopMode.LOOP
    )

    # 导出
    output_path = "custom_animation.json"
    exporter = ExpressionExporter()

    from facial_capture.core import ExtendedBlendshapeData
    sequence_bs = []
    for frame in sequence:
        bs = ExtendedBlendshapeData()
        bs.from_dict_extended(frame)
        sequence_bs.append(bs)

    exporter.export_json_sequence(sequence_bs, output_path, fps=fps)

    print(f"\n✓ 已保存到: {output_path}")


def convert_dict_to_blendshape_data(data_dict):
    """转换字典为 BlendshapeData"""
    from facial_capture.core import ExtendedBlendshapeData
    bs = ExtendedBlendshapeData()
    bs.from_dict_extended(data_dict)
    return bs


if __name__ == "__main__":
    main()
