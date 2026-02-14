#!/usr/bin/env python3
"""
对比标准 ARKit 和扩展参数的效果
展示有无扩展 Shape Keys 的区别
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture.core import EnhancedFacialDetector
from facial_capture.core import convert_detection_to_extended_blendshapes
import cv2


def main():
    image_path = input("输入图片路径: ")

    # 检测
    detector = EnhancedFacialDetector()
    image = cv2.imread(image_path)

    if image is None:
        print("无法加载图像")
        return

    detection = detector.detect_with_details(image)

    if not detection:
        print("未检测到面部")
        return

    # 转换
    blendshapes = convert_detection_to_extended_blendshapes(detection)

    print("\n" + "=" * 70)
    print("参数对比：标准 ARKit vs 扩展参数")
    print("=" * 70)

    # 舌头对比
    print("\n【舌头控制】")
    print("-" * 70)

    print("\n✅ 标准 ARKit (所有模型都有):")
    standard_tongue = blendshapes.get('tongueOut')
    print(f"  tongueOut: {standard_tongue:.3f}")
    print(f"  → 效果: 舌头伸出 {standard_tongue*100:.0f}%（简单的开/关）")

    print("\n⭐ 扩展参数 (需要在模型上创建):")
    tongue_info = blendshapes.get_tongue_info()
    print(f"  tongueOutExtension: {tongue_info['extension']:.3f}")
    print(f"  tongueUpDown:       {tongue_info['y']:.3f}")
    print(f"  tongueLeftRight:    {tongue_info['x']:.3f}")
    print(f"  → 效果: 舌头伸出 {tongue_info['extension']*100:.0f}%")
    print(f"           位置 X={tongue_info['x']:.2f}, Y={tongue_info['y']:.2f}")
    print(f"           (3D 精确定位！)")

    # 瞳孔对比
    print("\n【瞳孔控制】")
    print("-" * 70)

    print("\n✅ 标准 ARKit (所有模型都有):")
    print(f"  eyeLookUpLeft:   {blendshapes.get('eyeLookUpLeft'):.3f}")
    print(f"  eyeLookDownLeft: {blendshapes.get('eyeLookDownLeft'):.3f}")
    print(f"  eyeLookInLeft:   {blendshapes.get('eyeLookInLeft'):.3f}")
    print(f"  eyeLookOutLeft:  {blendshapes.get('eyeLookOutLeft'):.3f}")
    print(f"  → 效果: 4 个方向的粗略控制")

    print("\n⭐ 扩展参数 (需要在模型上创建或用骨骼):")
    left_pupil = blendshapes.get_pupil_info('left')
    print(f"  leftPupilPositionX: {left_pupil['x']:.3f}")
    print(f"  leftPupilPositionY: {left_pupil['y']:.3f}")
    print(f"  → 效果: 连续的 2D 位置 (X={left_pupil['x']:.2f}, Y={left_pupil['y']:.2f})")
    print(f"           (像素级精确！)")

    # 建议
    print("\n【建议】")
    print("-" * 70)

    has_tongue = tongue_info['extension'] > 0.1
    has_extreme_gaze = abs(left_pupil['y']) > 0.5

    if has_tongue and has_extreme_gaze:
        print("✅ 检测到极端表情（可能是阿黑颜）")
        print("   推荐: 创建扩展参数以获得最佳效果")
    elif has_tongue:
        print("✅ 检测到舌头伸出")
        print("   可选: 创建 tongueOutExtension 等参数")
        print("   不创建: 使用标准 tongueOut 也可以")
    else:
        print("✅ 普通表情")
        print("   标准 ARKit 52 个参数已足够")

    # 导出对比
    print("\n【导出文件】")
    print("-" * 70)

    # 仅标准参数
    standard_only = blendshapes.to_dict()
    with open('standard_only.json', 'w') as f:
        json.dump({
            'version': '1.0',
            'format': 'ARKit Standard',
            'blendshapes': standard_only,
            'note': '适用于所有标准 ARKit 模型'
        }, f, indent=2)
    print("✅ 已保存: standard_only.json (52 个标准参数)")

    # 完整参数
    extended_full = blendshapes.to_dict_extended()
    with open('extended_full.json', 'w') as f:
        json.dump({
            'version': '2.0',
            'format': 'ARKit Extended',
            'blendshapes': extended_full,
            'tongue_detail': detection['tongue'],
            'pupil_detail': detection['pupils'],
            'note': '需要模型支持扩展参数'
        }, f, indent=2)
    print("✅ 已保存: extended_full.json (70 个完整参数)")

    print("\n" + "=" * 70)
    print("对比完成！")
    print("=" * 70)

    print("\n使用建议:")
    print("1. 先用 standard_only.json 测试（适配所有模型）")
    print("2. 如需更精确效果，再创建扩展 Shape Keys")
    print("3. 然后使用 extended_full.json 获得完整效果")


if __name__ == "__main__":
    main()
