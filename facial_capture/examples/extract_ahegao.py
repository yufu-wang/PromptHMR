#!/usr/bin/env python3
"""
示例：提取阿黑颜表情（精确的舌头和瞳孔位置）
"""

import sys
from pathlib import Path
import cv2

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture.core.enhanced_detector import EnhancedFacialDetector
from facial_capture.core.extended_blendshapes import (
    convert_detection_to_extended_blendshapes,
    ExtendedBlendshapeData
)
from facial_capture.core.exporter import ExpressionExporter


def main():
    # 输入图像路径
    image_path = "ahegao_input.jpg"  # 替换为你的图像路径
    output_path = "ahegao_expression.json"

    print("=" * 60)
    print("阿黑颜表情提取器")
    print("=" * 60)

    # 创建增强检测器
    detector = EnhancedFacialDetector()

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像 {image_path}")
        return

    print(f"\n正在分析图像: {image_path}")

    # 检测面部（包括舌头和瞳孔详细信息）
    detection = detector.detect_with_details(image)

    if detection is None:
        print("错误: 未检测到面部")
        return

    print("\n✓ 面部检测成功")

    # 显示舌头信息
    print("\n" + "=" * 60)
    print("舌头信息:")
    print("-" * 60)

    tongue_info = detection['tongue']
    if tongue_info['detected']:
        print(f"✓ 检测到舌头")
        print(f"  伸出程度: {tongue_info['extension']:.3f}")
        print(f"  可见度:   {tongue_info['visibility']:.3f}")

        if tongue_info['position']:
            pos = tongue_info['position']
            print(f"  位置:")
            print(f"    X: {pos['x']:.3f}")
            print(f"    Y: {pos['y']:.3f}")
            print(f"    Z: {pos['z']:.3f}")

        if tongue_info['direction']:
            direction = tongue_info['direction']
            print(f"  方向: ({direction[0]:.3f}, {direction[1]:.3f})")
    else:
        print("✗ 未检测到舌头")

    # 显示瞳孔信息
    print("\n" + "=" * 60)
    print("瞳孔/虹膜信息:")
    print("-" * 60)

    pupils_info = detection['pupils']

    for eye_name in ['left', 'right']:
        eye_data = pupils_info[eye_name]
        print(f"\n{eye_name.upper()} 眼:")

        if eye_data['center']:
            center = eye_data['center']
            print(f"  虹膜中心: ({center['x']:.3f}, {center['y']:.3f}, {center['z']:.3f})")

        if eye_data['position_in_eye']:
            pos = eye_data['position_in_eye']
            print(f"  瞳孔在眼中的位置:")
            print(f"    X: {pos['x']:.3f} ({'向内' if pos['x'] < 0 else '向外'})")
            print(f"    Y: {pos['y']:.3f} ({'向上' if pos['y'] < 0 else '向下'})")

        if eye_data['gaze_direction']:
            gaze = eye_data['gaze_direction']
            print(f"  注视方向: ({gaze['x']:.3f}, {gaze['y']:.3f})")

        print(f"  虹膜大小: {eye_data['iris_size']:.4f}")

    # 分析阿黑颜特征
    print("\n" + "=" * 60)
    print("阿黑颜特征分析:")
    print("-" * 60)

    ahegao_features = detector.get_ahegao_features(detection)

    print(f"是否为阿黑颜: {'✓ 是' if ahegao_features['is_ahegao'] else '✗ 否'}")
    print(f"置信度: {ahegao_features['confidence']:.3f}")
    print(f"\n特征分数:")
    print(f"  舌头伸出: {ahegao_features['tongue_out']:.3f}")
    print(f"  眼睛翻白: {ahegao_features['eyes_rolled']:.3f}")
    print(f"  嘴巴张开: {ahegao_features['mouth_open']:.3f}")

    # 转换为扩展 blendshapes
    print("\n" + "=" * 60)
    print("转换为扩展 Blendshapes:")
    print("-" * 60)

    blendshapes = convert_detection_to_extended_blendshapes(detection)

    # 显示舌头 blendshapes
    tongue_bs = blendshapes.get_tongue_info()
    print("\n舌头 Blendshapes:")
    for key, value in tongue_bs.items():
        if abs(value) > 0.01:
            print(f"  {key:15s}: {value:.3f}")

    # 显示瞳孔 blendshapes
    print("\n左眼瞳孔 Blendshapes:")
    left_pupil = blendshapes.get_pupil_info('left')
    for key, value in left_pupil.items():
        if abs(value) > 0.01:
            print(f"  {key:15s}: {value:.3f}")

    print("\n右眼瞳孔 Blendshapes:")
    right_pupil = blendshapes.get_pupil_info('right')
    for key, value in right_pupil.items():
        if abs(value) > 0.01:
            print(f"  {key:15s}: {value:.3f}")

    # 显示主要的 ARKit blendshapes
    print("\n主要 ARKit Blendshapes:")
    all_bs = blendshapes.to_dict()
    sorted_bs = sorted(all_bs.items(), key=lambda x: abs(x[1]), reverse=True)

    for name, value in sorted_bs[:15]:
        if abs(value) > 0.05:
            print(f"  {name:25s}: {value:.3f}")

    # 导出为 JSON
    print("\n" + "=" * 60)
    print("导出结果:")
    print("-" * 60)

    exporter = ExpressionExporter()

    # 导出扩展格式
    import json
    extended_data = {
        'version': '2.0',
        'format': 'ARKit Extended',
        'blendshapes': blendshapes.to_dict_extended(),
        'ahegao_features': ahegao_features,
        'tongue_detail': tongue_info,
        'pupil_detail': pupils_info,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 已保存到: {output_path}")

    # 可视化
    print("\n显示可视化结果（按任意键关闭）...")
    vis_image = detector.visualize_enhanced(image, detection)

    # 添加阿黑颜标记
    if ahegao_features['is_ahegao']:
        h, w = vis_image.shape[:2]
        cv2.putText(vis_image, "AHEGAO DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(vis_image, f"Confidence: {ahegao_features['confidence']:.2f}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Ahegao Expression Analysis', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
