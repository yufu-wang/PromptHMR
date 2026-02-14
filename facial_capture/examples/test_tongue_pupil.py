#!/usr/bin/env python3
"""
示例：测试舌头和瞳孔的精确定位
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from facial_capture.core.enhanced_detector import EnhancedFacialDetector


def draw_coordinate_system(image, origin, scale=50):
    """绘制坐标系"""
    ox, oy = int(origin[0]), int(origin[1])

    # X 轴 (红色)
    cv2.arrowedLine(image, (ox, oy), (ox + scale, oy),
                   (0, 0, 255), 2, tipLength=0.2)
    cv2.putText(image, 'X', (ox + scale + 5, oy + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Y 轴 (绿色)
    cv2.arrowedLine(image, (ox, oy), (ox, oy + scale),
                   (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(image, 'Y', (ox + 5, oy + scale + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def create_position_grid(image, pupils_info):
    """创建瞳孔位置网格可视化"""
    h, w = image.shape[:2]
    grid = np.zeros((400, 800, 3), dtype=np.uint8)

    # 绘制左眼网格
    left_center = (200, 200)
    cv2.circle(grid, left_center, 100, (100, 100, 100), 2)
    cv2.circle(grid, left_center, 5, (255, 255, 255), -1)
    cv2.putText(grid, 'LEFT EYE', (120, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 绘制右眼网格
    right_center = (600, 200)
    cv2.circle(grid, right_center, 100, (100, 100, 100), 2)
    cv2.circle(grid, right_center, 5, (255, 255, 255), -1)
    cv2.putText(grid, 'RIGHT EYE', (520, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 绘制十字线
    for center in [left_center, right_center]:
        cv2.line(grid, (center[0] - 100, center[1]), (center[0] + 100, center[1]),
                (80, 80, 80), 1)
        cv2.line(grid, (center[0], center[1] - 100), (center[0], center[1] + 100),
                (80, 80, 80), 1)

    # 绘制瞳孔位置
    left_eye = pupils_info.get('left', {})
    if left_eye.get('position_in_eye'):
        pos = left_eye['position_in_eye']
        px = int(left_center[0] + pos['x'] * 100)
        py = int(left_center[1] + pos['y'] * 100)
        cv2.circle(grid, (px, py), 10, (0, 255, 255), -1)
        cv2.putText(grid, f"({pos['x']:.2f}, {pos['y']:.2f})",
                   (px - 50, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    right_eye = pupils_info.get('right', {})
    if right_eye.get('position_in_eye'):
        pos = right_eye['position_in_eye']
        px = int(right_center[0] + pos['x'] * 100)
        py = int(right_center[1] + pos['y'] * 100)
        cv2.circle(grid, (px, py), 10, (0, 255, 255), -1)
        cv2.putText(grid, f"({pos['x']:.2f}, {pos['y']:.2f})",
                   (px - 50, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return grid


def main():
    print("舌头和瞳孔精确定位测试")
    print("=" * 60)

    # 使用摄像头或图像文件
    use_webcam = input("使用摄像头? (y/n): ").lower() == 'y'

    detector = EnhancedFacialDetector()

    if use_webcam:
        print("\n打开摄像头...")
        print("提示：")
        print("- 伸出舌头测试舌头检测")
        print("- 转动眼球测试瞳孔检测")
        print("- 按 'q' 退出")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测
            detection = detector.detect_with_details(frame)

            if detection:
                # 可视化
                vis_frame = detector.visualize_enhanced(frame, detection)

                # 创建瞳孔位置网格
                pupil_grid = create_position_grid(frame, detection['pupils'])

                # 显示舌头信息
                tongue = detection['tongue']
                y_offset = 30
                if tongue['detected']:
                    cv2.putText(vis_frame, f"Tongue Extension: {tongue['extension']:.2f}",
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 30

                # 显示瞳孔信息
                for eye_name, eye_data in [('Left', detection['pupils']['left']),
                                           ('Right', detection['pupils']['right'])]:
                    if eye_data.get('position_in_eye'):
                        pos = eye_data['position_in_eye']
                        text = f"{eye_name} Pupil: ({pos['x']:.2f}, {pos['y']:.2f})"
                        cv2.putText(vis_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 30

                # 显示结果
                cv2.imshow('Face Detection', vis_frame)
                cv2.imshow('Pupil Position Grid', pupil_grid)
            else:
                cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # 从文件读取
        image_path = input("输入图像路径: ")
        image = cv2.imread(image_path)

        if image is None:
            print(f"错误: 无法加载 {image_path}")
            return

        print("\n分析中...")
        detection = detector.detect_with_details(image)

        if detection is None:
            print("错误: 未检测到面部")
            return

        # 显示详细信息
        print("\n检测结果:")
        print("-" * 60)

        # 舌头
        tongue = detection['tongue']
        print(f"\n舌头:")
        print(f"  检测到: {tongue['detected']}")
        if tongue['detected']:
            print(f"  伸出程度: {tongue['extension']:.3f}")
            if tongue['position']:
                print(f"  位置: X={tongue['position']['x']:.3f}, "
                      f"Y={tongue['position']['y']:.3f}, "
                      f"Z={tongue['position']['z']:.3f}")

        # 瞳孔
        pupils = detection['pupils']
        for eye_name in ['left', 'right']:
            eye_data = pupils[eye_name]
            print(f"\n{eye_name.upper()} 瞳孔:")
            if eye_data['center']:
                print(f"  中心: X={eye_data['center']['x']:.3f}, "
                      f"Y={eye_data['center']['y']:.3f}")
            if eye_data['position_in_eye']:
                pos = eye_data['position_in_eye']
                print(f"  眼内位置: X={pos['x']:.3f} ({['向内', '中央', '向外'][int(pos['x'] * 1.5 + 1.5)]}), "
                      f"Y={pos['y']:.3f} ({['向上', '中央', '向下'][int(pos['y'] * 1.5 + 1.5)]})")
            if eye_data['gaze_direction']:
                gaze = eye_data['gaze_direction']
                print(f"  注视方向: ({gaze['x']:.3f}, {gaze['y']:.3f})")

        # 可视化
        vis_image = detector.visualize_enhanced(image, detection)
        pupil_grid = create_position_grid(image, detection['pupils'])

        # 并排显示
        cv2.imshow('Detection Result', vis_image)
        cv2.imshow('Pupil Position Grid', pupil_grid)

        print("\n按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
