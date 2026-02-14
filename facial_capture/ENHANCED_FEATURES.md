# 增强功能 - 精确舌头和瞳孔定位

## 概述

在标准 ARKit 52 个 blendshapes 基础上，添加了**精确的舌头位置**和**瞳孔位置**检测，特别优化了**阿黑颜（Ahegao）**表情的捕捉。

## 新增功能

### 1. 精确舌头检测

能够检测舌头的：
- **伸出程度** (0-1): 舌头伸出嘴外的长度
- **3D 位置** (x, y, z): 舌尖的精确位置
- **方向向量**: 舌头伸出的方向
- **可见度**: 舌头的可见程度

#### 舌头 Blendshapes (6个)

| 名称 | 范围 | 说明 |
|------|------|------|
| `tongueOutExtension` | 0-1 | 舌头伸出程度 |
| `tongueUpDown` | -1到1 | 舌头上下位置 |
| `tongueLeftRight` | -1到1 | 舌头左右位置 |
| `tongueCurl` | 0-1 | 舌头卷曲 |
| `tongueWidth` | 0-1 | 舌头宽度 |
| `tongueThickness` | 0-1 | 舌头厚度 |

### 2. 精确瞳孔/虹膜检测

能够检测：
- **虹膜中心** (x, y, z): 5个关键点的虹膜位置
- **瞳孔在眼中的位置** (-1到1): 相对于眼睛边界的归一化位置
- **注视方向向量**: 眼球的精确注视方向
- **虹膜大小**: 瞳孔/虹膜的直径

#### 瞳孔 Blendshapes (每只眼 4个，共 8个)

| 名称 | 范围 | 说明 |
|------|------|------|
| `leftPupilPositionX` | -1到1 | 左瞳孔水平位置 (-1=向内, 1=向外) |
| `leftPupilPositionY` | -1到1 | 左瞳孔垂直位置 (-1=向上, 1=向下) |
| `leftPupilDilation` | 0-1 | 左瞳孔扩张 |
| `leftIrisRotation` | 0-1 | 左虹膜旋转 |
| `rightPupilPositionX` | -1到1 | 右瞳孔水平位置 |
| `rightPupilPositionY` | -1到1 | 右瞳孔垂直位置 |
| `rightPupilDilation` | 0-1 | 右瞳孔扩张 |
| `rightIrisRotation` | 0-1 | 右虹膜旋转 |

### 3. 阿黑颜表情识别

专门优化的阿黑颜表情检测，包括：

#### 阿黑颜特效 Blendshapes (4个)

| 名称 | 范围 | 说明 |
|------|------|------|
| `ahegaoIntensity` | 0-1 | 阿黑颜整体强度 |
| `eyeRollIntensity` | 0-1 | 眼睛翻白强度 |
| `droolEffect` | 0-1 | 流口水效果 |
| `heartPupils` | 0-1 | 心形瞳孔效果 |

#### 阿黑颜特征分析

自动分析以下特征：
- **舌头伸出**: 检测舌头是否伸出嘴外
- **眼睛翻白**: 检测瞳孔是否向上翻
- **嘴巴张开**: 检测嘴巴张开程度
- **整体置信度**: 综合判断是否为阿黑颜表情

## 使用方法

### Python API

#### 基础使用

```python
from facial_capture.core import EnhancedFacialDetector, convert_detection_to_extended_blendshapes

# 创建增强检测器
detector = EnhancedFacialDetector()

# 检测（包含舌头和瞳孔详情）
detection = detector.detect_with_details(image)

# 转换为扩展 blendshapes
blendshapes = convert_detection_to_extended_blendshapes(detection)
```

#### 舌头信息

```python
# 获取舌头详细信息
tongue_info = detection['tongue']

if tongue_info['detected']:
    print(f"舌头伸出程度: {tongue_info['extension']:.2f}")
    print(f"舌头位置: {tongue_info['position']}")
    print(f"舌头方向: {tongue_info['direction']}")

# 从 blendshapes 获取
tongue_bs = blendshapes.get_tongue_info()
print(f"舌头 X 位置: {tongue_bs['x']:.2f}")
print(f"舌头 Y 位置: {tongue_bs['y']:.2f}")
```

#### 瞳孔信息

```python
# 获取瞳孔详细信息
pupils = detection['pupils']

left_eye = pupils['left']
if left_eye['position_in_eye']:
    pos = left_eye['position_in_eye']
    print(f"左瞳孔位置: X={pos['x']:.2f}, Y={pos['y']:.2f}")
    print(f"注视方向: {left_eye['gaze_direction']}")

# 从 blendshapes 获取
left_pupil = blendshapes.get_pupil_info('left')
print(f"左瞳孔 X: {left_pupil['x']:.2f}")
print(f"左瞳孔 Y: {left_pupil['y']:.2f}")
```

#### 阿黑颜表情

```python
# 分析阿黑颜特征
ahegao = detector.get_ahegao_features(detection)

if ahegao['is_ahegao']:
    print(f"检测到阿黑颜表情！置信度: {ahegao['confidence']:.2f}")
    print(f"舌头伸出: {ahegao['tongue_out']:.2f}")
    print(f"眼睛翻白: {ahegao['eyes_rolled']:.2f}")

# 或手动设置阿黑颜表情
blendshapes.set_ahegao_expression(intensity=0.8)
```

#### 可视化

```python
# 增强可视化（显示舌头和瞳孔标记）
vis_image = detector.visualize_enhanced(image, detection)
cv2.imshow('Enhanced Detection', vis_image)
```

### 完整示例

```python
import cv2
from facial_capture.core import EnhancedFacialDetector, ExtendedBlendshapeData
from facial_capture.core import convert_detection_to_extended_blendshapes

# 创建检测器
detector = EnhancedFacialDetector()

# 加载图像
image = cv2.imread("ahegao_face.jpg")

# 检测
detection = detector.detect_with_details(image)

if detection:
    # 转换为 blendshapes
    blendshapes = convert_detection_to_extended_blendshapes(detection)

    # 获取所有数据（ARKit 52 + 扩展 18 = 70个参数）
    all_data = blendshapes.to_dict_extended()

    # 导出
    import json
    with open("output.json", 'w') as f:
        json.dump({
            'blendshapes': all_data,
            'tongue_detail': detection['tongue'],
            'pupil_detail': detection['pupils'],
        }, f, indent=2)

    # 可视化
    vis = detector.visualize_enhanced(image, detection)
    cv2.imshow('Result', vis)
    cv2.waitKey(0)
```

## 实时检测（摄像头）

```python
import cv2
from facial_capture.core import EnhancedFacialDetector

detector = EnhancedFacialDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detection = detector.detect_with_details(frame)

    if detection:
        # 可视化
        vis = detector.visualize_enhanced(frame, detection)

        # 显示舌头信息
        tongue = detection['tongue']
        if tongue['detected']:
            cv2.putText(vis, f"Tongue: {tongue['extension']:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 显示瞳孔信息
        left_pos = detection['pupils']['left'].get('position_in_eye')
        if left_pos:
            cv2.putText(vis, f"Left Pupil: ({left_pos['x']:.2f}, {left_pos['y']:.2f})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Live Detection', vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 导出格式

扩展的 JSON 格式包含：

```json
{
  "version": "2.0",
  "format": "ARKit Extended",
  "blendshapes": {
    // ARKit 标准 52 个
    "eyeBlinkLeft": 0.15,
    "mouthSmileLeft": 0.45,
    // ... 其他 ARKit blendshapes

    // 扩展的舌头参数
    "tongueOutExtension": 0.75,
    "tongueUpDown": 0.2,
    "tongueLeftRight": -0.1,

    // 扩展的瞳孔参数
    "leftPupilPositionX": -0.3,
    "leftPupilPositionY": -0.8,
    "rightPupilPositionX": 0.2,
    "rightPupilPositionY": -0.7,

    // 阿黑颜特效
    "ahegaoIntensity": 0.85,
    "eyeRollIntensity": 0.9
  },
  "tongue_detail": {
    "detected": true,
    "extension": 0.75,
    "position": {"x": 0.51, "y": 0.68, "z": 0.02},
    "direction": [0.05, 0.99]
  },
  "pupil_detail": {
    "left": {
      "center": {"x": 0.35, "y": 0.42, "z": -0.01},
      "position_in_eye": {"x": -0.3, "y": -0.8},
      "gaze_direction": {"x": -0.25, "y": -0.95}
    },
    "right": { /* ... */ }
  }
}
```

## Blender / Unity 使用

### Blender

扩展的 blendshapes 会自动导出到 JSON 文件，可以在 Blender 中：

1. 手动创建对应的 Shape Keys（如 `tongueOutExtension`）
2. 使用插件导入 JSON 时会自动映射
3. 如果没有对应的 Shape Key，扩展参数会被忽略

### Unity

Unity 插件会自动识别扩展参数，只需：

1. 确保模型有对应的 Blendshape 名称
2. 导入扩展 JSON 格式
3. 系统会自动应用所有可用的 blendshapes

## 技术细节

### MediaPipe 关键点

使用 MediaPipe Face Mesh 的 478 个关键点：
- **舌头**: 关键点 10, 13, 12, 14
- **虹膜**: 关键点 468-477 (10个虹膜点)
- **眼睛**: 完整的眼睛轮廓用于计算相对位置

### 坐标系统

- **X 轴**: -1 (左/内) 到 1 (右/外)
- **Y 轴**: -1 (上) 到 1 (下)
- **Z 轴**: 深度（负值=向后，正值=向前）

所有位置都是归一化的相对坐标。

## 示例脚本

- `examples/extract_ahegao.py` - 阿黑颜表情提取
- `examples/test_tongue_pupil.py` - 舌头和瞳孔测试（支持摄像头）

## 注意事项

1. **舌头检测精度**: 依赖面部角度和光线，正面效果最好
2. **瞳孔检测**: 需要清晰的眼部图像，虹膜必须可见
3. **阿黑颜识别**: 自动识别有一定误差，可手动调整参数
4. **性能**: 增强检测比基础检测慢约 10-20%

## 总结

扩展功能提供：
- **ARKit 52** + **扩展 18** = **共 70 个 blendshape 参数**
- 精确的舌头 3D 位置和方向
- 精确的瞳孔在眼中的相对位置
- 专门的阿黑颜表情识别和生成
- 完整的可视化和导出支持
