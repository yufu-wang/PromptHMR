# Facial Expression Capture Plugin

面部表情捕捉插件 - 从 2D 图像（包括漫画、插图）中提取面部表情并映射到 3D 模型

## 功能特性

- 🎭 **表情提取**: 从 2D 图像中提取面部表情（支持真人和动漫风格）
- 🎯 **ARKit 兼容**: 输出标准 ARKit 52 个 blendshape 参数
- 👅 **精确舌头检测**: 3D 位置、伸出程度、方向（专为阿黑颜优化）
- 👁️ **精确瞳孔定位**: 虹膜在眼中的相对位置、注视方向
- 🎨 **扩展参数**: 额外 18 个 blendshapes（舌头 6 + 瞳孔 8 + 特效 4）
- 😵 **阿黑颜识别**: 自动识别和生成阿黑颜表情
- 🔧 **插件支持**: 提供 Blender 和 Unity 插件
- 📦 **批量处理**: 支持单张图像或序列图像处理

## 架构设计

```
输入图像 (R18表情图)
    ↓
面部检测 & 关键点提取 (MediaPipe/OpenCV)
    ↓
表情特征分析
    ↓
ARKit Blendshape 映射 (52个参数)
    ↓
导出格式:
  - JSON (通用)
  - FBX (Blender/Unity)
  - Alembic (动画缓存)
```

## Blendshape 参数

### 标准 ARKit (52个)

支持的 52 个标准 ARKit 表情单元：

### 眼部 (16个)
- eyeBlinkLeft, eyeBlinkRight
- eyeLookDownLeft, eyeLookDownRight
- eyeLookInLeft, eyeLookInRight
- eyeLookOutLeft, eyeLookOutRight
- eyeLookUpLeft, eyeLookUpRight
- eyeSquintLeft, eyeSquintRight
- eyeWideLeft, eyeWideRight

### 眉毛 (8个)
- browDownLeft, browDownRight
- browInnerUp
- browOuterUpLeft, browOuterUpRight

### 嘴部 (28个)
- mouthClose
- mouthFunnel, mouthPucker
- mouthLeft, mouthRight
- mouthSmileLeft, mouthSmileRight
- mouthFrownLeft, mouthFrownRight
- mouthDimpleLeft, mouthDimpleRight
- mouthStretchLeft, mouthStretchRight
- mouthRollLower, mouthRollUpper
- mouthShrugLower, mouthShrugUpper
- mouthPressLeft, mouthPressRight
- mouthLowerDownLeft, mouthLowerDownRight
- mouthUpperUpLeft, mouthUpperUpRight

### 脸颊和下巴 (6个)
- cheekPuff
- cheekSquintLeft, cheekSquintRight
- jawOpen, jawForward
- jawLeft, jawRight

### 扩展参数 (18个) ⭐ 新增

#### 舌头控制 (6个)
- tongueOutExtension - 舌头伸出程度 (0-1)
- tongueUpDown - 舌头上下位置 (-1到1)
- tongueLeftRight - 舌头左右位置 (-1到1)
- tongueCurl - 舌头卷曲
- tongueWidth - 舌头宽度
- tongueThickness - 舌头厚度

#### 瞳孔精确控制 (8个)
- leftPupilPositionX/Y - 左瞳孔位置 (-1到1)
- rightPupilPositionX/Y - 右瞳孔位置 (-1到1)
- leftPupilDilation - 左瞳孔扩张
- rightPupilDilation - 右瞳孔扩张
- leftIrisRotation - 左虹膜旋转
- rightIrisRotation - 右虹膜旋转

#### 阿黑颜特效 (4个)
- ahegaoIntensity - 阿黑颜整体强度
- eyeRollIntensity - 眼睛翻白强度
- droolEffect - 流口水效果
- heartPupils - 心形瞳孔效果

**总计: ARKit 52 + 扩展 18 = 70 个 blendshape 参数**

> 📖 详细文档请查看 [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md)

## 安装

```bash
cd /home/user/PromptHMR/facial_capture
pip install -r requirements.txt
```

## 使用方法

### Python API

#### 基础使用

```python
from facial_capture import FacialExpressionCapture

# 初始化
capturer = FacialExpressionCapture()

# 从图像提取表情
blendshapes = capturer.extract_from_image("input.jpg")

# 导出为 ARKit 格式
capturer.export_arkit("output.json", blendshapes)
```

#### 增强功能 - 精确舌头和瞳孔检测 ⭐

```python
from facial_capture.core import EnhancedFacialDetector, convert_detection_to_extended_blendshapes

# 创建增强检测器
detector = EnhancedFacialDetector()

# 检测（包含舌头和瞳孔详情）
detection = detector.detect_with_details(image)

# 查看舌头信息
if detection['tongue']['detected']:
    print(f"舌头伸出: {detection['tongue']['extension']:.2f}")
    print(f"舌头位置: {detection['tongue']['position']}")

# 查看瞳孔信息
left_pupil = detection['pupils']['left']['position_in_eye']
print(f"左瞳孔位置: X={left_pupil['x']:.2f}, Y={left_pupil['y']:.2f}")

# 转换为扩展 blendshapes (70个参数)
blendshapes = convert_detection_to_extended_blendshapes(detection)

# 获取舌头和瞳孔 blendshapes
tongue_info = blendshapes.get_tongue_info()
pupil_info = blendshapes.get_pupil_info('left')

# 可视化（显示舌头和瞳孔标记）
vis = detector.visualize_enhanced(image, detection)
```

#### 阿黑颜表情识别

```python
# 分析阿黑颜特征
ahegao = detector.get_ahegao_features(detection)

if ahegao['is_ahegao']:
    print(f"检测到阿黑颜！置信度: {ahegao['confidence']:.2f}")
    print(f"舌头伸出: {ahegao['tongue_out']:.2f}")
    print(f"眼睛翻白: {ahegao['eyes_rolled']:.2f}")

# 手动设置阿黑颜表情
blendshapes.set_ahegao_expression(intensity=0.8)
```

### Blender 插件

1. 打开 Blender
2. Edit > Preferences > Add-ons > Install
3. 选择 `blender_addon/facial_capture_addon.py`
4. 启用插件
5. 在 3D View 侧边栏找到 "Facial Capture" 面板

### Unity 插件

1. 将 `unity_plugin/` 文件夹复制到 Unity 项目的 `Assets/` 目录
2. 在 Unity 中打开 Window > Facial Capture
3. 拖拽图像到面板并点击 "Extract Expression"

## 技术栈

- **面部检测**: MediaPipe Face Mesh (478 个关键点)
- **表情分析**: 基于几何特征和深度学习的混合方法
- **导出格式**: JSON, FBX, Alembic
- **插件框架**: Blender Python API, Unity C# API

## 文件结构

```
facial_capture/
├── core/
│   ├── detector.py          # 面部检测
│   ├── expression_analyzer.py # 表情分析
│   ├── arkit_mapper.py      # ARKit 映射
│   └── exporter.py          # 导出器
├── blender_addon/
│   └── facial_capture_addon.py
├── unity_plugin/
│   ├── FacialCapture.cs
│   └── ARKitBlendshapes.cs
├── models/                   # 预训练模型
└── examples/                 # 示例代码
```

## License

遵循 PromptHMR 项目的许可证
