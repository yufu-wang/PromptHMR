# 面部表情捕捉插件 - 使用指南

## 目录

1. [快速开始](#快速开始)
2. [Python API 使用](#python-api-使用)
3. [Blender 插件使用](#blender-插件使用)
4. [Unity 插件使用](#unity-插件使用)
5. [高级功能](#高级功能)
6. [故障排除](#故障排除)

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/user/PromptHMR/facial_capture
pip install -r requirements.txt
```

### 2. 测试安装

```python
from facial_capture import FacialExpressionCapture

capturer = FacialExpressionCapture()
print("安装成功！")
```

---

## Python API 使用

### 基本用法 - 单张图像

```python
from facial_capture import FacialExpressionCapture

# 创建捕捉器
capturer = FacialExpressionCapture()

# 从图像提取表情
blendshapes = capturer.extract_from_image("input.jpg", visualize=True)

# 查看结果
print(blendshapes.to_dict())

# 保存为 JSON
capturer.export_arkit("output.json", blendshapes)
```

### 处理视频

```python
# 从视频提取表情序列
sequence = capturer.extract_from_video(
    "input.mp4",
    output_dir="output",
    fps=30.0,
    visualize=False
)

# 导出为多种格式
capturer.export_arkit("output.json", sequence, fps=30.0, format='json')
capturer.export_arkit("output.csv", sequence, fps=30.0, format='csv')
capturer.export_arkit("output.npz", sequence, fps=30.0, format='numpy')
```

### 批量处理图像

```python
from pathlib import Path

# 收集图像文件
image_files = sorted(Path("images").glob("*.jpg"))

# 处理序列
sequence = capturer.extract_from_images(
    image_files,
    output_path="sequence.json",
    fps=30.0
)
```

### 支持的输出格式

| 格式 | 文件扩展名 | 用途 |
|------|-----------|------|
| JSON | `.json` | 通用格式，所有平台 |
| CSV | `.csv` | Excel, 数据分析 |
| NumPy | `.npz` | Python 数据分析 |
| Unity | `.unity.json` | Unity 动画曲线 |
| Blender | `.blender.json` | Blender Action |

---

## Blender 插件使用

### 1. 安装插件

1. 打开 Blender
2. `Edit > Preferences > Add-ons > Install`
3. 选择 `facial_capture/blender_addon/facial_capture_addon.py`
4. 启用 "Import-Export: Facial Expression Capture"

### 2. 准备模型

您的 3D 模型需要有 ARKit 标准的 Shape Keys：

```python
# 在 Blender 中创建 ARKit Shape Keys
1. 选择面部网格对象
2. 在插件面板点击 "Create ARKit Shape Keys"
3. 手动调整每个 Shape Key 的变形
```

或者使用已有 ARKit blendshapes 的模型（如 VRM 模型）。

### 3. 从图像提取表情

1. 打开 3D View 侧边栏 (`N` 键)
2. 切换到 "Facial Capture" 标签
3. 点击 "Extract Expression from Image"
4. 选择图像文件
5. 表情会自动应用到选中的对象

### 4. 导入 JSON 动画

```python
# 如果已经用 Python 导出了 JSON 文件
1. 选择带有 Shape Keys 的对象
2. 点击 "Import Expression JSON"
3. 选择 JSON 文件
4. 勾选 "Create Animation" 创建动画
5. 播放时间轴查看动画
```

### 5. 导出动画

导入的动画会作为 Action 保存在 Blender 中，可以：
- 在 Dope Sheet 中编辑
- 导出为 FBX/Alembic
- 混合多个表情动画

---

## Unity 插件使用

### 1. 安装插件

```bash
# 复制插件文件到 Unity 项目
cp -r facial_capture/unity_plugin /path/to/unity/project/Assets/FacialCapture
```

### 2. 准备模型

确保您的 3D 模型有 ARKit 标准的 Blendshapes：
- 可以从 Blender 导出带 ARKit blendshapes 的模型
- 或使用支持 ARKit 的模型（如 VRM）

### 3. 打开编辑器窗口

```
Unity 菜单: Window > Facial Expression Capture
```

### 4. 导入表情数据

1. 在编辑器窗口中选择 `Skinned Mesh Renderer`
2. 点击 "Browse" 选择 JSON 文件
3. 点击 "Import Single Frame" 或 "Import Sequence"
4. 点击 "Apply to Target" 应用表情

### 5. 创建动画

对于序列数据：
1. 导入序列 JSON
2. 点击 "Create Animation Clip"
3. 保存为 `.anim` 文件
4. 将动画添加到 Animator Controller

### 6. 运行时使用（C# 脚本）

```csharp
using FacialExpressionCapture;

public class FacialAnimationController : MonoBehaviour
{
    public SkinnedMeshRenderer faceRenderer;
    public TextAsset expressionJson;

    void Start()
    {
        // 加载表情数据
        var blendshapes = FacialCaptureImporter.ImportSingleFrame(
            Application.dataPath + "/expression.json"
        );

        // 应用表情
        FacialCaptureImporter.ApplyToSkinnedMesh(faceRenderer, blendshapes);
    }
}
```

---

## 高级功能

### 1. 校准中性表情

```python
# 使用中性表情图像校准
capturer.calibrate_neutral_expression("neutral_face.jpg")

# 之后的表情检测会更准确
blendshapes = capturer.extract_from_image("expression.jpg")
```

### 2. 动漫风格检测

```python
# 启用动漫检测器（实验性）
capturer = FacialExpressionCapture(use_anime_detector=True)

# 处理动漫/漫画图像
blendshapes = capturer.extract_from_image("anime_expression.jpg")
```

### 3. 自定义 Blendshape 映射

```python
from facial_capture.core import ExpressionAnalyzer

# 创建自定义分析器
analyzer = ExpressionAnalyzer()

# 修改分析参数
# 例如：调整眼睛闭合的阈值
# (需要修改 expression_analyzer.py 中的参数)
```

### 4. 实时表情捕捉

```python
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)
capturer = FacialExpressionCapture()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 实时提取表情
    blendshapes = capturer.extract_from_image(frame)

    if blendshapes:
        # 显示主要表情
        smile = blendshapes.get('mouthSmileLeft')
        print(f"Smile: {smile:.2f}")

    # 显示画面
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5. 数据分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载序列数据
from facial_capture.core import ExpressionExporter

exporter = ExpressionExporter()
sequence, fps, metadata = exporter.load_json_sequence("sequence.json")

# 提取特定 blendshape 的时间序列
smile_left = [bs.get('mouthSmileLeft') for bs in sequence]
smile_right = [bs.get('mouthSmileRight') for bs in sequence]

# 绘图
time = np.arange(len(smile_left)) / fps
plt.plot(time, smile_left, label='Left Smile')
plt.plot(time, smile_right, label='Right Smile')
plt.xlabel('Time (s)')
plt.ylabel('Blendshape Value')
plt.legend()
plt.show()
```

---

## 故障排除

### 问题：未检测到面部

**可能原因：**
- 图像质量太低
- 面部角度太大（侧脸）
- 光线太暗
- 面部被遮挡

**解决方案：**
```python
# 检查图像
import cv2
image = cv2.imread("input.jpg")
print(f"图像尺寸: {image.shape}")

# 确保图像清晰且面部正面可见
# 建议：分辨率至少 512x512，面部占图像的 30% 以上
```

### 问题：Blender 插件无法导入

**解决方案：**
```bash
# 确保安装了依赖
pip install mediapipe opencv-python numpy

# 检查 Python 路径
import sys
print(sys.executable)

# 在 Blender Preferences 中设置正确的 Python 路径
```

### 问题：Unity 无法找到 Blendshapes

**原因：** Blendshape 名称不匹配

**解决方案：**
```csharp
// 检查模型的 blendshape 名称
var mesh = GetComponent<SkinnedMeshRenderer>().sharedMesh;
for (int i = 0; i < mesh.blendShapeCount; i++)
{
    Debug.Log($"Blendshape {i}: {mesh.GetBlendShapeName(i)}");
}

// 确保名称与 ARKit 标准一致
// 或修改 FacialCaptureImporter.cs 中的名称映射
```

### 问题：表情不准确

**解决方案：**
```python
# 1. 使用中性表情校准
capturer.calibrate_neutral_expression("neutral.jpg")

# 2. 检查关键点检测
detection = capturer.detector.detect(image)
vis_image = capturer.detector.visualize(image, detection)
cv2.imshow('Landmarks', vis_image)
cv2.waitKey(0)

# 3. 调整检测参数
# 修改 expression_analyzer.py 中的阈值
```

### 问题：动漫图像检测失败

**说明：** 当前版本主要针对真人面部优化，动漫检测为实验性功能

**解决方案：**
- 使用清晰的动漫面部图像（正面，大头像）
- 考虑使用专门的动漫面部检测器
- 或手动调整关键点并导入

---

## 性能优化

### 批量处理优化

```python
# 使用多线程处理
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    capturer = FacialExpressionCapture()
    return capturer.extract_from_image(image_path)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_files))
```

### 视频处理优化

```python
# 降低帧率
sequence = capturer.extract_from_video(
    "input.mp4",
    fps=15.0  # 从 30fps 降到 15fps
)

# 或只处理关键帧
# (需要额外的关键帧检测代码)
```

---

## 示例项目

查看 `examples/` 目录中的完整示例：

- `extract_single_image.py` - 单图像提取
- `extract_video.py` - 视频处理
- `batch_process.py` - 批量处理

运行示例：
```bash
cd facial_capture/examples
python extract_single_image.py
```

---

## 技术支持

如有问题，请查看：
- GitHub Issues: [PromptHMR Issues](https://github.com/Arthur151/PromptHMR/issues)
- 文档: `facial_capture/README.md`
- API 参考: 查看代码注释

---

## 许可证

本插件遵循 PromptHMR 项目的许可证。
