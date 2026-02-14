# Blender 中创建 ARKit Blendshapes 教程

本教程教您如何在 Blender 中为模型手动创建 ARKit Blendshapes。

## 前提条件

- Blender 3.0+
- 您的角色模型（包含面部网格）
- 基础的 Blender 操作知识

## 步骤 1: 准备模型

1. 打开 Blender，导入您的模型
2. 选择面部网格对象
3. 确保模型在编辑模式下顶点分布合理

## 步骤 2: 创建 Basis Shape Key

1. 选择面部网格
2. 在右侧属性面板，找到 `Shape Keys` (形状键图标)
3. 点击 `+` 添加第一个 Shape Key，命名为 `Basis`
4. `Basis` 是中性表情，**不要修改**

## 步骤 3: 创建表情 Shape Keys

对于每个 ARKit blendshape，重复以下步骤：

### 示例：创建 `eyeBlinkLeft` (左眼闭合)

1. **添加新 Shape Key**
   - 点击 `+` 添加新的 Shape Key
   - 重命名为 `eyeBlinkLeft`（必须与 ARKit 名称完全一致）

2. **进入编辑模式**
   - 选中新创建的 `eyeBlinkLeft`
   - 按 `Tab` 进入编辑模式

3. **修改顶点**
   - 选择左眼上下眼睑的顶点
   - 使用 `G` (移动) 将上下眼睑顶点移动到闭合位置
   - 调整直到眼睛完全闭合

4. **退出编辑模式**
   - 按 `Tab` 退出编辑模式
   - 在 Shape Keys 面板中，拖动 `eyeBlinkLeft` 的值滑块（0-1）
   - 检查效果：0 = 睁眼，1 = 闭眼

5. **重置值为 0**
   - 创建完成后，将值设回 0（中性状态）

## 步骤 4: 批量创建所有 52 个标准 ARKit Blendshapes

按照同样方法创建以下所有 blendshapes：

### 眼部 (16个)
```
eyeBlinkLeft        - 左眼闭合（上下眼睑靠拢）
eyeBlinkRight       - 右眼闭合
eyeSquintLeft       - 左眼眯起（下眼睑上提，但不完全闭合）
eyeSquintRight      - 右眼眯起
eyeWideLeft         - 左眼睁大（眼睑拉开）
eyeWideRight        - 右眼睁大

eyeLookUpLeft       - 左眼向上看（虹膜/瞳孔向上移动）*
eyeLookUpRight      - 右眼向上看 *
eyeLookDownLeft     - 左眼向下看 *
eyeLookDownRight    - 右眼向下看 *
eyeLookInLeft       - 左眼向内看（向鼻子方向）*
eyeLookInRight      - 右眼向内看 *
eyeLookOutLeft      - 左眼向外看 *
eyeLookOutRight     - 右眼向外看 *
```

**注：** 带 `*` 的需要模型有独立的虹膜/瞳孔网格，或使用骨骼/UV 动画。如果没有，可以跳过，使用我们的扩展参数。

### 眉毛 (8个)
```
browDownLeft        - 左眉下压（皱眉）
browDownRight       - 右眉下压
browInnerUp         - 内眉上扬（惊讶/担忧）
browOuterUpLeft     - 左外眉上扬
browOuterUpRight    - 右外眉上扬
```

### 嘴部 (28个)
```
jawOpen             - 下巴张开（下颌骨下移）

mouthClose          - 嘴巴闭紧（唇部压紧）
mouthFunnel         - 嘟嘴（嘴唇向前呈 O 形）
mouthPucker         - 撅嘴（嘴唇收紧向前）

mouthLeft           - 嘴巴向左移
mouthRight          - 嘴巴向右移

mouthSmileLeft      - 左嘴角上扬（微笑）
mouthSmileRight     - 右嘴角上扬
mouthFrownLeft      - 左嘴角下垂（不悦）
mouthFrownRight     - 右嘴角下垂

mouthDimpleLeft     - 左酒窝（嘴角内凹）
mouthDimpleRight    - 右酒窝

mouthStretchLeft    - 嘴巴向左拉伸
mouthStretchRight   - 嘴巴向右拉伸

mouthRollLower      - 下唇卷入口内
mouthRollUpper      - 上唇卷入口内
mouthShrugLower     - 下唇向上推
mouthShrugUpper     - 上唇向下推

mouthPressLeft      - 左唇挤压
mouthPressRight     - 右唇挤压

mouthLowerDownLeft  - 左下唇下拉
mouthLowerDownRight - 右下唇下拉
mouthUpperUpLeft    - 左上唇上提
mouthUpperUpRight   - 右上唇上提
```

### 脸颊和鼻子
```
cheekPuff           - 鼓腮（脸颊向外鼓）
cheekSquintLeft     - 左脸颊上提（微笑时的脸颊）
cheekSquintRight    - 右脸颊上提

noseSneerLeft       - 左鼻翼皱起
noseSneerRight      - 右鼻翼皱起
```

### 下颌
```
jawForward          - 下巴向前
jawLeft             - 下巴向左
jawRight            - 下巴向右
```

### 舌头（标准）
```
tongueOut           - 舌头伸出（基础版本）
```

## 步骤 5: 创建扩展 Blendshapes（可选）⭐

如果要支持精确的舌头和瞳孔控制，创建这些额外的 Shape Keys：

### 舌头扩展 (6个)
```
tongueOutExtension  - 舌头伸出程度（比 tongueOut 更精确）
tongueUpDown        - 舌头上下移动
tongueLeftRight     - 舌头左右移动
tongueCurl          - 舌头卷曲
tongueWidth         - 舌头变宽
tongueThickness     - 舌头变厚
```

**提示：** 舌头需要独立的网格对象。如果模型没有舌头网格，需要先建模添加。

### 瞳孔扩展 (可选)
```
leftPupilPositionX  - 左瞳孔水平位置
leftPupilPositionY  - 左瞳孔垂直位置
leftPupilDilation   - 左瞳孔扩张
...
```

**注：** 这些需要独立的瞳孔/虹膜网格，或使用 UV 动画。如果没有，我们的插件会自动使用 `eyeLookUp/Down/In/Out` 代替。

## 步骤 6: 测试 Blendshapes

1. **手动测试**
   - 在 Shape Keys 面板中，逐个调整每个 blendshape 的值（0-1）
   - 确保每个表情变形正确
   - 检查是否有顶点错位

2. **组合测试**
   - 同时激活多个 blendshapes（如 `mouthSmileLeft` + `eyeSquintLeft`）
   - 确保表情可以自然组合

3. **极值测试**
   - 测试值为 0、0.5、1 时的效果
   - 确保没有异常变形

## 步骤 7: 导出模型

导出为支持 blendshapes 的格式：

### FBX 导出（Unity 推荐）
1. `File > Export > FBX (.fbx)`
2. 勾选 `Shape Keys` 选项
3. 导出

### GLTF/GLB 导出（通用）
1. `File > Export > glTF 2.0 (.glb/.gltf)`
2. 在导出选项中，勾选 `Shape Keys` / `Morph Targets`
3. 导出

### Blender 文件（Blender 内使用）
1. 直接保存 `.blend` 文件
2. 使用我们的 Blender 插件导入表情数据

## 快捷提示 💡

### 使用 Python 脚本批量创建 Basis

可以使用 Blender Python 脚本快速创建所有空白 Shape Keys：

```python
import bpy

# ARKit 标准名称列表
arkit_names = [
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'browDownLeft', 'browDownRight',
    'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight', 'cheekPuff',
    'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight',
    'tongueOut',
    # ... 添加其余的
]

# 选中的对象
obj = bpy.context.active_object

# 确保有 Basis
if not obj.data.shape_keys:
    obj.shape_key_add(name='Basis', from_mix=False)

# 批量创建 Shape Keys
for name in arkit_names:
    if name not in obj.data.shape_keys.key_blocks:
        obj.shape_key_add(name=name, from_mix=False)
        print(f"Created: {name}")
```

运行这个脚本后，所有 Shape Keys 会被创建，然后您只需逐个编辑形状。

### 使用对称工具

对于左右对称的表情（如眼睛、眉毛）：
1. 先创建一侧（如 `eyeBlinkLeft`）
2. 使用 Blender 的镜像功能复制到另一侧
3. 重命名为对应的右侧版本（如 `eyeBlinkRight`）

### 参考真实表情

- 对着镜子或拍照参考自己的表情
- 使用参考图像（搜索 "ARKit blendshapes reference"）
- 研究真人的面部肌肉运动

## 常见问题

### Q: 必须创建全部 52 个吗？
A: 不必须。可以先创建最常用的（眼睛、嘴巴、眉毛），其他的留空。插件会跳过不存在的 blendshapes。

### Q: 名称必须完全一致吗？
A: 是的。ARKit 标准名称区分大小写，必须完全匹配（如 `eyeBlinkLeft` 而不是 `eye_blink_left`）。

### Q: 可以用其他命名约定吗？
A: 可以在代码中添加名称映射。修改 `blender_addon/facial_capture_addon.py` 中的 `apply_blendshapes()` 函数。

### Q: 舌头和瞳孔怎么做？
A:
- **舌头**: 需要独立的舌头网格，然后为其创建 Shape Keys
- **瞳孔**: 推荐使用骨骼控制或 UV 动画，而不是 Shape Keys

### Q: 创建很耗时怎么办？
A:
- 优先创建常用表情（眼睛眨眼、嘴巴微笑、眉毛等）
- 或使用 AI 工具自动生成（如 Faceit for Blender）
- 或购买已有 ARKit blendshapes 的模型

## 推荐工具

### Blender 插件
- **Faceit**: 自动生成面部绑定和 ARKit blendshapes
  - https://faceit-doc.readthedocs.io/
  - 可以大幅减少手动工作

- **Auto-Rig Pro**: 包含面部绑定功能
  - https://blendermarket.com/products/auto-rig-pro

### 在线资源
- ARKit 官方文档: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
- Blendshape 参考图: 搜索 "ARKit 52 blendshapes reference"

## 总结

创建 ARKit Blendshapes 的工作流：

1. ✅ 准备模型（面部网格）
2. ✅ 创建 Basis Shape Key
3. ✅ 逐个创建 52 个标准表情 Shape Keys
4. ✅ （可选）创建 18 个扩展 Shape Keys
5. ✅ 测试和调整
6. ✅ 导出为 FBX/GLB
7. ✅ 在 Blender/Unity 中导入表情数据

**预计时间:**
- 完整手动创建：20-40 小时
- 使用 Faceit 等工具：2-4 小时
- 使用现成模型：0 小时 ⭐

建议优先使用现成的 ARKit 模型，或使用 Faceit 等插件自动生成！
