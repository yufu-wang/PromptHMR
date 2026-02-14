# 快速开始指南 - 三种方案

根据您的需求选择最适合的方案：

## 🎯 方案对比

| 方案 | 难度 | 时间 | 费用 | 推荐度 |
|------|------|------|------|--------|
| **方案 1: 使用现成 ARKit 模型** | ⭐ 简单 | 5 分钟 | 免费-$50 | ⭐⭐⭐⭐⭐ |
| **方案 2: 使用 Faceit 自动生成** | ⭐⭐ 中等 | 2-4 小时 | $40 | ⭐⭐⭐⭐ |
| **方案 3: 手动创建 Blendshapes** | ⭐⭐⭐⭐⭐ 困难 | 20-40 小时 | 免费 | ⭐⭐ |

---

## 方案 1: 使用现成 ARKit 模型（推荐）⭐

**适合：** 想快速开始，不介意使用通用模型

### 1.1 免费模型

#### VRoid Studio（动漫风格）
```bash
# 1. 下载 VRoid Studio (免费)
https://vroid.com/studio

# 2. 创建或导入角色
# 3. 导出为 VRM 格式
File > Export as VRM

# 4. 导入到 Blender
安装 VRM 插件: https://github.com/saturday06/VRM-Addon-for-Blender
File > Import > VRM (.vrm)
```

✅ **包含完整 52 个 ARKit blendshapes**
✅ 可自定义面部、发型、服装
✅ 完全免费

#### Ready Player Me（写实风格）
```bash
# 1. 访问网站
https://readyplayer.me/

# 2. 创建头像（可上传照片生成）
# 3. 下载 GLB 格式
# 4. 导入 Blender/Unity
```

✅ **包含标准面部 blendshapes**
✅ 可用照片自动生成
✅ 免费（高级功能付费）

### 1.2 购买商业模型

#### Booth（日本角色市场）
```
https://booth.pm/
搜索: "VRM" 或 "ARKit blendshape"
价格: $10-$100
```

#### Gumroad / Sketchfab
```
搜索: "ARKit face" "blendshapes" "VRM"
价格: $20-$200
```

### 1.3 测试流程

```python
# 下载示例脚本
cd facial_capture/examples

# 提取表情
python extract_single_image.py

# 在 Blender 中导入
# 1. 导入模型
# 2. 启用 Facial Capture 插件
# 3. 导入生成的 JSON 文件
# 4. 查看表情应用效果
```

---

## 方案 2: 使用 Faceit 自动生成

**适合：** 有自己的角色模型，想要专业的 ARKit 支持

### 2.1 安装 Faceit

```bash
# 1. 购买 Faceit for Blender
https://faceit-doc.readthedocs.io/
价格: ~$40

# 2. 在 Blender 中安装
Edit > Preferences > Add-ons > Install
选择下载的 Faceit .zip 文件

# 3. 启用插件
搜索 "Faceit" 并勾选
```

### 2.2 使用 Faceit 生成 Blendshapes

```bash
# 在 Blender 中：

# 1. 选择面部网格对象
# 2. 打开 Faceit 面板（侧边栏）
# 3. Setup（设置）
   - 点击 "Register Objects"
   - 标记面部区域（眼睛、嘴巴、眉毛等）

# 4. Rigging（绑定）
   - 选择 "ARKit" 预设
   - 点击 "Generate Rig"

# 5. Bake（烘焙）
   - 点击 "Bake to Shape Keys"
   - 等待自动生成所有 52 个 ARKit blendshapes

# 完成！模型现在有完整的 ARKit blendshapes
```

### 2.3 优点

✅ 自动生成，节省大量时间
✅ 专业品质，表情自然
✅ 支持 ARKit 标准
✅ 可后续手动微调

### 2.4 局限性

⚠️ 需要购买插件（$40）
⚠️ 需要基础的 Blender 知识
⚠️ 自动生成的表情可能需要微调

---

## 方案 3: 手动创建（仅在必要时）

**适合：** 需要 100% 自定义，有充足时间

详见：[CREATE_BLENDSHAPES_TUTORIAL.md](CREATE_BLENDSHAPES_TUTORIAL.md)

**预计时间:**
- 52 个标准 blendshapes: 20-30 小时
- 18 个扩展 blendshapes: 10-15 小时

**建议：** 仅当以上两种方案都不适用时才选择此方案。

---

## 🚀 推荐工作流程

### 初学者流程
```
1. 使用 VRoid Studio 创建免费角色（15分钟）
   ↓
2. 导出为 VRM，导入 Blender（5分钟）
   ↓
3. 安装 Facial Capture 插件（2分钟）
   ↓
4. 从 R18 图片提取表情（1分钟）
   ↓
5. 导入 JSON 到 Blender（1分钟）
   ↓
6. 查看效果，导出动画（5分钟）

总时间: ~30 分钟 ✅
```

### 专业流程
```
1. 有自己的角色模型
   ↓
2. 购买 Faceit ($40)（1分钟）
   ↓
3. 使用 Faceit 生成 ARKit blendshapes（2-4小时）
   ↓
4. 手动微调关键表情（2-4小时）
   ↓
5. 使用 Facial Capture 提取真实表情（随时）
   ↓
6. 应用到模型，导出动画

总时间: 一天工作量 ✅
费用: $40
```

---

## 🎨 舌头和瞳孔的特殊处理

我们的**扩展功能**（舌头和瞳孔精确控制）需要额外处理：

### 舌头（tongueOutExtension 等）

#### 选项 A: 独立舌头网格（推荐）
```
1. 在 Blender 中建模舌头
2. 为舌头创建 Shape Keys
   - tongueOutExtension（伸出）
   - tongueUpDown（上下）
   - tongueLeftRight（左右）

3. 父级到头部骨骼
```

#### 选项 B: 仅使用标准 tongueOut
```
如果不需要精确控制，只使用标准的 tongueOut blendshape
我们的插件会自动回退到标准参数
```

### 瞳孔（pupilPositionX/Y 等）

#### 选项 A: 使用骨骼控制（推荐）⭐
```
1. 为左右眼球创建骨骼
2. 用骨骼旋转控制眼球方向
3. 不使用 Shape Keys

我们的插件会输出瞳孔位置数据
您可以手动或通过脚本将其转换为骨骼旋转
```

#### 选项 B: UV 动画
```
1. 虹膜/瞳孔使用纹理
2. 通过 UV 偏移移动瞳孔位置
3. 使用着色器节点控制
```

#### 选项 C: 回退到标准 ARKit
```
如果不需要扩展控制，使用标准的：
- eyeLookUp/Down/In/Out (8个)

我们的插件会自动使用这些参数
```

---

## 📦 完整示例项目

### 示例 1: VRoid 角色 + 阿黑颜表情

```bash
# 1. 下载 VRoid Studio 示例
# （假设您已有 VRM 文件）

# 2. 提取阿黑颜表情
cd facial_capture/examples
python extract_ahegao.py

# 输入图片: 阿黑颜表情图片
# 输出: ahegao_expression.json

# 3. 在 Blender 中
# - 导入 VRM 模型
# - 使用 Facial Capture 插件导入 JSON
# - 查看舌头伸出 + 眼睛翻白的效果

# 4. 导出动画
# File > Export > FBX
```

### 示例 2: 实时摄像头捕捉

```bash
cd facial_capture/examples
python test_tongue_pupil.py

# 选择使用摄像头
# 伸出舌头，转动眼球测试
# 实时查看检测效果
```

---

## ❓ 常见问题

### Q: 我完全是新手，应该选择哪个方案？
**A:** 方案 1 - 使用 VRoid Studio（免费，30分钟搞定）

### Q: 我有自己的角色但没有 blendshapes？
**A:** 方案 2 - 使用 Faceit（$40，半天搞定）

### Q: ARKit 眼球注视不够精确？
**A:** 使用我们的扩展瞳孔参数（leftPupilPositionX/Y 等）

### Q: 舌头检测总是失败？
**A:**
1. 确保图像中舌头清晰可见
2. 正面拍摄效果最好
3. 查看 `detection['tongue']['detected']` 状态

### Q: 不需要全部 70 个参数怎么办？
**A:** 可以！只创建您需要的。插件会跳过不存在的 blendshapes。

### Q: 可以混合使用吗？
**A:** 可以！例如：
- 基础表情用标准 ARKit（52个）
- 特殊场景使用扩展参数（舌头+瞳孔）

---

## 🎯 推荐配置

### 最小配置（快速测试）
```
✅ ARKit 模型（52个标准 blendshapes）
✅ Python 环境 + MediaPipe
✅ Blender 插件

可以: 提取和应用大部分表情
不可: 精确舌头和瞳孔控制
```

### 标准配置（日常使用）
```
✅ ARKit 模型 + 标准 tongueOut
✅ 眼球骨骼控制（可选）
✅ Python + Blender/Unity 插件

可以: 完整的表情捕捉
包括: 基础舌头检测
```

### 完整配置（专业制作）
```
✅ ARKit 52 + 扩展 18 (共70个)
✅ 独立舌头网格 + Shape Keys
✅ 眼球骨骼/UV 动画
✅ 所有插件和工具

可以: 100% 精确的表情捕捉
包括: 阿黑颜、极端表情等
```

---

## 📚 延伸阅读

- [CREATE_BLENDSHAPES_TUTORIAL.md](CREATE_BLENDSHAPES_TUTORIAL.md) - 详细创建教程
- [ENHANCED_FEATURES.md](../ENHANCED_FEATURES.md) - 扩展功能文档
- [USAGE_GUIDE.md](../USAGE_GUIDE.md) - 使用指南

---

## 💡 最后建议

1. **先从简单开始**: 使用现成 ARKit 模型测试功能
2. **按需扩展**: 需要时再添加舌头/瞳孔控制
3. **保存工作**: 随时保存 JSON 文件，便于后续调整
4. **多次测试**: 不同角度、光线下测试提取效果

祝您使用愉快！🎉
