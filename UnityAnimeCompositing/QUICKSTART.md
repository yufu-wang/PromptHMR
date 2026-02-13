# 快速开始指南 - Anime Compositing System

## 5 分钟快速上手

### 步骤 1: 导入插件

将 `UnityAnimeCompositing` 文件夹复制到您的 Unity 项目的 `Assets` 文件夹中。

### 步骤 2: 设置摄像机

**方法 A: 使用向导（推荐）**

1. 打开 Unity 编辑器
2. 菜单栏: `Window → Anime Compositing → Setup Wizard`
3. 点击 "自动查找主摄像机"
4. 点击 "添加 Anime Compositing 组件"

**方法 B: 手动添加**

1. 选择场景中的主摄像机
2. 在 Inspector 中点击 `Add Component`
3. 添加 `AnimeRenderManager`
4. 添加 `AnimeLayerCompositor`

### 步骤 3: 创建材质

1. 在 Project 窗口右键: `Create → Material`
2. 命名为 `AnimeMaterial`
3. 在 Inspector 中，Shader 选择: `AnimeCompositing/CharacterMRT`

### 步骤 4: 应用到模型

1. 将角色模型拖入场景
2. 选择模型的 Renderer
3. 将 `AnimeMaterial` 拖到 Material 槽位

### 步骤 5: 调整参数

在 `AnimeLayerCompositor` 组件中：
- 点击 "标准动画" 预设按钮
- 或手动调整参数

✅ 完成！您现在应该能看到动画风格的渲染效果了。

---

## 常用参数调整

### 获得更强的阴影效果

在材质的 Shader 参数中：
- 降低 `Shadow Threshold`: 0.5 → 0.3
- 增加 `Shadow Intensity`: 1.0 → 1.5

### 增加高光

在材质的 Shader 参数中：
- 降低 `Highlight Threshold`: 0.8 → 0.6
- 增加 `Specular Power`: 32 → 64

### 添加描边

在 `AnimeLayerCompositor` 组件中：
- 勾选 `Enable Outline`
- 设置 `Outline Thickness`: 1.0 → 2.0
- 设置 `Outline Color`: 黑色

### 调整色彩风格

在 `AnimeLayerCompositor` 组件中：
- `Contrast`: 1.2 (增加对比度)
- `Saturation`: 1.1 (增加饱和度)
- `Color Tint`: 微调整体色调

---

## 热键参考

### LayerDebugger 热键（需要添加该组件）

- `1`: 显示合成结果
- `2`: 仅显示基础色层
- `3`: 仅显示阴影层
- `4`: 仅显示高光层
- `5`: 分屏显示所有层

### ExampleSceneSetup 热键（需要添加该组件）

- `Space`: 切换开关
- `N`: 切换下一个预设

---

## 快速预设

### 标准动画风格
```
Base Intensity: 1.0
Shadow Intensity: 1.0
Highlight Intensity: 1.0
Contrast: 1.2
Saturation: 1.1
Outline: 启用, 粗细 1.0
```

### 高对比度（戏剧性）
```
Base Intensity: 1.2
Shadow Intensity: 1.5
Highlight Intensity: 1.3
Contrast: 1.5
Saturation: 1.3
Outline: 启用, 粗细 1.5
```

### 柔和风格（治愈系）
```
Base Intensity: 0.9
Shadow Intensity: 0.7
Highlight Intensity: 0.8
Contrast: 0.9
Saturation: 0.9
Outline: 启用, 粗细 0.8
```

---

## 常见问题速查

### ❌ 看不到任何效果

**检查清单**:
- [ ] 摄像机上有 `AnimeRenderManager` 组件吗？
- [ ] 摄像机上有 `AnimeLayerCompositor` 组件吗？
- [ ] 材质使用的是 `AnimeCompositing/CharacterMRT` Shader 吗？
- [ ] 场景中有光源吗？

### ❌ Shader 找不到

**解决方案**:
1. 确保 `Shaders` 文件夹已导入
2. 在 Unity 中右键 `Shaders` 文件夹 → `Reimport`
3. 检查 Console 是否有 Shader 编译错误

### ❌ 性能太差

**优化方法**:
1. 降低 `renderScale` (在 AnimeRenderManager 中)
2. 禁用 `Enable Outline`
3. 使用更低精度的纹理格式
4. 减少场景中的角色数量

### ❌ 描边太粗/太细

**调整方法**:
- 在 `AnimeLayerCompositor` 中调整 `Outline Thickness`
- 建议范围: 0.5 - 2.0

---

## 下一步

- 📖 阅读完整的 [README.md](README.md)
- 🔧 查看详细的 [TECHNICAL.md](TECHNICAL.md)
- 💡 尝试不同的预设和参数组合
- 🎨 为不同角色创建自定义材质

---

## 支持和反馈

遇到问题？
1. 查看 [README.md](README.md) 的 FAQ 部分
2. 查看 [TECHNICAL.md](TECHNICAL.md) 的故障排除章节
3. 提交 Issue 到项目仓库

---

**祝您使用愉快！** ✨
