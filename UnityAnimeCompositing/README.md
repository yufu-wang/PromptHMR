# Unity Anime Compositing - 日本动画风格合成系统

基于 MRT (Multiple Render Targets) 技术的 Unity 插件，复现日本动画拍摄合成工作流程。

## 📖 简介

这个插件实现了传统日本动画制作中的三层级合成系统：

- **基础色层 (Base Color Layer)**: 角色的基本颜色和纹理
- **阴影层 (Shadow Layer)**: Toon Shading 风格的硬阴影
- **高光层 (Highlight Layer)**: 镜面高光和边缘光

通过将这三层分离渲染并合成，可以获得高度可控的动画风格渲染效果，类似于传统动画制作中的"セル画"（赛璐璐）技术。

## ✨ 特性

- ✅ **MRT 渲染**: 一次渲染输出三个独立图层
- ✅ **Toon Shading**: 动画风格的硬边阴影
- ✅ **灵活的混合模式**: 支持正片叠底、相加、滤色、叠加等
- ✅ **后处理效果**: 色彩调整、对比度、饱和度控制
- ✅ **描边系统**: Sobel 边缘检测实现的轮廓线
- ✅ **预设系统**: 内置多种动画风格预设
- ✅ **编辑器工具**: 可视化设置向导

## 📦 文件结构

```
UnityAnimeCompositing/
├── Scripts/
│   ├── AnimeRenderManager.cs        # MRT 渲染管理器
│   ├── AnimeLayerCompositor.cs      # 三层级合成器
│   └── ...
├── Shaders/
│   ├── AnimeCharacterMRT.shader     # MRT 角色 Shader
│   ├── Compositor.shader             # 合成 Shader
│   └── Outline.shader                # 描边 Shader
├── Editor/
│   └── AnimeCompositingEditor.cs    # 编辑器工具
├── Examples/
│   └── ExampleSceneSetup.cs         # 示例场景设置
└── README.md                         # 本文档
```

## 🚀 快速开始

### 方法 1: 使用设置向导（推荐）

1. **打开设置向导**
   - Unity 菜单: `Window → Anime Compositing → Setup Wizard`

2. **选择主摄像机**
   - 在向导中选择场景的主摄像机
   - 或点击"自动查找主摄像机"按钮

3. **添加组件**
   - 点击"添加 Anime Compositing 组件"按钮
   - 系统会自动添加 `AnimeRenderManager` 和 `AnimeLayerCompositor`

4. **创建材质**
   - 点击"创建 Anime Character 材质"
   - 将创建的材质应用到角色模型上

### 方法 2: 手动设置

1. **设置摄像机**
   ```csharp
   // 在主摄像机上添加组件
   Camera mainCamera = Camera.main;
   mainCamera.gameObject.AddComponent<AnimeRenderManager>();
   mainCamera.gameObject.AddComponent<AnimeLayerCompositor>();
   ```

2. **创建材质**
   - 创建新材质
   - 选择 Shader: `AnimeCompositing/CharacterMRT`
   - 应用到角色模型

3. **调整参数**
   - 在 Inspector 中调整 `AnimeLayerCompositor` 的参数
   - 或使用快速预设按钮

## 🎨 Shader 参数说明

### AnimeCharacterMRT Shader

**基础属性**
- `Base Texture`: 基础纹理贴图
- `Base Color`: 基础色调

**阴影属性**
- `Shadow Color`: 阴影颜色
- `Shadow Threshold`: 阴影阈值 (0-1)
- `Shadow Smoothness`: 阴影柔和度 (0-1)

**高光属性**
- `Highlight Color`: 高光颜色
- `Highlight Threshold`: 高光阈值 (0-1)
- `Highlight Smoothness`: 高光柔和度 (0-1)
- `Specular Power`: 镜面反射强度 (1-128)

**边缘光属性**
- `Rim Color`: 边缘光颜色
- `Rim Power`: 边缘光强度 (0.1-8)
- `Rim Intensity`: 边缘光亮度 (0-1)

## 🎬 合成器参数

### 层强度控制
- `Base Color Intensity`: 基础色层强度 (0-2)
- `Shadow Intensity`: 阴影层强度 (0-2)
- `Highlight Intensity`: 高光层强度 (0-2)

### 混合模式
- `Shadow Blend Mode`: 阴影混合模式
  - Normal: 正常
  - Multiply: 正片叠底（默认，适合阴影）
  - Add: 相加
  - Screen: 滤色
  - Overlay: 叠加

- `Highlight Blend Mode`: 高光混合模式
  - Normal: 正常
  - Multiply: 正片叠底
  - Add: 相加（默认，适合高光）
  - Screen: 滤色
  - Overlay: 叠加

### 后处理
- `Enable Color Grading`: 启用色彩调整
- `Color Tint`: 色调偏移
- `Contrast`: 对比度 (0.5-2.0)
- `Saturation`: 饱和度 (0-2.0)

### 描边
- `Enable Outline`: 启用描边
- `Outline Color`: 描边颜色
- `Outline Thickness`: 描边粗细 (0-5)

## 🎯 内置预设

系统提供了多种预设风格：

1. **标准动画 (Standard)**
   - 平衡的参数设置
   - 适合大多数动画场景

2. **高对比度 (High Contrast)**
   - 强烈的明暗对比
   - 适合戏剧性场景

3. **柔和风格 (Soft)**
   - 柔和的阴影和高光
   - 适合温馨、治愈系风格

4. **戏剧性 (Dramatic)**
   - 强烈的阴影效果
   - 适合动作场景

5. **粉彩风格 (Pastel)**
   - 淡雅的色彩
   - 适合少女系、梦幻风格

## 💻 代码示例

### 运行时切换预设

```csharp
using AnimeCompositing;

public class RuntimeControl : MonoBehaviour
{
    private AnimeLayerCompositor compositor;

    void Start()
    {
        compositor = Camera.main.GetComponent<AnimeLayerCompositor>();
    }

    // 应用标准预设
    public void ApplyStandardStyle()
    {
        compositor.baseColorIntensity = 1.0f;
        compositor.shadowIntensity = 1.0f;
        compositor.highlightIntensity = 1.0f;
        compositor.contrast = 1.2f;
        compositor.saturation = 1.1f;
    }

    // 动态调整阴影强度
    public void SetShadowIntensity(float intensity)
    {
        compositor.shadowIntensity = intensity;
    }
}
```

### 使用示例场景脚本

```csharp
using AnimeCompositing.Examples;

public class GameManager : MonoBehaviour
{
    private ExampleSceneSetup sceneSetup;

    void Start()
    {
        sceneSetup = gameObject.AddComponent<ExampleSceneSetup>();

        // 设置初始预设
        sceneSetup.presetType = ExampleSceneSetup.AnimePresetType.Standard;
        sceneSetup.allowRuntimeChange = true;
    }

    // 切换到高对比度预设
    public void SwitchToHighContrast()
    {
        sceneSetup.ApplyPreset(ExampleSceneSetup.AnimePresetType.HighContrast);
    }
}
```

## 🔧 技术细节

### MRT 工作原理

1. **渲染阶段**: `AnimeCharacterMRT.shader` 在一次渲染中输出三个目标:
   - `SV_Target0`: 基础色
   - `SV_Target1`: 阴影（alpha 通道存储遮罩）
   - `SV_Target2`: 高光（alpha 通道存储遮罩）

2. **合成阶段**: `Compositor.shader` 按顺序合成:
   ```
   结果 = 基础色
   结果 = 应用阴影混合(结果, 阴影层)
   结果 = 应用高光混合(结果, 高光层)
   结果 = 应用色彩调整(结果)
   ```

3. **后处理**: `Outline.shader` 使用 Sobel 算子检测边缘并添加描边

### 性能考虑

- **MRT 开销**: 相比多次渲染，MRT 只进行一次几何处理，性能更优
- **填充率**: 需要写入多个渲染目标，填充率开销增加
- **建议**: 在移动平台降低分辨率或禁用部分后处理

### 兼容性

- Unity 版本: 2019.4 或更高
- 渲染管线:
  - ✅ Built-in Render Pipeline
  - ⚠️ URP (需要自定义 Renderer Feature)
  - ⚠️ HDRP (需要自定义 Pass)
- 平台:
  - ✅ Windows/Mac/Linux
  - ✅ Android/iOS (需要 GPU 支持 MRT)
  - ✅ WebGL 2.0

## 🎓 最佳实践

1. **光照设置**
   - 使用单一方向光作为主光源
   - 避免过多动态光源影响 Toon Shading 效果

2. **材质调整**
   - 先调整 Shader 参数获得基础效果
   - 再通过 Compositor 微调合成效果

3. **性能优化**
   - 适当降低 `renderScale` (在 AnimeRenderManager 中)
   - 移动平台禁用描边或降低 `outlineThickness`
   - 使用 LOD 系统减少远处物体的渲染复杂度

4. **美术风格**
   - 高对比度纹理配合低阴影阈值
   - 使用色调偏移营造特定氛围
   - 边缘光增强轮廓感

## 📝 常见问题

**Q: 为什么看不到任何效果？**
A: 确保：
- 角色使用了 `AnimeCompositing/CharacterMRT` Shader
- 摄像机上同时添加了 `AnimeRenderManager` 和 `AnimeLayerCompositor`
- 场景中有光源

**Q: 描边效果不明显怎么办？**
A: 增加 `Outline Thickness` 参数，或调整 `Outline Color` 的对比度

**Q: 如何在 URP 中使用？**
A: 需要创建自定义 Renderer Feature，将 MRT 集成到 URP 管线中

**Q: 性能有问题怎么办？**
A:
- 降低渲染分辨率
- 禁用描边
- 减少后处理效果
- 优化模型面数

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

灵感来源于日本传统动画制作技术和现代实时渲染技术的结合。

---

**创建时间**: 2025-12-30
**版本**: 1.0.0
