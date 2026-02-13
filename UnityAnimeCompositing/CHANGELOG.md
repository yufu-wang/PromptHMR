# Changelog

All notable changes to the Anime Compositing System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-30

### Added

#### 核心功能
- ✨ MRT (Multiple Render Targets) 渲染系统
- ✨ 三层级渲染架构（基础色/阴影/高光）
- ✨ Toon Shading 风格的角色渲染
- ✨ 灵活的层合成系统

#### Shaders
- 🎨 `AnimeCharacterMRT.shader` - MRT 角色着色器
  - 基础颜色层输出
  - Toon 风格阴影计算
  - 镜面高光和边缘光
  - 可调节的阴影/高光阈值
- 🎨 `Compositor.shader` - 层合成着色器
  - 5 种混合模式（Normal/Multiply/Add/Screen/Overlay）
  - 色彩调整（对比度/饱和度/色调）
  - 分层强度控制
- 🎨 `Outline.shader` - 描边着色器
  - Sobel 边缘检测
  - 可调节描边粗细和颜色

#### C# 脚本
- 📜 `AnimeRenderManager.cs` - 渲染管理器
  - RenderTexture 管理
  - MRT 设置和配置
  - 分辨率缩放支持
  - Debug 可视化
- 📜 `AnimeLayerCompositor.cs` - 层合成器
  - 实时层合成
  - 混合模式控制
  - 后处理管线
  - Inspector 参数调整
- 📜 `LayerDebugger.cs` - 调试工具
  - 实时层预览
  - 性能统计
  - 多种显示模式
  - 热键支持

#### 编辑器工具
- 🛠️ `AnimeCompositingEditor.cs` - 编辑器窗口
  - 设置向导
  - 一键配置
  - 材质创建工具
  - 自定义 Inspector

#### 示例和预设
- 📚 `ExampleSceneSetup.cs` - 示例场景脚本
  - 5 种内置预设（标准/高对比度/柔和/戏剧性/粉彩）
  - 运行时预设切换
  - 热键控制
  - GUI 调试界面

#### 文档
- 📖 `README.md` - 用户指南
  - 快速开始教程
  - 详细参数说明
  - 代码示例
  - 最佳实践
  - FAQ
- 📖 `TECHNICAL.md` - 技术文档
  - 架构设计说明
  - 渲染流程图
  - 性能分析
  - URP/HDRP 移植指南
  - 扩展开发指南

#### 其他
- 📦 `package.json` - Unity Package Manager 配置
- 📄 `LICENSE.txt` - MIT 许可证
- 📝 `CHANGELOG.md` - 版本更新日志

### Features

#### 渲染特性
- 支持多达 3 个同时输出的渲染目标
- 基于物理的 Toon Shading 光照模型
- 实时阴影接收和投射
- 边缘光（Rim Light）效果
- 可配置的渲染分辨率缩放

#### 合成特性
- 5 种 Photoshop 风格混合模式
- 独立的层强度控制
- 色彩分级（Color Grading）
- 对比度和饱和度调整
- 色调偏移

#### 后处理
- Sobel 算子边缘检测
- 可调节描边效果
- 性能友好的单 Pass 实现

#### 工具特性
- 场景内实时预览
- 多种调试显示模式
- 性能监控（FPS/分辨率）
- 热键快速切换
- 编辑器设置向导

### Technical Details

- **Unity 版本**: 2019.4+
- **渲染管线**: Built-in Render Pipeline（主要支持）
- **Shader Model**: 3.0+
- **平台支持**:
  - ✅ Windows/Mac/Linux
  - ✅ Android/iOS (需要 MRT 支持)
  - ✅ WebGL 2.0

### Performance

- **内存占用**: ~95 MB (1920x1080, ARGBFloat)
- **性能影响**: ~20-30% (相比标准渲染)
- **优化**: 支持分辨率缩放和格式调整

### Known Limitations

- URP 和 HDRP 需要额外的集成工作
- 某些旧移动设备可能不支持 MRT
- 描边效果在低分辨率下可能不够清晰

### Dependencies

无外部依赖，仅使用 Unity 内置功能。

---

## [Unreleased]

### Planned Features

- URP Renderer Feature 支持
- HDRP Custom Pass 支持
- 更多混合模式（Color Dodge, Color Burn, etc.）
- 纹理空间描边（Texture-space Outlining）
- 多光源支持优化
- 性能分析工具
- 更多预设样式

### Under Consideration

- 材质库和预设管理器
- 运行时材质编辑器
- GIF/视频导出工具
- VR 支持
- 移动端优化版本

---

## Version History

- **1.0.0** (2025-12-30) - Initial release

---

## Support

如有问题或建议，请：
- 提交 Issue
- 查看文档
- 联系开发团队

## Contributors

感谢所有为这个项目做出贡献的开发者！

---

**Note**: 这是初始版本发布。我们欢迎社区反馈和贡献！
