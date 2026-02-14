# 程序化表情动画生成

自动在两个或多个 R18 表情之间生成平滑的循环动画，支持从 MMD 视频提取表情。

## 🎬 功能概述

### 1. 两个表情之间的过渡

从两张图片自动生成平滑过渡动画：

```python
from facial_capture.core import EnhancedFacialDetector, convert_detection_to_extended_blendshapes
from facial_capture.core.animation_generator import ProceduralAnimationGenerator, EasingType, LoopMode

# 检测两个表情
detector = EnhancedFacialDetector()

detection_a = detector.detect_with_details(image_a)
detection_b = detector.detect_with_details(image_b)

bs_a = convert_detection_to_extended_blendshapes(detection_a)
bs_b = convert_detection_to_extended_blendshapes(detection_b)

# 生成动画
generator = ProceduralAnimationGenerator()
sequence = generator.create_simple_transition(
    bs_a.to_dict_extended(),
    bs_b.to_dict_extended(),
    duration=2.0,              # 2秒
    fps=30.0,                  # 30帧/秒
    easing=EasingType.EASE_IN_OUT,  # 缓动类型
    loop_mode=LoopMode.PING_PONG     # 来回循环
)

# sequence 包含 60 帧动画数据
```

### 2. 多阶段关键帧动画

使用多个关键帧创建复杂动画：

```python
# 定义关键帧
keyframes = [
    (0.0, neutral_expression, EasingType.EASE_IN),
    (0.3, slight_reaction, EasingType.EASE_OUT),
    (0.7, intense_expression, EasingType.ELASTIC),
    (1.0, peak_expression, EasingType.SPRING),
]

generator = ProceduralAnimationGenerator()
sequence = generator.create_multi_stage_animation(
    keyframes,
    duration=4.0,
    fps=30.0,
    loop_mode=LoopMode.LOOP
)
```

### 3. 从 MMD 视频提取表情

从 MMD 视频中提取表情并生成循环：

```python
from facial_capture.core.video_expression_extractor import VideoExpressionExtractor

extractor = VideoExpressionExtractor()

# 提取关键帧
keyframes = extractor.extract_keyframes(
    "mmd_video.mp4",
    num_keyframes=5,
    method="peaks"  # 表情峰值
)

# 创建循环
loop_sequence = extractor.create_loop_from_video(
    "mmd_video.mp4",
    loop_duration=2.0,
    method="seamless"  # 无缝循环
)
```

---

## 🎨 缓动类型（EasingType）

### 可用的 11 种缓动效果：

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `LINEAR` | 线性，匀速 | 机械动作 |
| `EASE_IN` | 缓入（慢→快） | 开始加速 |
| `EASE_OUT` | 缓出（快→慢） | 结束减速 |
| `EASE_IN_OUT` | 缓入缓出 | **通用，最自然** |
| `SINE` | 正弦曲线 | 平滑波动 |
| `CUBIC` | 三次曲线 | 柔和过渡 |
| `EXPO` | 指数曲线 | 剧烈加速/减速 |
| `ELASTIC` | 弹性（回弹） | **阿黑颜等剧烈表情** |
| `BOUNCE` | 弹跳效果 | 惊讶、震惊 |
| `SPRING` | 弹簧效果 | 颤抖、震动 |
| `OVERSHOOT` | 过冲（超过再回） | 夸张表情 |

### 示例对比：

```python
# 柔和过渡（日常表情）
EasingType.EASE_IN_OUT

# 夸张过渡（R18 表情）
EasingType.ELASTIC  # 会有明显的回弹效果

# 机械过渡
EasingType.LINEAR
```

---

## 🔄 循环模式（LoopMode）

| 模式 | 描述 | 效果 |
|------|------|------|
| `PING_PONG` | 来回循环 | A→B→A→B... |
| `SEAMLESS` | 无缝循环 | A→B（平滑过渡回A） |
| `LOOP` | 直接循环 | A→B→(跳回A)→B... |
| `HOLD` | 单次过渡 | A→B（停止） |

### 推荐使用：

- **R18 循环动画**：`PING_PONG` 或 `SEAMLESS`
- **一次性过渡**：`HOLD`
- **快速循环**：`LOOP`

---

## 📖 使用示例

### 示例 1：简单的阿黑颜循环

```bash
cd facial_capture/examples
python generate_loop_animation.py

# 选择模式 1（从两张图片）
# 输入起始表情：neutral.jpg
# 输入结束表情：ahegao.jpg
# 时长：2.0
# FPS：30
# 缓动：4 (EASE_IN_OUT)
# 循环：1 (PING_PONG)

# 输出：loop_animation.json
```

### 示例 2：使用预设的阿黑颜动画

```bash
python generate_loop_animation.py

# 选择模式 2（阿黑颜预设）
# 强度：0.8
# 时长：4.0
# FPS：30

# 自动生成 4 个关键帧的渐进动画
# 输出：ahegao_animation.json
```

### 示例 3：从 MMD 视频提取

```bash
python extract_from_mmd_video.py

# 选择功能 2（提取关键帧并生成循环）
# 输入 MMD 视频：dance.mp4
# 关键帧数量：5
# 方法：2 (peaks)
# 循环时长：2
# FPS：30

# 输出：mmd_loop_animation.json
```

---

## 🎯 进阶功能

### 1. 添加程序化变化（更自然）

```python
# 基础动画
sequence = generator.create_simple_transition(bs_a, bs_b, 2.0, 30.0)

# 添加微小随机变化
natural_sequence = generator.add_procedural_variations(
    sequence,
    noise_amount=0.05,  # 5% 噪声
    target_blendshapes=['eyeLookUpLeft', 'eyeLookUpRight', 'tongueOut']
)
```

### 2. 混合两个动画

```python
# 有两个不同的动画
anim_a = generate_animation_a()
anim_b = generate_animation_b()

# 混合它们
blended = generator.blend_animations(
    anim_a,
    anim_b,
    blend_factor=0.5  # 50% A + 50% B
)
```

### 3. 自定义缓动函数

```python
def custom_easing(t):
    """自定义缓动：快速开始，缓慢结束，中间有停顿"""
    if t < 0.2:
        return t * 5  # 快速
    elif t < 0.7:
        return 1.0  # 停顿
    else:
        return 1.0 + (t - 0.7) * 0.33  # 缓慢

# 使用
result = generator.interpolate_blendshapes(
    bs_a, bs_b, t, custom_easing
)
```

### 4. 分析 MMD 表情模式

```python
from facial_capture.core.video_expression_extractor import MMDExpressionAnalyzer

analyzer = MMDExpressionAnalyzer()
result = analyzer.extract_mmd_patterns("mmd_dance.mp4")

# 查看结果
print(f"眨眼次数: {len(result['mmd_patterns']['blinks'])}")
print(f"微笑峰值: {len(result['mmd_patterns']['smile_peaks'])}")

# 最活跃的 blendshapes
for name, data in result['analysis']['top_active']:
    print(f"{name}: {data['active_ratio']*100:.1f}% 活跃")
```

---

## 💡 实用技巧

### 技巧 1：阿黑颜渐进效果

使用多个关键帧模拟渐进的高潮过程：

```python
from facial_capture.core.animation_generator import R18AnimationPresets

# 获取预设关键帧
keyframes = R18AnimationPresets.ahegao_transition(intensity=0.8)

# 关键帧包括：
# 0.0 - 中性表情
# 0.3 - 开始反应（眼睛睁大，嘴巴微开）
# 0.7 - 高潮阶段（舌头伸出，眼睛翻白）
# 1.0 - 持续高潮（微小震颤）

generator.create_multi_stage_animation(keyframes, 4.0, 30.0)
```

### 技巧 2：呼吸循环（增加真实感）

```python
breathing = R18AnimationPresets.breathing_cycle()

# 在主动画上叠加呼吸
main_animation = generate_main_expression()
breathing_animation = generator.create_multi_stage_animation(breathing, 2.0, 30.0)

# 混合（主表情 90% + 呼吸 10%）
final = generator.blend_animations(main_animation, breathing_animation, 0.1)
```

### 技巧 3：从视频特定片段提取

```python
extractor = VideoExpressionExtractor()

# 只提取 5-10 秒的片段
sequence, fps = extractor.extract_from_video(
    "mmd_video.mp4",
    start_time=5.0,
    end_time=10.0,
    skip_frames=1  # 不跳帧，保留所有细节
)
```

### 技巧 4：找视频中最佳循环点

```python
# 自动找相似的起止点
loop = extractor.create_loop_from_video(
    "mmd_video.mp4",
    loop_duration=2.0,
    method="seamless"  # 会自动寻找最佳循环点
)
```

---

## 📤 导出格式

生成的动画可以导出为多种格式：

### JSON（通用）
```python
exporter.export_json_sequence(sequence, "animation.json", fps=30.0)
```

### Blender Action
```python
exporter.export_blender_action(sequence, "animation_blender.json", fps=30.0)
```

### Unity AnimationClip
```python
exporter.export_unity_animation(sequence, "animation_unity.json", fps=30.0)
```

### CSV（数据分析）
```python
exporter.export_csv(sequence, "animation.csv", fps=30.0)
```

---

## 🎥 完整工作流程

### 工作流 A：两张图片 → 循环动画

```
1. 准备两张 R18 表情图片
   - neutral.jpg（起始）
   - peak.jpg（高潮）

2. 运行生成脚本
   python generate_loop_animation.py

3. 选择参数
   - 时长：2 秒
   - 缓动：ELASTIC（弹性）
   - 循环：PING_PONG

4. 导入 Blender/Unity
   - 使用插件导入 JSON
   - 应用到模型

5. 渲染/导出
```

### 工作流 B：MMD 视频 → 循环动画

```
1. 找一个 MMD 舞蹈/表情视频

2. 提取关键帧
   python extract_from_mmd_video.py
   选择功能 2

3. 选择参数
   - 关键帧数：5-10
   - 方法：peaks（峰值）

4. 生成循环
   - 循环时长：1-3 秒
   - 方法：seamless

5. 应用到模型
```

### 工作流 C：自定义多阶段

```
1. 准备多张表情图片
   - stage1.jpg（0%）
   - stage2.jpg（30%）
   - stage3.jpg（70%）
   - stage4.jpg（100%）

2. 运行自定义脚本
   python generate_loop_animation.py
   选择模式 3

3. 输入关键帧
   0.0 stage1.jpg 4
   0.3 stage2.jpg 5
   0.7 stage3.jpg 8
   1.0 stage4.jpg 10

4. 生成并导出
```

---

## ⚡ 性能优化

### 视频处理加速

```python
# 方法 1：跳帧
sequence, fps = extractor.extract_from_video(
    "video.mp4",
    skip_frames=3  # 只处理 1/3 的帧
)

# 方法 2：只处理特定片段
sequence, fps = extractor.extract_from_video(
    "video.mp4",
    start_time=10.0,
    end_time=15.0  # 只处理 5 秒
)

# 方法 3：降低输出帧率
sequence, fps = extractor.extract_from_video(
    "video.mp4",
    fps=15.0  # 从 30fps 降到 15fps
)
```

---

## 🐛 故障排除

### Q: 动画太僵硬，不自然
**A:** 尝试：
1. 使用 `ELASTIC` 或 `SPRING` 缓动
2. 添加程序化变化 `add_procedural_variations()`
3. 增加关键帧数量

### Q: MMD 视频检测失败
**A:**
1. 确保视频中面部清晰可见
2. 使用 `skip_frames=1` 不跳帧
3. 尝试裁剪聚焦面部区域

### Q: 循环不平滑，有跳跃
**A:**
1. 使用 `LoopMode.SEAMLESS` 而不是 `LOOP`
2. 确保起止表情相似
3. 增加动画时长

### Q: 生成的动画太快/太慢
**A:**
1. 调整 `duration` 参数（秒）
2. 调整 `fps` 参数
3. 使用不同的缓动类型

---

## 📚 API 参考

### ProceduralAnimationGenerator

主要方法：
- `create_simple_transition()` - 两个表情过渡
- `create_multi_stage_animation()` - 多关键帧动画
- `add_procedural_variations()` - 添加变化
- `blend_animations()` - 混合动画

### VideoExpressionExtractor

主要方法：
- `extract_from_video()` - 提取完整序列
- `extract_keyframes()` - 提取关键帧
- `create_loop_from_video()` - 创建循环
- `analyze_expression_patterns()` - 分析模式

### MMDExpressionAnalyzer

主要方法：
- `extract_mmd_patterns()` - 提取 MMD 特有模式

---

## 🎉 总结

这个动画生成系统提供：

✅ **11 种缓动效果** - 从线性到弹性
✅ **4 种循环模式** - 满足各种需求
✅ **MMD 视频支持** - 从视频提取表情
✅ **程序化变化** - 更自然的动画
✅ **多格式导出** - Blender、Unity、JSON

适用于：
- R18 表情循环动画
- 阿黑颜渐进效果
- MMD 舞蹈表情提取
- 自定义多阶段动画

**开始创作您的动画吧！** 🚀
