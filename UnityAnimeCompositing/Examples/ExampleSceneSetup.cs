using UnityEngine;

namespace AnimeCompositing.Examples
{
    /// <summary>
    /// 示例场景设置脚本
    /// 演示如何在运行时动态设置 Anime Compositing 系统
    /// </summary>
    public class ExampleSceneSetup : MonoBehaviour
    {
        [Header("Preset Configurations")]
        public AnimePresetType presetType = AnimePresetType.Standard;

        public enum AnimePresetType
        {
            Standard,       // 标准动画风格
            HighContrast,   // 高对比度
            Soft,           // 柔和风格
            Dramatic,       // 戏剧性
            Pastel          // 粉彩风格
        }

        [Header("Runtime Controls")]
        public bool allowRuntimeChange = true;
        public KeyCode toggleKey = KeyCode.Space;
        public KeyCode nextPresetKey = KeyCode.N;

        private AnimeLayerCompositor compositor;
        private int currentPresetIndex = 0;

        void Start()
        {
            // 查找或创建合成器
            compositor = Camera.main?.GetComponent<AnimeLayerCompositor>();

            if (compositor == null)
            {
                Debug.LogError("[ExampleSceneSetup] 未找到 AnimeLayerCompositor！请确保主摄像机上已添加该组件。");
                return;
            }

            // 应用预设
            ApplyPreset(presetType);
        }

        void Update()
        {
            if (!allowRuntimeChange || compositor == null) return;

            // 切换开关
            if (Input.GetKeyDown(toggleKey))
            {
                compositor.enabled = !compositor.enabled;
                Debug.Log($"Anime Compositing: {(compositor.enabled ? "启用" : "禁用")}");
            }

            // 切换预设
            if (Input.GetKeyDown(nextPresetKey))
            {
                currentPresetIndex = (currentPresetIndex + 1) % System.Enum.GetValues(typeof(AnimePresetType)).Length;
                presetType = (AnimePresetType)currentPresetIndex;
                ApplyPreset(presetType);
                Debug.Log($"切换到预设: {presetType}");
            }
        }

        void OnGUI()
        {
            if (!allowRuntimeChange) return;

            GUILayout.BeginArea(new Rect(10, 10, 300, 200));
            GUILayout.BeginVertical("box");

            GUILayout.Label("Anime Compositing 控制", GUI.skin.box);
            GUILayout.Space(5);

            GUILayout.Label($"当前预设: {presetType}");
            GUILayout.Label($"状态: {(compositor != null && compositor.enabled ? "启用" : "禁用")}");

            GUILayout.Space(10);

            GUILayout.Label($"按 [{toggleKey}] 切换开关");
            GUILayout.Label($"按 [{nextPresetKey}] 切换预设");

            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        public void ApplyPreset(AnimePresetType preset)
        {
            if (compositor == null) return;

            switch (preset)
            {
                case AnimePresetType.Standard:
                    ApplyStandardPreset();
                    break;

                case AnimePresetType.HighContrast:
                    ApplyHighContrastPreset();
                    break;

                case AnimePresetType.Soft:
                    ApplySoftPreset();
                    break;

                case AnimePresetType.Dramatic:
                    ApplyDramaticPreset();
                    break;

                case AnimePresetType.Pastel:
                    ApplyPastelPreset();
                    break;
            }
        }

        void ApplyStandardPreset()
        {
            compositor.baseColorIntensity = 1.0f;
            compositor.shadowIntensity = 1.0f;
            compositor.highlightIntensity = 1.0f;

            compositor.shadowBlendMode = AnimeLayerCompositor.BlendMode.Multiply;
            compositor.highlightBlendMode = AnimeLayerCompositor.BlendMode.Add;

            compositor.enableColorGrading = true;
            compositor.colorTint = Color.white;
            compositor.contrast = 1.2f;
            compositor.saturation = 1.1f;

            compositor.enableOutline = true;
            compositor.outlineColor = Color.black;
            compositor.outlineThickness = 1.0f;
        }

        void ApplyHighContrastPreset()
        {
            compositor.baseColorIntensity = 1.2f;
            compositor.shadowIntensity = 1.5f;
            compositor.highlightIntensity = 1.3f;

            compositor.shadowBlendMode = AnimeLayerCompositor.BlendMode.Multiply;
            compositor.highlightBlendMode = AnimeLayerCompositor.BlendMode.Add;

            compositor.enableColorGrading = true;
            compositor.colorTint = new Color(1.0f, 1.0f, 1.0f);
            compositor.contrast = 1.5f;
            compositor.saturation = 1.3f;

            compositor.enableOutline = true;
            compositor.outlineColor = Color.black;
            compositor.outlineThickness = 1.5f;
        }

        void ApplySoftPreset()
        {
            compositor.baseColorIntensity = 0.9f;
            compositor.shadowIntensity = 0.7f;
            compositor.highlightIntensity = 0.8f;

            compositor.shadowBlendMode = AnimeLayerCompositor.BlendMode.Multiply;
            compositor.highlightBlendMode = AnimeLayerCompositor.BlendMode.Screen;

            compositor.enableColorGrading = true;
            compositor.colorTint = new Color(1.0f, 0.98f, 0.95f);
            compositor.contrast = 0.9f;
            compositor.saturation = 0.9f;

            compositor.enableOutline = true;
            compositor.outlineColor = new Color(0.2f, 0.2f, 0.2f);
            compositor.outlineThickness = 0.8f;
        }

        void ApplyDramaticPreset()
        {
            compositor.baseColorIntensity = 1.1f;
            compositor.shadowIntensity = 1.8f;
            compositor.highlightIntensity = 1.5f;

            compositor.shadowBlendMode = AnimeLayerCompositor.BlendMode.Multiply;
            compositor.highlightBlendMode = AnimeLayerCompositor.BlendMode.Add;

            compositor.enableColorGrading = true;
            compositor.colorTint = new Color(0.95f, 0.95f, 1.0f);
            compositor.contrast = 1.6f;
            compositor.saturation = 1.2f;

            compositor.enableOutline = true;
            compositor.outlineColor = Color.black;
            compositor.outlineThickness = 2.0f;
        }

        void ApplyPastelPreset()
        {
            compositor.baseColorIntensity = 0.85f;
            compositor.shadowIntensity = 0.6f;
            compositor.highlightIntensity = 1.2f;

            compositor.shadowBlendMode = AnimeLayerCompositor.BlendMode.Multiply;
            compositor.highlightBlendMode = AnimeLayerCompositor.BlendMode.Screen;

            compositor.enableColorGrading = true;
            compositor.colorTint = new Color(1.0f, 0.95f, 0.98f);
            compositor.contrast = 0.8f;
            compositor.saturation = 0.85f;

            compositor.enableOutline = true;
            compositor.outlineColor = new Color(0.3f, 0.25f, 0.3f);
            compositor.outlineThickness = 0.7f;
        }
    }
}
