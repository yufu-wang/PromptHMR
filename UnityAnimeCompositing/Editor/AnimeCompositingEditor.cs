using UnityEngine;
using UnityEditor;

namespace AnimeCompositing.Editor
{
    /// <summary>
    /// Anime Compositing 系统的编辑器工具
    /// </summary>
    public class AnimeCompositingEditor : EditorWindow
    {
        [MenuItem("Window/Anime Compositing/Setup Wizard")]
        public static void ShowWindow()
        {
            AnimeCompositingEditor window = GetWindow<AnimeCompositingEditor>("Anime Compositing Setup");
            window.minSize = new Vector2(400, 500);
            window.Show();
        }

        private GameObject selectedCamera;
        private Material testMaterial;

        void OnGUI()
        {
            GUILayout.Label("日本动画风格合成系统设置向导", EditorStyles.boldLabel);
            GUILayout.Space(10);

            EditorGUILayout.HelpBox(
                "这个向导将帮助您在场景中设置 Anime Compositing 系统。\n" +
                "系统使用 MRT (Multiple Render Targets) 技术实现三层级渲染：\n" +
                "• 基础色层 (Base Color)\n" +
                "• 阴影层 (Shadow Layer)\n" +
                "• 高光层 (Highlight Layer)",
                MessageType.Info);

            GUILayout.Space(15);

            // 步骤 1：选择摄像机
            GUILayout.Label("步骤 1: 选择主摄像机", EditorStyles.boldLabel);
            selectedCamera = EditorGUILayout.ObjectField("主摄像机", selectedCamera, typeof(GameObject), true) as GameObject;

            if (GUILayout.Button("自动查找主摄像机"))
            {
                Camera mainCam = Camera.main;
                if (mainCam != null)
                {
                    selectedCamera = mainCam.gameObject;
                }
                else
                {
                    EditorUtility.DisplayDialog("未找到", "场景中没有标记为 MainCamera 的摄像机", "确定");
                }
            }

            GUILayout.Space(10);

            // 步骤 2：设置组件
            GUILayout.Label("步骤 2: 设置渲染组件", EditorStyles.boldLabel);

            GUI.enabled = selectedCamera != null;
            if (GUILayout.Button("添加 Anime Compositing 组件", GUILayout.Height(30)))
            {
                SetupAnimeCompositing();
            }
            GUI.enabled = true;

            GUILayout.Space(15);

            // 步骤 3：创建测试材质
            GUILayout.Label("步骤 3: 创建测试材质", EditorStyles.boldLabel);

            if (GUILayout.Button("创建 Anime Character 材质"))
            {
                CreateTestMaterial();
            }

            if (testMaterial != null)
            {
                EditorGUILayout.ObjectField("创建的材质", testMaterial, typeof(Material), false);
            }

            GUILayout.Space(15);

            // 帮助信息
            GUILayout.Label("使用说明", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "1. 确保场景中有主摄像机\n" +
                "2. 点击 '添加 Anime Compositing 组件' 按钮\n" +
                "3. 为您的角色模型创建并应用 Anime Character 材质\n" +
                "4. 调整合成器参数以获得期望的效果",
                MessageType.None);

            GUILayout.Space(10);

            if (GUILayout.Button("打开文档"))
            {
                Application.OpenURL("https://github.com/your-repo/anime-compositing/wiki");
            }
        }

        void SetupAnimeCompositing()
        {
            if (selectedCamera == null)
            {
                EditorUtility.DisplayDialog("错误", "请先选择一个摄像机", "确定");
                return;
            }

            // 添加 AnimeRenderManager
            AnimeRenderManager renderManager = selectedCamera.GetComponent<AnimeRenderManager>();
            if (renderManager == null)
            {
                renderManager = selectedCamera.AddComponent<AnimeRenderManager>();
                Debug.Log($"已添加 AnimeRenderManager 到 {selectedCamera.name}");
            }

            // 添加 AnimeLayerCompositor
            AnimeLayerCompositor compositor = selectedCamera.GetComponent<AnimeLayerCompositor>();
            if (compositor == null)
            {
                compositor = selectedCamera.AddComponent<AnimeLayerCompositor>();
                Debug.Log($"已添加 AnimeLayerCompositor 到 {selectedCamera.name}");
            }

            EditorUtility.DisplayDialog("成功", "Anime Compositing 组件已成功添加到摄像机！", "确定");
        }

        void CreateTestMaterial()
        {
            string path = "Assets/AnimeCharacterMaterial.mat";

            // 检查 Shader 是否存在
            Shader shader = Shader.Find("AnimeCompositing/CharacterMRT");
            if (shader == null)
            {
                EditorUtility.DisplayDialog("错误",
                    "找不到 'AnimeCompositing/CharacterMRT' Shader。\n" +
                    "请确保已将 Shader 文件导入到项目中。",
                    "确定");
                return;
            }

            // 创建材质
            testMaterial = new Material(shader);

            // 保存材质
            AssetDatabase.CreateAsset(testMaterial, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorUtility.DisplayDialog("成功",
                $"材质已创建：{path}\n" +
                "您现在可以将此材质应用到角色模型上。",
                "确定");

            // 选中创建的材质
            Selection.activeObject = testMaterial;
            EditorGUIUtility.PingObject(testMaterial);
        }
    }

    /// <summary>
    /// AnimeLayerCompositor 的自定义 Inspector
    /// </summary>
    [CustomEditor(typeof(AnimeLayerCompositor))]
    public class AnimeLayerCompositorInspector : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            AnimeLayerCompositor compositor = (AnimeLayerCompositor)target;

            EditorGUILayout.HelpBox(
                "日本动画风格合成器 - 控制三层级渲染的合成参数",
                MessageType.Info);

            GUILayout.Space(10);

            DrawDefaultInspector();

            GUILayout.Space(10);

            // 预设按钮
            GUILayout.Label("快速预设", EditorStyles.boldLabel);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("标准动画"))
            {
                ApplyStandardPreset(compositor);
            }
            if (GUILayout.Button("高对比度"))
            {
                ApplyHighContrastPreset(compositor);
            }
            if (GUILayout.Button("柔和风格"))
            {
                ApplySoftPreset(compositor);
            }
            GUILayout.EndHorizontal();
        }

        void ApplyStandardPreset(AnimeLayerCompositor compositor)
        {
            compositor.baseColorIntensity = 1.0f;
            compositor.shadowIntensity = 1.0f;
            compositor.highlightIntensity = 1.0f;
            compositor.contrast = 1.2f;
            compositor.saturation = 1.1f;
            compositor.enableOutline = true;
            compositor.outlineThickness = 1.0f;
            EditorUtility.SetDirty(compositor);
        }

        void ApplyHighContrastPreset(AnimeLayerCompositor compositor)
        {
            compositor.baseColorIntensity = 1.2f;
            compositor.shadowIntensity = 1.5f;
            compositor.highlightIntensity = 1.3f;
            compositor.contrast = 1.5f;
            compositor.saturation = 1.3f;
            compositor.enableOutline = true;
            compositor.outlineThickness = 1.5f;
            EditorUtility.SetDirty(compositor);
        }

        void ApplySoftPreset(AnimeLayerCompositor compositor)
        {
            compositor.baseColorIntensity = 0.9f;
            compositor.shadowIntensity = 0.7f;
            compositor.highlightIntensity = 0.8f;
            compositor.contrast = 0.9f;
            compositor.saturation = 0.9f;
            compositor.enableOutline = true;
            compositor.outlineThickness = 0.8f;
            EditorUtility.SetDirty(compositor);
        }
    }
}
