using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

namespace FacialExpressionCapture.Editor
{
    /// <summary>
    /// Unity 编辑器窗口 - 面部表情捕捉
    /// </summary>
    public class FacialCaptureWindow : EditorWindow
    {
        private string jsonFilePath = "";
        private SkinnedMeshRenderer targetRenderer;
        private ARKitBlendshapeData currentExpression;
        private List<ARKitBlendshapeData> expressionSequence;
        private float sequenceFps = 30f;
        private bool isSequence = false;

        // Python 脚本路径（用于直接从图像提取）
        private string pythonScriptPath = "";
        private string imageFilePath = "";

        [MenuItem("Window/Facial Expression Capture")]
        public static void ShowWindow()
        {
            GetWindow<FacialCaptureWindow>("Facial Capture");
        }

        void OnGUI()
        {
            GUILayout.Label("Facial Expression Capture", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            // 目标对象
            EditorGUILayout.LabelField("Target Object", EditorStyles.boldLabel);
            targetRenderer = (SkinnedMeshRenderer)EditorGUILayout.ObjectField(
                "Skinned Mesh Renderer",
                targetRenderer,
                typeof(SkinnedMeshRenderer),
                true
            );
            EditorGUILayout.Space();

            // 导入 JSON
            EditorGUILayout.LabelField("Import Expression Data", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            jsonFilePath = EditorGUILayout.TextField("JSON File", jsonFilePath);
            if (GUILayout.Button("Browse", GUILayout.Width(80)))
            {
                jsonFilePath = EditorUtility.OpenFilePanel("Select Expression JSON", "", "json");
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Import Single Frame"))
            {
                ImportSingleFrame();
            }
            if (GUILayout.Button("Import Sequence"))
            {
                ImportSequence();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space();

            // 应用表情
            if (currentExpression != null && !isSequence)
            {
                EditorGUILayout.LabelField("Current Expression", EditorStyles.boldLabel);

                if (GUILayout.Button("Apply to Target"))
                {
                    ApplyCurrentExpression();
                }

                // 显示 blendshape 值
                if (GUILayout.Button(showBlendshapeValues ? "Hide Values" : "Show Values"))
                {
                    showBlendshapeValues = !showBlendshapeValues;
                }

                if (showBlendshapeValues)
                {
                    DrawBlendshapeValues();
                }
            }

            // 序列数据
            if (expressionSequence != null && expressionSequence.Count > 0)
            {
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Expression Sequence", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Frames: {expressionSequence.Count}");
                EditorGUILayout.LabelField($"FPS: {sequenceFps}");
                EditorGUILayout.LabelField($"Duration: {expressionSequence.Count / sequenceFps:F2}s");

                if (GUILayout.Button("Create Animation Clip"))
                {
                    CreateAnimationClipFromSequence();
                }
            }

            EditorGUILayout.Space();

            // 从图像提取（需要 Python）
            DrawImageExtractionSection();

            EditorGUILayout.Space();

            // 工具
            DrawToolsSection();
        }

        private bool showBlendshapeValues = false;

        void DrawBlendshapeValues()
        {
            EditorGUILayout.BeginVertical("box");

            for (int i = 0; i < ARKitBlendshapeNames.Count; i++)
            {
                string name = ARKitBlendshapeNames.AllNames[i];
                float value = currentExpression.GetByIndex(i);

                if (value > 0.1f) // 只显示非零值
                {
                    EditorGUILayout.LabelField($"{name}: {value:F2}");
                }
            }

            EditorGUILayout.EndVertical();
        }

        void DrawImageExtractionSection()
        {
            EditorGUILayout.LabelField("Extract from Image (Experimental)", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "需要安装 Python 和 facial_capture 模块\n" +
                "pip install mediapipe opencv-python numpy",
                MessageType.Info
            );

            EditorGUILayout.BeginHorizontal();
            imageFilePath = EditorGUILayout.TextField("Image File", imageFilePath);
            if (GUILayout.Button("Browse", GUILayout.Width(80)))
            {
                imageFilePath = EditorUtility.OpenFilePanel("Select Image", "", "jpg,png,jpeg");
            }
            EditorGUILayout.EndHorizontal();

            if (GUILayout.Button("Extract Expression from Image"))
            {
                ExtractFromImage();
            }
        }

        void DrawToolsSection()
        {
            EditorGUILayout.LabelField("Tools", EditorStyles.boldLabel);

            if (GUILayout.Button("Create ARKit Blendshapes on Target"))
            {
                CreateARKitBlendshapes();
            }

            if (GUILayout.Button("Reset All Blendshapes"))
            {
                ResetAllBlendshapes();
            }
        }

        void ImportSingleFrame()
        {
            if (string.IsNullOrEmpty(jsonFilePath))
            {
                EditorUtility.DisplayDialog("Error", "Please select a JSON file", "OK");
                return;
            }

            currentExpression = FacialCaptureImporter.ImportSingleFrame(jsonFilePath);
            isSequence = false;

            if (currentExpression != null)
            {
                EditorUtility.DisplayDialog("Success", "Expression imported successfully", "OK");
            }
        }

        void ImportSequence()
        {
            if (string.IsNullOrEmpty(jsonFilePath))
            {
                EditorUtility.DisplayDialog("Error", "Please select a JSON file", "OK");
                return;
            }

            expressionSequence = FacialCaptureImporter.ImportSequence(jsonFilePath, out sequenceFps);
            isSequence = true;

            if (expressionSequence != null && expressionSequence.Count > 0)
            {
                EditorUtility.DisplayDialog(
                    "Success",
                    $"Imported {expressionSequence.Count} frames",
                    "OK"
                );
            }
        }

        void ApplyCurrentExpression()
        {
            if (targetRenderer == null)
            {
                EditorUtility.DisplayDialog("Error", "Please select a target renderer", "OK");
                return;
            }

            if (currentExpression == null)
            {
                EditorUtility.DisplayDialog("Error", "No expression data loaded", "OK");
                return;
            }

            FacialCaptureImporter.ApplyToSkinnedMesh(targetRenderer, currentExpression);
        }

        void CreateAnimationClipFromSequence()
        {
            if (targetRenderer == null)
            {
                EditorUtility.DisplayDialog("Error", "Please select a target renderer", "OK");
                return;
            }

            if (expressionSequence == null || expressionSequence.Count == 0)
            {
                EditorUtility.DisplayDialog("Error", "No sequence data loaded", "OK");
                return;
            }

            string path = EditorUtility.SaveFilePanelInProject(
                "Save Animation Clip",
                "FacialExpression",
                "anim",
                "Save animation clip"
            );

            if (string.IsNullOrEmpty(path))
                return;

            // 获取相对路径
            string meshPath = "";
            var clip = FacialCaptureImporter.CreateAnimationClip(
                expressionSequence,
                sequenceFps,
                meshPath
            );

            AssetDatabase.CreateAsset(clip, path);
            AssetDatabase.SaveAssets();

            EditorUtility.DisplayDialog("Success", $"Animation clip created: {path}", "OK");
        }

        void ExtractFromImage()
        {
            EditorUtility.DisplayDialog(
                "Not Implemented",
                "从图像直接提取需要 Python 环境。\n" +
                "请使用 Python 脚本先提取表情为 JSON，然后导入。\n\n" +
                "示例命令:\n" +
                "python extract_expression.py input.jpg output.json",
                "OK"
            );
        }

        void CreateARKitBlendshapes()
        {
            if (targetRenderer == null || targetRenderer.sharedMesh == null)
            {
                EditorUtility.DisplayDialog("Error", "Please select a valid target", "OK");
                return;
            }

            EditorUtility.DisplayDialog(
                "Info",
                "Unity 不支持运行时创建 blendshapes。\n" +
                "请在 3D 软件（如 Blender）中创建 ARKit blendshapes，\n" +
                "然后导入到 Unity。",
                "OK"
            );
        }

        void ResetAllBlendshapes()
        {
            if (targetRenderer == null)
            {
                EditorUtility.DisplayDialog("Error", "Please select a target renderer", "OK");
                return;
            }

            var mesh = targetRenderer.sharedMesh;
            int count = mesh.blendShapeCount;

            for (int i = 0; i < count; i++)
            {
                targetRenderer.SetBlendShapeWeight(i, 0f);
            }

            Debug.Log($"Reset {count} blendshapes");
        }
    }
}
