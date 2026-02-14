using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace FacialExpressionCapture
{
    /// <summary>
    /// JSON 数据格式 (单帧)
    /// </summary>
    [Serializable]
    public class ExpressionDataSingle
    {
        public string version;
        public string format;
        public Dictionary<string, float> blendshapes;
    }

    /// <summary>
    /// JSON 数据格式 (序列)
    /// </summary>
    [Serializable]
    public class ExpressionDataSequence
    {
        public string version;
        public string format;
        public float fps;
        public int num_frames;
        public float duration;
        public List<FrameData> frames;
    }

    [Serializable]
    public class FrameData
    {
        public int frame;
        public float time;
        public Dictionary<string, float> blendshapes;
    }

    /// <summary>
    /// 面部表情导入器
    /// </summary>
    public class FacialCaptureImporter
    {
        /// <summary>
        /// 从 JSON 文件导入单帧数据
        /// </summary>
        public static ARKitBlendshapeData ImportSingleFrame(string jsonPath)
        {
            try
            {
                string json = File.ReadAllText(jsonPath);
                var data = JsonUtility.FromJson<ExpressionDataSingle>(json);

                if (data == null || data.blendshapes == null)
                {
                    Debug.LogError("Invalid JSON format");
                    return null;
                }

                var blendshapes = new ARKitBlendshapeData();

                // 从 0-1 范围转换为 0-100 范围 (Unity 标准)
                foreach (var kvp in data.blendshapes)
                {
                    blendshapes.Set(kvp.Key, kvp.Value * 100f);
                }

                Debug.Log($"Imported single frame expression from: {jsonPath}");
                return blendshapes;
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to import expression: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// 从 JSON 文件导入序列数据
        /// </summary>
        public static List<ARKitBlendshapeData> ImportSequence(string jsonPath, out float fps)
        {
            fps = 30f;

            try
            {
                string json = File.ReadAllText(jsonPath);
                var data = JsonUtility.FromJson<ExpressionDataSequence>(json);

                if (data == null || data.frames == null)
                {
                    Debug.LogError("Invalid JSON format");
                    return null;
                }

                fps = data.fps;
                var sequence = new List<ARKitBlendshapeData>();

                foreach (var frameData in data.frames)
                {
                    var blendshapes = new ARKitBlendshapeData();

                    // 从 0-1 范围转换为 0-100 范围
                    foreach (var kvp in frameData.blendshapes)
                    {
                        blendshapes.Set(kvp.Key, kvp.Value * 100f);
                    }

                    sequence.Add(blendshapes);
                }

                Debug.Log($"Imported {sequence.Count} frames at {fps} FPS from: {jsonPath}");
                return sequence;
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to import sequence: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// 应用 blendshapes 到 SkinnedMeshRenderer
        /// </summary>
        public static void ApplyToSkinnedMesh(SkinnedMeshRenderer renderer, ARKitBlendshapeData blendshapes)
        {
            if (renderer == null || renderer.sharedMesh == null)
            {
                Debug.LogError("Invalid SkinnedMeshRenderer");
                return;
            }

            int appliedCount = 0;
            var mesh = renderer.sharedMesh;

            for (int i = 0; i < ARKitBlendshapeNames.Count; i++)
            {
                string blendshapeName = ARKitBlendshapeNames.AllNames[i];
                float value = blendshapes.GetByIndex(i);

                // 查找对应的 blendshape 索引
                int blendshapeIndex = mesh.GetBlendShapeIndex(blendshapeName);

                if (blendshapeIndex >= 0)
                {
                    renderer.SetBlendShapeWeight(blendshapeIndex, value);
                    appliedCount++;
                }
            }

            if (appliedCount == 0)
            {
                Debug.LogWarning("No matching blendshapes found. Check blendshape names.");
            }
            else
            {
                Debug.Log($"Applied {appliedCount} blendshapes");
            }
        }

        /// <summary>
        /// 创建动画片段
        /// </summary>
        public static AnimationClip CreateAnimationClip(List<ARKitBlendshapeData> sequence, float fps, string meshPath)
        {
            var clip = new AnimationClip();
            clip.frameRate = fps;

            // 为每个 blendshape 创建动画曲线
            for (int i = 0; i < ARKitBlendshapeNames.Count; i++)
            {
                string blendshapeName = ARKitBlendshapeNames.AllNames[i];
                var curve = new AnimationCurve();

                for (int frameIdx = 0; frameIdx < sequence.Count; frameIdx++)
                {
                    float time = frameIdx / fps;
                    float value = sequence[frameIdx].GetByIndex(i);
                    curve.AddKey(time, value);
                }

                // 绑定到 SkinnedMeshRenderer 的 blendshape
                string propertyPath = $"blendShape.{blendshapeName}";
                clip.SetCurve(meshPath, typeof(SkinnedMeshRenderer), propertyPath, curve);
            }

            Debug.Log($"Created animation clip with {sequence.Count} frames");
            return clip;
        }
    }
}
