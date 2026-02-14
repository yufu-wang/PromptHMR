using System;
using System.Collections.Generic;
using UnityEngine;

namespace FacialExpressionCapture
{
    /// <summary>
    /// ARKit 标准 52 个 Blendshape 定义
    /// </summary>
    public static class ARKitBlendshapeNames
    {
        // 所有 52 个 ARKit blendshape 名称
        public static readonly string[] AllNames = new string[]
        {
            // 眼部 (16个)
            "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookInLeft", "eyeLookInRight",
            "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight",
            "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight",

            // 眉毛 (8个)
            "browDownLeft", "browDownRight",
            "browInnerUp",
            "browOuterUpLeft", "browOuterUpRight",

            // 嘴部 (28个)
            "mouthClose",
            "mouthFunnel", "mouthPucker",
            "mouthLeft", "mouthRight",
            "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight",
            "mouthDimpleLeft", "mouthDimpleRight",
            "mouthStretchLeft", "mouthStretchRight",
            "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper",
            "mouthPressLeft", "mouthPressRight",
            "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",

            // 脸颊和下巴 (6个)
            "cheekPuff",
            "cheekSquintLeft", "cheekSquintRight",
            "jawOpen", "jawForward",
            "jawLeft", "jawRight",

            // 鼻子 (2个)
            "noseSneerLeft", "noseSneerRight",

            // 舌头 (1个)
            "tongueOut"
        };

        public static readonly int Count = AllNames.Length;
    }

    /// <summary>
    /// ARKit Blendshape 数据类
    /// </summary>
    [Serializable]
    public class ARKitBlendshapeData
    {
        [SerializeField]
        private float[] values = new float[ARKitBlendshapeNames.Count];

        public ARKitBlendshapeData()
        {
            Reset();
        }

        /// <summary>
        /// 重置所有值为 0
        /// </summary>
        public void Reset()
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 0f;
            }
        }

        /// <summary>
        /// 设置 blendshape 值 (0-100 范围，Unity 标准)
        /// </summary>
        public void Set(string name, float value)
        {
            int index = Array.IndexOf(ARKitBlendshapeNames.AllNames, name);
            if (index >= 0)
            {
                values[index] = Mathf.Clamp(value, 0f, 100f);
            }
            else
            {
                Debug.LogWarning($"Invalid blendshape name: {name}");
            }
        }

        /// <summary>
        /// 设置 blendshape 值 (通过索引)
        /// </summary>
        public void SetByIndex(int index, float value)
        {
            if (index >= 0 && index < values.Length)
            {
                values[index] = Mathf.Clamp(value, 0f, 100f);
            }
        }

        /// <summary>
        /// 获取 blendshape 值
        /// </summary>
        public float Get(string name)
        {
            int index = Array.IndexOf(ARKitBlendshapeNames.AllNames, name);
            return index >= 0 ? values[index] : 0f;
        }

        /// <summary>
        /// 获取 blendshape 值 (通过索引)
        /// </summary>
        public float GetByIndex(int index)
        {
            if (index >= 0 && index < values.Length)
                return values[index];
            return 0f;
        }

        /// <summary>
        /// 获取所有值的数组
        /// </summary>
        public float[] GetAllValues()
        {
            return (float[])values.Clone();
        }

        /// <summary>
        /// 从数组加载值
        /// </summary>
        public void FromArray(float[] arr)
        {
            if (arr.Length != values.Length)
            {
                Debug.LogError($"Array length must be {values.Length}");
                return;
            }

            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Mathf.Clamp(arr[i], 0f, 100f);
            }
        }

        /// <summary>
        /// 转换为字典
        /// </summary>
        public Dictionary<string, float> ToDictionary()
        {
            var dict = new Dictionary<string, float>();
            for (int i = 0; i < ARKitBlendshapeNames.AllNames.Length; i++)
            {
                dict[ARKitBlendshapeNames.AllNames[i]] = values[i];
            }
            return dict;
        }

        /// <summary>
        /// 从字典加载
        /// </summary>
        public void FromDictionary(Dictionary<string, float> dict)
        {
            foreach (var kvp in dict)
            {
                Set(kvp.Key, kvp.Value);
            }
        }
    }
}
