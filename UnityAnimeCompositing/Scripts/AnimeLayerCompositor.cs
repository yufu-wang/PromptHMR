using UnityEngine;

namespace AnimeCompositing
{
    /// <summary>
    /// 日本动画风格的三层级合成器
    /// Three-layer compositor for Japanese animation style rendering
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [RequireComponent(typeof(AnimeRenderManager))]
    public class AnimeLayerCompositor : MonoBehaviour
    {
        [Header("Layer Composition Settings")]
        [Tooltip("基础色层强度")]
        [Range(0, 2)]
        public float baseColorIntensity = 1.0f;

        [Tooltip("阴影层强度")]
        [Range(0, 2)]
        public float shadowIntensity = 1.0f;

        [Tooltip("高光层强度")]
        [Range(0, 2)]
        public float highlightIntensity = 1.0f;

        [Header("Composition Modes")]
        [Tooltip("阴影混合模式")]
        public BlendMode shadowBlendMode = BlendMode.Multiply;

        [Tooltip("高光混合模式")]
        public BlendMode highlightBlendMode = BlendMode.Add;

        [Header("Post-Processing")]
        [Tooltip("启用色调调整")]
        public bool enableColorGrading = true;

        [Tooltip("色调偏移")]
        public Color colorTint = Color.white;

        [Tooltip("对比度")]
        [Range(0.5f, 2.0f)]
        public float contrast = 1.0f;

        [Tooltip("饱和度")]
        [Range(0f, 2.0f)]
        public float saturation = 1.0f;

        [Tooltip("启用边缘增强")]
        public bool enableOutline = true;

        [Tooltip("边缘颜色")]
        public Color outlineColor = Color.black;

        [Tooltip("边缘粗细")]
        [Range(0f, 5.0f)]
        public float outlineThickness = 1.0f;

        // 混合模式枚举
        public enum BlendMode
        {
            Normal,      // 正常
            Multiply,    // 正片叠底
            Add,         // 相加
            Screen,      // 滤色
            Overlay      // 叠加
        }

        private AnimeRenderManager renderManager;
        private Material compositeMaterial;
        private Material outlineMaterial;

        void Awake()
        {
            renderManager = GetComponent<AnimeRenderManager>();
            InitializeMaterials();
        }

        void OnDestroy()
        {
            if (compositeMaterial != null)
                Destroy(compositeMaterial);
            if (outlineMaterial != null)
                Destroy(outlineMaterial);
        }

        /// <summary>
        /// 初始化材质
        /// </summary>
        void InitializeMaterials()
        {
            // 创建合成材质
            Shader compositeShader = Shader.Find("Hidden/AnimeCompositing/Compositor");
            if (compositeShader == null)
            {
                Debug.LogError("[AnimeLayerCompositor] 找不到合成 Shader！请确保 Compositor.shader 已导入。");
                return;
            }
            compositeMaterial = new Material(compositeShader);

            // 创建描边材质
            Shader outlineShader = Shader.Find("Hidden/AnimeCompositing/Outline");
            if (outlineShader != null)
            {
                outlineMaterial = new Material(outlineShader);
            }
        }

        /// <summary>
        /// 图像后处理回调
        /// </summary>
        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            if (compositeMaterial == null || renderManager == null)
            {
                Graphics.Blit(source, destination);
                return;
            }

            // 获取三个渲染层
            RenderTexture baseColor = renderManager.BaseColorTexture;
            RenderTexture shadowLayer = renderManager.ShadowLayerTexture;
            RenderTexture highlightLayer = renderManager.HighlightLayerTexture;

            if (baseColor == null || shadowLayer == null || highlightLayer == null)
            {
                Graphics.Blit(source, destination);
                return;
            }

            // 设置材质属性
            compositeMaterial.SetTexture("_BaseColorTex", baseColor);
            compositeMaterial.SetTexture("_ShadowTex", shadowLayer);
            compositeMaterial.SetTexture("_HighlightTex", highlightLayer);

            compositeMaterial.SetFloat("_BaseIntensity", baseColorIntensity);
            compositeMaterial.SetFloat("_ShadowIntensity", shadowIntensity);
            compositeMaterial.SetFloat("_HighlightIntensity", highlightIntensity);

            compositeMaterial.SetInt("_ShadowBlendMode", (int)shadowBlendMode);
            compositeMaterial.SetInt("_HighlightBlendMode", (int)highlightBlendMode);

            // 色彩调整
            compositeMaterial.SetColor("_ColorTint", colorTint);
            compositeMaterial.SetFloat("_Contrast", contrast);
            compositeMaterial.SetFloat("_Saturation", saturation);
            compositeMaterial.SetInt("_EnableColorGrading", enableColorGrading ? 1 : 0);

            // 第一步：合成三个层
            RenderTexture compositeResult = RenderTexture.GetTemporary(
                source.width, source.height, 0, RenderTextureFormat.ARGBFloat);
            Graphics.Blit(source, compositeResult, compositeMaterial, 0);

            // 第二步：可选的描边处理
            if (enableOutline && outlineMaterial != null)
            {
                RenderTexture outlineResult = RenderTexture.GetTemporary(
                    source.width, source.height, 0, RenderTextureFormat.ARGBFloat);

                outlineMaterial.SetColor("_OutlineColor", outlineColor);
                outlineMaterial.SetFloat("_OutlineThickness", outlineThickness);
                outlineMaterial.SetTexture("_MainTex", compositeResult);

                Graphics.Blit(compositeResult, outlineResult, outlineMaterial, 0);

                Graphics.Blit(outlineResult, destination);
                RenderTexture.ReleaseTemporary(outlineResult);
            }
            else
            {
                Graphics.Blit(compositeResult, destination);
            }

            RenderTexture.ReleaseTemporary(compositeResult);
        }

        #if UNITY_EDITOR
        /// <summary>
        /// 在Inspector中显示帮助信息
        /// </summary>
        void OnValidate()
        {
            // 确保值在有效范围内
            baseColorIntensity = Mathf.Max(0, baseColorIntensity);
            shadowIntensity = Mathf.Max(0, shadowIntensity);
            highlightIntensity = Mathf.Max(0, highlightIntensity);
        }
        #endif
    }
}
