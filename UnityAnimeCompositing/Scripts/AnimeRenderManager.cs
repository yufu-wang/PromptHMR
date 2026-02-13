using UnityEngine;
using UnityEngine.Rendering;

namespace AnimeCompositing
{
    /// <summary>
    /// 管理 MRT 渲染流程的核心类
    /// Manages the MRT (Multiple Render Targets) rendering pipeline
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class AnimeRenderManager : MonoBehaviour
    {
        [Header("Render Targets")]
        [Tooltip("渲染分辨率倍数")]
        public float renderScale = 1.0f;

        [Tooltip("是否在Scene视图中显示调试信息")]
        public bool showDebugInScene = true;

        // MRT 渲染目标
        private RenderTexture baseColorRT;
        private RenderTexture shadowLayerRT;
        private RenderTexture highlightLayerRT;

        // 渲染缓冲
        private RenderBuffer[] colorBuffers = new RenderBuffer[3];
        private RenderBuffer depthBuffer;

        private Camera renderCamera;
        private CommandBuffer commandBuffer;

        // 公共访问接口
        public RenderTexture BaseColorTexture => baseColorRT;
        public RenderTexture ShadowLayerTexture => shadowLayerRT;
        public RenderTexture HighlightLayerTexture => highlightLayerRT;

        void Awake()
        {
            renderCamera = GetComponent<Camera>();
            InitializeRenderTargets();
            SetupCommandBuffer();
        }

        void OnDestroy()
        {
            ReleaseRenderTargets();
            if (commandBuffer != null)
            {
                renderCamera.RemoveCommandBuffer(CameraEvent.BeforeImageEffects, commandBuffer);
                commandBuffer.Release();
            }
        }

        void OnPreRender()
        {
            // 更新 MRT 设置
            UpdateMRTSetup();
        }

        void OnPostRender()
        {
            // 恢复默认渲染目标
            Graphics.SetRenderTarget(null);
        }

        /// <summary>
        /// 初始化渲染目标纹理
        /// </summary>
        void InitializeRenderTargets()
        {
            int width = Mathf.RoundToInt(Screen.width * renderScale);
            int height = Mathf.RoundToInt(Screen.height * renderScale);

            // 创建三个渲染目标
            baseColorRT = CreateRenderTexture(width, height, "BaseColor");
            shadowLayerRT = CreateRenderTexture(width, height, "ShadowLayer");
            highlightLayerRT = CreateRenderTexture(width, height, "HighlightLayer");

            Debug.Log($"[AnimeRenderManager] 已初始化 MRT 渲染目标: {width}x{height}");
        }

        /// <summary>
        /// 创建渲染纹理
        /// </summary>
        RenderTexture CreateRenderTexture(int width, int height, string name)
        {
            RenderTexture rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            rt.name = name;
            rt.filterMode = FilterMode.Bilinear;
            rt.wrapMode = TextureWrapMode.Clamp;
            rt.Create();
            return rt;
        }

        /// <summary>
        /// 释放渲染目标
        /// </summary>
        void ReleaseRenderTargets()
        {
            if (baseColorRT != null) baseColorRT.Release();
            if (shadowLayerRT != null) shadowLayerRT.Release();
            if (highlightLayerRT != null) highlightLayerRT.Release();
        }

        /// <summary>
        /// 设置命令缓冲区
        /// </summary>
        void SetupCommandBuffer()
        {
            commandBuffer = new CommandBuffer();
            commandBuffer.name = "AnimeCompositing MRT Setup";

            renderCamera.AddCommandBuffer(CameraEvent.BeforeImageEffects, commandBuffer);
        }

        /// <summary>
        /// 更新 MRT 设置
        /// </summary>
        void UpdateMRTSetup()
        {
            if (baseColorRT == null || shadowLayerRT == null || highlightLayerRT == null)
            {
                InitializeRenderTargets();
            }

            // 设置多渲染目标
            colorBuffers[0] = baseColorRT.colorBuffer;
            colorBuffers[1] = shadowLayerRT.colorBuffer;
            colorBuffers[2] = highlightLayerRT.colorBuffer;
            depthBuffer = baseColorRT.depthBuffer;

            Graphics.SetRenderTarget(colorBuffers, depthBuffer);
        }

        /// <summary>
        /// 在窗口重置大小时重新初始化渲染目标
        /// </summary>
        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            // 这里不做任何处理，让合成器来处理最终输出
            // 如果没有合成器，则直接传递源纹理
            Graphics.Blit(source, destination);
        }

        #if UNITY_EDITOR
        /// <summary>
        /// 在Scene视图中显示调试信息
        /// </summary>
        void OnGUI()
        {
            if (!showDebugInScene) return;

            int previewSize = 200;
            int padding = 10;
            int yOffset = padding;

            // 显示标题
            GUI.Label(new Rect(padding, yOffset, 300, 20), "Anime MRT Layers (Debug)");
            yOffset += 25;

            // 显示三个渲染目标
            if (baseColorRT != null)
            {
                GUI.Label(new Rect(padding, yOffset, previewSize, 20), "Base Color");
                GUI.DrawTexture(new Rect(padding, yOffset + 20, previewSize, previewSize), baseColorRT);
                yOffset += previewSize + 30;
            }

            if (shadowLayerRT != null)
            {
                GUI.Label(new Rect(padding, yOffset, previewSize, 20), "Shadow Layer");
                GUI.DrawTexture(new Rect(padding, yOffset + 20, previewSize, previewSize), shadowLayerRT);
                yOffset += previewSize + 30;
            }

            if (highlightLayerRT != null)
            {
                GUI.Label(new Rect(padding, yOffset, previewSize, 20), "Highlight Layer");
                GUI.DrawTexture(new Rect(padding, yOffset + 20, previewSize, previewSize), highlightLayerRT);
            }
        }
        #endif
    }
}
