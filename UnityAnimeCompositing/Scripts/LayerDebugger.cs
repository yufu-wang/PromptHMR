using UnityEngine;

namespace AnimeCompositing
{
    /// <summary>
    /// 层调试工具 - 用于可视化和调试三个渲染层
    /// </summary>
    [RequireComponent(typeof(AnimeRenderManager))]
    public class LayerDebugger : MonoBehaviour
    {
        [Header("Debug Display Settings")]
        [Tooltip("显示模式")]
        public DisplayMode displayMode = DisplayMode.Composite;

        [Tooltip("调试窗口大小")]
        [Range(100, 500)]
        public int debugWindowSize = 200;

        [Tooltip("显示位置")]
        public DebugPosition position = DebugPosition.TopLeft;

        [Tooltip("启用实时统计")]
        public bool showStats = true;

        public enum DisplayMode
        {
            Composite,      // 合成结果
            BaseColorOnly,  // 仅基础色
            ShadowOnly,     // 仅阴影
            HighlightOnly,  // 仅高光
            SplitView       // 分屏显示
        }

        public enum DebugPosition
        {
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight
        }

        private AnimeRenderManager renderManager;
        private Material debugMaterial;
        private int frameCount = 0;
        private float fps = 0f;
        private float deltaTime = 0f;

        void Awake()
        {
            renderManager = GetComponent<AnimeRenderManager>();
            InitializeDebugMaterial();
        }

        void Update()
        {
            // 计算 FPS
            frameCount++;
            deltaTime += Time.unscaledDeltaTime;

            if (deltaTime >= 1.0f)
            {
                fps = frameCount / deltaTime;
                frameCount = 0;
                deltaTime = 0f;
            }

            // 热键切换显示模式
            if (Input.GetKeyDown(KeyCode.Alpha1))
                displayMode = DisplayMode.Composite;
            else if (Input.GetKeyDown(KeyCode.Alpha2))
                displayMode = DisplayMode.BaseColorOnly;
            else if (Input.GetKeyDown(KeyCode.Alpha3))
                displayMode = DisplayMode.ShadowOnly;
            else if (Input.GetKeyDown(KeyCode.Alpha4))
                displayMode = DisplayMode.HighlightOnly;
            else if (Input.GetKeyDown(KeyCode.Alpha5))
                displayMode = DisplayMode.SplitView;
        }

        void InitializeDebugMaterial()
        {
            Shader debugShader = Shader.Find("Hidden/AnimeCompositing/DebugViewer");
            if (debugShader != null)
            {
                debugMaterial = new Material(debugShader);
            }
        }

        void OnGUI()
        {
            if (renderManager == null) return;

            // 获取渲染纹理
            RenderTexture baseColor = renderManager.BaseColorTexture;
            RenderTexture shadowLayer = renderManager.ShadowLayerTexture;
            RenderTexture highlightLayer = renderManager.HighlightLayerTexture;

            if (baseColor == null) return;

            // 计算位置
            Vector2 pos = GetDebugPosition();
            int size = debugWindowSize;

            switch (displayMode)
            {
                case DisplayMode.Composite:
                    DrawCompositeView(pos, size, baseColor, shadowLayer, highlightLayer);
                    break;

                case DisplayMode.BaseColorOnly:
                    DrawSingleLayer(pos, size, baseColor, "Base Color Layer");
                    break;

                case DisplayMode.ShadowOnly:
                    DrawSingleLayer(pos, size, shadowLayer, "Shadow Layer");
                    break;

                case DisplayMode.HighlightOnly:
                    DrawSingleLayer(pos, size, highlightLayer, "Highlight Layer");
                    break;

                case DisplayMode.SplitView:
                    DrawSplitView(pos, size, baseColor, shadowLayer, highlightLayer);
                    break;
            }

            // 显示统计信息
            if (showStats)
            {
                DrawStats();
            }

            // 显示热键提示
            DrawHotkeys();
        }

        Vector2 GetDebugPosition()
        {
            int margin = 10;
            int size = debugWindowSize;

            switch (position)
            {
                case DebugPosition.TopLeft:
                    return new Vector2(margin, margin);

                case DebugPosition.TopRight:
                    return new Vector2(Screen.width - size - margin, margin);

                case DebugPosition.BottomLeft:
                    return new Vector2(margin, Screen.height - size - margin - 100);

                case DebugPosition.BottomRight:
                    return new Vector2(Screen.width - size - margin, Screen.height - size - margin - 100);

                default:
                    return new Vector2(margin, margin);
            }
        }

        void DrawSingleLayer(Vector2 pos, int size, RenderTexture texture, string label)
        {
            GUI.Box(new Rect(pos.x - 5, pos.y - 25, size + 10, size + 35), "");
            GUI.Label(new Rect(pos.x, pos.y - 20, size, 20), label);
            GUI.DrawTexture(new Rect(pos.x, pos.y, size, size), texture);
        }

        void DrawCompositeView(Vector2 pos, int size, RenderTexture baseColor, RenderTexture shadowLayer, RenderTexture highlightLayer)
        {
            int previewSize = size / 3;
            GUI.Box(new Rect(pos.x - 5, pos.y - 25, size + 10, size + 35), "");
            GUI.Label(new Rect(pos.x, pos.y - 20, size, 20), "Layer Preview");

            // 显示三个小预览
            GUI.Label(new Rect(pos.x, pos.y, previewSize, 15), "Base", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x, pos.y + 15, previewSize - 5, previewSize - 5), baseColor);

            GUI.Label(new Rect(pos.x + previewSize, pos.y, previewSize, 15), "Shadow", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x + previewSize, pos.y + 15, previewSize - 5, previewSize - 5), shadowLayer);

            GUI.Label(new Rect(pos.x + previewSize * 2, pos.y, previewSize, 15), "Highlight", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x + previewSize * 2, pos.y + 15, previewSize - 5, previewSize - 5), highlightLayer);
        }

        void DrawSplitView(Vector2 pos, int size, RenderTexture baseColor, RenderTexture shadowLayer, RenderTexture highlightLayer)
        {
            int smallSize = size / 2;

            GUI.Box(new Rect(pos.x - 5, pos.y - 25, size + 10, size + 80), "");
            GUI.Label(new Rect(pos.x, pos.y - 20, size, 20), "Split View");

            // 左上: 基础色
            GUI.Label(new Rect(pos.x, pos.y, smallSize, 15), "Base Color", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x, pos.y + 15, smallSize - 2, smallSize - 2), baseColor);

            // 右上: 阴影
            GUI.Label(new Rect(pos.x + smallSize, pos.y, smallSize, 15), "Shadow", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x + smallSize, pos.y + 15, smallSize - 2, smallSize - 2), shadowLayer);

            // 左下: 高光
            GUI.Label(new Rect(pos.x, pos.y + smallSize + 15, smallSize, 15), "Highlight", GUI.skin.box);
            GUI.DrawTexture(new Rect(pos.x, pos.y + smallSize + 30, smallSize - 2, smallSize - 2), highlightLayer);

            // 右下: 合成结果信息
            GUI.Label(new Rect(pos.x + smallSize, pos.y + smallSize + 15, smallSize, 15), "Info", GUI.skin.box);
            GUI.Box(new Rect(pos.x + smallSize, pos.y + smallSize + 30, smallSize - 2, smallSize - 2),
                   $"Mode: {displayMode}\nFPS: {fps:F1}\nRes: {baseColor.width}x{baseColor.height}");
        }

        void DrawStats()
        {
            Rect statsRect = new Rect(10, Screen.height - 80, 250, 70);
            GUI.Box(statsRect, "");

            GUIStyle style = new GUIStyle(GUI.skin.label);
            style.fontSize = 11;

            string stats = $"FPS: {fps:F1}\n";
            stats += $"Display Mode: {displayMode}\n";
            stats += $"Resolution: {renderManager.BaseColorTexture.width}x{renderManager.BaseColorTexture.height}\n";

            GUI.Label(new Rect(statsRect.x + 5, statsRect.y + 5, statsRect.width - 10, statsRect.height - 10), stats, style);
        }

        void DrawHotkeys()
        {
            Rect hotkeysRect = new Rect(Screen.width - 200, Screen.height - 130, 190, 120);
            GUI.Box(hotkeysRect, "");

            GUIStyle style = new GUIStyle(GUI.skin.label);
            style.fontSize = 10;

            string hotkeys = "=== 热键 ===\n";
            hotkeys += "[1] Composite\n";
            hotkeys += "[2] Base Color\n";
            hotkeys += "[3] Shadow\n";
            hotkeys += "[4] Highlight\n";
            hotkeys += "[5] Split View";

            GUI.Label(new Rect(hotkeysRect.x + 5, hotkeysRect.y + 5, hotkeysRect.width - 10, hotkeysRect.height - 10), hotkeys, style);
        }

        void OnDestroy()
        {
            if (debugMaterial != null)
            {
                Destroy(debugMaterial);
            }
        }
    }
}
