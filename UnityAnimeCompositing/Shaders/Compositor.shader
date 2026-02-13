Shader "Hidden/AnimeCompositing/Compositor"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        _BaseColorTex ("Base Color", 2D) = "white" {}
        _ShadowTex ("Shadow Layer", 2D) = "white" {}
        _HighlightTex ("Highlight Layer", 2D) = "white" {}

        _BaseIntensity ("Base Intensity", Float) = 1.0
        _ShadowIntensity ("Shadow Intensity", Float) = 1.0
        _HighlightIntensity ("Highlight Intensity", Float) = 1.0

        _ShadowBlendMode ("Shadow Blend Mode", Int) = 1
        _HighlightBlendMode ("Highlight Blend Mode", Int) = 2

        _ColorTint ("Color Tint", Color) = (1,1,1,1)
        _Contrast ("Contrast", Float) = 1.0
        _Saturation ("Saturation", Float) = 1.0
        _EnableColorGrading ("Enable Color Grading", Int) = 1
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        ZTest Always
        ZWrite Off
        Cull Off

        Pass
        {
            Name "Composite"

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            sampler2D _MainTex;
            sampler2D _BaseColorTex;
            sampler2D _ShadowTex;
            sampler2D _HighlightTex;

            float _BaseIntensity;
            float _ShadowIntensity;
            float _HighlightIntensity;

            int _ShadowBlendMode;
            int _HighlightBlendMode;

            fixed4 _ColorTint;
            float _Contrast;
            float _Saturation;
            int _EnableColorGrading;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // 混合模式函数
            fixed3 BlendNormal(fixed3 base, fixed3 blend)
            {
                return blend;
            }

            fixed3 BlendMultiply(fixed3 base, fixed3 blend)
            {
                return base * blend;
            }

            fixed3 BlendAdd(fixed3 base, fixed3 blend)
            {
                return min(base + blend, 1.0);
            }

            fixed3 BlendScreen(fixed3 base, fixed3 blend)
            {
                return 1.0 - (1.0 - base) * (1.0 - blend);
            }

            fixed3 BlendOverlay(fixed3 base, fixed3 blend)
            {
                fixed3 result;
                result.r = base.r < 0.5 ? (2.0 * base.r * blend.r) : (1.0 - 2.0 * (1.0 - base.r) * (1.0 - blend.r));
                result.g = base.g < 0.5 ? (2.0 * base.g * blend.g) : (1.0 - 2.0 * (1.0 - base.g) * (1.0 - blend.g));
                result.b = base.b < 0.5 ? (2.0 * base.b * blend.b) : (1.0 - 2.0 * (1.0 - base.b) * (1.0 - blend.b));
                return result;
            }

            // 应用混合模式
            fixed3 ApplyBlendMode(fixed3 base, fixed3 blend, float alpha, int mode)
            {
                fixed3 blended;

                if (mode == 0) // Normal
                    blended = BlendNormal(base, blend);
                else if (mode == 1) // Multiply
                    blended = BlendMultiply(base, blend);
                else if (mode == 2) // Add
                    blended = BlendAdd(base, blend);
                else if (mode == 3) // Screen
                    blended = BlendScreen(base, blend);
                else if (mode == 4) // Overlay
                    blended = BlendOverlay(base, blend);
                else
                    blended = blend;

                return lerp(base, blended, alpha);
            }

            // 色彩调整
            fixed3 ApplyColorGrading(fixed3 color, fixed3 tint, float contrast, float saturation)
            {
                // 应用色调
                color *= tint;

                // 应用对比度
                color = (color - 0.5) * contrast + 0.5;

                // 应用饱和度
                float luminance = dot(color, fixed3(0.299, 0.587, 0.114));
                color = lerp(fixed3(luminance, luminance, luminance), color, saturation);

                return saturate(color);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 采样三个层
                fixed4 baseColor = tex2D(_BaseColorTex, i.uv);
                fixed4 shadowLayer = tex2D(_ShadowTex, i.uv);
                fixed4 highlightLayer = tex2D(_HighlightTex, i.uv);

                // 应用层强度
                baseColor.rgb *= _BaseIntensity;
                shadowLayer.rgb *= _ShadowIntensity;
                highlightLayer.rgb *= _HighlightIntensity;

                // 开始合成
                fixed3 result = baseColor.rgb;

                // 第一步：合成阴影层（通常使用正片叠底）
                result = ApplyBlendMode(result, shadowLayer.rgb, shadowLayer.a, _ShadowBlendMode);

                // 第二步：合成高光层（通常使用相加）
                result = ApplyBlendMode(result, highlightLayer.rgb, highlightLayer.a, _HighlightBlendMode);

                // 应用色彩调整
                if (_EnableColorGrading == 1)
                {
                    result = ApplyColorGrading(result, _ColorTint.rgb, _Contrast, _Saturation);
                }

                return fixed4(result, 1.0);
            }
            ENDCG
        }
    }

    FallBack Off
}
