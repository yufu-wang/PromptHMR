Shader "Hidden/AnimeCompositing/Outline"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
        _OutlineColor ("Outline Color", Color) = (0,0,0,1)
        _OutlineThickness ("Outline Thickness", Float) = 1.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        ZTest Always
        ZWrite Off
        Cull Off

        Pass
        {
            Name "Outline"

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
            float4 _MainTex_TexelSize;
            fixed4 _OutlineColor;
            float _OutlineThickness;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Sobel 边缘检测
            float SobelEdgeDetection(float2 uv)
            {
                float2 texelSize = _MainTex_TexelSize.xy * _OutlineThickness;

                // Sobel 核
                // Gx
                float gx = 0.0;
                gx += tex2D(_MainTex, uv + float2(-texelSize.x, texelSize.y)).r * -1.0;
                gx += tex2D(_MainTex, uv + float2(-texelSize.x, 0)).r * -2.0;
                gx += tex2D(_MainTex, uv + float2(-texelSize.x, -texelSize.y)).r * -1.0;
                gx += tex2D(_MainTex, uv + float2(texelSize.x, texelSize.y)).r * 1.0;
                gx += tex2D(_MainTex, uv + float2(texelSize.x, 0)).r * 2.0;
                gx += tex2D(_MainTex, uv + float2(texelSize.x, -texelSize.y)).r * 1.0;

                // Gy
                float gy = 0.0;
                gy += tex2D(_MainTex, uv + float2(-texelSize.x, texelSize.y)).r * 1.0;
                gy += tex2D(_MainTex, uv + float2(0, texelSize.y)).r * 2.0;
                gy += tex2D(_MainTex, uv + float2(texelSize.x, texelSize.y)).r * 1.0;
                gy += tex2D(_MainTex, uv + float2(-texelSize.x, -texelSize.y)).r * -1.0;
                gy += tex2D(_MainTex, uv + float2(0, -texelSize.y)).r * -2.0;
                gy += tex2D(_MainTex, uv + float2(texelSize.x, -texelSize.y)).r * -1.0;

                // 计算边缘强度
                float edge = sqrt(gx * gx + gy * gy);
                return edge;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 color = tex2D(_MainTex, i.uv);

                // 计算亮度用于边缘检测
                float luminance = dot(color.rgb, fixed3(0.299, 0.587, 0.114));

                // 边缘检测
                float edge = SobelEdgeDetection(i.uv);

                // 如果检测到边缘，应用描边颜色
                float edgeThreshold = 0.1;
                if (edge > edgeThreshold)
                {
                    color.rgb = lerp(color.rgb, _OutlineColor.rgb, edge * _OutlineColor.a);
                }

                return color;
            }
            ENDCG
        }
    }

    FallBack Off
}
