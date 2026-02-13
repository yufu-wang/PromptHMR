Shader "AnimeCompositing/CharacterMRT"
{
    Properties
    {
        _MainTex ("Base Texture", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1,1,1,1)

        // Shadow Properties
        _ShadowColor ("Shadow Color", Color) = (0.5, 0.5, 0.8, 1)
        _ShadowThreshold ("Shadow Threshold", Range(0, 1)) = 0.5
        _ShadowSmoothness ("Shadow Smoothness", Range(0, 1)) = 0.05

        // Highlight Properties
        _HighlightColor ("Highlight Color", Color) = (1, 1, 1, 1)
        _HighlightThreshold ("Highlight Threshold", Range(0, 1)) = 0.8
        _HighlightSmoothness ("Highlight Smoothness", Range(0, 1)) = 0.05
        _SpecularPower ("Specular Power", Range(1, 128)) = 32

        // Rim Light
        _RimColor ("Rim Color", Color) = (1, 1, 1, 1)
        _RimPower ("Rim Power", Range(0.1, 8)) = 3
        _RimIntensity ("Rim Intensity", Range(0, 1)) = 0.5
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "LightMode"="ForwardBase" }
        LOD 200

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_fwdbase

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 worldNormal : TEXCOORD1;
                float3 worldPos : TEXCOORD2;
                float3 viewDir : TEXCOORD3;
                SHADOW_COORDS(4)
            };

            // Multiple Render Targets output
            struct MRTOutput
            {
                fixed4 baseColor : SV_Target0;    // 基础颜色层
                fixed4 shadowLayer : SV_Target1;   // 阴影层
                fixed4 highlightLayer : SV_Target2; // 高光层
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _BaseColor;
            fixed4 _ShadowColor;
            fixed4 _HighlightColor;
            fixed4 _RimColor;
            float _ShadowThreshold;
            float _ShadowSmoothness;
            float _HighlightThreshold;
            float _HighlightSmoothness;
            float _SpecularPower;
            float _RimPower;
            float _RimIntensity;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.viewDir = normalize(UnityWorldSpaceViewDir(o.worldPos));
                TRANSFER_SHADOW(o);
                return o;
            }

            // Toon shading step function with smoothstep
            fixed ToonStep(float value, float threshold, float smoothness)
            {
                return smoothstep(threshold - smoothness, threshold + smoothness, value);
            }

            MRTOutput frag (v2f i)
            {
                MRTOutput output;

                // Sample base texture
                fixed4 texColor = tex2D(_MainTex, i.uv);

                // Normalize vectors
                float3 normal = normalize(i.worldNormal);
                float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
                float3 viewDir = normalize(i.viewDir);
                float3 halfDir = normalize(lightDir + viewDir);

                // Calculate lighting
                float NdotL = dot(normal, lightDir);
                float NdotH = dot(normal, halfDir);
                float NdotV = dot(normal, viewDir);

                // Shadow attenuation
                float atten = SHADOW_ATTENUATION(i);
                float lightIntensity = NdotL * atten;

                // ========== BASE COLOR LAYER ==========
                // 基础颜色层：原始纹理颜色
                output.baseColor = texColor * _BaseColor;
                output.baseColor.rgb *= _LightColor0.rgb;

                // ========== SHADOW LAYER ==========
                // 阴影层：使用 Toon Shading 风格的硬阴影
                fixed shadowStep = 1.0 - ToonStep(lightIntensity, _ShadowThreshold, _ShadowSmoothness);
                output.shadowLayer = fixed4(_ShadowColor.rgb, shadowStep);

                // Add ambient shadow
                float ambientShadow = 1.0 - ToonStep(NdotL + 0.5, 0.5, 0.1);
                output.shadowLayer.a = max(shadowStep, ambientShadow * 0.5);

                // ========== HIGHLIGHT LAYER ==========
                // 高光层：包含镜面高光和边缘光

                // Specular highlight (Toon style)
                float specular = pow(max(0, NdotH), _SpecularPower);
                fixed specularStep = ToonStep(specular, _HighlightThreshold, _HighlightSmoothness);

                // Rim light (Fresnel effect)
                float rim = 1.0 - saturate(NdotV);
                rim = pow(rim, _RimPower) * _RimIntensity;
                fixed rimStep = ToonStep(rim, 0.5, 0.1);

                // Combine specular and rim
                fixed highlightMask = max(specularStep, rimStep);
                output.highlightLayer = fixed4(_HighlightColor.rgb, highlightMask);

                // Add rim color influence
                output.highlightLayer.rgb = lerp(output.highlightLayer.rgb,
                                                 _RimColor.rgb,
                                                 rimStep * 0.5);

                return output;
            }
            ENDCG
        }

        // Shadow caster pass
        UsePass "Legacy Shaders/VertexLit/SHADOWCASTER"
    }

    FallBack "Diffuse"
}
