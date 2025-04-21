#version 440 core
in vec2 TexCoord;

uniform sampler2D inputTexture;
uniform float blurFalloff;  // controls spatial weighting
uniform float blurScale;    // controls range (color intensity) weighting
uniform vec2 direction;

out vec4 FragColor;

void main(){
    float centerDepth = texture(inputTexture, TexCoord).r;
    if (centerDepth == 1.0)
        discard;

    vec2 texelSize = 1.0 / textureSize(inputTexture, 0);
    float halfKernel = 40.0;  // kernel radius (num samples)
    float wsum = 0.0;
    vec3 result = vec3(0.0);

    for (float x = -halfKernel; x <= halfKernel; x++){
        vec2 offset = direction * x * texelSize * (1 - (centerDepth));
        float depthSample = texture(inputTexture, TexCoord + offset).r;

        float r = x * blurScale;    // distance component
        float w = exp(-r*r);

        float r2 = (depthSample - centerDepth) * blurFalloff;       // color range component
        float g = exp(-r2*r2);

        result += depthSample * w * g;
        wsum += w * g;
    }
    
    if (wsum > 0.0)
        result /= wsum;

    FragColor = vec4(result, 1.0);

}