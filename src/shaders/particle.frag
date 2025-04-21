#version 440 core
#define maxHeight 2.5

in float density;
in float height;
in float velocity;

out vec4 FragColor;

// density threshold for normalization 
uniform float maxDensity;

uniform vec3 lightPos;  // Position of the light source in view space
float maxVelocity = 10.;

subroutine vec4 colorMode();

vec3 color1 = vec3(0.1, 0.3, 0.6);
vec3 color2 = vec3(0.8, 0.3, 0.2);

subroutine uniform colorMode Color_Shader;

vec3 rgb2hsv(vec3 c) {
    float cMax = max(c.r, max(c.g, c.b));
    float cMin = min(c.r, min(c.g, c.b));
    float delta = cMax - cMin;
    float h = 0.0;
    if (delta > 0.00001) {
        if (cMax == c.r) {
            h = mod((c.g - c.b) / delta, 6.0);
        } else if (cMax == c.g) {
            h = ((c.b - c.r) / delta) + 2.0;
        } else {  // cMax == c.b
            h = ((c.r - c.g) / delta) + 4.0;
        }
        h /= 6.0;
    }
    float s = (cMax <= 0.0) ? 0.0 : (delta / cMax);
    float v = cMax;
    return vec3(h, s, v);
}

// Convert an HSV color back to RGB
vec3 hsv2rgb(vec3 c) {
    float h = c.x * 6.0;
    float s = c.y;
    float v = c.z;
    float i = floor(h);
    float f = h - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    vec3 rgb;
    if (i == 0.0)
        rgb = vec3(v, t, p);
    else if (i == 1.0)
        rgb = vec3(q, v, p);
    else if (i == 2.0)
        rgb = vec3(p, v, t);
    else if (i == 3.0)
        rgb = vec3(p, q, v);
    else if (i == 4.0)
        rgb = vec3(t, p, v);
    else
        rgb = vec3(v, p, q);
    return rgb;
}

subroutine(colorMode)
vec4 HeightGradient(){
    float normHeight = clamp(height / maxHeight, 0.0, 1.0);
    vec3 hsvStart = rgb2hsv(color1);
    vec3 hsvEnd = rgb2hsv(color2);
    vec3 hsvColor = mix(hsvStart, hsvEnd, normHeight);
    return vec4(hsv2rgb(hsvColor), 1.0);
}

subroutine(colorMode)
vec4 DensityGradient(){
    float normDensity = clamp(density / maxDensity, 0.0, 1.0);
    vec3 hsvStart = rgb2hsv(color1);
    vec3 hsvEnd = rgb2hsv(color2);
    vec3 hsvColor = mix(hsvStart, hsvEnd, normDensity);
    return vec4(hsv2rgb(hsvColor), 1.0);
}

subroutine(colorMode)
vec4 VelocityGradient(){
    float safeVelocity = max(velocity, 0.5);
    float normVelocity = clamp(log( safeVelocity) / log(0.5 + maxVelocity), 0.0, 1.0);
    vec3 hsvStart = rgb2hsv(color1);
    vec3 hsvEnd = rgb2hsv(color2);
    vec3 hsvColor = mix(hsvStart, hsvEnd, normVelocity);
    return vec4(hsv2rgb(hsvColor), 1.0);
}


void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // Convert to [-1, 1] range
    float centerDist = sqrt(dot(coord, coord));
    
    if (centerDist > 1.0) discard;  // Discard pixels outside the circle

    vec4 color = Color_Shader();

    vec3 normal = normalize(vec3(coord, 1.0 - centerDist));

    // Lighting calculations
    vec3 lightDir = normalize(lightPos);  // Light direction (assuming directional light)

    // Final color (simple Lambert + Phong ambient)
    float ambient = 0.5;
    float diffuse = max(dot(normal, lightDir), 0.05);  // Diffuse shading

    FragColor = color * vec4(vec3(ambient + diffuse), 1);
}