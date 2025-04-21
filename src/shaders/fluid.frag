#version 440 core

in vec2 TexCoords;  // From the vertex shader
out vec4 FragColor;

// Textures
uniform sampler2D depthMap;       
uniform sampler2D blurDepthMap;      
uniform sampler2D thicknessMap;   
uniform sampler2D backgroundTexture;   
uniform samplerCube skybox; 

// transform matrices
uniform mat4 inverseView;
uniform mat4 inverseProjection;
uniform mat4 projection;
uniform mat4 view;

uniform float refractIndex; // Refractive index of the fluid (1/1.33)
uniform vec3 fluidColor;    
uniform vec3 cameraPos;     // world coordinates of the camera
uniform uint fragSelector;

vec3 planeCenter = vec3(7.5, 0.0, 4.0);
vec3 planeNormal = vec3(0.0, 1.0, 0.0); // Normal of the plane
vec2 planeSize = vec2(40.0, 40.0); // Size of the plane in X and Z directions

vec3 planeIntersection(vec3 rayOrigin, vec3 rayDir){
    if (rayOrigin.y > planeCenter.y && rayDir.y > 0.0)
        return vec3(0.0);
    
    // Distance to the plane
    float d = dot(planeCenter - rayOrigin, planeNormal) / dot(rayDir, planeNormal);

    // ray-plane intersection point
    vec3 intersectionPoint = rayOrigin + d * rayDir;

    // Check if the intersection point is within the bounds of the plane
    if (abs(intersectionPoint.x - planeCenter.x) > planeSize.x / 2.0 || 
        abs(intersectionPoint.z - planeCenter.z) > planeSize.y / 2.0)
        return vec3(0.0);

    return intersectionPoint;
}

vec2 worldToUV(vec3 worldPos){
    vec4 viewPos = view * vec4(worldPos, 1.0);
    vec4 clipPos = projection * viewPos;
    // Convert to NDC coordinates
    clipPos /= clipPos.w; 

    return clipPos.xy * 0.5 + 0.5; // Convert to [0,1] range

}

// vec3 ReconstructWorldPosFromDepth(vec2 uv, float depth){ // not working
//     // Convert [0,1] depth to clip-space z in [-1,1].
//     // Create a clip-space position
//     vec4 clipPos = vec4(
//         uv.x * 2.0 - 1.0,       // x in clip space
//         uv.y * 2.0 - 1.0,       // y in clip space
//         depth * 2.0 - 1.0,      // z in clip space
//         1.0
//     );
//     // Convert clip-space to view-space
//     vec4 viewPosH = inverseProjection * clipPos;
//     viewPosH /= viewPosH.w;
//     return (inverseView * viewPosH).xyz; // world-space position
// }

vec3 reconstructViewDir(vec2 uv) {
    // NDC ray direction
    vec4 rayClip = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    
    // Transform to view space
    vec4 rayView = inverseProjection * rayClip;
    rayView = vec4(rayView.xy, -1.0, 0.0); // Forward in view space is -z
    
    // Transform to world space
    vec4 rayWorld = inverseView * rayView;
    return normalize(rayWorld.xyz);
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec3 rayDir = reconstructViewDir(uv);
    
    // Calculate world position using the ray and depth
    float linearDepth = depth * 40; 
    return cameraPos + rayDir * linearDepth;
}


vec3 computeNormalFromDepth(vec2 uv) {
    float depth = texture(blurDepthMap, uv).r;
    if (depth == 1.0)
        return vec3(0.0);
    
    // Sample neighboring pixels
    vec2 texelSize = 1.0 / textureSize(blurDepthMap, 0);
    vec3 centerPos = reconstructWorldPos(uv, depth);
    
    vec3 ddx = reconstructWorldPos(uv + vec2(texelSize.x, 0), 
                                            texture(blurDepthMap, uv + vec2(texelSize.x, 0)).r) - centerPos;
    vec3 ddx2 = centerPos - reconstructWorldPos(uv + vec2(texelSize.x, 0), 
                                            texture(blurDepthMap, uv + vec2(texelSize.x, 0)).r) ;
    if (abs(ddx.z) < abs(ddx2.z)) 
        ddx = ddx2;
    
    vec3 ddy = reconstructWorldPos(uv + vec2(0, texelSize.y), 
                                            texture(blurDepthMap, uv + vec2(0, texelSize.y)).r) - centerPos;
    vec3 ddy2 = centerPos - reconstructWorldPos(uv + vec2(0, texelSize.y), 
                                            texture(blurDepthMap, uv + vec2(0, texelSize.y)).r) ;
    if (abs(ddy2.z) < abs(ddy.z)) 
        ddy = ddy2;
                     
    return normalize(cross(ddx, ddy));
}

float fresnelReflectance(vec3 viewDir, vec3 surfaceNormal, float ior) {
    vec3 halfVector = normalize(viewDir + surfaceNormal);
    float cosTheta = dot(surfaceNormal, halfVector);
    float r0 = pow((1.0 - ior) / (1.0 + ior), 2.0); 
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}

vec4 sampleColorFromViewRay(vec3 rayOrigin, vec3 rayDir) {
    vec3 planeIntersectPos = planeIntersection(rayOrigin, rayDir);
    vec2 uv = worldToUV(planeIntersectPos);

    // if there is no intersection with the plane sample the skybox
    if (planeIntersectPos == vec3(0.0)) 
        return texture(skybox, rayDir);

    return texture(backgroundTexture, uv);
}

void main()
{
    float depthVal = texture(depthMap, TexCoords).r; 
    vec3 fluidNormal = computeNormalFromDepth(TexCoords);
    float thickness = texture(thicknessMap, TexCoords).r; 

    vec3 hitPos = reconstructWorldPos(TexCoords, depthVal);
    vec3 viewDir = reconstructViewDir(TexCoords);

    vec4 planeColor = texture(backgroundTexture, TexCoords);
    vec4 skyColor = texture(skybox, viewDir);
    vec4 refractColor = vec4(0.0);
    vec4 reflectColor = vec4(0.0);

    vec3 refractDir = refract(viewDir, fluidNormal, refractIndex);
    vec3 reflectDir = reflect(viewDir, fluidNormal);

    refractColor = sampleColorFromViewRay(hitPos, refractDir);
    reflectColor = sampleColorFromViewRay(hitPos, reflectDir);

    float reflectance = clamp(fresnelReflectance(viewDir, fluidNormal, 1.33) + 0.7, 0.0, 1.0);

    vec3 transmittance = exp((fluidColor - 1) * thickness);
    refractColor.rgb *= transmittance;

    vec3 finalColor = mix(reflectColor, refractColor, reflectance).rgb;

    switch (fragSelector) {
        case 0: // depth map
            FragColor = vec4(vec3(depthVal*1.2), 1.0);
            break;
        case 1: // blurred depth map
            FragColor = vec4(vec3(texture(blurDepthMap, TexCoords).r*1.2), 1.0);
            break;
        case 2: // normal map
            FragColor = vec4(fluidNormal, 1.0);
            break;
        case 3: // thickness map
            FragColor = vec4(vec3(thickness), 1.0);
            break;
        case 4: // background texture
            FragColor = (planeColor.rgb == vec3(0.0)) ? skyColor : planeColor;
            break;
        case 5: // refractions only
            FragColor = vec4(refractColor.rgb/transmittance, 1.0);
            break;
        case 6: // refractions + reflections + volumetric absorptions
            FragColor = vec4(finalColor, 1.0);
            break;
    }       
    
}
