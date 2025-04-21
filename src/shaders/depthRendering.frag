#version 440 core

in vec4 particlePosition;
out vec4 FragColor;

uniform mat4 projectionMat;

void main() {
    float maxDistance = 40;
    float minDistance = 0.01;
    // Normalize the distance to the range [0, 1]

    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // Convert to [-1, 1] range
    float centerDist = dot(coord, coord);
    
    if (centerDist > 1.0) 
        discard;

    float z = sqrt(1.0 - centerDist);
    float sphereFactor = (1 - z) * 0.5;
    vec4 posEye = vec4(particlePosition.xyz, 1.0);
    posEye.z -= sphereFactor;

    vec4 clipPos = projectionMat * posEye;
    
    float t = clamp(clipPos.z / maxDistance, 0.0, 0.99);

    gl_FragDepth = t;

    vec3 color = mix(vec3(0.0) , vec3(0.99), gl_FragDepth);

    FragColor = vec4(color, 1);

}