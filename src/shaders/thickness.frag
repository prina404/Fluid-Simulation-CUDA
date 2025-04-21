#version 440 core

// This fragment shader outputs a constant thickness contribution.
out vec4 FragColor;
uniform float particleweight;

void main(){
    // Each fragment contributes a thickness of 1.0 (adjust as needed)
    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // Convert to [-1, 1] range
    float distSquared = dot(coord, coord);
    
    if (distSquared > 1.0) discard;
    FragColor = vec4(vec3(particleweight), 1);
}