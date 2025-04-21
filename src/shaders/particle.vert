#version 440 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aVelocity;
layout(location = 2) in float aDensity;

out float velocity;
out float density;
out float height;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float pointSize;

void main() {

    vec4 worldPosition = viewMatrix  * vec4(aPos, 1.0);
    gl_Position = projectionMatrix  * worldPosition;
    gl_PointSize = pointSize / gl_Position.z;

    velocity = length(aVelocity);
    density = aDensity; 
    height = aPos.y;
}