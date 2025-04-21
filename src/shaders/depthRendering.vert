#version 440 core
layout(location = 0) in vec3 aPos;


out vec4 particlePosition;

uniform mat4 viewMatrix;
uniform mat4 projectionMat;

float pointSize = 200;

float maxDistance = 4;
float minDistance = 0;

void main() {

    vec4 worldPos = viewMatrix * vec4(aPos, 1.0);
    particlePosition = worldPos;

    gl_Position = projectionMat  * worldPos;
    gl_PointSize = pointSize / gl_Position.w; // make point size proportional to distance
}