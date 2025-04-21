#version 440 core

layout (location = 0) in vec3 inPosition;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

float pointSize = 100;

void main()
{
    gl_Position = projectionMatrix * viewMatrix  * vec4(inPosition, 1.0);
    gl_PointSize = pointSize / gl_Position.w;
}