#version 440 core

layout (location = 0) in vec3 aPos;        // vertex position
layout (location = 1) in vec3 aNormal;     // vertex normal
layout (location = 2) in vec2 aTexCoord;   // texture coordinate

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform float repeat;

out vec3 Normal;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = modelMatrix * vec4(aPos, 1.0);
    Normal = normalize(normalMatrix * aNormal);
    TexCoord = aTexCoord * repeat;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}