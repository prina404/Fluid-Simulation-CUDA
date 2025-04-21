#version 440 core

layout (location = 5) in vec2 vertexPos;        // vertex position
layout (location = 6) in vec2 uvCoord;   // texture coordinate

out vec3 FragPos;
out vec2 TexCoords;

void main() {
    gl_Position = vec4(vertexPos, 0.0, 1.0);
    TexCoords = uvCoord;
}