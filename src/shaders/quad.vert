#version 440 core
layout(location = 5) in vec2 aPos;
layout(location = 6) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}