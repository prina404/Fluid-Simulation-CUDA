#version 440 core

out vec4 FragColor;

in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D tex;
vec3 lightDir = vec3(1.0);      // static light direction
vec3 ambientColor = normalize( vec3(0.2)); 
vec3 diffuseColor = normalize( vec3(0.9));  

void main()
{
    // Diffuse term using Lambert's cosine law
    float diff = max(dot(normalize(Normal), normalize(-lightDir)), 0.0);
    
    vec3 textureColor = texture(tex, TexCoord).rgb + 0.2;
    vec3 ambient = ambientColor * textureColor;
    vec3 diffuse = diffuseColor * diff * textureColor;
    
    vec3 result = ambient + diffuse;
    FragColor = vec4(result, 1.0);
}