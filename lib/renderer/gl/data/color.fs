#version 330 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 FragNormal;
layout (location = 2) out vec4 FragDepth;

in vec3 Color;
in vec3 CamNormal;
in vec3 depth;


void main() 
{
    FragColor = vec4(Color,1.0);

    vec3 cam_norm_normalized = normalize(CamNormal);
    vec3 rgb = (cam_norm_normalized + 1.0) / 2.0;
	FragNormal = vec4(rgb, 1.0);
    FragDepth = vec4(depth.xyz, 1.0);
}
