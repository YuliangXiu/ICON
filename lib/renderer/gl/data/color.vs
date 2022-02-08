#version 330 core

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Color;
layout (location = 2) in vec3 a_Normal;

out vec3 CamNormal;
out vec3 CamPos;
out vec3 Color;
out vec3 depth;


uniform mat3 RotMat;
uniform mat4 NormMat;
uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
    vec3 a_Position = (NormMat * vec4(a_Position,1.0)).xyz;
    gl_Position = PerspMat * ModelMat * vec4(RotMat * a_Position, 1.0);
    Color = a_Color;

    mat3 R = mat3(ModelMat) * RotMat;
    CamNormal = (R * a_Normal);

    depth = vec3(gl_Position.z / gl_Position.w);
   
}