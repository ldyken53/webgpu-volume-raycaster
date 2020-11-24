#version 450 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

layout(location = 0) out vec4 vcolor;

layout(set = 0, binding = 0, std140) uniform ViewParams {
    mat4 proj_view;
};

void main(void) {
    vcolor = vec4(color, 1);
    gl_Position = proj_view * vec4(pos - 0.5, 1);
}

