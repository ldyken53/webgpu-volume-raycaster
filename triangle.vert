// Vertex shader
#version 450 core

layout(location = 0) in vec3 pos;

layout(location = 0) out vec3 vray_dir;
layout(location = 1) flat out vec3 transformed_eye;

// Our uniform buffer containing the projection * view matrix
layout(set = 0, binding = 0, std140) uniform ViewParams {
    mat4 proj_view;
    vec3 eye_pos;
    vec3 volume_scale;
};

void main(void) {
	// TODO: For non-uniform size volumes we need to transform them differently as well
	// to center them properly
	vec3 volume_translation = vec3(0.5) - volume_scale * 0.5;
	gl_Position = proj_view * vec4(pos * volume_scale + volume_translation, 1);
	transformed_eye = (eye_pos - volume_translation) / volume_scale;
	vray_dir = pos - transformed_eye;
}