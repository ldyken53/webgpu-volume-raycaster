#version 450 core

layout(location = 0) in vec3 vray_dir;
layout(location = 1) flat in vec3 transformed_eye;

layout(location = 0) out vec4 color;

vec2 intersect_box(vec3 orig, vec3 dir) {
    const vec3 box_min = vec3(0);
    const vec3 box_max = vec3(1);
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin_tmp = (box_min - orig) * inv_dir;
    vec3 tmax_tmp = (box_max - orig) * inv_dir;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}

void main(void) {
    vec3 ray_dir = normalize(vray_dir);

    color = vec4((ray_dir + vec3(1)) * 0.5, 1);
}

