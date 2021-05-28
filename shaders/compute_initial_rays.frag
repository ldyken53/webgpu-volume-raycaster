#version 450 core
#define UINT_MAX uint(0xffffffff)

layout(location = 0) in vec3 vray_dir;
layout(location = 1) flat in vec3 transformed_eye;

struct RayInfo {
    uint block_id;
    vec3 ray_dir;
    float t;
}

layout(set = 0, binding = 1, std430) buffer RayInformation {
    RayInfo rays[];
};

layout(set = 0, binding = 2) uniform VolumeParams {
	ivec3 volume_dims;
    uint image_width;
};

vec2 intersect_box(vec3 orig, vec3 dir, const vec3 box_min, const vec3 box_max) {
    vec3 inv_dir = 1.0 / dir;
    vec3 tmin_tmp = (box_min - orig) * inv_dir;
    vec3 tmax_tmp = (box_max - orig) * inv_dir;
    vec3 tmin = min(tmin_tmp, tmax_tmp);
    vec3 tmax = max(tmin_tmp, tmax_tmp);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return vec2(t0, t1);
}

void main() {
    vec3 ray_dir = normalize(vray_dir);

    // Transform the ray into the dual grid space and intersect with the dual grid bounds
	const vec3 vol_eye = transformed_eye * volume_dims - vec3(0.5);
    const vec3 grid_ray_dir = normalize(ray_dir * volume_dims);

	vec2 t_hit = intersect_box(vol_eye, grid_ray_dir, vec3(0), volume_dims - 1);
    int index = gl_FragCoord.x + image_width * gl_FragCoord.y
	if (t_hit.x > t_hit.y) {
        int index = gl_FragCoord.x + imgWidth * gl_FragCoord.y
        rays[index].block_id = UINT_MAX;
        rays[index].ray_dir = ray_dir;
        rays[index].t = t_hit.x;
	} else {
        rays[index].block_id = UINT_MAX;
        rays[index].ray_dir = ray_dir;
        rays[index].t = 1000;
    }
}