#version 450 core

layout(location = 0) in vec3 vray_dir;
layout(location = 1) flat in vec3 transformed_eye;

layout(set = 0, binding = 1) buffer VolumeData {
	uint data[];
} volumeData;
// layout(set = 0, binding = 2) uniform texture2D colormap;
// layout(set = 0, binding = 3) uniform sampler mySampler;

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

    // Step 2: Intersect the ray with the volume bounds to find the interval
	// along the ray overlapped by the volume.
	vec2 t_hit = intersect_box(transformed_eye, ray_dir);
	if (t_hit.x > t_hit.y) {
		discard;
	}
	// We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
	t_hit.x = max(t_hit.x, 0.0);

	// Step 3: Compute the step size to march through the volume grid
	vec3 dt_vec = 1.0 / (vec3(64.0, 64.0, 64.0) * abs(ray_dir));
	float dt = min(dt_vec.x, min(dt_vec.y, dt_vec.z));

	// Step 4: Starting from the entry point, march the ray through the volume
	// and sample it
	vec3 p = transformed_eye + t_hit.x * ray_dir;
	for (float t = t_hit.x; t < t_hit.y; t += dt) {
		// Step 4.1: Sample the volume, and color it by the transfer function.
		// Note that here we don't use the opacity from the transfer function,
		// and just use the sample value as the opacity
		ivec3 h = ivec3(p);
		uint val = volumeData.data[h.x+h.y*64+h.z*64*64];
		vec4 val_color = vec4(vec3(val/255.0), val/255.0);

		// Step 4.2: Accumulate the color and opacity using the front-to-back
		// compositing equation
		color.rgb += (1.0 - color.a) * val_color.a * val_color.rgb;
		color.a += (1.0 - color.a) * val_color.a;

		// Optimization: break out of the loop when the color is near opaque
		if (color.a >= 0.95) {
			break;
		}
		p += ray_dir * dt;
	}
}

