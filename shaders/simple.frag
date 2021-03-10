#version 450 core

layout(location = 0) in vec3 vray_dir;
layout(location = 1) flat in vec3 transformed_eye;

layout(set = 0, binding = 1) buffer VolumeData {
	uint data[];
} volumeData;
layout(set = 0, binding = 2) uniform texture2D colormap;
layout(set = 0, binding = 3) uniform sampler mySampler;
layout(set = 0, binding = 4) uniform volumeParams {
	ivec3 volumeDims;
};
layout(set = 0, binding = 6) uniform Isovalue {
	float isovalue;
};

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

float trilinear_interpolate(vec3 p) {
	// Multiply elements of p by volume dimensions to get appropriate x, y, and z values
	vec3 fitted_p = vec3(p.x*(volumeDims.x-1), p.y*(volumeDims.y-1), p.z*(volumeDims.z-1));
	ivec3 p1 = ivec3(fitted_p);
	// Need to apply minimum so function doesn't try to interpolate when on x, y, or z edges
	ivec3 p2 = ivec3(min(volumeDims.x-1, p1.x+1), min(volumeDims.x-1, p1.y+1), min(volumeDims.x-1, p1.z+1));
	// Keep track of the decimal portions 
	vec3 differences = fitted_p - p1;
	// Interpolate across x, then y, then z, and return the value normalized between 0 and 1
	float c11 = volumeData.data[p1.x+p1.y*volumeDims.x+p1.z*volumeDims.x*volumeDims.y] * (1 - differences.x) + volumeData.data[p2.x+p1.y*volumeDims.x+p1.z*volumeDims.x*volumeDims.y] * differences.x;
	float c12 = volumeData.data[p1.x+p1.y*volumeDims.x+p2.z*volumeDims.x*volumeDims.y] * (1 - differences.x) + volumeData.data[p2.x+p1.y*volumeDims.x+p2.z*volumeDims.x*volumeDims.y] * differences.x;
	float c21 = volumeData.data[p1.x+p2.y*volumeDims.x+p1.z*volumeDims.x*volumeDims.y] * (1 - differences.x) + volumeData.data[p2.x+p2.y*volumeDims.x+p1.z*volumeDims.x*volumeDims.y] * differences.x;
	float c22 = volumeData.data[p1.x+p2.y*volumeDims.x+p2.z*volumeDims.x*volumeDims.y] * (1 - differences.x) + volumeData.data[p2.x+p2.y*volumeDims.x+p2.z*volumeDims.x*volumeDims.y] * differences.x;
	float c1 = c11 * (1 - differences.y) + c21 * differences.y;
	float c2 = c12 * (1 - differences.y) + c22 * differences.y;
	float c = c1 * (1 - differences.z) + c2 * differences.z;
	return c / 255.0;
}

vec4 polynomial(vec3 origin, vec3 b0) {
	// Multiply elements of p by volume dimensions to get appropriate x, y, and z values
	vec3 fitted_p = vec3(origin.x*(volumeDims.x-1), origin.y*(volumeDims.y-1), origin.z*(volumeDims.z-1));
	ivec3 p0 = ivec3(fitted_p);
	// Need to apply minimum so function doesn't try to interpolate when on x, y, or z edges
	ivec3 p1 = ivec3(min(volumeDims.x-1, p0.x+1), min(volumeDims.x-1, p0.y+1), min(volumeDims.x-1, p0.z+1));
	// Keep track of the decimal portions 
	const vec3 a[2] = vec3[2](
		fitted_p-p0,
		vec3(1) - (fitted_p-p0)
	);
	const vec3 b[2] = vec3[2](
		b0,
		vec3(1) - b0
	);
	const ivec3 p[2] = ivec3[2](
		p0,
		p1
	);
	vec4 poly = vec4(0);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				poly.x += b[i].x*b[j].y*b[k].z*volumeData.data[p[i].x+p[j].y*volumeDims.x+p[k].z*volumeDims.x*volumeDims.y];
				poly.y +=	(a[i].x*b[j].y*b[k].z+b[i].x*a[j].y*b[k].z+b[i].x*b[j].y*a[k].z) * volumeData.data[p[i].x+p[j].y*volumeDims.x+p[k].z*volumeDims.x*volumeDims.y];
				poly.z += (b[i].x*a[j].y*a[k].z+a[i].x*b[j].y*a[k].z+a[i].x*a[j].y*b[k].z) * volumeData.data[p[i].x+p[j].y*volumeDims.x+p[k].z*volumeDims.x*volumeDims.y];
				poly.w += a[i].x*a[j].y*a[k].z*volumeData.data[p[i].x+p[j].y*volumeDims.x+p[k].z*volumeDims.x*volumeDims.y];
			}
		}
	}
	return poly;
}

vec3 shading(vec3 N, vec3 V, vec3 L) {
	vec3 Kd = vec3(0.6);
	vec3 Ks = vec3(0.2);
	float mean = 0.7;
	float scale = 0.2;

	vec3 lightIntensity = vec3(1);
	vec3 H = normalize(L+V);
	float n_h = dot(N,H);
	float n_v = dot(N,V);
	float v_h = dot(V,H);
	float n_l = dot(N,L);

	vec3 diffuse = Kd * max(n_l, 0);
	float fresnel = pow(1.0 + v_h, 4);
	float delta = acos(n_h).x;
	float exponent = -pow((delta/mean), 2);
	float microfacets = scale * exp(exponent);

	float term1 = 2 * n_h * n_v/v_h;
	float term2 = 2 * n_h * n_l/v_h;
	float selfshadow = min(1, min(term1, term2));

	vec3 specular = Ks * fresnel * microfacets * selfshadow / n_v;
	return vec3(1) * (diffuse + specular);
}

float func(vec4 poly, float t) {
	return (poly.x*pow(t,3)+poly.y*pow(t,2)+poly.z*t+poly.w)/255.0-isovalue;
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
	vec3 dt_vec = 1.0 / (volumeDims * abs(ray_dir));
	float dt = min(dt_vec.x, min(dt_vec.y, dt_vec.z));

	// Step 4: Starting from the entry point, march the ray through the volume
	// and sample it
	vec3 origin = transformed_eye + t_hit.x * ray_dir;
	vec3 p = origin;
	int dt_steps = 0;

	// SIMPLE LINEAR INTERPOLATION
	// for (float t = t_hit.x; t < t_hit.y; t += dt) {
	// 	// Step 4.1: Sample the volume, and color it by the transfer function.
	// 	// Note that here we don't use the opacity from the transfer function,
	// 	// and just use the sample value as the opacity
	// 	float val_in = trilinear_interpolate(p);
	// 	p += ray_dir * dt;
	// 	float val_out = trilinear_interpolate(p);
	// 	if (sign(val_in - isovalue) != sign(val_out - isovalue)){
	// 		color = vec4(0, 1, 0.5, 1);
	// 		p = (p - ray_dir * dt) + (ray_dir * dt) * ((isovalue - val_in)/(val_out - val_in));
	// 		vec3 sample1 = vec3(trilinear_interpolate(p - vec3(0.01, 0, 0)), 
	// 			trilinear_interpolate(p - vec3(0, 0.01, 0)), trilinear_interpolate(p - vec3(0, 0, 0.01)));
	// 		vec3 sample2 = vec3(trilinear_interpolate(p + vec3(0.01, 0, 0)), 
	// 			trilinear_interpolate(p + vec3(0, 0.01, 0)), trilinear_interpolate(p + vec3(0, 0, 0.01)));
	// 		vec3 N = normalize(sample2-sample1);
	// 		vec3 L = normalize(vec3(255,0,200)-vray_dir);
	// 		vec3 V = normalize(transformed_eye-vray_dir);
	// 		color.rgb += shading(N, V, L);
	// 		break;
	// 	}
	// }

	// FAST AND ACCURATE INTERSECTIOn
	for (float t = t_hit.x; t < t_hit.y; t += dt) {
		vec4 poly = polynomial(p, ray_dir);
		p += ray_dir * dt;
		float t0 = t;
		float t1 = t+dt;
		float f0 = func(poly, t);
		float f1 = func(poly, t+dt);
		//solve quadratic for extrema
		float D = (2*poly.y) * (2*poly.y) - 4 * (3*poly.x) * poly.z; // calculate discriminant squared
		if (D > 0) {
			D = sqrt(D);
			vec2 roots = vec2((-(2*poly.y) - D) / (2 * (3*poly.x)), (-(2*poly.y) + D) / (2 * (3*poly.x))); //return roots
			float e0 = min(roots.x, roots.y);
			if (e0 < t1 && e0 > t0) {
				if (sign(func(poly, e0)) == sign(f0)) {
					t0 = e0;
					f0 = func(poly, e0);
				} else {
					t1 = e0;
					f1 = func(poly, e0);
				}
			}
			float e1 = max(roots.x, roots.y);
			if (e1 < t1 && e1 > t0) {
				if (sign(func(poly, e1)) == sign(f0)) {
					t0 = e1;
					f0 = func(poly, e1);
				} else {
					t1 = e1;
					f1 = func(poly, e1);
				}
			}
		} if (sign(f0) != sign(f1)) {
			color = vec4(1);
			break;
		}
	}
}

