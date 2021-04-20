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

float trilinear_interpolate(vec3 fitted_p) {
	ivec3 p1 = clamp(ivec3(fitted_p), ivec3(0), ivec3(volumeDims - 1));
	// Need to apply minimum so function doesn't try to interpolate when on x, y, or z edges
    ivec3 p2 = clamp(p1 + 1, ivec3(0), ivec3(volumeDims - 1));
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

/*
vec4 polynomial(vec3 fitted_origin, vec3 b0, ivec3 steps, ivec3 current_voxel) {
	ivec3 current_voxel1 = current_voxel + steps;
	// Keep track of the decimal portions 
	const vec3 a[2] = vec3[2](
		(current_voxel - fitted_origin),
		(fitted_origin - current_voxel)
	);
	const vec3 b[2] = vec3[2](
		b0,
		(-1 * b0)
	);
	const ivec3 p[2] = ivec3[2](
		current_voxel,
		current_voxel1
	);
	vec4 poly = vec4(0);
	poly.w = -1 * 255 * isovalue;
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
*/

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
	return (poly.x*pow(t,3)+poly.y*pow(t,2)+poly.z*t+poly.w);
}

bool outside_grid(const vec3 p) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, vec3(volumeDims)));
}

bool outside_dual_grid(const vec3 p) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, vec3(volumeDims - 1)));
}

const ivec3 index_to_vertex[8] = {
    ivec3(0, 0, 0), // v000 = 0
    ivec3(1, 0, 0), // v100 = 1
    ivec3(0, 1, 0), // v010 = 2
    ivec3(1, 1, 0), // v110 = 3
    ivec3(0, 0, 1), // v001 = 4
    ivec3(1, 0, 1), // v101 = 5
    ivec3(0, 1, 1), // v011 = 6
    ivec3(1, 1, 1)  // v111 = 7
};

// Load the vertex values for the dual cell's vertices with its bottom-near-left corner
// at v000. Vertex values will be returned in the order:
// [v000, v100, v110, v010, v001, v101, v111, v011]
void load_vertex_values(const ivec3 v000, out float values[8], out vec2 cell_range) {
    cell_range.x = 1e20f;
    cell_range.y = -1e20f;
    for (int i = 0; i < 8; ++i) { 
        ivec3 v = v000 + index_to_vertex[i];
        values[i] = volumeData.data[v.x + volumeDims.x * (v.y + v.z * volumeDims.y)] / 255.0;
        cell_range.x = min(cell_range.x, values[i]);
        cell_range.y = max(cell_range.y, values[i]);
    }
}

// Trilinear interpolation at the given point within the cell with its origin at v000
// (origin = bottom-left-near point)
float trilinear_interpolate_in_cell(const vec3 p, const ivec3 v000, in float values[8]) {
    const vec3 diff = p - v000;
	// Interpolate across x, then y, then z, and return the value normalized between 0 and 1
    // WILL note: renamed t0 c00/c11 to match wikipedia notation
	const float c00 = values[0] * (1.f - diff.x) + values[1] * diff.x;
	const float c01 = values[4] * (1.f - diff.x) + values[5] * diff.x;
    const float c10 = values[2] * (1.f - diff.x) + values[3] * diff.x;
    const float c11 = values[6] * (1.f - diff.x) + values[7] * diff.x;
	const float c0 = c00 * (1.f - diff.y) + c10 * diff.y;
	const float c1 = c01 * (1.f - diff.y) + c11 * diff.y;
	return c0 * (1.f - diff.z) + c1 * diff.z;
}

#define USE_POLYNOMIAL 1
#define MARMITT 1
#define SHOW_VOLUME 0

// Compute the polynomial for the cell with the given vertex values
vec4 compute_polynomial(const vec3 p, const vec3 dir, const vec3 v000, in float values[8]) {
    const vec3 v111 = v000 + vec3(1);
    // Note: Grid voxels sizes are 1^3
    const vec3 a[2] = {v111 - p, p - v000};
    const vec3 b[2] = {dir, -dir};

    vec4 poly = vec4(0);
	poly.w -= isovalue;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                const float val = values[i + 2 * (j + 2 * k)];

                poly.x += b[i].x * b[j].y * b[k].z * val;

                poly.y += (a[i].x * b[j].y * b[k].z +
                           b[i].x * a[j].y * b[k].z +
                           b[i].x * b[j].y * a[k].z) * val;

                poly.z += (b[i].x * a[j].y * a[k].z +
                           a[i].x * b[j].y * a[k].z +
                           a[i].x * a[j].y * b[k].z) * val;

                poly.w += a[i].x * a[j].y * a[k].z * val;
            }
        }
    }

	return poly;
}

float evaluate_polynomial(const vec4 poly, const float t) {
    return poly.x * pow(t, 3.f) + poly.y * pow(t, 2.f) + poly.z * t + poly.w;
}

// Returns true if the quadratic has real roots
bool solve_quadratic(const vec3 poly, out float roots[2]) {
    // Check for case when poly is just Bt + c = 0
    if (poly.x == 0) {
        roots[0] = -poly.z/poly.y;
        roots[1] = -poly.z/poly.y;
        return true;
    }
    float discriminant = pow(poly.y, 2.f) - 4.f * poly.x * poly.z;
    if (discriminant < 0.f) {
        return false;
    }
    discriminant = sqrt(discriminant);
    vec2 r = 0.5f * vec2(-poly.y + discriminant, -poly.y - discriminant) / poly.x;
    roots[0] = min(r.x, r.y);
    roots[1] = max(r.x, r.y);
    return true;
}

void main() {
    vec3 ray_dir = normalize(vray_dir);

    // Transform the ray into the dual grid space and intersect with the dual grid bounds
	const vec3 vol_eye = transformed_eye * volumeDims - vec3(0.5);
    const vec3 grid_ray_dir = normalize(ray_dir * volumeDims);

	vec2 t_hit = intersect_box(vol_eye, grid_ray_dir, vec3(0), volumeDims - 1);
	if (t_hit.x > t_hit.y) {
		discard;
	}
	// We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
	t_hit.x = max(t_hit.x, 0.0);

	vec3 p = (vol_eye + t_hit.x * grid_ray_dir);
    p = clamp(p, vec3(0), vec3(volumeDims - 2));
    const vec3 inv_grid_ray_dir = 1.0 / grid_ray_dir;
    const vec3 start_cell = floor(p);
    const vec3 t_max_neg = (start_cell - vol_eye) * inv_grid_ray_dir;
    const vec3 t_max_pos = (start_cell + vec3(1) - vol_eye) * inv_grid_ray_dir;
    const bvec3 is_neg_dir = lessThan(grid_ray_dir, vec3(0));
    // Pick between positive/negative t_max based on the ray sign
    vec3 t_max = mix(t_max_pos, t_max_neg, is_neg_dir);
    const ivec3 grid_step = ivec3(sign(grid_ray_dir));
    // Note: each voxel is a 1^3 box on the grid
    const vec3 t_delta = abs(inv_grid_ray_dir);

    /* Note: For isosurface rendering we want to be traversing the dual grid,
     * where values are at the cell vertices. This traverses the cell-centered
     * grid where values are at cell centers. Switching to the dual grid just
     * is an additional -0.5 offset in voxel space (shifting by 0.5 grid cells down)
     */

    float prev_vol_t = t_hit.x;
    float t_prev = t_hit.x;
    float vertex_values[8];
    vec2 cell_range;
    color = vec4(0);
    // Traverse the grid 
    while (!outside_dual_grid(p) && color.a <= 0.99) {
        const ivec3 v000 = ivec3(p);
        load_vertex_values(v000, vertex_values, cell_range);

        // Simple rule of signs isosurface within the cell. First compute
        // the field value at the ray's enter and exit points
        const float t_next = min(t_max.x, min(t_max.y, t_max.z));

        bool skip_cell = false;
#if !SHOW_VOLUME
        // Skip cells that we know don't contain the surface
        skip_cell = isovalue < cell_range.x || isovalue > cell_range.y;
#endif

        if (!skip_cell) {
#if USE_POLYNOMIAL
            // The text seems to not say explicitly, but I think it is required to have
            // the ray "origin" within the cell for the cell-local coordinates for a to
            // be computed properly. So here I set the cell_p to be at the midpoint of the
            // ray's overlap with the cell, which makes it easy to compute t_in/t_out and
            // avoid numerical issues with cell_p being right at the edge of the cell.
            const vec3 cell_p = vol_eye + grid_ray_dir * (t_prev + (t_next - t_prev) * 0.5f);
            float t_in = -(t_next - t_prev) * 0.5f;
            float t_out = (t_next - t_prev) * 0.5f;
            const vec4 poly = compute_polynomial(cell_p, grid_ray_dir, v000, vertex_values);

            float f_in = evaluate_polynomial(poly, t_in);
            float f_out = evaluate_polynomial(poly, t_out);

#else
            // Non-polynomial mode, just do trilinear interpolation
            const vec3 p_enter = vol_eye + grid_ray_dir * t_prev;
            float f_in = trilinear_interpolate_in_cell(p_enter, v000, vertex_values) - isovalue;

            const vec3 p_exit = vol_eye + grid_ray_dir * t_next;
            float f_out = trilinear_interpolate_in_cell(p_exit, v000, vertex_values) - isovalue;
#endif

            vec4 val_color = vec4(0);
#if MARMITT
            float roots[2] = {0, 0};
            // TODO: Seeming to get some holes in the surface with the Marmitt intersector
            if (solve_quadratic(vec3(3.f * poly.x, 2.f * poly.y, poly.z), roots)) {
                if (roots[0] >= t_in && roots[0] <= t_out) {
                    float f_root0 = evaluate_polynomial(poly, roots[0]);
                    if (sign(f_root0) == sign(f_in)) {
                        t_in = roots[0];
                        f_in = f_root0;
                    } else {
                        t_out = roots[0];
                        f_out = f_root0;
                    }
                }
                if (roots[1] >= t_in && roots[1] <= t_out) {
                    float f_root1 = evaluate_polynomial(poly, roots[1]);
                    if (sign(f_root1) == sign(f_in)) {
                        t_in = roots[1];
                        f_in = f_root1;
                    } else {
                        t_out = roots[1];
                        f_out = f_root1;
                    }
                }
            }
            // If the signs aren't equal we know there's an intersection in the cell
            if (sign(f_in) != sign(f_out)) {
                // Find the intersection via repeated linear interpolation
                for (int i = 0; i < 2; i++) {
                    float t = t_in + (t_out - t_in) * (-f_in) / (f_out - f_in);
                    float f_t = evaluate_polynomial(poly, t);
                    if (sign(f_t) == sign(f_in)) {
                        t_in = t;
                        f_in = f_t;
                    } else {
                        t_out = t;
                        f_out = f_t;
                    }
                }
                float t_hit = t_in + (t_out - t_in) * (-f_in) / (f_out - f_in);
                // This t_hit value is relative to cell_p, so now find the depth
                // along the original ray
                vec3 hit_p = cell_p + grid_ray_dir * t_hit;
                t_hit = length(hit_p - vol_eye);
                // Apply some scaling factor so the depth values are within [0, 1]
                // to be displayed as a color. Here I'm just dividing by the volume
                // dimensions to scale it
                val_color.xyz = vec3(t_hit) / length(volumeDims);
                val_color.w = 1;
            }
#else
            if (sign(f_in) != sign(f_out)) {
                val_color = vec4(1);
            }
#endif
#if SHOW_VOLUME
            else {
				// Have to add two * isovalue because f_in, f_out have isovalue subtracted from them for isosurface extraction
                float val = (f_in + f_out + 2 * isovalue) * 0.5f;
                val_color = vec4(textureLod(sampler2D(colormap, mySampler), vec2(val, 0.5), 0.f).rgb, val * 0.5);
                // Opacity correction applied to the val_color.a to account for
                // variable interval of ray overlap with each cell
                val_color.a = clamp(1.f - pow(1.f - val_color.a, t_next - t_prev), 0.f, 1.f);
            }
#endif
            color.rgb += (1.0 - color.a) * val_color.a * val_color.rgb;
            color.a += (1.0 - color.a) * val_color.a;
        }

        t_prev = t_next;
        // Advance in the grid
        if (t_next == t_max.x) {
            p.x += grid_step.x;
            t_max.x += t_delta.x;
        } else if (t_next == t_max.y) {
            p.y += grid_step.y;
            t_max.y += t_delta.y;
        } else {
            p.z += grid_step.z;
            t_max.z += t_delta.z;
        }
    }
}

#if 0
void old_main(void) {
    vec3 ray_dir = normalize(vray_dir);

    // Intersect the ray with the volume bounds to find the interval
	// along the ray overlapped by the volume.
	vec2 t_hit = intersect_box(transformed_eye, ray_dir, vec3(0), vec3(1));
	if (t_hit.x > t_hit.y) {
		discard;
	}
	// We don't want to sample voxels behind the eye if it's
	// inside the volume, so keep the starting point at or in front
	// of the eye
	t_hit.x = max(t_hit.x, 0.0);

	vec3 origin = transformed_eye + t_hit.x * ray_dir;
	vec3 fitted_origin = vec3(origin.x*(volumeDims.x-1), origin.y*(volumeDims.y-1), origin.z*(volumeDims.z-1));
	ivec3 steps = ivec3(sign(ray_dir.x), sign(ray_dir.y), sign(ray_dir.z));
	ivec3 current_voxel = ivec3(fitted_origin.x + -0.5*(steps.x-1), fitted_origin.y + -0.5*(steps.y-1), fitted_origin.z + -0.5*(steps.z-1));
	vec3 tMax = vec3(
		((current_voxel.x + steps.x) - fitted_origin.x)/ray_dir.x,
		((current_voxel.y + steps.y) - fitted_origin.y)/ray_dir.y,
		((current_voxel.z + steps.z) - fitted_origin.z)/ray_dir.z
	);
	vec3 tDelta = vec3(
		1/ray_dir.x,
		1/ray_dir.y,
		1/ray_dir.z
	);
	float tIn = 0;
	float tOut = 0;
	float t0;
	float t1;
	float f0;
	float f1;
	vec4 poly;
	//traverse through intersecting voxels
	do {
		if (tMax.x < tMax.y) {
			if (tMax.x < tMax.z) {
				current_voxel.x += steps.x;
				current_voxel.x = min(current_voxel.x, volumeDims.x-1);
				current_voxel.x = max(current_voxel.x, 0);
				tOut = tMax.x;
				tMax.x += tDelta.x;
			} else {
				current_voxel.z += steps.z;
				current_voxel.z = min(current_voxel.z, volumeDims.z-1);
				current_voxel.z = max(current_voxel.z, 0);
				tOut = tMax.z;
				tMax.z += tDelta.z;
			}
		} else {
			if (tMax.y < tMax.z) {
				current_voxel.y += steps.y;
				current_voxel.y = min(current_voxel.y, volumeDims.y-1);
				current_voxel.y = max(current_voxel.y, 0);
				tOut = tMax.y;
				tMax.y += tDelta.y;
			} else {
				current_voxel.z += steps.z;
				current_voxel.z = min(current_voxel.z, volumeDims.z-1);
				current_voxel.z = max(current_voxel.z, 0);
				tOut = tMax.z;
				tMax.z += tDelta.z;
			}
		} 
		poly = polynomial(fitted_origin, ray_dir, current_voxel);
		t0 = tIn;
		t1 = tOut;
		f0 = func(poly, t0);
		f1 = func(poly, t1);
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
		tIn = tOut;
	} while (0 < min(current_voxel.x, min(current_voxel.y, current_voxel.z)) && (volumeDims.x - 1) > current_voxel.x && (volumeDims.y - 1) > current_voxel.y && (volumeDims.z - 1) > current_voxel.z);

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
	// for (float t = t_hit.x; t < t_hit.y; t += dt) {
	// 	vec4 poly = polynomial(p, ray_dir);
	// 	p += ray_dir * dt;
	// 	float t0 = t;
	// 	float t1 = t+dt;
	// 	float f0 = func(poly, t);
	// 	float f1 = func(poly, t+dt);
	// 	//solve quadratic for extrema
	// 	float D = (2*poly.y) * (2*poly.y) - 4 * (3*poly.x) * poly.z; // calculate discriminant squared
	// 	if (D > 0) {
	// 		D = sqrt(D);
	// 		vec2 roots = vec2((-(2*poly.y) - D) / (2 * (3*poly.x)), (-(2*poly.y) + D) / (2 * (3*poly.x))); //return roots
	// 		float e0 = min(roots.x, roots.y);
	// 		if (e0 < t1 && e0 > t0) {
	// 			if (sign(func(poly, e0)) == sign(f0)) {
	// 				t0 = e0;
	// 				f0 = func(poly, e0);
	// 			} else {
	// 				t1 = e0;
	// 				f1 = func(poly, e0);
	// 			}
	// 		}
	// 		float e1 = max(roots.x, roots.y);
	// 		if (e1 < t1 && e1 > t0) {
	// 			if (sign(func(poly, e1)) == sign(f0)) {
	// 				t0 = e1;
	// 				f0 = func(poly, e1);
	// 			} else {
	// 				t1 = e1;
	// 				f1 = func(poly, e1);
	// 			}
	// 		}
	// 	} if (sign(f0) != sign(f1)) {
	// 		color = vec4(1);
	// 		break;
	// 	}
	// }
}
#endif

