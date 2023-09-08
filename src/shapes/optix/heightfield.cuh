#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixHeightfieldData {
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    size_t res_x;
    size_t res_y;
    float* grid_data;
    float max_height;
};

struct IntersectResult {
    bool found_intersection;
    float t_closest;
    float u_closest;
    float v_closest;
};

#ifdef __CUDACC__
__device__ void get_tile_vertices(unsigned int tile_index, Vector3f* vertices, OptixHeightfieldData* heightfield) {
    float cell_size[2] = { 2.0f / (heightfield->res_x  - 1), 2.0f / (heightfield->res_y - 1)};

    unsigned int amount_rows = heightfield->res_y - 1;
    unsigned int amount_bboxes_per_row = heightfield->res_x- 1;

    unsigned int values_per_row = heightfield->res_x;
    unsigned int row_nr = tile_index / amount_bboxes_per_row;
    unsigned int row_offset = tile_index % (amount_bboxes_per_row);

    // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
    // row offset to this to get the absolute offset to obtain the corresponding texel of the
    // current AABB (+ 1 in both dimensions to get right and top texels)  
    unsigned int left_bottom_idx = (amount_rows - row_nr) * values_per_row + row_offset;
    unsigned int right_bottom_idx = (amount_rows - row_nr) * values_per_row + row_offset + 1;
    unsigned int left_top_idx = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
    unsigned int right_top_idx = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

    // Compute the fractional bounds of the cell we're testing the intersection for
    Vector2f local_min_target_bounds = { -1.0f + row_offset * cell_size[0], -1.0f + row_nr * cell_size[1] };
    Vector2f local_max_target_bounds = { -1.0f + (row_offset + 1) * cell_size[0], -1.0f + (row_nr + 1) * cell_size[1] };

    vertices[0] = heightfield->to_world.transform_point(Vector3f{local_min_target_bounds.x(), local_max_target_bounds.y(), heightfield->grid_data[left_top_idx] * heightfield->max_height });
    vertices[1] = heightfield->to_world.transform_point(Vector3f{local_min_target_bounds.x(), local_min_target_bounds.y(), heightfield->grid_data[left_bottom_idx] * heightfield->max_height });
    vertices[2] = heightfield->to_world.transform_point(Vector3f{local_max_target_bounds.x(), local_min_target_bounds.y(), heightfield->grid_data[right_bottom_idx] * heightfield->max_height });
    vertices[3] = heightfield->to_world.transform_point(Vector3f{local_max_target_bounds.x(), local_max_target_bounds.y(), heightfield->grid_data[right_top_idx] * heightfield->max_height });
}


__device__ IntersectResult moeller_trumbore_two_triangles(const Ray3f &ray, Vector3f vertices[4]) {
    bool intersect_t1_found = true;
    bool intersect_t2_found = true;

    Vector3f e1t1 = vertices[1] - vertices[0], e1t2 = vertices[0] - vertices[3], e2t1 = vertices[2] - vertices[0], e2t2 = vertices[2] - vertices[3];

    Vector3f pvec_t1 = cross(ray.d, e2t1);
    Vector3f pvec_t2 = cross(ray.d, e2t2);
    float inv_det_t1 = 1.0f / dot(e1t1, pvec_t1);
    float inv_det_t2 = 1.0f / dot(e1t2, pvec_t1);

    Vector3f tvec_t1 = ray.o - vertices[0];
    Vector3f tvec_t2 = ray.o - vertices[3];

    float u1 = dot(tvec_t1, pvec_t1) * inv_det_t1;
    float u2 = dot(tvec_t2, pvec_t2) * inv_det_t2;
    intersect_t1_found &= u1 >= 0.0f && u1 <= 1.0f;
    intersect_t2_found &= u2 >= 0.0f && u2 <= 1.0f;

    Vector3f qvec1 = cross(tvec_t1, e1t1);
    Vector3f qvec2 = cross(tvec_t2, e1t2);
    float v1 = dot(ray.d, qvec1) * inv_det_t1;
    float v2 = dot(ray.d, qvec2) * inv_det_t2;
    intersect_t1_found &= v1 >= 0.0f && u1 + v1 <= 1.0f;
    intersect_t2_found &= v2 >= 0.0f && u2 + v2 <= 1.0f;

    float t1 = dot(e2t1, qvec1) * inv_det_t1;
    float t2 = dot(e2t2, qvec2) * inv_det_t2;
    intersect_t1_found &= t1 >= 0.f && t1 <= ray.maxt;
    intersect_t2_found &= t2 >= 0.f && t2 <= ray.maxt;

    float closest_t = (t1 < t2 && (intersect_t1_found && intersect_t2_found)) || (intersect_t1_found && !intersect_t2_found) ? t1 : t2; 
    float closest_u = (t1 < t2 && (intersect_t1_found && intersect_t2_found)) || (intersect_t1_found && !intersect_t2_found) ? u1 : u2; 
    float closest_v = (t1 < t2 && (intersect_t1_found && intersect_t2_found)) || (intersect_t1_found && !intersect_t2_found) ? v1 : v2; 

    return IntersectResult{intersect_t1_found || intersect_t2_found, closest_t, closest_u, closest_v };
}

extern "C" __global__ void __intersection__heightfield() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixHeightfieldData *heightfield = (OptixHeightfieldData *)sbt_data->data;
    unsigned int aabb_index = optixGetPrimitiveIndex();

    // Ray in instance-space
    Ray3f ray = get_ray();
    
    Vector3f tile_vertices[4];
    get_tile_vertices(aabb_index, tile_vertices, heightfield);

    IntersectResult res = moeller_trumbore_two_triangles(ray, tile_vertices);   

    if(res.found_intersection)
        optixReportIntersection(res.t_closest, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
}

extern "C" __global__ void __closesthit__heightfield() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    unsigned int prim_index = optixGetPrimitiveIndex();
    OptixHeightfieldData *heightfield = (OptixHeightfieldData *)sbt_data->data;
    
    Ray3f ray = get_ray();
    Vector3f inters_p = ray(optixGetRayTmax());

    Vector3f tile_vertices[4];
    get_tile_vertices(prim_index, tile_vertices, heightfield); 

    IntersectResult res = moeller_trumbore_two_triangles(ray, tile_vertices);

    set_preliminary_intersection_to_payload(res.t_closest, Vector2f(res.u_closest, res.v_closest), prim_index,
                                            sbt_data->shape_registry_id);
}
#endif
