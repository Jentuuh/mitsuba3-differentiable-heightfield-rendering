#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixHeightfieldData {
    optix::Transform4f to_object;
    size_t res_x;
    size_t res_y;
    float* grid_data;
    float max_height;
};


#ifdef __CUDACC__
extern "C" __global__ void __intersection__heightfield() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixHeightfieldData *heightfield = (OptixHeightfieldData *)sbt_data->data;
    unsigned int aabb_index = optixGetPrimitiveIndex();

    float cell_size[2] = { 2.0f / (heightfield->res_x  - 1), 2.0f / (heightfield->res_y - 1)};
    unsigned int amount_rows = heightfield->res_x - 1;
    unsigned int values_per_row = heightfield->res_x ;
    unsigned int row_nr = floor((float)aabb_index / (float) amount_rows);
    unsigned int row_offset = aabb_index % (amount_rows);
    
    // Compute the fractional bounds of the cell we're testing the intersection for
    Vector2f local_min_target_bounds = { -1.0f + (float)row_offset * cell_size[0], -1.0f + (float)row_nr * cell_size[1] };
    Vector2f local_max_target_bounds = { -1.0f + (float)(row_offset + 1) * cell_size[0], -1.0f + (float)(row_nr + 1) * cell_size[1] };
    
    // Ray in instance-space
    Ray3f ray = get_ray();
    // Ray in object-space
    ray = heightfield->to_object.transform_ray(ray);

    // `(amount_rows - row_nr) * values_per_row` gives us the offset to get to the current row, we add the 
    // row offset to this to get the absolute offset to obtain the corresponding texel of the
    // current AABB (+ 1 in both dimensions to get right and top texels)  
    unsigned int left_bottom_idx = (amount_rows - row_nr) * values_per_row + row_offset;
    unsigned int right_bottom_idx = (amount_rows - row_nr) * values_per_row + row_offset + 1;
    unsigned int left_top_idx = (amount_rows - (row_nr + 1)) * values_per_row + row_offset;
    unsigned int right_top_idx = (amount_rows - (row_nr + 1)) * values_per_row + row_offset + 1;

    float max_displacement_in_tile = fmaxf(heightfield->grid_data[left_bottom_idx], 
                                fmaxf(heightfield->grid_data[right_bottom_idx],
                                fmaxf(heightfield->grid_data[left_top_idx],
                                      heightfield->grid_data[right_top_idx])));

    // Check how high the heightfield is at current cell (we will intersect a plane at this height parallel to the XY plane)
    float z_displacement = max_displacement_in_tile * heightfield->max_height;

    // We intersect with the plane Z = `z_displacement`, parallel to the XY plane (flat heightfield in local space is defined as a rectangle aligned with XY)
    float t = (z_displacement - ray.o.z()) / ray.d.z();
    Vector3f local = ray(t);  

    // Is intersection within ray segment and heightfield cell?
    if (local.x() >= local_min_target_bounds.x() 
        && local.y() >= local_min_target_bounds.y()
        && local.x() <= local_max_target_bounds.x()
        && local.y() <= local_max_target_bounds.y())
        optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
}

extern "C" __global__ void __closesthit__heightfield() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    unsigned int prim_index = optixGetPrimitiveIndex();
    set_preliminary_intersection_to_payload(optixGetRayTmax(), Vector2f(), prim_index,
                                            sbt_data->shape_registry_id);
}
#endif
