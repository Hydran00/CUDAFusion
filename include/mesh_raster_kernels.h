#pragma once

#include "types.h"

__global__ void rasterize_triangles_z_kernel(
    const float3 *canonical_vertices,
    const float3 *warped_vertices,
    const int3 *triangles,
    int num_vertices,
    int num_triangles,
    CameraIntrinsics cam,
    Mat4 T_cam_world,
    unsigned long long *ztri);

__global__ void resolve_rasterized_triangles_kernel(
    const float3 *canonical_vertices,
    const float3 *warped_vertices,
    const int3 *triangles,
    int num_vertices,
    CameraIntrinsics cam,
    Mat4 T_cam_world,
    float3 tsdf_origin,
    float voxel_size,
    int3 tsdf_dims,
    const unsigned long long *ztri,
    float3 *out_vertices_live,
    float3 *out_normals_live,
    int *out_hit_voxel_idx);
