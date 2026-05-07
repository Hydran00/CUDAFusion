#pragma once

#include "types.h"

__global__ void skin_vertices_from_voxel_knn_kernel(
    const float3 *canonical_vertices,
    float3 *warped_vertices,
    int num_vertices,
    float3 origin,
    float voxel_size,
    int3 dims,
    const DeformNode *nodes,
    const DualQuat *transforms,
    int num_nodes,
    const int *voxel_knn,
    const float *voxel_knn_w);
