#pragma once
#include "types.h"

__global__ void tsdf_integrate_kernel(
    TSDFVoxel*        voxels,
    int3              dims,
    float3            origin,
    float             voxel_size,
    float             truncation,
    const float*      depth,
    int               img_w,
    int               img_h,
    CameraIntrinsics  cam,
    Mat4              T_cam_world,
    const DeformNode* nodes,
    const Mat4*       transforms,
    int               num_nodes,
    const int*        voxel_knn,
    const float*      voxel_knn_w,
    bool              use_warp);

__global__ void raycast_kernel(
    const TSDFVoxel*  voxels,
    int3              dims,
    float3            origin,
    float             voxel_size,
    float             truncation,
    float3*           out_vertices,
    float3*           out_normals,
    int               img_w,
    int               img_h,
    CameraIntrinsics  cam,
    Mat4              T_world_cam,
    const DeformNode* nodes,
    const Mat4*       transforms,
    int               num_nodes,
    const int*        voxel_knn,
    const float*      voxel_knn_w,
    bool              use_warp);

__global__ void compute_depth_normals_kernel(
    const float*     depth,
    float3*          normals,
    int              w,
    int              h,
    CameraIntrinsics cam);
