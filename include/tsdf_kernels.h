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
    int*              out_canonical_vidx,
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

__global__ void find_correspondences_kernel(
    const float3*      live_vertices,
    const float3*      live_normals,
    const float*       depth_map,
    const float3*      depth_normals,
    Correspondence*    corrs,
    int*               num_valid,
    int                img_w,
    int                img_h,
    CameraIntrinsics   cam,
    Mat4               T_cam_world,
    const int*         pixel_knn,
    const float*       pixel_knn_w,
    float              dist_threshold,
    float              angle_threshold,
    float              view_threshold,
    int                search_radius_px,
    int*               debug_stats);

__global__ void compute_pixel_knn_kernel(
    const int*    canonical_vidx,
    int           img_w,
    int           img_h,
    const int*    voxel_knn,
    const float*  voxel_knn_w,
    int*          pixel_knn,
    float*        pixel_knn_w);

__global__ void bbox_filter_kernel(
    float3* vertices,
    int     n,
    float3  min_pt,
    float3  max_pt);

__global__ void depth_bbox_filter_kernel(
    float*           depth,
    int              w,
    int              h,
    CameraIntrinsics cam,
    float3           min_pt,
    float3           max_pt);
