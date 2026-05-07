#include "tsdf_volume.h"
#include "tsdf_kernels.h"

#include <cstring>
#include <vector>

// ─────────────────────────────────────────────
//  TSDFVolume
// ─────────────────────────────────────────────

TSDFVolume::TSDFVolume(const Params& p) : params_(p) {
  int total = p.dims.x * p.dims.y * p.dims.z;
  d_voxels_.allocate(total);

  // Inizializza voxel: TSDF=1 (tutto "fuori"), weight=0
  std::vector<TSDFVoxel> init(total);
  d_voxels_.upload(init);
}

TSDFVolume::~TSDFVolume() {}

void TSDFVolume::integrate(const DeviceArray<float>& depth,
                           const DeviceArray<float3>& normals,
                           const CameraIntrinsics& cam, const Mat4& camera_pose,
                           const DeformNode* nodes, const DualQuat* transforms,
                           int num_nodes, const int* voxel_knn,
                           const float* voxel_knn_w,
                           const int* voxel_opt_counts,
                           int min_opt_count) {
  bool use_warp = (nodes != nullptr && num_nodes > 0);

  // T_cam_world = inversa di camera_pose (world→camera)
  // Approssimazione per matrici rigide: R^T, -R^T t
  Mat4 T_cam_world;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      T_cam_world.m[i][j] = camera_pose.m[j][i];  // trasposizione R
  // traslazione: -R^T * t
  T_cam_world.m[0][3] = -(T_cam_world.m[0][0] * camera_pose.m[0][3] +
                          T_cam_world.m[0][1] * camera_pose.m[1][3] +
                          T_cam_world.m[0][2] * camera_pose.m[2][3]);
  T_cam_world.m[1][3] = -(T_cam_world.m[1][0] * camera_pose.m[0][3] +
                          T_cam_world.m[1][1] * camera_pose.m[1][3] +
                          T_cam_world.m[1][2] * camera_pose.m[2][3]);
  T_cam_world.m[2][3] = -(T_cam_world.m[2][0] * camera_pose.m[0][3] +
                          T_cam_world.m[2][1] * camera_pose.m[1][3] +
                          T_cam_world.m[2][2] * camera_pose.m[2][3]);

  dim3 block(8, 8, 1);
  dim3 grid((params_.dims.x + block.x - 1) / block.x,
            (params_.dims.y + block.y - 1) / block.y, params_.dims.z);

  // Chiama kernel definito in tsdf_kernels.cu
  tsdf_integrate_kernel<<<grid, block>>>(
      d_voxels_.data, params_.dims, params_.origin, params_.voxel_size,
      params_.truncation, depth.data, cam.width, cam.height, cam, T_cam_world,
      nodes, transforms, num_nodes, voxel_knn, voxel_knn_w, voxel_opt_counts,
      min_opt_count, use_warp);

  cudaDeviceSynchronize();
}

void TSDFVolume::raycast(DeviceArray<float3>& vertices,
                         DeviceArray<float3>& normals,
                         const CameraIntrinsics& cam, const Mat4& camera_pose,
                         const DeformNode* nodes, const DualQuat* transforms,
                         int num_nodes, const int* voxel_knn,
                         const float* voxel_knn_w,
                         int* out_canonical_vidx) {
  int n_pixels = cam.width * cam.height;
  vertices.allocate(n_pixels);
  normals.allocate(n_pixels);

  bool use_warp = (nodes != nullptr && num_nodes > 0);

  dim3 block(16, 16);
  dim3 grid((cam.width + block.x - 1) / block.x,
            (cam.height + block.y - 1) / block.y);

  raycast_kernel<<<grid, block>>>(
      d_voxels_.data, params_.dims, params_.origin, params_.voxel_size,
      params_.truncation, vertices.data, normals.data, out_canonical_vidx,
      cam.width, cam.height, cam, camera_pose, nodes, transforms, num_nodes,
      voxel_knn, voxel_knn_w, use_warp);

  cudaDeviceSynchronize();
}

