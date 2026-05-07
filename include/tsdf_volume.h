#pragma once
#include <string>

#include "types.h"

// ─────────────────────────────────────────────
//  TSDF Volume — gestisce il volume canonico
// ─────────────────────────────────────────────

class TSDFVolume {
 public:
  struct Params {
    int3 dims = {256, 256, 256};               // voxel dimensions
    float voxel_size = 0.006f;                 // 6mm per voxel
    float truncation = 0.03f;                  // 5x voxel_size
    float3 origin = {-0.768f, -0.768f, 0.5f};  // world origin
    // TSDF decay parameters (for dynamic scenes)
    float decay_alpha = 0.90f;  // multiply previous weight by this each frame
    float max_weight = 50.0f;   // cap for voxel weight to keep reactivity
  };

  explicit TSDFVolume(const Params &p);
  ~TSDFVolume();

  // Integra una depth map nel volume usando un warp field
  // Se nodes/transforms sono nullptr → integrazione rigida (KinectFusion)
  void integrate(const DeviceArray<float> &depth,
                 const DeviceArray<float3> &normals,
                 const CameraIntrinsics &cam, const Mat4 &camera_pose,
                 // Warp field (DynamicFusion)
                 const DeformNode *nodes = nullptr,
                 const DualQuat *transforms = nullptr, int num_nodes = 0,
                 const int *voxel_knn = nullptr,      // [vol_size * K]
                 const float *voxel_knn_w = nullptr,  // [vol_size * K]
                 const int *voxel_opt_counts = nullptr, int min_opt_count = 0);

  // Raycasting → mappa di vertici e normali (live frame)
  void raycast(DeviceArray<float3> &vertices, DeviceArray<float3> &normals,
               const CameraIntrinsics &cam, const Mat4 &camera_pose,
               const DeformNode *nodes = nullptr,
               const DualQuat *transforms = nullptr, int num_nodes = 0,
               const int *voxel_knn = nullptr,
               const float *voxel_knn_w = nullptr,
               int *out_canonical_vidx = nullptr);

  // Estrai mesh (marching cubes semplificato — output su CPU)
  void extract_surface(std::vector<float3> &vertices,
                       std::vector<float3> &normals,
                       std::vector<int3> &triangles) const;

  // Estrai mesh direttamente su GPU. Usato nel tracking per evitare il
  // round-trip CPU->GPU prima di skinning e rasterizzazione.
  void extract_surface_device(DeviceArray<float3> &vertices,
                              DeviceArray<int3> &triangles) const;

  // Accessors
  const Params &params() const { return params_; }
  TSDFVoxel *device_data() { return d_voxels_.data; }
  const TSDFVoxel *device_data() const { return d_voxels_.data; }
  int total_voxels() const {
    return params_.dims.x * params_.dims.y * params_.dims.z;
  }

  // Conversione indice ↔ posizione 3D
  __host__ int3 idx_to_xyz(int idx) const {
    int x = idx % params_.dims.x;
    int y = (idx / params_.dims.x) % params_.dims.y;
    int z = idx / (params_.dims.x * params_.dims.y);
    return make_int3(x, y, z);
  }

  __host__ float3 voxel_to_world(int3 xyz) const {
    return make_float3(params_.origin.x + xyz.x * params_.voxel_size,
                       params_.origin.y + xyz.y * params_.voxel_size,
                       params_.origin.z + xyz.z * params_.voxel_size);
  }

 private:
  Params params_;
  DeviceArray<TSDFVoxel> d_voxels_;
};
