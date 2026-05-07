#include "skinning_kernels.h"

#include "se3_math.cuh"

namespace
{

__device__ __forceinline__ int voxel_index(int x, int y, int z, int3 dims)
{
  return z * dims.x * dims.y + y * dims.x + x;
}

__device__ __forceinline__ int canonical_voxel_index_device(float3 p,
                                                            float3 origin,
                                                            float voxel_size,
                                                            int3 dims)
{
  int vx = __float2int_rd((p.x - origin.x) / voxel_size);
  int vy = __float2int_rd((p.y - origin.y) / voxel_size);
  int vz = __float2int_rd((p.z - origin.z) / voxel_size);
  if (vx < 0 || vx >= dims.x || vy < 0 || vy >= dims.y || vz < 0 ||
      vz >= dims.z)
    return -1;
  return voxel_index(vx, vy, vz, dims);
}

} // namespace

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
    const float *voxel_knn_w)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_vertices)
    return;

  float3 p = canonical_vertices[i];
  if (num_nodes <= 0 || !nodes || !transforms || !voxel_knn || !voxel_knn_w)
  {
    warped_vertices[i] = p;
    return;
  }

  int vidx = canonical_voxel_index_device(p, origin, voxel_size, dims);
  if (vidx < 0)
  {
    warped_vertices[i] = p;
    return;
  }

  const int *ids = voxel_knn + vidx * K_NEIGHBORS;
  const float *ws = voxel_knn_w + vidx * K_NEIGHBORS;
  warped_vertices[i] = warp_point(p, nodes, transforms, ids, ws);
}
