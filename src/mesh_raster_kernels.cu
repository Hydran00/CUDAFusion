#include "mesh_raster_kernels.h"

#include <math_constants.h>

namespace
{

__device__ __forceinline__ float edge_fn(float2 a, float2 b, float2 p)
{
  return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b)
{
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float norm3(float3 v)
{
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 normalize3(float3 v)
{
  float n = norm3(v);
  return (n > 1e-9f) ? make_float3(v.x / n, v.y / n, v.z / n) : v;
}

__device__ __forceinline__ int voxel_index(float3 p, float3 origin,
                                           float voxel_size, int3 dims)
{
  int vx = __float2int_rd((p.x - origin.x) / voxel_size);
  int vy = __float2int_rd((p.y - origin.y) / voxel_size);
  int vz = __float2int_rd((p.z - origin.z) / voxel_size);
  if (vx < 0 || vx >= dims.x || vy < 0 || vy >= dims.y || vz < 0 ||
      vz >= dims.z)
    return -1;
  return vz * dims.x * dims.y + vy * dims.x + vx;
}

__device__ __forceinline__ bool project_triangle(
    const float3 *warped_vertices,
    const int ids[3],
    CameraIntrinsics cam,
    Mat4 T_cam_world,
    float3 wc[3],
    float3 cc[3],
    float2 p[3],
    float &area)
{
  wc[0] = warped_vertices[ids[0]];
  wc[1] = warped_vertices[ids[1]];
  wc[2] = warped_vertices[ids[2]];

  cc[0] = T_cam_world.transform_point(wc[0]);
  cc[1] = T_cam_world.transform_point(wc[1]);
  cc[2] = T_cam_world.transform_point(wc[2]);
  if (cc[0].z <= 0.0f || cc[1].z <= 0.0f || cc[2].z <= 0.0f)
    return false;

  p[0] = cam.project(cc[0]);
  p[1] = cam.project(cc[1]);
  p[2] = cam.project(cc[2]);
  area = edge_fn(p[0], p[1], p[2]);
  return fabsf(area) >= 1e-6f;
}

} // namespace

__global__ void rasterize_triangles_z_kernel(
    const float3 *canonical_vertices,
    const float3 *warped_vertices,
    const int3 *triangles,
    int num_vertices,
    int num_triangles,
    CameraIntrinsics cam,
    Mat4 T_cam_world,
    unsigned long long *ztri)
{
  int tri_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_i >= num_triangles)
    return;

  int3 tri = triangles[tri_i];
  int ids[3] = {tri.x, tri.y, tri.z};
  if (ids[0] < 0 || ids[1] < 0 || ids[2] < 0 || ids[0] >= num_vertices ||
      ids[1] >= num_vertices || ids[2] >= num_vertices)
    return;

  float3 wc[3], cc[3];
  float2 p[3];
  float area = 0.0f;
  if (!project_triangle(warped_vertices, ids, cam, T_cam_world, wc, cc, p,
                        area))
    return;

  int min_u = max(0, __float2int_rd(fminf(p[0].x, fminf(p[1].x, p[2].x))));
  int max_u = min(cam.width - 1,
                  __float2int_ru(fmaxf(p[0].x, fmaxf(p[1].x, p[2].x))));
  int min_v = max(0, __float2int_rd(fminf(p[0].y, fminf(p[1].y, p[2].y))));
  int max_v = min(cam.height - 1,
                  __float2int_ru(fmaxf(p[0].y, fmaxf(p[1].y, p[2].y))));
  if (min_u > max_u || min_v > max_v)
    return;

  float3 face_n = normalize3(cross3(sub3(wc[1], wc[0]), sub3(wc[2], wc[0])));
  if (norm3(face_n) < 1e-6f)
    return;

  for (int y = min_v; y <= max_v; y++)
  {
    for (int x = min_u; x <= max_u; x++)
    {
      float2 q = make_float2((float)x + 0.5f, (float)y + 0.5f);
      float w0 = edge_fn(p[1], p[2], q) / area;
      float w1 = edge_fn(p[2], p[0], q) / area;
      float w2 = edge_fn(p[0], p[1], q) / area;
      if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f)
        continue;

      float z = w0 * cc[0].z + w1 * cc[1].z + w2 * cc[2].z;
      if (z <= 0.0f || !isfinite(z))
        continue;

      unsigned int z_key = __float_as_uint(z);
      unsigned long long packed =
          ((unsigned long long)z_key << 32) | (unsigned int)tri_i;
      atomicMin(&ztri[y * cam.width + x], packed);
    }
  }
}

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
    int *out_hit_voxel_idx)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int n_pixels = cam.width * cam.height;
  if (px >= n_pixels)
    return;

  unsigned long long packed = ztri[px];
  if (packed == 0xffffffffffffffffULL)
    return;

  int tri_i = (int)(packed & 0xffffffffULL);
  int3 tri = triangles[tri_i];
  int ids[3] = {tri.x, tri.y, tri.z};
  if (ids[0] < 0 || ids[1] < 0 || ids[2] < 0 || ids[0] >= num_vertices ||
      ids[1] >= num_vertices || ids[2] >= num_vertices)
    return;

  float3 wc[3], cc[3];
  float2 p[3];
  float area = 0.0f;
  if (!project_triangle(warped_vertices, ids, cam, T_cam_world, wc, cc, p,
                        area))
    return;

  int x = px % cam.width;
  int y = px / cam.width;
  float2 q = make_float2((float)x + 0.5f, (float)y + 0.5f);
  float w0 = edge_fn(p[1], p[2], q) / area;
  float w1 = edge_fn(p[2], p[0], q) / area;
  float w2 = edge_fn(p[0], p[1], q) / area;
  if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f)
    return;

  out_vertices_live[px] =
      make_float3(w0 * wc[0].x + w1 * wc[1].x + w2 * wc[2].x,
                  w0 * wc[0].y + w1 * wc[1].y + w2 * wc[2].y,
                  w0 * wc[0].z + w1 * wc[1].z + w2 * wc[2].z);
  out_normals_live[px] =
      normalize3(cross3(sub3(wc[1], wc[0]), sub3(wc[2], wc[0])));

  float3 c0 = canonical_vertices[ids[0]];
  float3 c1 = canonical_vertices[ids[1]];
  float3 c2 = canonical_vertices[ids[2]];
  float3 cp = make_float3(w0 * c0.x + w1 * c1.x + w2 * c2.x,
                          w0 * c0.y + w1 * c1.y + w2 * c2.y,
                          w0 * c0.z + w1 * c1.z + w2 * c2.z);
  out_hit_voxel_idx[px] = voxel_index(cp, tsdf_origin, voxel_size, tsdf_dims);
}
