#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "se3_math.cuh"
#include "warp_field.h"

static float host_norm3(float3 v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static float3 host_cross3(float3 a, float3 b)
{
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

static float host_dot3(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float3 host_normalize3(float3 v)
{
  float n = host_norm3(v);
  return (n > 1e-9f) ? make_float3(v.x / n, v.y / n, v.z / n) : v;
}

static float3 warp_point_dual_quat_host(float3 p,
                                        const std::vector<DeformNode> &nodes,
                                        const std::vector<DualQuat> &transforms,
                                        const int node_ids[K_NEIGHBORS],
                                        const float node_ws[K_NEIGHBORS])
{
  DualQuat dq_blend;
  dq_blend.real = make_float4(0, 0, 0, 0);
  dq_blend.dual = make_float4(0, 0, 0, 0);

  float w_sum = 0.0f;
  for (int k = 0; k < K_NEIGHBORS; k++)
  {
    int nid = node_ids[k];
    float w = node_ws[k];
    if (nid < 0 || nid >= (int)nodes.size() || nid >= (int)transforms.size() ||
        w < 1e-8f)
      continue;

    DualQuat dq = dq_centered(transforms[nid], nodes[nid].pos);
    if (quat_dot(dq.real, dq_blend.real) < 0.0f)
    {
      dq.real.x *= -1;
      dq.real.y *= -1;
      dq.real.z *= -1;
      dq.real.w *= -1;
      dq.dual.x *= -1;
      dq.dual.y *= -1;
      dq.dual.z *= -1;
      dq.dual.w *= -1;
    }

    dq_blend.real.x += w * dq.real.x;
    dq_blend.real.y += w * dq.real.y;
    dq_blend.real.z += w * dq.real.z;
    dq_blend.real.w += w * dq.real.w;
    dq_blend.dual.x += w * dq.dual.x;
    dq_blend.dual.y += w * dq.dual.y;
    dq_blend.dual.z += w * dq.dual.z;
    dq_blend.dual.w += w * dq.dual.w;
    w_sum += w;
  }

  if (w_sum < 1e-8f)
    return p;
  return dq_transform_point(dq_normalize(dq_blend), p);
}

static Mat4 inverse_rigid_host(const Mat4 &pose)
{
  Mat4 inv;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      inv.m[i][j] = pose.m[j][i];
  inv.m[0][3] = -(inv.m[0][0] * pose.m[0][3] + inv.m[0][1] * pose.m[1][3] +
                  inv.m[0][2] * pose.m[2][3]);
  inv.m[1][3] = -(inv.m[1][0] * pose.m[0][3] + inv.m[1][1] * pose.m[1][3] +
                  inv.m[1][2] * pose.m[2][3]);
  inv.m[2][3] = -(inv.m[2][0] * pose.m[0][3] + inv.m[2][1] * pose.m[1][3] +
                  inv.m[2][2] * pose.m[2][3]);
  inv.m[3][0] = inv.m[3][1] = inv.m[3][2] = 0.0f;
  inv.m[3][3] = 1.0f;
  return inv;
}

static Mat4 exp_se3_host(const float dx[6])
{
  return exp_se3(dx);
}
