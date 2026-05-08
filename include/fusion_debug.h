#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

#include "se3_math.cuh"
#include "types.h"
#include "warp_field.h"

class ScopedStdoutSilence
{
public:
  explicit ScopedStdoutSilence(bool enabled) : enabled_(enabled)
  {
    if (!enabled_)
      return;
    fflush(stdout);
    saved_stdout_ = dup(STDOUT_FILENO);
    null_fd_ = open("/dev/null", O_WRONLY);
    if (saved_stdout_ >= 0 && null_fd_ >= 0)
      dup2(null_fd_, STDOUT_FILENO);
  }

  ~ScopedStdoutSilence()
  {
    if (!enabled_)
      return;
    fflush(stdout);
    if (saved_stdout_ >= 0)
    {
      dup2(saved_stdout_, STDOUT_FILENO);
      close(saved_stdout_);
    }
    if (null_fd_ >= 0)
      close(null_fd_);
  }

private:
  bool enabled_ = false;
  int saved_stdout_ = -1;
  int null_fd_ = -1;
};

static cv::Mat dbg_depth(const std::vector<float> &d, int W, int H,
                         float max_m = 4.0f)
{
  cv::Mat grey(H, W, CV_8U);
  for (int i = 0; i < H * W; i++)
  {
    float v = d[i];
    grey.data[i] = (v > 0.01f && v < max_m) ? (uint8_t)(v / max_m * 255.f) : 0;
  }
  cv::Mat color;
  cv::applyColorMap(grey, color, cv::COLORMAP_JET);
  for (int i = 0; i < H * W; i++)
    if (d[i] <= 0.01f || d[i] >= max_m)
      color.data[i * 3] = color.data[i * 3 + 1] = color.data[i * 3 + 2] = 0;
  return color;
}

static cv::Mat dbg_normals(const std::vector<float3> &n, int W, int H)
{
  cv::Mat img(H, W, CV_8UC3);
  for (int i = 0; i < H * W; i++)
  {
    float len = sqrtf(n[i].x * n[i].x + n[i].y * n[i].y + n[i].z * n[i].z);
    if (len < 0.5f)
    {
      img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] = 0;
    }
    else
    {
      img.data[i * 3 + 0] = (uint8_t)((n[i].z * 0.5f + 0.5f) * 255);
      img.data[i * 3 + 1] = (uint8_t)((n[i].y * 0.5f + 0.5f) * 255);
      img.data[i * 3 + 2] = (uint8_t)((n[i].x * 0.5f + 0.5f) * 255);
    }
  }
  return img;
}

static cv::Mat dbg_verts(const std::vector<float3> &verts,
                         const CameraIntrinsics &cam, const Mat4 &T_cam_world,
                         const std::vector<DeformNode> *nodes = nullptr,
                         const std::vector<DualQuat> *transforms = nullptr)
{
  cv::Mat depth(cam.height, cam.width, CV_32F, cv::Scalar(0.f));
  for (const auto &vw : verts)
  {
    float3 v = T_cam_world.transform_point(vw);
    if (v.z <= 0.01f)
      continue;
    int u = (int)(cam.fx * v.x / v.z + cam.cx);
    int vv = (int)(cam.fy * v.y / v.z + cam.cy);
    if (u >= 0 && u < cam.width && vv >= 0 && vv < cam.height)
      depth.at<float>(vv, u) = v.z;
  }
  cv::Mat grey, color;
  cv::normalize(depth, grey, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::applyColorMap(grey, color, cv::COLORMAP_TURBO);
  for (int r = 0; r < cam.height; r++)
    for (int c = 0; c < cam.width; c++)
      if (depth.at<float>(r, c) <= 0.01f)
        color.at<cv::Vec3b>(r, c) = {0, 0, 0};

  auto project = [&](float3 p, cv::Point &q)
  {
    p = T_cam_world.transform_point(p);
    if (p.z <= 0.01f)
      return false;
    q = {(int)(cam.fx * p.x / p.z + cam.cx), (int)(cam.fy * p.y / p.z + cam.cy)};
    return q.x >= 0 && q.x < cam.width && q.y >= 0 && q.y < cam.height;
  };

  if (nodes && transforms)
  {
    int n = std::min((int)nodes->size(), (int)transforms->size());
    for (int i = 0; i < n; i++)
    {
      float3 pi = dq_transform_point(dq_centered((*transforms)[i], (*nodes)[i].pos),
                                     (*nodes)[i].pos);
      cv::Point a;
      if (!project(pi, a))
        continue;
      for (int k = 0; k < (*nodes)[i].num_neighbors; k++)
      {
        int j = (*nodes)[i].neighbors[k];
        if (j < 0 || j >= n)
          continue;
        if (j < i)
          continue;
        float3 pj = dq_transform_point(
            dq_centered((*transforms)[j], (*nodes)[j].pos), (*nodes)[j].pos);
        cv::Point b;
        if (project(pj, b))
        {
          float3 d = make_float3((*nodes)[i].pos.x - (*nodes)[j].pos.x,
                                 (*nodes)[i].pos.y - (*nodes)[j].pos.y,
                                 (*nodes)[i].pos.z - (*nodes)[j].pos.z);
          float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
          float ref = std::max((*nodes)[i].radius, (*nodes)[j].radius);
          cv::Scalar edge_col = (len > 2.5f * ref) ? cv::Scalar(0, 0, 255)
                                                    : cv::Scalar(100, 0, 255);
          cv::line(color, a, b, edge_col, 1, cv::LINE_AA);
        }
      }
      cv::circle(color, a, 2, {0, 250, 255}, -1, cv::LINE_AA);
    }
  }
  cv::putText(color, "live surface", {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
              {220, 220, 220}, 1);
  return color;
}

static cv::Mat dbg_corrs(const std::vector<Correspondence> &corrs,
                         const CameraIntrinsics &cam, const Mat4 &T_cam_world,
                         int n_valid)
{
  cv::Mat img(cam.height, cam.width, CV_8UC3, cv::Scalar(15, 15, 15));
  int drawn = 0;
  for (const auto &c : corrs)
  {
    if (!c.valid)
      continue;

    float3 dst = T_cam_world.transform_point(c.dst);
    float3 src = T_cam_world.transform_point(c.src);
    if (dst.z > 0.01f)
    {
      int u = (int)(cam.fx * dst.x / dst.z + cam.cx);
      int v = (int)(cam.fy * dst.y / dst.z + cam.cy);
      if (u >= 0 && u < cam.width && v >= 0 && v < cam.height)
      {
        img.at<cv::Vec3b>(v, u)[2] = 180;
        drawn++;
      }
    }

    if (src.z > 0.01f)
    {
      int u = (int)(cam.fx * src.x / src.z + cam.cx);
      int v = (int)(cam.fy * src.y / src.z + cam.cy);
      if (u >= 0 && u < cam.width && v >= 0 && v < cam.height)
        img.at<cv::Vec3b>(v, u)[1] = 180;
    }

    if (src.z > 0.01f && dst.z > 0.01f && drawn % 80 == 0)
    {
      int u1 = (int)(cam.fx * src.x / src.z + cam.cx);
      int v1 = (int)(cam.fy * src.y / src.z + cam.cy);
      int u2 = (int)(cam.fx * dst.x / dst.z + cam.cx);
      int v2 = (int)(cam.fy * dst.y / dst.z + cam.cy);
      if (u1 >= 0 && u1 < cam.width && v1 >= 0 && v1 < cam.height && u2 >= 0 &&
          u2 < cam.width && v2 >= 0 && v2 < cam.height)
        cv::line(img, {u1, v1}, {u2, v2}, {200, 200, 200}, 1);
    }
  }

  cv::Mat grown;
  cv::dilate(img, grown, cv::Mat(), cv::Point(-1, -1), 1);
  cv::putText(grown,
              "G=src R=dst valid=" + std::to_string(n_valid) +
                  " shown=" + std::to_string(drawn),
              {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
  return grown;
}
