#pragma once
#include <cuda_runtime.h>
#include <open3d/Open3D.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "cudaImage.h"
#include "cudaSift.h"
#include "se3_math.cuh"
#include "solver.h"
#include "tsdf_kernels.h"
#include "tsdf_volume.h"
#include "types.h"
#include "warp_field.h"
#include "yaml_helper.h"

namespace fs = std::filesystem;

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
  float ox = dx[0], oy = dx[1], oz = dx[2];
  float vx = dx[3], vy = dx[4], vz = dx[5];
  float theta2 = ox * ox + oy * oy + oz * oz;
  float theta = std::sqrt(theta2);
  float A, B, C;
  if (theta < 1e-6f)
  {
    A = 1.0f;
    B = 0.5f;
    C = 1.0f / 6.0f;
  }
  else
  {
    float inv_t = 1.0f / theta;
    float inv_t2 = inv_t * inv_t;
    A = std::sin(theta) * inv_t;
    B = (1.0f - std::cos(theta)) * inv_t2;
    C = (theta - std::sin(theta)) * inv_t2 * inv_t;
  }
  Mat4 T;
  T.m[0][0] = 1.f - B * (oy * oy + oz * oz);
  T.m[0][1] = B * ox * oy - A * oz;
  T.m[0][2] = B * ox * oz + A * oy;
  T.m[1][0] = B * ox * oy + A * oz;
  T.m[1][1] = 1.f - B * (ox * ox + oz * oz);
  T.m[1][2] = B * oy * oz - A * ox;
  T.m[2][0] = B * ox * oz - A * oy;
  T.m[2][1] = B * oy * oz + A * ox;
  T.m[2][2] = 1.f - B * (ox * ox + oy * oy);
  float Vxx = 1.f - C * (oy * oy + oz * oz);
  float Vxy = C * ox * oy - B * oz;
  float Vxz = C * ox * oz + B * oy;
  float Vyx = C * ox * oy + B * oz;
  float Vyy = 1.f - C * (ox * ox + oz * oz);
  float Vyz = C * oy * oz - B * ox;
  float Vzx = C * ox * oz - B * oy;
  float Vzy = C * oy * oz + B * ox;
  float Vzz = 1.f - C * (ox * ox + oy * oy);
  T.m[0][3] = Vxx * vx + Vxy * vy + Vxz * vz;
  T.m[1][3] = Vyx * vx + Vyy * vy + Vyz * vz;
  T.m[2][3] = Vzx * vx + Vzy * vy + Vzz * vz;
  return T;
}

// ─────────────────────────────────────────────
//  Depth sequence loader
//  Supporta: TUM RGB-D, ICL-NUIM, cartella raw
// ─────────────────────────────────────────────

struct DepthFrame
{
  cv::Mat depth_m; // float32, metri
  cv::Mat color_gray;
  double timestamp;
  int index;
};

class DepthSequence
{
public:
  enum class Format
  {
    TUM,
    ICL,
    RAW_PNG,
    RAW_EXR
  };

  struct Config
  {
    std::string path;
    Format format = Format::TUM;
    float depth_scale = 0.001f; // uint16 raw → metres
    int start_frame = 0;
    int max_frames = -1; // -1 = tutti
    CameraIntrinsics cam;
  };

  explicit DepthSequence(const Config &cfg) : cfg_(cfg)
  {
    switch (cfg.format)
    {
    case Format::TUM:
      load_tum();
      break;
    case Format::ICL:
      load_icl();
      break;
    case Format::RAW_PNG:
    case Format::RAW_EXR:
      load_raw();
      break;
    }
    std::cout << "[DepthSequence] Loaded " << frames_.size()
              << " frames from: " << cfg.path << "\n";
  }

  bool has_next() const { return current_idx_ < (int)frames_.size(); }

  DepthFrame next()
  {
    auto &f = frames_[current_idx_++];
    return f;
  }

  int total() const { return (int)frames_.size(); }

  const CameraIntrinsics &camera() const { return cfg_.cam; }

private:
  Config cfg_;
  std::vector<DepthFrame> frames_;
  int current_idx_ = 0;

  void load_tum()
  {
    // Formato TUM: depth/*.png, associations.txt
    // depth images uint16, scaled according to config
    std::string assoc_path = cfg_.path + "/associations.txt";
    std::ifstream assoc(assoc_path);
    if (!assoc.is_open())
    {
      // Fallback: scansiona cartella depth/ usando la scala del config.
      load_raw_dir(cfg_.path + "/depth", ".png");
      return;
    }

    std::string line;
    int count = 0;
    while (std::getline(assoc, line))
    {
      if (line.empty() || line[0] == '#')
        continue;
      if (count < cfg_.start_frame)
      {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames)
        break;

      std::istringstream ss(line);
      double ts_rgb, ts_depth;
      std::string rgb_file, depth_file;
      ss >> ts_rgb >> rgb_file >> ts_depth >> depth_file;

      std::string full_path = cfg_.path + "/" + depth_file;
      cv::Mat raw = cv::imread(full_path, cv::IMREAD_ANYDEPTH);
      if (raw.empty())
        continue;

      cv::Mat depth_m;
      raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);

      cv::Mat color =
          cv::imread(cfg_.path + "/" + rgb_file, cv::IMREAD_GRAYSCALE);
      if (!color.empty() &&
          (color.cols != depth_m.cols || color.rows != depth_m.rows))
      {
        cv::resize(color, color, depth_m.size(), 0.0, 0.0, cv::INTER_LINEAR);
      }

      frames_.push_back({depth_m, color, ts_depth, count});
      count++;
    }
  }

  void load_icl()
  {
    // Formato ICL-NUIM: depth/*.png (float in mm * 1000)
    load_raw_dir(cfg_.path + "/depth", ".png");
  }

  void load_raw()
  {
    std::string ext = (cfg_.format == Format::RAW_EXR) ? ".exr" : ".png";
    load_raw_dir(cfg_.path, ext);
  }

  void load_raw_dir(const std::string &dir, const std::string &ext,
                    float depth_scale_override = -1.0f)
  {
    const float depth_scale =
        (depth_scale_override > 0.0f) ? depth_scale_override : cfg_.depth_scale;

    if (!fs::exists(dir))
    {
      std::cerr << "[DepthSequence] Directory not found: " << dir << "\n";
      return;
    }
    // std::cout << "[DepthSequence] file: " << p << std::endl;
    std::vector<std::string> paths;
    for (const auto &entry : fs::directory_iterator(dir))
    {
      if (entry.path().extension() == ext)
        paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());

    int count = 0;
    for (const auto &p : paths)
    {
      if (count < cfg_.start_frame)
      {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames)
        break;

      cv::Mat raw = cv::imread(p, cv::IMREAD_ANYDEPTH);
      if (raw.empty())
      {
        // Prova EXR
        raw = cv::imread(p, cv::IMREAD_UNCHANGED);
      }
      if (raw.empty())
      {
        count++;
        continue;
      }

      cv::Mat depth_m;
      if (raw.type() == CV_32F)
      {
        depth_m = raw; // già float in metri
      }
      else if (raw.type() == CV_16U)
      {
        raw.convertTo(depth_m, CV_32F, depth_scale);
      }
      else
      {
        raw.convertTo(depth_m, CV_32F);
      }

      // Filtra valori invalidi
      cv::threshold(depth_m, depth_m, 0.1f, 0, cv::THRESH_TOZERO);
      cv::threshold(depth_m, depth_m, 6.0f, 0, cv::THRESH_TOZERO_INV);

      cv::Mat color;
      depth_to_gray(depth_m, color);
      frames_.push_back({depth_m, color, (double)count, count});
      count++;
    }
  }

public:
  static void depth_to_gray(const cv::Mat &depth_m, cv::Mat &gray)
  {
    double min_v = 0.0, max_v = 0.0;
    cv::Mat valid = depth_m > 0.01f;
    cv::minMaxLoc(depth_m, &min_v, &max_v, nullptr, nullptr, valid);
    if (max_v <= min_v)
    {
      gray = cv::Mat(depth_m.size(), CV_8UC1, cv::Scalar(0));
      return;
    }
    cv::Mat norm;
    depth_m.convertTo(norm, CV_8UC1, -255.0 / (max_v - min_v),
                      255.0 * max_v / (max_v - min_v));
    norm.setTo(0, ~valid);
    gray = norm;
  }
};

// ─────────────────────────────────────────────
//  Debug visualization helpers
// ─────────────────────────────────────────────

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
      img.data[i * 3 + 0] = (uint8_t)((n[i].z * 0.5f + 0.5f) * 255); // B=z
      img.data[i * 3 + 1] = (uint8_t)((n[i].y * 0.5f + 0.5f) * 255); // G=y
      img.data[i * 3 + 2] = (uint8_t)((n[i].x * 0.5f + 0.5f) * 255); // R=x
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
      float3 pi = dq_transform_point(dq_centered((*transforms)[i], (*nodes)[i].pos), (*nodes)[i].pos);
      cv::Point a;
      if (!project(pi, a))
        continue;
      for (int k = 0; k < (*nodes)[i].num_neighbors; k++)
      {
        int j = (*nodes)[i].neighbors[k];
        if (j < 0 || j >= n)
          continue;
        float3 pj = dq_transform_point(dq_centered((*transforms)[j], (*nodes)[j].pos), (*nodes)[j].pos);
        cv::Point b;
        if (project(pj, b))
          cv::line(color, a, b, {100, 0, 255}, 1, cv::LINE_AA);
      }
      cv::circle(color, a, 2, {0, 250, 255}, -1, cv::LINE_AA);
    }
  }
  cv::putText(color, "live surface (raycasted)", {8, 18},
              cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
  return color;
}

static cv::Mat dbg_corrs(const std::vector<Correspondence> &corrs,
                         const CameraIntrinsics &cam, const Mat4 &T_cam_world,
                         int n_valid, bool show_o3d = false)
{
  cv::Mat img(cam.height, cam.width, CV_8UC3, cv::Scalar(15, 15, 15));

  int drawn = 0;

  // --- Open3D containers ---
  std::shared_ptr<open3d::geometry::PointCloud> pcd_src, pcd_dst;
  std::shared_ptr<open3d::geometry::LineSet> lines;

  std::vector<Eigen::Vector3d> pts_src, pts_dst;
  std::vector<Eigen::Vector2i> line_idx;
  std::vector<Eigen::Vector3d> line_colors;

  int idx = 0;

  for (const auto &c : corrs)
  {
    if (!c.valid)
      continue;

    float3 dst = T_cam_world.transform_point(c.dst);
    float3 src = T_cam_world.transform_point(c.src);

    // --- OpenCV debug 2D ---
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
      {
        img.at<cv::Vec3b>(v, u)[1] = 180;
      }
    }

    // linea ogni N per non intasare
    if (src.z > 0.01f && dst.z > 0.01f && drawn % 80 == 0)
    {
      int u1 = (int)(cam.fx * src.x / src.z + cam.cx);
      int v1 = (int)(cam.fy * src.y / src.z + cam.cy);
      int u2 = (int)(cam.fx * dst.x / dst.z + cam.cx);
      int v2 = (int)(cam.fy * dst.y / dst.z + cam.cy);

      if (u1 >= 0 && u1 < cam.width && v1 >= 0 && v1 < cam.height && u2 >= 0 &&
          u2 < cam.width && v2 >= 0 && v2 < cam.height)
      {
        cv::line(img, {u1, v1}, {u2, v2}, {200, 200, 200}, 1);
      }
    }

    // --- Open3D debug 3D ---
    if (show_o3d && src.z > 0.01f && dst.z > 0.01f)
    {
      Eigen::Vector3d ps(src.x, src.y, src.z);
      Eigen::Vector3d pd(dst.x, dst.y, dst.z);

      pts_src.push_back(ps);
      pts_dst.push_back(pd);

      // linea
      line_idx.emplace_back(idx, idx + 1);

      double len = (ps - pd).norm();
      double t = std::min(len / 0.05, 1.0); // clamp 5cm

      line_colors.emplace_back(t, 1.0 - t, 0.0); // rosso->verde

      idx += 2;

      // if (idx > 4000) break;  // evita esplosione viewer
    }
  }

  cv::Mat grown;
  cv::dilate(img, grown, cv::Mat(), cv::Point(-1, -1), 1);

  cv::putText(grown,
              "G=src  R=dst  valid=" + std::to_string(n_valid) +
                  " shown=" + std::to_string(drawn),
              {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);

  // --- mostra Open3D ---
  if (show_o3d && !pts_src.empty())
  {
    pcd_src = std::make_shared<open3d::geometry::PointCloud>();
    pcd_dst = std::make_shared<open3d::geometry::PointCloud>();
    lines = std::make_shared<open3d::geometry::LineSet>();

    std::vector<Eigen::Vector3d> all_pts;
    for (size_t i = 0; i < pts_src.size(); ++i)
    {
      all_pts.push_back(pts_src[i]);
      all_pts.push_back(pts_dst[i]);
    }

    pcd_src->points_ = pts_src;
    pcd_dst->points_ = pts_dst;

    pcd_src->PaintUniformColor({0.0, 1.0, 0.0}); // verde
    pcd_dst->PaintUniformColor({1.0, 0.0, 0.0}); // rosso

    lines->points_ = all_pts;
    lines->lines_ = line_idx;
    lines->colors_ = line_colors;

    std::cout << "[dbg_corrs] Open3D viewer: " << pts_src.size()
              << " corrispondenze" << std::endl;
    open3d::visualization::DrawGeometries({pcd_src, pcd_dst, lines},
                                          "corrs 3D");
  }

  return grown;
}

static cv::Mat dbg_delta_x(const std::vector<float> &dx, int n_nodes)
{
  const int bar_w = 4, bar_h = 200, pad = 2;
  int img_w = std::max(300, (bar_w + pad) * n_nodes + pad);
  cv::Mat img(bar_h + 20, img_w, CV_8UC3, cv::Scalar(20, 20, 20));
  float max_norm = 0.f;
  std::vector<float> norms(n_nodes, 0.f);
  for (int i = 0; i < n_nodes; i++)
  {
    for (int j = 0; j < 6; j++)
      norms[i] += dx[i * 6 + j] * dx[i * 6 + j];
    norms[i] = sqrtf(norms[i]);
    max_norm = std::max(max_norm, norms[i]);
  }
  if (max_norm < 1e-8f)
    max_norm = 1.f;
  int max_bars = (img_w - pad) / (bar_w + pad);
  for (int i = 0; i < n_nodes && i < max_bars; i++)
  {
    int h = (int)(norms[i] / max_norm * bar_h);
    int x0 = pad + i * (bar_w + pad);
    cv::rectangle(img, {x0, bar_h - h}, {x0 + bar_w, bar_h}, {0, 200, 200}, -1);
  }
  cv::putText(img, "delta_x norm/node  max=" + std::to_string(max_norm),
              {4, bar_h + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200, 200, 200},
              1);
  return img;
}

// ─────────────────────────────────────────────
//  Pipeline DynamicFusion (self-contained per test)
// ─────────────────────────────────────────────

class DynamicFusionPipeline
{
public:
  struct Params
  {
    TSDFVolume::Params tsdf;
    GaussNewtonSolver::Params solver;
    CameraIntrinsics cam;
    float node_radius = 0.04f;
    float node_min_dist = 0.025f;
    float dist_threshold = 0.05f; // ICP distanza max
    float angle_threshold = 0.7f; // cos(angolo) min normali
    float view_threshold = 0.2f;  // cos minimo contro silhouette
    int search_radius_px = 3;     // ricerca locale attorno alla proiezione
    int min_valid_corrs = 2000;
    int max_nodes = 4096;
    int max_corrs = 300000;
    int node_update_every_n = 5;
    float max_dx_mean = 0.03f;
    float max_update_rot = 0.05f;
    float max_update_trans = 0.03f;
    float update_scale = 0.25f;
    BBoxFilter bbox;
    bool integrate_warped = false;
    bool integrate_unwarped_fallback = false;
    int integrate_min_optimized_count = 3;
    bool rigid_tracking = true;
    bool use_sift = false;
    int sift_max_features = 8192;
    int sift_max_history = 30;
    int sift_max_matches_per_frame = 64;
    int sift_octaves = 5;
    float sift_threshold = 3.0f;
    float sift_max_match_error = 5.0f;
    float sift_max_ambiguity = 0.85f;
    float sift_max_3d_dist = 0.20f;
    float sift_weight = 1.0f;
    bool debug_vis = false;
    int debug_every_n = 1;
  };

  explicit DynamicFusionPipeline(const Params &p)
      : params_(p), frame_count_(0)
  {
    volume_ = std::make_unique<TSDFVolume>(p.tsdf);
    warp_field_ = std::make_unique<WarpField>(p.node_radius, p.max_nodes);
    solver_ = std::make_unique<GaussNewtonSolver>(p.solver, p.max_nodes);

    int n_pixels = p.cam.width * p.cam.height;
    d_depth_.allocate(n_pixels);
    d_depth_normals_.allocate(n_pixels);
    d_vertices_live_.allocate(n_pixels);
    d_normals_live_.allocate(n_pixels);
    int sift_slots =
        p.use_sift ? 3 * p.sift_max_history * p.sift_max_matches_per_frame : 0;
    d_corrs_.allocate(n_pixels + sift_slots);
    d_delta_x_.allocate(p.max_nodes * 6);
    d_hit_voxel_idx_.allocate(n_pixels);
    d_num_valid_.allocate(1);
    d_corr_stats_.allocate(6);
    int n_voxels = p.tsdf.dims.x * p.tsdf.dims.y * p.tsdf.dims.z;
    d_voxel_opt_counts_.allocate(n_voxels);
    d_voxel_opt_frame_.allocate(n_voxels);
    d_voxel_opt_counts_.zero();
    d_voxel_opt_frame_.zero();

    camera_pose_ = Mat4::identity();
    if (params_.use_sift)
      InitCuda(0);
  }

  bool last_frame_integrated() const { return last_frame_integrated_; }

  void process_frame(const cv::Mat &depth_m,
                     const cv::Mat &color_gray = cv::Mat())
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    last_frame_integrated_ = false;
    current_color_gray_ = color_gray;

    const bool do_dbg =
        params_.debug_vis && (frame_count_ % params_.debug_every_n == 0);

    if (frame_count_ == 0)
      print_cpu_depth_stats(depth_m, "cpu depth input");

    // 1. Upload depth
    {
      std::vector<float> h_depth((float *)depth_m.datastart,
                                 (float *)depth_m.dataend);
      if (frame_count_ == 0)
        print_host_depth_stats(h_depth, "host upload vector");
      d_depth_.upload(h_depth);
    }

    // if (do_dbg) {
    //   std::vector<float> h_d;
    //   d_depth_.download(h_d);
    //   cv::Mat vis = dbg_depth(h_d, params_.cam.width, params_.cam.height);
    //   cv::putText(vis, "frame " + std::to_string(frame_count_), {8, 18},
    //               cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
    //   cv::imshow("1_depth_raw", vis);
    //   cv::waitKey(1);
    // }

    if (frame_count_ == 0)
      print_depth_stats("depth before bbox");

    // 1b. Bounding box filter on depth (before integration)
    if (params_.bbox.enabled)
    {
      apply_depth_bbox_filter();
    }

    if (frame_count_ == 0)
      print_depth_stats("depth after bbox");

    // 2. Calcola normali dal depth
    compute_normals();

    if (do_dbg)
    {
      std::vector<float3> h_n;
      d_depth_normals_.download(h_n);
      cv::imshow("2_depth_normals",
                 dbg_normals(h_n, params_.cam.width, params_.cam.height));
      cv::waitKey(1);
    }

    if (frame_count_ == 0)
    {
      // Primo frame: solo integra
      volume_->integrate(d_depth_, d_depth_normals_, params_.cam, camera_pose_);
      last_frame_integrated_ = true;
      print_tsdf_stats("tsdf after first integrate");
      initialize_node_graph();
    }
    else
    {
      bool warp_update_ok = false;

      // Warp field: salva trasformazioni precedenti (warm start)
      warp_field_->save_transforms();

      int num_corrs = 0;
      int accepted_updates = 0;
      double last_rmse = std::numeric_limits<double>::infinity();
      for (int gn = 0; gn < std::max(1, params_.solver.gn_iterations); gn++)
      {
        bool iter_ok = false;

        // 3. VolumeDeform-style prediction: extract canonical mesh, deform
        // vertices, then rasterize the deformed mesh into the current camera.
        rasterize_deformed_mesh_live_surface();

        if (do_dbg && gn == 0)
        {
          std::vector<float3> h_v;
          d_vertices_live_.download(h_v);
          auto h_nodes = warp_field_->download_nodes();
          auto h_transforms = warp_field_->download_transforms();
          cv::imshow(
              "3_live_surface",
              dbg_verts(h_v, params_.cam, inverse_rigid_host(camera_pose_),
                        &h_nodes, &h_transforms));
          cv::waitKey(1);
        }

        // 3b. Bounding box filter on live vertices
        if (params_.bbox.enabled)
          apply_bbox_filter(d_vertices_live_);

        // 4. Trova corrispondenze ICP. SIFT history is consumed once per frame;
        // later GN iterations refresh only dense projective correspondences.
        num_corrs = find_correspondences(do_dbg && gn == 0, gn == 0);
        if (params_.rigid_tracking && gn == 0 && num_corrs >= 64)
        {
          if (estimate_rigid_tracking_update())
          {
            rasterize_deformed_mesh_live_surface();
            if (params_.bbox.enabled)
              apply_bbox_filter(d_vertices_live_);
            num_corrs = find_correspondences(do_dbg && gn == 0, false);

            // Re-apply SIFT augmentation after the rigid update so sparse
            // SIFT rows stay present and are consistent with the new pose.
            if (params_.use_sift)
            {
              num_corrs = augment_correspondences_with_sift(num_corrs,
                                                            do_dbg && gn == 0,
                                                            false);
              std::cout << "  [rigid update] corrispondenze dopo secondo SIFT: " << num_corrs
                        << "\n";
            }
          }
        }
        std::cout << "  [GN " << gn << "] Corrispondenze valide: " << num_corrs
                  << "\n";
        double rmse = correspondence_rmse();
        if (gn == 0)
        {
          std::cout << "  [pre-opt] corr_rmse=" << std::fixed
                    << std::setprecision(5) << rmse << "\n";
        }
        std::cout << "  [GN " << gn << "] corr_rmse=" << std::fixed
                  << std::setprecision(5) << rmse << "\n";

        // if (gn > 0 && rmse > last_rmse * 1.02)
        // {
        //   warp_field_->restore_transforms();
        //   accepted_updates = std::max(0, accepted_updates - 1);
        //   warp_update_ok = accepted_updates > 0;
        //   std::cout << "  [GN " << gn
        //             << "] reject previous update (rmse increased from "
        //             << last_rmse << " to " << rmse << ")\n";
        //   break;
        // }
        last_rmse = rmse;

        if (do_dbg && gn == 0)
        {
          std::vector<Correspondence> h_c;
          d_corrs_.download(h_c);
          cv::imshow("4_correspondences",
                     dbg_corrs(h_c, params_.cam,
                               inverse_rigid_host(camera_pose_), num_corrs));
          cv::waitKey(1);
        }

        if (num_corrs < params_.min_valid_corrs)
        {
          std::cout
              << "  [solve] skip warp update (too few correspondences, min="
              << params_.min_valid_corrs << ")\n";
          break;
        }

        // 5. Ottimizzazione Gauss-Newton
        solver_->solve(d_corrs_, (int)d_corrs_.size(),
                       warp_field_->device_nodes(),
                       warp_field_->device_transforms(),
                       warp_field_->num_nodes(), d_delta_x_);

        // 6. Valida incremento prima di applicarlo.
        {
          std::vector<float> h_dx;
          d_delta_x_.download(h_dx);

          float rot_sum = 0.0f, trans_sum = 0.0f;
          float rot_max = 0.0f, trans_max = 0.0f;

          float eff_rot_sum = 0.0f, eff_trans_sum = 0.0f;
          float eff_rot_max = 0.0f, eff_trans_max = 0.0f;
          bool finite_raw_update = true;

          int n_nodes = warp_field_->num_nodes();

          for (int i = 0; i < n_nodes; i++)
          {
            float *dx = h_dx.data() + i * 6;

            float rn = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
            float tn = std::sqrt(dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]);
            finite_raw_update =
                finite_raw_update && std::isfinite(rn) && std::isfinite(tn);

            rot_sum += rn;
            trans_sum += tn;

            rot_max = std::max(rot_max, rn);
            trans_max = std::max(trans_max, tn);

            // ---- clamp + scaling (VALORI REALI USATI) ----
            float er = params_.update_scale * rn;
            float et = params_.update_scale * tn;

            if (params_.max_update_rot > 0.0f)
              er = std::min(er, params_.max_update_rot);

            if (params_.max_update_trans > 0.0f)
              et = std::min(et, params_.max_update_trans);

            eff_rot_sum += er;
            eff_trans_sum += et;

            eff_rot_max = std::max(eff_rot_max, er);
            eff_trans_max = std::max(eff_trans_max, et);
          }

          float rot_mean = rot_sum / std::max(1, n_nodes);
          float trans_mean = trans_sum / std::max(1, n_nodes);

          float eff_rot_mean = eff_rot_sum / std::max(1, n_nodes);
          float eff_trans_mean = eff_trans_sum / std::max(1, n_nodes);

          // ---- reject MOLTO più sensato ----
          const float reject_rot = 5.0f * params_.max_update_rot;
          const float reject_trans = 5.0f * params_.max_update_trans;

          // A single node can legitimately move more than the mean when the
          // deformation is localized. Keep this as a diagnostic, but only
          // reject if the scaled raw update is catastrophically beyond the
          // configured per-node clamp.
          bool has_ratio_spike =
              (rot_max > 10.0f * rot_mean) || (trans_max > 10.0f * trans_mean);
          bool has_bad_spike =
              (params_.update_scale * rot_max > reject_rot) ||
              (params_.update_scale * trans_max > reject_trans);
          bool mean_ok = (params_.max_dx_mean <= 0.0f) ||
                         (eff_rot_mean <= params_.max_dx_mean &&
                          eff_trans_mean <= params_.max_dx_mean);
          std::cout << "has ratio_spike=" << (has_ratio_spike ? "yes" : "no")
                    << " has_bad_spike=" << (has_bad_spike ? "yes" : "no")
                    << " mean_ok=" << (mean_ok ? "yes" : "no") << "\n";

          iter_ok =
              finite_raw_update && std::isfinite(eff_rot_mean) &&
              std::isfinite(eff_trans_mean) && std::isfinite(eff_rot_max) &&
              std::isfinite(eff_trans_max) && (eff_rot_max < reject_rot) &&
              (eff_trans_max < reject_trans) && mean_ok && !has_bad_spike;

          {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(5)
               << "  [dx] rot_mean=" << rot_mean << " rot_max=" << rot_max
               << " trans_mean=" << trans_mean << " trans_max=" << trans_max
               << " eff_rot_mean=" << eff_rot_mean
               << " eff_rot_max=" << eff_rot_max
               << " eff_trans_mean=" << eff_trans_mean
               << " eff_trans_max=" << eff_trans_max
               << " ratio_spike=" << (has_ratio_spike ? "yes" : "no")
               << " bad_spike=" << (has_bad_spike ? "yes" : "no")
               << " mean_ok=" << (mean_ok ? "yes" : "no")
               << " nodes=" << n_nodes << " max_rot=" << params_.max_update_rot
               << " max_trans=" << params_.max_update_trans
               << " reject_rot=" << reject_rot
               << " reject_trans=" << reject_trans;

            std::cout << ss.str() << "\n";
          }

          // if (do_dbg && gn == 0) {
          //   cv::imshow("5_delta_x",
          //              dbg_delta_x(h_dx, warp_field_->num_nodes()));
          //   cv::waitKey(1);
          // }
        }

        // ---- APPLY SOLO SE OK ----
        if (iter_ok)
        {
          warp_field_->save_transforms();

          warp_field_->apply_twist_increment(d_delta_x_, params_.max_update_rot,
                                             params_.max_update_trans,
                                             params_.update_scale);

          warp_update_ok = true;
          accepted_updates++;
        }
        else
        {
          std::cout << "  [warp] skip applying unstable / invalid update\n";
          break;
        }
      }
      // 7. Integra depth. Unwarped fusion after frame 0 is only safe for a
      // static scene/camera; otherwise it smears incompatible live frames into
      // the canonical TSDF.
      if (params_.integrate_warped)
      {
        if (warp_update_ok)
        {
          mark_optimized_voxels();
          volume_->integrate(
              d_depth_, d_depth_normals_, params_.cam, camera_pose_,
              warp_field_->device_nodes(), warp_field_->device_transforms(),
              warp_field_->num_nodes(), d_voxel_knn_.data, d_voxel_knn_w_.data,
              d_voxel_opt_counts_.data, params_.integrate_min_optimized_count);
          last_frame_integrated_ = true;
        }
        else
        {
          std::cout << "  [integrate] skip warped fusion"
                    << " (warp update not stable)\n";
        }
      }
      else if (params_.integrate_unwarped_fallback)
      {
        volume_->integrate(d_depth_, d_depth_normals_, params_.cam,
                           camera_pose_);
        last_frame_integrated_ = true;
        std::cout << "  [integrate] fused depth without warp"
                  << " (fallback enabled)\n";
      }
      else
      {
        std::cout << "  [integrate] skip fusion"
                  << " (warped fusion disabled)\n";
      }

      // 8. Aggiorna grafo con nuovi nodi
      if (params_.node_update_every_n > 0 &&
          frame_count_ % params_.node_update_every_n == 0)
        update_node_graph();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Frame " << std::setw(4) << frame_count_ << "] " << std::fixed
              << std::setprecision(1) << ms
              << " ms | nodes: " << warp_field_->num_nodes()
              << " | integrated: " << (last_frame_integrated_ ? "yes" : "no")
              << "\n";

    frame_count_++;
  }

  // Fill existing O3D mesh in-place with the deformed live mesh.
  void update_o3d_mesh(open3d::geometry::TriangleMesh &mesh) const
  {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    float mean_disp = 0.0f, max_disp = 0.0f;
    extract_warped_surface(verts, norms, tris, &mean_disp, &max_disp);
    std::cout << "warped verts: " << verts.size() << " tris: " << tris.size()
              << " | warp_disp mean=" << mean_disp << " max=" << max_disp
              << std::endl;

    mesh.vertices_.clear();
    mesh.triangles_.clear();
    mesh.vertices_.reserve(verts.size());
    mesh.triangles_.reserve(tris.size());
    for (auto &v : verts)
      mesh.vertices_.push_back({v.x, v.y, v.z});
    for (auto &t : tris)
      mesh.triangles_.push_back({t.x, t.y, t.z});
    mesh.ComputeVertexNormals();
    mesh.PaintUniformColor({1.0, 1.0, 1.0});
  }

  // Salva la mesh warped (live frame) come PLY
  // CPU skinning: per ogni vertice canonico, applica blend delle trasformazioni
  // nodi più vicini
  void save_warped_mesh_ply(const std::string &path) const
  {
    std::vector<float3> warped, norms;
    std::vector<int3> tris;
    extract_warped_surface(warped, norms, tris);

    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << warped.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "element face " << tris.size() << "\n";
    f << "property list uchar int vertex_indices\nend_header\n";
    for (size_t i = 0; i < warped.size(); i++)
      f << warped[i].x << " " << warped[i].y << " " << warped[i].z << " "
        << norms[i].x << " " << norms[i].y << " " << norms[i].z << "\n";
    for (const auto &t : tris)
      f << "3 " << t.x << " " << t.y << " " << t.z << "\n";
    std::cout << "[PLY warped] " << path << " (" << warped.size() << " v)\n";
  }

  // Salva la mesh canonica corrente come PLY
  void save_mesh_ply(const std::string &path) const
  {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    volume_->extract_surface(verts, norms, tris);

    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << verts.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "element face " << tris.size() << "\n";
    f << "property list uchar int vertex_indices\n";
    f << "end_header\n";
    for (size_t i = 0; i < verts.size(); i++)
    {
      f << verts[i].x << " " << verts[i].y << " " << verts[i].z << " "
        << norms[i].x << " " << norms[i].y << " " << norms[i].z << "\n";
    }
    for (const auto &t : tris)
      f << "3 " << t.x << " " << t.y << " " << t.z << "\n";

    std::cout << "[PLY] Salvato: " << path << " (" << verts.size()
              << " vertici)\n";
  }

private:
  struct SiftFeatureFrame;

  Params params_;
  int frame_count_;

  std::unique_ptr<TSDFVolume> volume_;
  std::unique_ptr<WarpField> warp_field_;
  std::unique_ptr<GaussNewtonSolver> solver_;

  DeviceArray<float> d_depth_;
  DeviceArray<float3> d_depth_normals_;
  DeviceArray<float3> d_vertices_live_;
  DeviceArray<float3> d_normals_live_;
  DeviceArray<Correspondence> d_corrs_;
  DeviceArray<float> d_delta_x_;
  DeviceArray<int> d_voxel_knn_;
  DeviceArray<float> d_voxel_knn_w_;
  DeviceArray<int> d_pixel_knn_;
  DeviceArray<float> d_pixel_knn_w_;
  DeviceArray<int> d_num_valid_;
  DeviceArray<int> d_corr_stats_;
  DeviceArray<int> d_hit_voxel_idx_;
  DeviceArray<int> d_voxel_opt_counts_;
  DeviceArray<int> d_voxel_opt_frame_;

  Mat4 camera_pose_;
  bool last_frame_integrated_ = false;
  cv::Mat current_color_gray_;
  std::deque<std::unique_ptr<SiftFeatureFrame>> sift_history_;

  void compute_normals()
  {
    dim3 block(16, 16);
    dim3 grid((params_.cam.width + block.x - 1) / block.x,
              (params_.cam.height + block.y - 1) / block.y);

    compute_depth_normals_kernel<<<grid, block>>>(
        d_depth_.data, d_depth_normals_.data, params_.cam.width,
        params_.cam.height, params_.cam);
    cudaDeviceSynchronize();
  }

  void initialize_node_graph()
  {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    auto t0_extract = std::chrono::high_resolution_clock::now();
    volume_->extract_surface(verts, norms, tris);
    auto t1_extract = std::chrono::high_resolution_clock::now();
    double ms_extract =
        std::chrono::duration<double, std::milli>(t1_extract - t0_extract)
            .count();
    std::cout << "  Surface extracted: " << verts.size() << " vertices in "
              << ms_extract << " ms\n";
    auto t0_add = std::chrono::high_resolution_clock::now();
    int added = warp_field_->add_nodes_from_surface(verts, params_.node_min_dist,
                                                    &norms);
    auto t1_add = std::chrono::high_resolution_clock::now();
    double ms_add =
        std::chrono::duration<double, std::milli>(t1_add - t0_add).count();
    std::cout << "  Nodes initialized: " << added << " in " << ms_add
              << " ms\n";
    auto t0_knn = std::chrono::high_resolution_clock::now();
    warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
    auto t1_knn = std::chrono::high_resolution_clock::now();
    double ms_knn =
        std::chrono::duration<double, std::milli>(t1_knn - t0_knn).count();
    std::cout << "  Voxel k-NN computed in " << ms_knn << " ms\n";

    std::cout << "[Init] Nodi aggiunti: " << added << "\n";
  }

  void update_node_graph()
  {
    // Download canonical voxel hit-map from raycast → convert to canonical
    // world pos Avoids MC; gives correct canonical (not warped) positions for
    // new nodes
    int n_pixels = params_.cam.width * params_.cam.height;
    std::vector<int> h_vidx(n_pixels);
    cudaMemcpy(h_vidx.data(), d_hit_voxel_idx_.data, n_pixels * sizeof(int),
               cudaMemcpyDeviceToHost);

    const auto &tp = params_.tsdf;
    std::vector<float3> canonical_pts;
    std::vector<float3> canonical_norms;
    canonical_pts.reserve(n_pixels / 4);
    canonical_norms.reserve(n_pixels / 4);
    std::vector<float3> h_depth_normals(n_pixels);
    cudaMemcpy(h_depth_normals.data(), d_depth_normals_.data,
               n_pixels * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int px = 0; px < n_pixels; px++)
    {
      int vidx = h_vidx[px];
      if (vidx < 0)
        continue;
      int vx = vidx % tp.dims.x;
      int vy = (vidx / tp.dims.x) % tp.dims.y;
      int vz = vidx / (tp.dims.x * tp.dims.y);
      canonical_pts.push_back(make_float3(tp.origin.x + vx * tp.voxel_size,
                                          tp.origin.y + vy * tp.voxel_size,
                                          tp.origin.z + vz * tp.voxel_size));
      canonical_norms.push_back(
          host_normalize3(camera_pose_.transform_normal(h_depth_normals[px])));
    }

    int added = warp_field_->add_nodes_from_surface(canonical_pts,
                                                    params_.node_min_dist,
                                                    &canonical_norms);
    if (added > 0)
    {
      warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
      std::cout << "[Update] Nuovi nodi: " << added << "\n";
    }
  }

  void apply_depth_bbox_filter()
  {
    dim3 block(16, 16);
    dim3 grid((params_.cam.width + block.x - 1) / block.x,
              (params_.cam.height + block.y - 1) / block.y);
    depth_bbox_filter_kernel<<<grid, block>>>(
        d_depth_.data, params_.cam.width, params_.cam.height, params_.cam,
        params_.bbox.min_pt, params_.bbox.max_pt);
    cudaDeviceSynchronize();
  }

  void apply_bbox_filter(DeviceArray<float3> &verts)
  {
    int n = (int)verts.size();
    bbox_filter_kernel<<<grid1d(n), 256>>>(verts.data, n, params_.bbox.min_pt,
                                           params_.bbox.max_pt);
    cudaDeviceSynchronize();
  }

  void extract_warped_surface(std::vector<float3> &warped,
                              std::vector<float3> &norms,
                              std::vector<int3> &tris,
                              float *mean_disp = nullptr,
                              float *max_disp = nullptr) const
  {
    std::vector<float3> verts;
    volume_->extract_surface(verts, norms, tris);
    warped.resize(verts.size());

    if (mean_disp)
      *mean_disp = 0.0f;
    if (max_disp)
      *max_disp = 0.0f;

    int num_nodes = warp_field_->num_nodes();
    if (num_nodes == 0)
    {
      warped = verts;
      return;
    }

    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();

    double disp_sum = 0.0;
    float disp_max = 0.0f;
    for (size_t vi = 0; vi < verts.size(); vi++)
    {
      float3 p = verts[vi];

      float best_d2[K_NEIGHBORS];
      int best_id[K_NEIGHBORS];
      for (int k = 0; k < K_NEIGHBORS; k++)
      {
        best_d2[k] = 1e30f;
        best_id[k] = -1;
      }
      for (int ni = 0; ni < num_nodes; ni++)
      {
        float3 d = make_float3(p.x - h_nodes[ni].pos.x, p.y - h_nodes[ni].pos.y,
                               p.z - h_nodes[ni].pos.z);
        float d2 = host_dot3(d, d);
        if (d2 < best_d2[K_NEIGHBORS - 1])
        {
          best_d2[K_NEIGHBORS - 1] = d2;
          best_id[K_NEIGHBORS - 1] = ni;
          for (int k = K_NEIGHBORS - 2; k >= 0; k--)
          {
            if (best_d2[k + 1] >= best_d2[k])
              break;
            std::swap(best_d2[k], best_d2[k + 1]);
            std::swap(best_id[k], best_id[k + 1]);
          }
        }
      }

      float w_sum = 0.0f;
      float weights[K_NEIGHBORS] = {};
      for (int k = 0; k < K_NEIGHBORS; k++)
      {
        if (best_id[k] < 0)
          continue;
        float r = std::max(h_nodes[best_id[k]].radius, 1e-6f);
        weights[k] = std::exp(-best_d2[k] / (2.0f * r * r));
        w_sum += weights[k];
      }

      float norm_ws[K_NEIGHBORS] = {};
      if (w_sum > 1e-8f)
        for (int k = 0; k < K_NEIGHBORS; k++)
          norm_ws[k] = weights[k] / w_sum;
      warped[vi] =
          warp_point_dual_quat_host(p, h_nodes, h_transforms, best_id, norm_ws);

      float3 d = make_float3(warped[vi].x - p.x, warped[vi].y - p.y,
                             warped[vi].z - p.z);
      float disp = host_norm3(d);
      disp_sum += disp;
      disp_max = std::max(disp_max, disp);
    }

    if (mean_disp)
      *mean_disp = verts.empty() ? 0.0f : (float)(disp_sum / verts.size());
    if (max_disp)
      *max_disp = disp_max;
  }

  int canonical_voxel_index(float3 p) const
  {
    const auto &tp = params_.tsdf;
    int vx = (int)std::floor((p.x - tp.origin.x) / tp.voxel_size);
    int vy = (int)std::floor((p.y - tp.origin.y) / tp.voxel_size);
    int vz = (int)std::floor((p.z - tp.origin.z) / tp.voxel_size);
    if (vx < 0 || vx >= tp.dims.x || vy < 0 || vy >= tp.dims.y || vz < 0 ||
        vz >= tp.dims.z)
      return -1;
    return vz * tp.dims.x * tp.dims.y + vy * tp.dims.x + vx;
  }

  void skin_point_host(float3 p, const std::vector<DeformNode> &nodes,
                       const std::vector<DualQuat> &transforms, float3 &out,
                       int node_ids[K_NEIGHBORS],
                       float node_ws[K_NEIGHBORS]) const
  {
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
      node_ids[k] = -1;
      node_ws[k] = 0.0f;
    }
    if (nodes.empty())
    {
      out = p;
      return;
    }

    float best_d2[K_NEIGHBORS];
    for (int k = 0; k < K_NEIGHBORS; k++)
      best_d2[k] = 1e30f;
    for (int ni = 0; ni < (int)nodes.size(); ni++)
    {
      float3 d = make_float3(p.x - nodes[ni].pos.x, p.y - nodes[ni].pos.y,
                             p.z - nodes[ni].pos.z);
      float d2 = host_dot3(d, d);
      if (d2 >= best_d2[K_NEIGHBORS - 1])
        continue;
      best_d2[K_NEIGHBORS - 1] = d2;
      node_ids[K_NEIGHBORS - 1] = ni;
      for (int k = K_NEIGHBORS - 2; k >= 0; k--)
      {
        if (best_d2[k + 1] >= best_d2[k])
          break;
        std::swap(best_d2[k], best_d2[k + 1]);
        std::swap(node_ids[k], node_ids[k + 1]);
      }
    }

    float w_sum = 0.0f;
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
      int nid = node_ids[k];
      if (nid < 0)
        continue;
      float r = std::max(nodes[nid].radius, 1e-6f);
      node_ws[k] = std::exp(-best_d2[k] / (2.0f * r * r));
      w_sum += node_ws[k];
    }

    if (w_sum > 1e-8f)
      for (int k = 0; k < K_NEIGHBORS; k++)
        node_ws[k] /= w_sum;
    out = warp_point_dual_quat_host(p, nodes, transforms, node_ids, node_ws);
  }

  void rasterize_deformed_mesh_live_surface()
  {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    volume_->extract_surface(verts, norms, tris);

    int n_pixels = params_.cam.width * params_.cam.height;
    std::vector<float3> out_v(n_pixels, make_float3(0, 0, 0));
    std::vector<float3> out_n(n_pixels, make_float3(0, 0, 0));
    std::vector<int> out_vidx(n_pixels, -1);
    std::vector<float> zbuf(n_pixels, std::numeric_limits<float>::infinity());

    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();
    std::vector<float3> warped(verts.size());
    for (size_t i = 0; i < verts.size(); i++)
    {
      int ids[K_NEIGHBORS];
      float ws[K_NEIGHBORS];
      skin_point_host(verts[i], h_nodes, h_transforms, warped[i], ids, ws);
    }

    Mat4 T_cam_world = inverse_rigid_host(camera_pose_);
    auto edge = [](float2 a, float2 b, float2 p)
    {
      return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    };

    for (const auto &tri : tris)
    {
      int ids[3] = {tri.x, tri.y, tri.z};
      if (ids[0] < 0 || ids[1] < 0 || ids[2] < 0 ||
          ids[0] >= (int)verts.size() || ids[1] >= (int)verts.size() ||
          ids[2] >= (int)verts.size())
        continue;

      float3 wc[3] = {warped[ids[0]], warped[ids[1]], warped[ids[2]]};
      float3 cc[3] = {T_cam_world.transform_point(wc[0]),
                      T_cam_world.transform_point(wc[1]),
                      T_cam_world.transform_point(wc[2])};
      if (cc[0].z <= 0.0f || cc[1].z <= 0.0f || cc[2].z <= 0.0f)
        continue;

      float2 p[3] = {params_.cam.project(cc[0]), params_.cam.project(cc[1]),
                     params_.cam.project(cc[2])};
      float area = edge(p[0], p[1], p[2]);
      if (std::fabs(area) < 1e-6f)
        continue;

      int min_u =
          std::max(0, (int)std::floor(std::min({p[0].x, p[1].x, p[2].x})));
      int max_u = std::min(params_.cam.width - 1,
                           (int)std::ceil(std::max({p[0].x, p[1].x, p[2].x})));
      int min_v =
          std::max(0, (int)std::floor(std::min({p[0].y, p[1].y, p[2].y})));
      int max_v = std::min(params_.cam.height - 1,
                           (int)std::ceil(std::max({p[0].y, p[1].y, p[2].y})));
      if (min_u > max_u || min_v > max_v)
        continue;

      float3 e1 =
          make_float3(wc[1].x - wc[0].x, wc[1].y - wc[0].y, wc[1].z - wc[0].z);
      float3 e2 =
          make_float3(wc[2].x - wc[0].x, wc[2].y - wc[0].y, wc[2].z - wc[0].z);
      float3 face_n = host_normalize3(host_cross3(e1, e2));
      if (host_norm3(face_n) < 1e-6f)
        continue;

      for (int y = min_v; y <= max_v; y++)
      {
        for (int x = min_u; x <= max_u; x++)
        {
          float2 q = make_float2((float)x + 0.5f, (float)y + 0.5f);
          float w0 = edge(p[1], p[2], q) / area;
          float w1 = edge(p[2], p[0], q) / area;
          float w2 = edge(p[0], p[1], q) / area;
          if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f)
            continue;
          float z = w0 * cc[0].z + w1 * cc[1].z + w2 * cc[2].z;
          int px = y * params_.cam.width + x;
          if (z <= 0.0f || z >= zbuf[px])
            continue;
          zbuf[px] = z;
          out_v[px] = make_float3(w0 * wc[0].x + w1 * wc[1].x + w2 * wc[2].x,
                                  w0 * wc[0].y + w1 * wc[1].y + w2 * wc[2].y,
                                  w0 * wc[0].z + w1 * wc[1].z + w2 * wc[2].z);
          out_n[px] = face_n;
          float3 cp = make_float3(w0 * verts[ids[0]].x + w1 * verts[ids[1]].x +
                                      w2 * verts[ids[2]].x,
                                  w0 * verts[ids[0]].y + w1 * verts[ids[1]].y +
                                      w2 * verts[ids[2]].y,
                                  w0 * verts[ids[0]].z + w1 * verts[ids[1]].z +
                                      w2 * verts[ids[2]].z);
          out_vidx[px] = canonical_voxel_index(cp);
        }
      }
    }

    d_vertices_live_.upload(out_v);
    d_normals_live_.upload(out_n);
    d_hit_voxel_idx_.upload(out_vidx);
  }

  void mark_optimized_voxels()
  {
    int n_pixels = params_.cam.width * params_.cam.height;
    mark_optimized_voxels_kernel<<<grid1d(n_pixels), 256>>>(
        d_corrs_.data, d_hit_voxel_idx_.data, n_pixels, params_.tsdf.dims,
        frame_count_, d_voxel_opt_counts_.data, d_voxel_opt_frame_.data);
    cudaDeviceSynchronize();
  }

  void print_depth_stats(const char *label)
  {
    std::vector<float> h_depth;
    d_depth_.download(h_depth);
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (float d : h_depth)
    {
      if (d <= 0.0f)
        continue;
      valid++;
      min_v = std::min(min_v, d);
      max_v = std::max(max_v, d);
      sum += d;
    }
    std::cout << "  [" << label << "] valid=" << valid
              << " min=" << (valid ? min_v : 0.0f)
              << " max=" << (valid ? max_v : 0.0f)
              << " mean=" << (valid ? (sum / valid) : 0.0) << "\n";
  }

  void print_cpu_depth_stats(const cv::Mat &depth, const char *label)
  {
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (int r = 0; r < depth.rows; r++)
    {
      const float *row = depth.ptr<float>(r);
      for (int c = 0; c < depth.cols; c++)
      {
        float d = row[c];
        if (d <= 0.0f)
          continue;
        valid++;
        min_v = std::min(min_v, d);
        max_v = std::max(max_v, d);
        sum += d;
      }
    }
    std::cout << "  [" << label << "] type=" << depth.type()
              << " continuous=" << depth.isContinuous() << " valid=" << valid
              << " min=" << (valid ? min_v : 0.0f)
              << " max=" << (valid ? max_v : 0.0f)
              << " mean=" << (valid ? (sum / valid) : 0.0) << "\n";
  }

  void print_host_depth_stats(const std::vector<float> &depth,
                              const char *label)
  {
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (float d : depth)
    {
      if (d <= 0.0f)
        continue;
      valid++;
      min_v = std::min(min_v, d);
      max_v = std::max(max_v, d);
      sum += d;
    }
    std::cout << "  [" << label << "] size=" << depth.size()
              << " valid=" << valid << " min=" << (valid ? min_v : 0.0f)
              << " max=" << (valid ? max_v : 0.0f)
              << " mean=" << (valid ? (sum / valid) : 0.0) << "\n";
  }

  void print_tsdf_stats(const char *label)
  {
    std::vector<TSDFVoxel> h_voxels;
    h_voxels.resize(volume_->total_voxels());
    cudaMemcpy(h_voxels.data(), volume_->device_data(),
               h_voxels.size() * sizeof(TSDFVoxel), cudaMemcpyDeviceToHost);

    int observed = 0, neg = 0, near_zero = 0;
    float min_tsdf = 1.0f;
    float max_tsdf = -1.0f;
    for (const auto &voxel : h_voxels)
    {
      if (voxel.weight <= 0.0f)
        continue;
      observed++;
      min_tsdf = std::min(min_tsdf, voxel.tsdf);
      max_tsdf = std::max(max_tsdf, voxel.tsdf);
      if (voxel.tsdf < 0.0f)
        neg++;
      if (fabsf(voxel.tsdf) < 0.2f)
        near_zero++;
    }
    std::cout << "  [" << label << "] observed=" << observed << " neg=" << neg
              << " near_zero=" << near_zero
              << " min=" << (observed ? min_tsdf : 0.0f)
              << " max=" << (observed ? max_tsdf : 0.0f) << "\n";
  }

  struct SparseFeature
  {
    float3 canonical_src;
    int node_ids[K_NEIGHBORS];
    float node_ws[K_NEIGHBORS];
  };

  struct SiftFeatureFrame
  {
    SiftData sift{};
    std::vector<SparseFeature> features;
    bool initialized = false;

    ~SiftFeatureFrame()
    {
      if (initialized)
        FreeSiftData(sift);
    }
  };

  cv::Mat sift_input_image(const std::vector<float> &depth) const
  {
    cv::Mat gray = current_color_gray_;
    if (gray.empty())
    {
      DepthSequence::depth_to_gray(
          cv::Mat(params_.cam.height, params_.cam.width, CV_32FC1,
                  const_cast<float *>(depth.data())),
          gray);
    }
    if (gray.channels() != 1)
      cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    if (gray.cols != params_.cam.width || gray.rows != params_.cam.height)
    {
      cv::resize(gray, gray, {params_.cam.width, params_.cam.height});
    }
    cv::Mat fimg;
    gray.convertTo(fimg, CV_32FC1);
    return fimg;
  }

  std::unique_ptr<SiftFeatureFrame> extract_sift_feature_frame(
      const cv::Mat &gray32, const std::vector<float3> &h_live,
      const std::vector<float> &h_depth, const std::vector<int> &h_pixel_knn,
      const std::vector<float> &h_pixel_knn_w,
      const std::vector<int> &h_hit_voxel_idx) const
  {
    auto frame = std::make_unique<SiftFeatureFrame>();
    CudaImage cuda_img;
    cuda_img.Allocate(params_.cam.width, params_.cam.height,
                      iAlignUp(params_.cam.width, 128), false, nullptr,
                      reinterpret_cast<float *>(gray32.data));
    cuda_img.Download();

    InitSiftData(frame->sift, params_.sift_max_features, true, true);
    frame->initialized = true;
    float *tmp = AllocSiftTempMemory(params_.cam.width, params_.cam.height,
                                     params_.sift_octaves, false);
    ExtractSift(frame->sift, cuda_img, params_.sift_octaves, 1.0f,
                params_.sift_threshold, 0.0f, false, tmp);
    FreeSiftTempMemory(tmp);

    frame->features.resize(frame->sift.numPts);
    for (int i = 0; i < frame->sift.numPts; i++)
    {
      const SiftPoint &sp = frame->sift.h_data[i];
      int u = static_cast<int>(std::lround(sp.xpos));
      int v = static_cast<int>(std::lround(sp.ypos));
      SparseFeature feat{};
      feat.canonical_src = make_float3(0, 0, 0);
      for (int k = 0; k < K_NEIGHBORS; k++)
      {
        feat.node_ids[k] = -1;
        feat.node_ws[k] = 0.0f;
      }
      if (u >= 0 && u < params_.cam.width && v >= 0 && v < params_.cam.height)
      {
        int px = v * params_.cam.width + u;
        float d = h_depth[px];
        float3 model_p = h_live[px];
        int vidx = h_hit_voxel_idx[px];
        if (d > 0.01f && model_p.z > 0.01f && vidx >= 0)
        {
          const auto &tp = params_.tsdf;
          int vx = vidx % tp.dims.x;
          int vy = (vidx / tp.dims.x) % tp.dims.y;
          int vz = vidx / (tp.dims.x * tp.dims.y);
          feat.canonical_src =
              make_float3(tp.origin.x + (vx + 0.5f) * tp.voxel_size,
                          tp.origin.y + (vy + 0.5f) * tp.voxel_size,
                          tp.origin.z + (vz + 0.5f) * tp.voxel_size);
          for (int k = 0; k < K_NEIGHBORS; k++)
          {
            feat.node_ids[k] = h_pixel_knn[px * K_NEIGHBORS + k];
            feat.node_ws[k] = h_pixel_knn_w[px * K_NEIGHBORS + k];
          }
        }
      }
      frame->features[i] = feat;
    }
    return frame;
  }

  int add_sparse_axis_constraint(std::vector<Correspondence> &h_corrs, int idx,
                                 const SparseFeature &src_feat,
                                 float3 warped_src, float3 dst,
                                 int axis) const
  {
    if (idx >= (int)h_corrs.size())
      return idx;
    Correspondence corr{};
    corr.src = warped_src;
    corr.dst = camera_pose_.transform_point(dst);
    float3 n_cam = make_float3(axis == 0 ? 1.0f : 0.0f, axis == 1 ? 1.0f : 0.0f,
                               axis == 2 ? 1.0f : 0.0f);
    corr.normal = host_normalize3(camera_pose_.transform_normal(n_cam));
    corr.weight = params_.sift_weight;
    corr.valid = true;
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
      corr.node_ids[k] = src_feat.node_ids[k];
      corr.node_ws[k] = src_feat.node_ws[k];
    }
    h_corrs[idx] = corr;
    return idx + 1;
  }

  static float3 warp_point_host(float3 p, const SparseFeature &feat,
                                const std::vector<DeformNode> &nodes,
                                const std::vector<DualQuat> &transforms)
  {
    float norm_ws[K_NEIGHBORS] = {};
    float w_sum = 0.f;
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
      norm_ws[k] = feat.node_ws[k];
      if (feat.node_ids[k] >= 0)
        w_sum += norm_ws[k];
    }
    if (w_sum > 1e-8f)
      for (int k = 0; k < K_NEIGHBORS; k++)
        norm_ws[k] /= w_sum;
    return warp_point_dual_quat_host(p, nodes, transforms, feat.node_ids,
                                     norm_ws);
  }

  int augment_correspondences_with_sift(int current_count, bool debug,
                                        bool update_history = true)
  {
    std::vector<float3> h_live, h_depth_normals;
    std::vector<float> h_depth, h_pixel_knn_w;
    std::vector<int> h_pixel_knn, h_hit_voxel_idx;
    std::vector<Correspondence> h_corrs;
    d_vertices_live_.download(h_live);
    d_depth_.download(h_depth);
    d_depth_normals_.download(h_depth_normals);
    d_pixel_knn_.download(h_pixel_knn);
    d_pixel_knn_w_.download(h_pixel_knn_w);
    d_hit_voxel_idx_.download(h_hit_voxel_idx);
    d_corrs_.download(h_corrs);
    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();

    int n_pixels = params_.cam.width * params_.cam.height;
    for (size_t i = n_pixels; i < h_corrs.size(); i++)
    {
      h_corrs[i].valid = false;
      h_corrs[i].weight = 0.0f;
    }

    cv::Mat gray32 = sift_input_image(h_depth);
    auto current = extract_sift_feature_frame(
        gray32, h_live, h_depth, h_pixel_knn, h_pixel_knn_w, h_hit_voxel_idx);
    int added = 0;
    int append_idx = n_pixels;
    for (auto &hist : sift_history_)
    {
      MatchSiftData(current->sift, hist->sift);
      int frame_added = 0;
      for (int i = 0; i < current->sift.numPts &&
                      frame_added < params_.sift_max_matches_per_frame;
           i++)
      {
        const SiftPoint &sp = current->sift.h_data[i];
        if (sp.match < 0 || sp.match >= hist->sift.numPts)
          continue;
        if (sp.match_error > params_.sift_max_match_error ||
            sp.ambiguity > params_.sift_max_ambiguity)
          continue;

        int u = static_cast<int>(std::lround(sp.xpos));
        int v = static_cast<int>(std::lround(sp.ypos));
        if (u < 0 || u >= params_.cam.width || v < 0 || v >= params_.cam.height)
          continue;

        int dst_px = v * params_.cam.width + u;
        float d = h_depth[dst_px];
        if (d <= 0.01f)
          continue;
        float3 dst = params_.cam.unproject(u, v, d);

        const SparseFeature &src_feat = hist->features[sp.match];
        if (src_feat.canonical_src.z <= 0.01f || src_feat.node_ids[0] < 0)
          continue;
        float3 warped_src = warp_point_host(src_feat.canonical_src, src_feat,
                                            h_nodes, h_transforms);
        float3 dst_world = camera_pose_.transform_point(dst);

        float3 diff =
            make_float3(warped_src.x - dst_world.x, warped_src.y - dst_world.y,
                        warped_src.z - dst_world.z);
        if (host_norm3(diff) > params_.sift_max_3d_dist)
          continue;

        append_idx = add_sparse_axis_constraint(h_corrs, append_idx, src_feat,
                                                warped_src, dst, 0);
        append_idx = add_sparse_axis_constraint(h_corrs, append_idx, src_feat,
                                                warped_src, dst, 1);
        append_idx = add_sparse_axis_constraint(h_corrs, append_idx, src_feat,
                                                warped_src, dst, 2);
        added += 3;
        frame_added++;
      }
    }

    d_corrs_.upload(h_corrs);
    if (debug || added > 0)
    {
      std::cout << "  [sift] current=" << current->sift.numPts
                << " history=" << sift_history_.size()
                << " sparse_rows=" << added << "\n";
    }

    if (update_history)
    {
      sift_history_.push_back(std::move(current));
      while ((int)sift_history_.size() > params_.sift_max_history)
      {
        sift_history_.pop_front();
      }
    }
    return current_count + added;
  }

  double correspondence_rmse() const
  {
    std::vector<Correspondence> h_corrs;
    d_corrs_.download(h_corrs);
    double sum = 0.0;
    double wsum = 0.0;
    for (const auto &c : h_corrs)
    {
      if (!c.valid || c.weight <= 0.0f)
        continue;
      float3 diff =
          make_float3(c.src.x - c.dst.x, c.src.y - c.dst.y, c.src.z - c.dst.z);
      double r =
          c.normal.x * diff.x + c.normal.y * diff.y + c.normal.z * diff.z;
      sum += (double)c.weight * r * r;
      wsum += c.weight;
    }
    return (wsum > 0.0) ? std::sqrt(sum / wsum)
                        : std::numeric_limits<double>::infinity();
  }

  static bool solve_6x6(double A[6][6], double b[6], double x[6])
  {
    double M[6][7];
    for (int r = 0; r < 6; r++)
    {
      for (int c = 0; c < 6; c++)
        M[r][c] = A[r][c];
      M[r][6] = b[r];
    }
    for (int i = 0; i < 6; i++)
    {
      int pivot = i;
      for (int r = i + 1; r < 6; r++)
        if (std::fabs(M[r][i]) > std::fabs(M[pivot][i]))
          pivot = r;
      if (std::fabs(M[pivot][i]) < 1e-10)
        return false;
      if (pivot != i)
        for (int c = i; c < 7; c++)
          std::swap(M[i][c], M[pivot][c]);
      double inv = 1.0 / M[i][i];
      for (int c = i; c < 7; c++)
        M[i][c] *= inv;
      for (int r = 0; r < 6; r++)
      {
        if (r == i)
          continue;
        double f = M[r][i];
        for (int c = i; c < 7; c++)
          M[r][c] -= f * M[i][c];
      }
    }
    for (int i = 0; i < 6; i++)
      x[i] = M[i][6];
    return true;
  }

  bool estimate_rigid_tracking_update()
  {
    std::vector<Correspondence> h_corrs;
    d_corrs_.download(h_corrs);
    Mat4 T_cam_world = inverse_rigid_host(camera_pose_);

    double A[6][6] = {};
    double b[6] = {};
    int used = 0;
    for (const auto &c : h_corrs)
    {
      if (!c.valid || c.weight <= 0.0f)
        continue;
      float3 p = T_cam_world.transform_point(c.src);
      float3 dst = T_cam_world.transform_point(c.dst);
      float3 n = host_normalize3(T_cam_world.transform_normal(c.normal));
      float3 diff = make_float3(p.x - dst.x, p.y - dst.y, p.z - dst.z);
      double r = host_dot3(n, diff);
      float3 rot = host_cross3(p, n);
      double J[6] = {rot.x, rot.y, rot.z, n.x, n.y, n.z};
      double w = std::max(0.0f, c.weight);
      for (int i = 0; i < 6; i++)
      {
        b[i] -= w * J[i] * r;
        for (int j = 0; j < 6; j++)
          A[i][j] += w * J[i] * J[j];
      }
      used++;
    }
    if (used < 64)
      return false;
    for (int i = 0; i < 6; i++)
      A[i][i] += 1e-6;

    double x[6] = {};
    if (!solve_6x6(A, b, x))
      return false;
    float dx[6];
    double rot_n = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    double trans_n = std::sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]);
    double rot_scale =
        rot_n > params_.max_update_rot ? params_.max_update_rot / rot_n : 1.0;
    double trans_scale = trans_n > params_.max_update_trans
                             ? params_.max_update_trans / trans_n
                             : 1.0;
    double scale = std::min({rot_scale, trans_scale, 1.0}) * 0.5;
    for (int i = 0; i < 6; i++)
      dx[i] = (float)(x[i] * scale);

    Mat4 dT = exp_se3_host(dx);
    Mat4 new_T_cam_world = dT * T_cam_world;
    camera_pose_ = inverse_rigid_host(new_T_cam_world);
    std::cout << "  [rigid] corr=" << used << " rot=" << std::fixed
              << std::setprecision(5) << rot_n * scale
              << " trans=" << trans_n * scale << "\n";
    return true;
  }

  int find_correspondences(bool debug, bool use_sift_aug)
  {
    int n_pixels = params_.cam.width * params_.cam.height;

    // Ensure pixel k-NN arrays sized correctly
    if (d_pixel_knn_.size() != (size_t)(n_pixels * K_NEIGHBORS))
    {
      d_pixel_knn_.allocate(n_pixels * K_NEIGHBORS);
      d_pixel_knn_w_.allocate(n_pixels * K_NEIGHBORS);
    }
    d_num_valid_.zero();
    d_corr_stats_.zero();
    d_corrs_.zero();

    dim3 block(16, 16);
    dim3 grid((params_.cam.width + block.x - 1) / block.x,
              (params_.cam.height + block.y - 1) / block.y);

    // Step 1: map canonical voxel idx → pixel k-NN (exact, no live≈canonical
    // approx)
    compute_pixel_knn_kernel<<<grid, block>>>(
        d_hit_voxel_idx_.data, params_.cam.width, params_.cam.height,
        d_voxel_knn_.data, d_voxel_knn_w_.data, d_pixel_knn_.data,
        d_pixel_knn_w_.data);

    // Step 2: projective ICP — fill d_corrs_
    Mat4 T_cam_world = inverse_rigid_host(camera_pose_);
    find_correspondences_kernel<<<grid, block>>>(
        d_vertices_live_.data, d_normals_live_.data, d_depth_.data,
        d_depth_normals_.data, d_corrs_.data, d_num_valid_.data,
        params_.cam.width, params_.cam.height, params_.cam, T_cam_world,
        camera_pose_, d_pixel_knn_.data, d_pixel_knn_w_.data,
        params_.dist_threshold, params_.angle_threshold, params_.view_threshold,
        params_.search_radius_px, debug ? d_corr_stats_.data : nullptr);
    cudaDeviceSynchronize();

    int h_valid = 0, h_before = 0;
    cudaMemcpy(&h_valid, d_num_valid_.data, sizeof(int),
               cudaMemcpyDeviceToHost);
    h_before = h_valid;
    if (params_.use_sift && use_sift_aug)
    {
      h_valid = augment_correspondences_with_sift(h_valid, debug);
    }
    if (debug)
    {
      std::vector<int> h_stats;
      d_corr_stats_.download(h_stats);
      std::cout << "  [corr debug] invalid_live=" << h_stats[0]
                << " out_proj=" << h_stats[1] << " no_depth=" << h_stats[2]
                << " far=" << h_stats[3] << " angle=" << h_stats[4]
                << " valid=" << h_stats[5] << " sift_aug=" << (h_valid - h_before) << "\n";
    }
    return h_valid;
  }
};

struct AppConfig
{
  DepthSequence::Config seq;
  DynamicFusionPipeline::Params df;
  bool use_vis = false;
};

AppConfig load_config(const std::string &path)
{
  YAML::Node cfg = YAML::LoadFile(path);
  AppConfig out;

  // ───── sequence ─────
  auto s = cfg["sequence"];
  out.seq.path = s["path"].as<std::string>();

  std::string fmt = s["format"].as<std::string>();
  if (fmt == "tum")
    out.seq.format = DepthSequence::Format::TUM;
  else if (fmt == "icl")
    out.seq.format = DepthSequence::Format::ICL;
  else
    out.seq.format = DepthSequence::Format::RAW_PNG;

  out.seq.depth_scale = s["depth_scale"].as<float>();
  out.seq.start_frame = s["start_frame"].as<int>();
  out.seq.max_frames = s["max_frames"].as<int>();

  // ───── camera ─────
  auto c = cfg["camera"];
  out.seq.cam.fx = c["fx"].as<float>();
  out.seq.cam.fy = c["fy"].as<float>();
  out.seq.cam.cx = c["cx"].as<float>();
  out.seq.cam.cy = c["cy"].as<float>();
  out.seq.cam.width = c["width"].as<int>();
  out.seq.cam.height = c["height"].as<int>();

  out.df.cam = out.seq.cam;

  // ───── tsdf ─────
  auto t = cfg["tsdf"];
  auto dims = read_vec_i(t["dims"]);
  out.df.tsdf.dims = {dims[0], dims[1], dims[2]};
  out.df.tsdf.voxel_size = t["voxel_size"].as<float>();
  out.df.tsdf.truncation = t["truncation"].as<float>();
  auto origin = read_vec_f(t["origin"]);
  out.df.tsdf.origin = make_float3(origin[0], origin[1], origin[2]);

  // ───── solver ─────
  auto sol = cfg["solver"];
  out.df.solver.gn_iterations = sol["gn_iterations"].as<int>();
  out.df.solver.pcg_iterations = sol["pcg_iterations"].as<int>();
  out.df.solver.lambda_smooth = sol["lambda_smooth"].as<float>();
  if (sol["lambda_damping"])
    out.df.solver.lambda_damping = sol["lambda_damping"].as<float>();

  // ───── warp ─────
  auto w = cfg["warp"];
  out.df.node_radius = w["node_radius"].as<float>();
  out.df.node_min_dist = w["node_min_dist"].as<float>();
  out.df.max_nodes = w["max_nodes"].as<int>();
  if (w["node_update_every_n"])
    out.df.node_update_every_n = w["node_update_every_n"].as<int>();
  if (w["max_dx_mean"])
    out.df.max_dx_mean = w["max_dx_mean"].as<float>();
  if (w["max_update_rot"])
    out.df.max_update_rot = w["max_update_rot"].as<float>();
  if (w["max_update_trans"])
    out.df.max_update_trans = w["max_update_trans"].as<float>();
  if (w["update_scale"])
    out.df.update_scale = w["update_scale"].as<float>();
  if (w["integrate_warped"])
    out.df.integrate_warped = w["integrate_warped"].as<bool>();
  if (w["integrate_unwarped_fallback"])
    out.df.integrate_unwarped_fallback =
        w["integrate_unwarped_fallback"].as<bool>();
  if (w["integrate_min_optimized_count"])
    out.df.integrate_min_optimized_count =
        w["integrate_min_optimized_count"].as<int>();

  // ───── icp ─────
  auto icp = cfg["icp"];
  out.df.dist_threshold = icp["dist_threshold"].as<float>();
  out.df.angle_threshold = icp["angle_threshold"].as<float>();
  if (icp["view_threshold"])
    out.df.view_threshold = icp["view_threshold"].as<float>();
  if (icp["search_radius_px"])
    out.df.search_radius_px = icp["search_radius_px"].as<int>();
  if (icp["min_valid_corrs"])
    out.df.min_valid_corrs = icp["min_valid_corrs"].as<int>();
  if (icp["rigid_tracking"])
    out.df.rigid_tracking = icp["rigid_tracking"].as<bool>();

  if (cfg["sift"])
  {
    auto sift = cfg["sift"];
    if (sift["enabled"])
      out.df.use_sift = sift["enabled"].as<bool>();
    if (sift["max_features"])
      out.df.sift_max_features = sift["max_features"].as<int>();
    if (sift["max_history"])
      out.df.sift_max_history = sift["max_history"].as<int>();
    if (sift["max_matches_per_frame"])
      out.df.sift_max_matches_per_frame =
          sift["max_matches_per_frame"].as<int>();
    if (sift["octaves"])
      out.df.sift_octaves = sift["octaves"].as<int>();
    if (sift["threshold"])
      out.df.sift_threshold = sift["threshold"].as<float>();
    if (sift["max_match_error"])
      out.df.sift_max_match_error = sift["max_match_error"].as<float>();
    if (sift["max_ambiguity"])
      out.df.sift_max_ambiguity = sift["max_ambiguity"].as<float>();
    if (sift["max_3d_dist"])
      out.df.sift_max_3d_dist = sift["max_3d_dist"].as<float>();
    if (sift["weight"])
      out.df.sift_weight = sift["weight"].as<float>();
  }

  // ───── bbox ─────
  auto b = cfg["bbox"];
  out.df.bbox.enabled = b["enabled"].as<bool>();
  auto bmin = read_vec_f(b["min_pt"]);
  auto bmax = read_vec_f(b["max_pt"]);
  out.df.bbox.min_pt = make_float3(bmin[0], bmin[1], bmin[2]);
  out.df.bbox.max_pt = make_float3(bmax[0], bmax[1], bmax[2]);

  // ───── visualizer ─────
  out.use_vis = cfg["visualizer"]["enabled"].as<bool>();

  return out;
}
// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
