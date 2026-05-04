#pragma once
#include <cuda_runtime.h>
#include <open3d/Open3D.h>

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

static float host_norm3(float3 v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ─────────────────────────────────────────────
//  Depth sequence loader
//  Supporta: TUM RGB-D, ICL-NUIM, cartella raw
// ─────────────────────────────────────────────

struct DepthFrame {
  cv::Mat depth_m;  // float32, metri
  cv::Mat color_gray;
  double timestamp;
  int index;
};

class DepthSequence {
 public:
  enum class Format { TUM, ICL, RAW_PNG, RAW_EXR };

  struct Config {
    std::string path;
    Format format = Format::TUM;
    float depth_scale = 0.001f;  // uint16 raw → metres
    int start_frame = 0;
    int max_frames = -1;  // -1 = tutti
    CameraIntrinsics cam;
  };

  explicit DepthSequence(const Config &cfg) : cfg_(cfg) {
    switch (cfg.format) {
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

  DepthFrame next() {
    auto &f = frames_[current_idx_++];
    return f;
  }

  int total() const { return (int)frames_.size(); }

  const CameraIntrinsics &camera() const { return cfg_.cam; }

 private:
  Config cfg_;
  std::vector<DepthFrame> frames_;
  int current_idx_ = 0;

  void load_tum() {
    // Formato TUM: depth/*.png, associations.txt
    // depth images uint16, scaled according to config
    std::string assoc_path = cfg_.path + "/associations.txt";
    std::ifstream assoc(assoc_path);
    if (!assoc.is_open()) {
      // Fallback: scansiona cartella depth/ usando la scala del config.
      load_raw_dir(cfg_.path + "/depth", ".png");
      return;
    }

    std::string line;
    int count = 0;
    while (std::getline(assoc, line)) {
      if (line.empty() || line[0] == '#') continue;
      if (count < cfg_.start_frame) {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames) break;

      std::istringstream ss(line);
      double ts_rgb, ts_depth;
      std::string rgb_file, depth_file;
      ss >> ts_rgb >> rgb_file >> ts_depth >> depth_file;

      std::string full_path = cfg_.path + "/" + depth_file;
      cv::Mat raw = cv::imread(full_path, cv::IMREAD_ANYDEPTH);
      if (raw.empty()) continue;

      cv::Mat depth_m;
      raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);

      cv::Mat color =
          cv::imread(cfg_.path + "/" + rgb_file, cv::IMREAD_GRAYSCALE);
      if (!color.empty() &&
          (color.cols != depth_m.cols || color.rows != depth_m.rows)) {
        cv::resize(color, color, depth_m.size(), 0.0, 0.0, cv::INTER_LINEAR);
      }

      frames_.push_back({depth_m, color, ts_depth, count});
      count++;
    }
  }

  void load_icl() {
    // Formato ICL-NUIM: depth/*.png (float in mm * 1000)
    load_raw_dir(cfg_.path + "/depth", ".png");
  }

  void load_raw() {
    std::string ext = (cfg_.format == Format::RAW_EXR) ? ".exr" : ".png";
    load_raw_dir(cfg_.path, ext);
  }

  void load_raw_dir(const std::string &dir, const std::string &ext,
                    float depth_scale_override = -1.0f) {
    const float depth_scale =
        (depth_scale_override > 0.0f) ? depth_scale_override : cfg_.depth_scale;

    if (!fs::exists(dir)) {
      std::cerr << "[DepthSequence] Directory not found: " << dir << "\n";
      return;
    }
    // std::cout << "[DepthSequence] file: " << p << std::endl;
    std::vector<std::string> paths;
    for (const auto &entry : fs::directory_iterator(dir)) {
      if (entry.path().extension() == ext)
        paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());

    int count = 0;
    for (const auto &p : paths) {
      if (count < cfg_.start_frame) {
        count++;
        continue;
      }
      if (cfg_.max_frames > 0 && (int)frames_.size() >= cfg_.max_frames) break;

      cv::Mat raw = cv::imread(p, cv::IMREAD_ANYDEPTH);
      if (raw.empty()) {
        // Prova EXR
        raw = cv::imread(p, cv::IMREAD_UNCHANGED);
      }
      if (raw.empty()) {
        count++;
        continue;
      }

      cv::Mat depth_m;
      if (raw.type() == CV_32F) {
        depth_m = raw;  // già float in metri
      } else if (raw.type() == CV_16U) {
        raw.convertTo(depth_m, CV_32F, depth_scale);
      } else {
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
  static void depth_to_gray(const cv::Mat &depth_m, cv::Mat &gray) {
    double min_v = 0.0, max_v = 0.0;
    cv::Mat valid = depth_m > 0.01f;
    cv::minMaxLoc(depth_m, &min_v, &max_v, nullptr, nullptr, valid);
    if (max_v <= min_v) {
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
                         float max_m = 4.0f) {
  cv::Mat grey(H, W, CV_8U);
  for (int i = 0; i < H * W; i++) {
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

static cv::Mat dbg_normals(const std::vector<float3> &n, int W, int H) {
  cv::Mat img(H, W, CV_8UC3);
  for (int i = 0; i < H * W; i++) {
    float len = sqrtf(n[i].x * n[i].x + n[i].y * n[i].y + n[i].z * n[i].z);
    if (len < 0.5f) {
      img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] = 0;
    } else {
      img.data[i * 3 + 0] = (uint8_t)((n[i].z * 0.5f + 0.5f) * 255);  // B=z
      img.data[i * 3 + 1] = (uint8_t)((n[i].y * 0.5f + 0.5f) * 255);  // G=y
      img.data[i * 3 + 2] = (uint8_t)((n[i].x * 0.5f + 0.5f) * 255);  // R=x
    }
  }
  return img;
}

static cv::Mat dbg_verts(const std::vector<float3> &verts,
                         const CameraIntrinsics &cam) {
  cv::Mat depth(cam.height, cam.width, CV_32F, cv::Scalar(0.f));
  for (const auto &v : verts) {
    if (v.z <= 0.01f) continue;
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
      if (depth.at<float>(r, c) <= 0.01f) color.at<cv::Vec3b>(r, c) = {0, 0, 0};
  cv::putText(color, "live surface (raycasted)", {8, 18},
              cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
  return color;
}

static cv::Mat dbg_corrs(const std::vector<Correspondence> &corrs,
                         const CameraIntrinsics &cam, int n_valid) {
  cv::Mat img(cam.height, cam.width, CV_8UC3, cv::Scalar(15, 15, 15));
  int drawn = 0;
  for (const auto &c : corrs) {
    if (!c.valid) continue;
    if (c.dst.z > 0.01f) {
      int u = (int)(cam.fx * c.dst.x / c.dst.z + cam.cx);
      int v = (int)(cam.fy * c.dst.y / c.dst.z + cam.cy);
      if (u >= 0 && u < cam.width && v >= 0 && v < cam.height) {
        img.at<cv::Vec3b>(v, u)[2] = 220;
        drawn++;
      }
    }
    if (c.src.z > 0.01f) {
      int u = (int)(cam.fx * c.src.x / c.src.z + cam.cx);
      int v = (int)(cam.fy * c.src.y / c.src.z + cam.cy);
      if (u >= 0 && u < cam.width && v >= 0 && v < cam.height) {
        img.at<cv::Vec3b>(v, u)[1] = 220;
      }
    }
    // draw line
    if (c.src.z > 0.01f && c.dst.z > 0.01f && drawn % 100 == 0) {
      int u1 = (int)(cam.fx * c.src.x / c.src.z + cam.cx);
      int v1 = (int)(cam.fy * c.src.y / c.src.z + cam.cy);
      int u2 = (int)(cam.fx * c.dst.x / c.dst.z + cam.cx);
      int v2 = (int)(cam.fy * c.dst.y / c.dst.z + cam.cy);
      if (u1 >= 0 && u1 < cam.width && v1 >= 0 && v1 < cam.height && u2 >= 0 &&
          u2 < cam.width && v2 >= 0 && v2 < cam.height) {
        cv::line(img, {u1, v1}, {u2, v2}, {220, 220, 220}, 1);
      }
    }
  }
  cv::Mat grown;
  cv::dilate(img, grown, cv::Mat(), cv::Point(-1, -1), 1);
  cv::putText(grown,
              "G=live  R=depth  valid=" + std::to_string(n_valid) +
                  " shown=" + std::to_string(drawn),
              {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
  return grown;
}

static cv::Mat dbg_delta_x(const std::vector<float> &dx, int n_nodes) {
  const int bar_w = 4, bar_h = 200, pad = 2;
  int img_w = std::max(300, (bar_w + pad) * n_nodes + pad);
  cv::Mat img(bar_h + 20, img_w, CV_8UC3, cv::Scalar(20, 20, 20));
  float max_norm = 0.f;
  std::vector<float> norms(n_nodes, 0.f);
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < 6; j++) norms[i] += dx[i * 6 + j] * dx[i * 6 + j];
    norms[i] = sqrtf(norms[i]);
    max_norm = std::max(max_norm, norms[i]);
  }
  if (max_norm < 1e-8f) max_norm = 1.f;
  int max_bars = (img_w - pad) / (bar_w + pad);
  for (int i = 0; i < n_nodes && i < max_bars; i++) {
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

class DynamicFusionPipeline {
 public:
  struct Params {
    TSDFVolume::Params tsdf;
    GaussNewtonSolver::Params solver;
    CameraIntrinsics cam;
    float node_radius = 0.04f;
    float node_min_dist = 0.025f;
    float dist_threshold = 0.05f;  // ICP distanza max
    float angle_threshold = 0.7f;  // cos(angolo) min normali
    float view_threshold = 0.2f;   // cos minimo contro silhouette
    int search_radius_px = 3;      // ricerca locale attorno alla proiezione
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
      : params_(p), frame_count_(0) {
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

    camera_pose_ = Mat4::identity();
    if (params_.use_sift) InitCuda(0);
  }

  bool last_frame_integrated() const { return last_frame_integrated_; }

  void process_frame(const cv::Mat &depth_m,
                     const cv::Mat &color_gray = cv::Mat()) {
    auto t0 = std::chrono::high_resolution_clock::now();
    last_frame_integrated_ = false;
    current_color_gray_ = color_gray;

    const bool do_dbg =
        params_.debug_vis && (frame_count_ % params_.debug_every_n == 0);

    if (frame_count_ == 0) print_cpu_depth_stats(depth_m, "cpu depth input");

    // 1. Upload depth
    {
      std::vector<float> h_depth((float *)depth_m.datastart,
                                 (float *)depth_m.dataend);
      if (frame_count_ == 0)
        print_host_depth_stats(h_depth, "host upload vector");
      d_depth_.upload(h_depth);
    }

    if (do_dbg) {
      std::vector<float> h_d;
      d_depth_.download(h_d);
      cv::Mat vis = dbg_depth(h_d, params_.cam.width, params_.cam.height);
      cv::putText(vis, "frame " + std::to_string(frame_count_), {8, 18},
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
      cv::imshow("1_depth_raw", vis);
      cv::waitKey(1);
    }

    if (frame_count_ == 0) print_depth_stats("depth before bbox");

    // 1b. Bounding box filter on depth (before integration)
    if (params_.bbox.enabled) {
      apply_depth_bbox_filter();
    }

    if (frame_count_ == 0) print_depth_stats("depth after bbox");

    // 2. Calcola normali dal depth
    compute_normals();

    if (do_dbg) {
      std::vector<float3> h_n;
      d_depth_normals_.download(h_n);
      cv::imshow("2_depth_normals",
                 dbg_normals(h_n, params_.cam.width, params_.cam.height));
      cv::waitKey(1);
    }

    if (frame_count_ == 0) {
      // Primo frame: solo integra
      volume_->integrate(d_depth_, d_depth_normals_, params_.cam, camera_pose_);
      last_frame_integrated_ = true;
      print_tsdf_stats("tsdf after first integrate");
      initialize_node_graph();
    } else {
      bool warp_update_ok = false;

      // Warp field: salva trasformazioni precedenti (warm start)
      warp_field_->save_transforms();

      int num_corrs = 0;
      int accepted_updates = 0;
      double last_rmse = std::numeric_limits<double>::infinity();
      for (int gn = 0; gn < std::max(1, params_.solver.gn_iterations); gn++) {
        bool iter_ok = false;

        // 3. Raycasting modello canonico → live surface
        volume_->raycast(d_vertices_live_, d_normals_live_, params_.cam,
                         camera_pose_, warp_field_->device_nodes(),
                         warp_field_->device_transforms(),
                         warp_field_->num_nodes(), d_voxel_knn_.data,
                         d_voxel_knn_w_.data, d_hit_voxel_idx_.data);

        if (do_dbg && gn == 0) {
          std::vector<float3> h_v;
          d_vertices_live_.download(h_v);
          cv::imshow("3_live_surface", dbg_verts(h_v, params_.cam));
          cv::waitKey(1);
        }

        // 3b. Bounding box filter on live vertices
        if (params_.bbox.enabled) apply_bbox_filter(d_vertices_live_);

        // 4. Trova corrispondenze ICP. SIFT history is consumed once per frame;
        // later GN iterations refresh only dense projective correspondences.
        num_corrs = find_correspondences(do_dbg && gn == 0, gn == 0);
        std::cout << "  [GN " << gn << "] Corrispondenze valide: " << num_corrs
                  << "\n";
        double rmse = correspondence_rmse();
        std::cout << "  [GN " << gn << "] corr_rmse=" << std::fixed
                  << std::setprecision(5) << rmse << "\n";

        if (gn > 0 && rmse > last_rmse * 1.02) {
          warp_field_->restore_transforms();
          accepted_updates = std::max(0, accepted_updates - 1);
          warp_update_ok = accepted_updates > 0;
          std::cout << "  [GN " << gn
                    << "] reject previous update (rmse increased from "
                    << last_rmse << " to " << rmse << ")\n";
          break;
        }
        last_rmse = rmse;

        if (do_dbg && gn == 0) {
          std::vector<Correspondence> h_c;
          d_corrs_.download(h_c);
          cv::imshow("4_correspondences",
                     dbg_corrs(h_c, params_.cam, num_corrs));
          cv::waitKey(1);
        }

        if (num_corrs < params_.min_valid_corrs) {
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
          int n_nodes = warp_field_->num_nodes();
          for (int i = 0; i < n_nodes; i++) {
            float *dx = h_dx.data() + i * 6;
            float rn = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
            float tn = std::sqrt(dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]);
            rot_sum += rn;
            trans_sum += tn;
            rot_max = std::max(rot_max, rn);
            trans_max = std::max(trans_max, tn);
          }
          float rot_mean = rot_sum / std::max(1, n_nodes);
          float trans_mean = trans_sum / std::max(1, n_nodes);
          const float reject_rot = 100.0f * params_.max_update_rot;
          const float reject_trans = 100.0f * params_.max_update_trans;
          iter_ok = std::isfinite(rot_mean) && std::isfinite(trans_mean) &&
                    std::isfinite(rot_max) && std::isfinite(trans_max) &&
                    rot_max < reject_rot && trans_max < reject_trans;
          {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(5)
               << "  [dx] rot_mean=" << rot_mean << " rot_max=" << rot_max
               << " trans_mean=" << trans_mean << " trans_max=" << trans_max
               << " nodes=" << n_nodes << " max_rot=" << params_.max_update_rot
               << " max_trans=" << params_.max_update_trans
               << " reject_rot=" << reject_rot
               << " reject_trans=" << reject_trans
               << " clamp=" << ((rot_max > params_.max_update_rot ||
                                  trans_max > params_.max_update_trans)
                                     ? "yes"
                                     : "no");
            std::cout << ss.str() << "\n";
          }
          if (do_dbg && gn == 0) {
            cv::imshow("5_delta_x",
                       dbg_delta_x(h_dx, warp_field_->num_nodes()));
            cv::waitKey(1);
          }
        }
        if (iter_ok) {
          warp_field_->save_transforms();
          warp_field_->apply_twist_increment(d_delta_x_, params_.max_update_rot,
                                             params_.max_update_trans,
                                             params_.update_scale);
          warp_update_ok = true;
          accepted_updates++;
        } else {
          std::cout << "  [warp] skip applying exploding/non-finite update\n";
          break;
        }
      }

      // 7. Integra depth. Unwarped fusion after frame 0 is only safe for a
      // static scene/camera; otherwise it smears incompatible live frames into
      // the canonical TSDF.
      if (params_.integrate_warped) {
        if (warp_update_ok) {
          volume_->integrate(
              d_depth_, d_depth_normals_, params_.cam, camera_pose_,
              warp_field_->device_nodes(), warp_field_->device_transforms(),
              warp_field_->num_nodes(), d_voxel_knn_.data, d_voxel_knn_w_.data);
          last_frame_integrated_ = true;
        } else {
          std::cout << "  [integrate] skip warped fusion"
                    << " (warp update not stable)\n";
        }
      } else if (params_.integrate_unwarped_fallback) {
        volume_->integrate(d_depth_, d_depth_normals_, params_.cam,
                           camera_pose_);
        last_frame_integrated_ = true;
        std::cout << "  [integrate] fused depth without warp"
                  << " (fallback enabled)\n";
      } else {
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

  // Fill existing O3D mesh in-place (for vis update)
  void update_o3d_mesh(open3d::geometry::TriangleMesh &mesh) const {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    volume_->extract_surface(verts, norms, tris);
    std::cout << "verts: " << verts.size() << " tris: " << tris.size()
              << std::endl;

    mesh.vertices_.clear();
    mesh.triangles_.clear();
    mesh.vertices_.reserve(verts.size());
    mesh.triangles_.reserve(tris.size());
    for (auto &v : verts) mesh.vertices_.push_back({v.x, v.y, v.z});
    for (auto &t : tris) mesh.triangles_.push_back({t.x, t.y, t.z});
    mesh.ComputeVertexNormals();
    mesh.PaintUniformColor({1.0, 1.0, 1.0});
  }

  // Salva la mesh warped (live frame) come PLY
  // CPU skinning: per ogni vertice canonico, applica blend delle trasformazioni
  // nodi più vicini
  void save_warped_mesh_ply(const std::string &path) const {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    volume_->extract_surface(verts, norms, tris);

    int num_nodes = warp_field_->num_nodes();
    if (num_nodes == 0) {
      save_mesh_ply(path);
      return;
    }

    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();

    std::vector<float3> warped(verts.size());
    for (size_t vi = 0; vi < verts.size(); vi++) {
      float3 p = verts[vi];

      // Brute-force K-NN among nodes
      float best_d2[K_NEIGHBORS];
      int best_id[K_NEIGHBORS];
      for (int k = 0; k < K_NEIGHBORS; k++) {
        best_d2[k] = 1e30f;
        best_id[k] = -1;
      }
      for (int ni = 0; ni < num_nodes; ni++) {
        float3 d = {p.x - h_nodes[ni].pos.x, p.y - h_nodes[ni].pos.y,
                    p.z - h_nodes[ni].pos.z};
        float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if (d2 < best_d2[K_NEIGHBORS - 1]) {
          best_d2[K_NEIGHBORS - 1] = d2;
          best_id[K_NEIGHBORS - 1] = ni;
          for (int k = K_NEIGHBORS - 2; k >= 0; k--) {
            if (best_d2[k + 1] < best_d2[k]) {
              std::swap(best_d2[k], best_d2[k + 1]);
              std::swap(best_id[k], best_id[k + 1]);
            }
          }
        }
      }

      float w_sum = 0;
      float weights[K_NEIGHBORS] = {};
      for (int k = 0; k < K_NEIGHBORS; k++) {
        if (best_id[k] < 0) continue;
        float r = h_nodes[best_id[k]].radius;
        weights[k] = expf(-best_d2[k] / (2.f * r * r));
        w_sum += weights[k];
      }

      float3 wp = {0, 0, 0};
      for (int k = 0; k < K_NEIGHBORS; k++) {
        if (best_id[k] < 0 || w_sum < 1e-8f) continue;
        float w = weights[k] / w_sum;
        float3 tp = h_transforms[best_id[k]].transform_point_centered(
            p, h_nodes[best_id[k]].pos);
        wp.x += w * tp.x;
        wp.y += w * tp.y;
        wp.z += w * tp.z;
      }
      warped[vi] = (w_sum > 1e-8f) ? wp : p;
    }

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
  void save_mesh_ply(const std::string &path) const {
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
    for (size_t i = 0; i < verts.size(); i++) {
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

  Mat4 camera_pose_;
  bool last_frame_integrated_ = false;
  cv::Mat current_color_gray_;
  std::deque<std::unique_ptr<SiftFeatureFrame>> sift_history_;

  void compute_normals() {
    dim3 block(16, 16);
    dim3 grid((params_.cam.width + block.x - 1) / block.x,
              (params_.cam.height + block.y - 1) / block.y);

    compute_depth_normals_kernel<<<grid, block>>>(
        d_depth_.data, d_depth_normals_.data, params_.cam.width,
        params_.cam.height, params_.cam);
    cudaDeviceSynchronize();
  }

  void initialize_node_graph() {
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
    int added =
        warp_field_->add_nodes_from_surface(verts, params_.node_min_dist);
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

  void update_node_graph() {
    // Download canonical voxel hit-map from raycast → convert to canonical
    // world pos Avoids MC; gives correct canonical (not warped) positions for
    // new nodes
    int n_pixels = params_.cam.width * params_.cam.height;
    std::vector<int> h_vidx(n_pixels);
    cudaMemcpy(h_vidx.data(), d_hit_voxel_idx_.data, n_pixels * sizeof(int),
               cudaMemcpyDeviceToHost);

    const auto &tp = params_.tsdf;
    std::vector<float3> canonical_pts;
    canonical_pts.reserve(n_pixels / 4);
    for (int px = 0; px < n_pixels; px++) {
      int vidx = h_vidx[px];
      if (vidx < 0) continue;
      int vx = vidx % tp.dims.x;
      int vy = (vidx / tp.dims.x) % tp.dims.y;
      int vz = vidx / (tp.dims.x * tp.dims.y);
      canonical_pts.push_back(make_float3(tp.origin.x + vx * tp.voxel_size,
                                          tp.origin.y + vy * tp.voxel_size,
                                          tp.origin.z + vz * tp.voxel_size));
    }

    int added = warp_field_->add_nodes_from_surface(canonical_pts,
                                                    params_.node_min_dist);
    if (added > 0) {
      warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
      std::cout << "[Update] Nuovi nodi: " << added << "\n";
    }
  }

  void apply_depth_bbox_filter() {
    dim3 block(16, 16);
    dim3 grid((params_.cam.width + block.x - 1) / block.x,
              (params_.cam.height + block.y - 1) / block.y);
    depth_bbox_filter_kernel<<<grid, block>>>(
        d_depth_.data, params_.cam.width, params_.cam.height, params_.cam,
        params_.bbox.min_pt, params_.bbox.max_pt);
    cudaDeviceSynchronize();
  }

  void apply_bbox_filter(DeviceArray<float3> &verts) {
    int n = (int)verts.size();
    bbox_filter_kernel<<<grid1d(n), 256>>>(verts.data, n, params_.bbox.min_pt,
                                           params_.bbox.max_pt);
    cudaDeviceSynchronize();
  }

  void print_depth_stats(const char *label) {
    std::vector<float> h_depth;
    d_depth_.download(h_depth);
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (float d : h_depth) {
      if (d <= 0.0f) continue;
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

  void print_cpu_depth_stats(const cv::Mat &depth, const char *label) {
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (int r = 0; r < depth.rows; r++) {
      const float *row = depth.ptr<float>(r);
      for (int c = 0; c < depth.cols; c++) {
        float d = row[c];
        if (d <= 0.0f) continue;
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
                              const char *label) {
    int valid = 0;
    float min_v = std::numeric_limits<float>::max();
    float max_v = 0.0f;
    double sum = 0.0;
    for (float d : depth) {
      if (d <= 0.0f) continue;
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

  void print_tsdf_stats(const char *label) {
    std::vector<TSDFVoxel> h_voxels;
    h_voxels.resize(volume_->total_voxels());
    cudaMemcpy(h_voxels.data(), volume_->device_data(),
               h_voxels.size() * sizeof(TSDFVoxel), cudaMemcpyDeviceToHost);

    int observed = 0, neg = 0, near_zero = 0;
    float min_tsdf = 1.0f;
    float max_tsdf = -1.0f;
    for (const auto &voxel : h_voxels) {
      if (voxel.weight <= 0.0f) continue;
      observed++;
      min_tsdf = std::min(min_tsdf, voxel.tsdf);
      max_tsdf = std::max(max_tsdf, voxel.tsdf);
      if (voxel.tsdf < 0.0f) neg++;
      if (fabsf(voxel.tsdf) < 0.2f) near_zero++;
    }
    std::cout << "  [" << label << "] observed=" << observed << " neg=" << neg
              << " near_zero=" << near_zero
              << " min=" << (observed ? min_tsdf : 0.0f)
              << " max=" << (observed ? max_tsdf : 0.0f) << "\n";
  }

  struct SparseFeature {
    float3 canonical_src;
    int node_ids[K_NEIGHBORS];
    float node_ws[K_NEIGHBORS];
  };

  struct SiftFeatureFrame {
    SiftData sift{};
    std::vector<SparseFeature> features;
    bool initialized = false;

    ~SiftFeatureFrame() {
      if (initialized) FreeSiftData(sift);
    }
  };

  cv::Mat sift_input_image(const std::vector<float> &depth) const {
    cv::Mat gray = current_color_gray_;
    if (gray.empty()) {
      DepthSequence::depth_to_gray(
          cv::Mat(params_.cam.height, params_.cam.width, CV_32FC1,
                  const_cast<float *>(depth.data())),
          gray);
    }
    if (gray.channels() != 1) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    if (gray.cols != params_.cam.width || gray.rows != params_.cam.height) {
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
      const std::vector<int> &h_hit_voxel_idx) const {
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
    for (int i = 0; i < frame->sift.numPts; i++) {
      const SiftPoint &sp = frame->sift.h_data[i];
      int u = static_cast<int>(std::lround(sp.xpos));
      int v = static_cast<int>(std::lround(sp.ypos));
      SparseFeature feat{};
      feat.canonical_src = make_float3(0, 0, 0);
      for (int k = 0; k < K_NEIGHBORS; k++) {
        feat.node_ids[k] = -1;
        feat.node_ws[k] = 0.0f;
      }
      if (u >= 0 && u < params_.cam.width && v >= 0 && v < params_.cam.height) {
        int px = v * params_.cam.width + u;
        float d = h_depth[px];
        float3 model_p = h_live[px];
        int vidx = h_hit_voxel_idx[px];
        if (d > 0.01f && model_p.z > 0.01f && vidx >= 0) {
          const auto &tp = params_.tsdf;
          int vx = vidx % tp.dims.x;
          int vy = (vidx / tp.dims.x) % tp.dims.y;
          int vz = vidx / (tp.dims.x * tp.dims.y);
          feat.canonical_src =
              make_float3(tp.origin.x + (vx + 0.5f) * tp.voxel_size,
                          tp.origin.y + (vy + 0.5f) * tp.voxel_size,
                          tp.origin.z + (vz + 0.5f) * tp.voxel_size);
          for (int k = 0; k < K_NEIGHBORS; k++) {
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
                                 int axis) const {
    if (idx >= (int)h_corrs.size()) return idx;
    Correspondence corr{};
    corr.src = warped_src;
    corr.dst = dst;
    corr.normal = make_float3(axis == 0 ? 1.0f : 0.0f, axis == 1 ? 1.0f : 0.0f,
                              axis == 2 ? 1.0f : 0.0f);
    corr.weight = params_.sift_weight;
    corr.valid = true;
    for (int k = 0; k < K_NEIGHBORS; k++) {
      corr.node_ids[k] = src_feat.node_ids[k];
      corr.node_ws[k] = src_feat.node_ws[k];
    }
    h_corrs[idx] = corr;
    return idx + 1;
  }

  static float3 warp_point_host(float3 p, const SparseFeature &feat,
                                const std::vector<DeformNode> &nodes,
                                const std::vector<Mat4> &transforms) {
    float3 out = make_float3(0, 0, 0);
    float w_sum = 0.0f;
    for (int k = 0; k < K_NEIGHBORS; k++) {
      int nid = feat.node_ids[k];
      float w = feat.node_ws[k];
      if (nid < 0 || nid >= (int)nodes.size() || w < 1e-8f) continue;
      float3 wp = transforms[nid].transform_point_centered(p, nodes[nid].pos);
      out.x += w * wp.x;
      out.y += w * wp.y;
      out.z += w * wp.z;
      w_sum += w;
    }
    if (w_sum > 1e-8f) {
      out.x /= w_sum;
      out.y /= w_sum;
      out.z /= w_sum;
      return out;
    }
    return p;
  }

  int augment_correspondences_with_sift(int current_count, bool debug) {
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
    for (size_t i = n_pixels; i < h_corrs.size(); i++) {
      h_corrs[i].valid = false;
      h_corrs[i].weight = 0.0f;
    }

    cv::Mat gray32 = sift_input_image(h_depth);
    auto current = extract_sift_feature_frame(
        gray32, h_live, h_depth, h_pixel_knn, h_pixel_knn_w, h_hit_voxel_idx);
    int added = 0;
    int append_idx = n_pixels;
    for (auto &hist : sift_history_) {
      MatchSiftData(current->sift, hist->sift);
      int frame_added = 0;
      for (int i = 0; i < current->sift.numPts &&
                      frame_added < params_.sift_max_matches_per_frame;
           i++) {
        const SiftPoint &sp = current->sift.h_data[i];
        if (sp.match < 0 || sp.match >= hist->sift.numPts) continue;
        if (sp.match_error > params_.sift_max_match_error ||
            sp.ambiguity > params_.sift_max_ambiguity)
          continue;

        int u = static_cast<int>(std::lround(sp.xpos));
        int v = static_cast<int>(std::lround(sp.ypos));
        if (u < 0 || u >= params_.cam.width || v < 0 || v >= params_.cam.height)
          continue;

        int dst_px = v * params_.cam.width + u;
        float d = h_depth[dst_px];
        if (d <= 0.01f) continue;
        float3 dst = params_.cam.unproject(u, v, d);

        const SparseFeature &src_feat = hist->features[sp.match];
        if (src_feat.canonical_src.z <= 0.01f || src_feat.node_ids[0] < 0)
          continue;
        float3 warped_src = warp_point_host(src_feat.canonical_src, src_feat,
                                            h_nodes, h_transforms);

        float3 diff = make_float3(warped_src.x - dst.x, warped_src.y - dst.y,
                                  warped_src.z - dst.z);
        if (host_norm3(diff) > params_.sift_max_3d_dist) continue;

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
    if (debug || added > 0) {
      std::cout << "  [sift] current=" << current->sift.numPts
                << " history=" << sift_history_.size()
                << " sparse_rows=" << added << "\n";
    }

    sift_history_.push_back(std::move(current));
    while ((int)sift_history_.size() > params_.sift_max_history) {
      sift_history_.pop_front();
    }
    return current_count + added;
  }

  double correspondence_rmse() const {
    std::vector<Correspondence> h_corrs;
    d_corrs_.download(h_corrs);
    double sum = 0.0;
    double wsum = 0.0;
    for (const auto &c : h_corrs) {
      if (!c.valid || c.weight <= 0.0f) continue;
      float3 diff = make_float3(c.src.x - c.dst.x, c.src.y - c.dst.y,
                                c.src.z - c.dst.z);
      double r = c.normal.x * diff.x + c.normal.y * diff.y +
                 c.normal.z * diff.z;
      sum += (double)c.weight * r * r;
      wsum += c.weight;
    }
    return (wsum > 0.0) ? std::sqrt(sum / wsum) : std::numeric_limits<double>::infinity();
  }

  int find_correspondences(bool debug, bool use_sift_aug) {
    int n_pixels = params_.cam.width * params_.cam.height;

    // Ensure pixel k-NN arrays sized correctly
    if (d_pixel_knn_.size() != (size_t)(n_pixels * K_NEIGHBORS)) {
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
    Mat4 T_cam_world = Mat4::identity();  // camera_pose_ = identity
    find_correspondences_kernel<<<grid, block>>>(
        d_vertices_live_.data, d_normals_live_.data, d_depth_.data,
        d_depth_normals_.data, d_corrs_.data, d_num_valid_.data,
        params_.cam.width, params_.cam.height, params_.cam, T_cam_world,
        d_pixel_knn_.data, d_pixel_knn_w_.data, params_.dist_threshold,
        params_.angle_threshold, params_.view_threshold,
        params_.search_radius_px, debug ? d_corr_stats_.data : nullptr);
    cudaDeviceSynchronize();

    int h_valid = 0;
    cudaMemcpy(&h_valid, d_num_valid_.data, sizeof(int),
               cudaMemcpyDeviceToHost);
    if (params_.use_sift && use_sift_aug) {
      h_valid = augment_correspondences_with_sift(h_valid, debug);
    }
    if (debug) {
      std::vector<int> h_stats;
      d_corr_stats_.download(h_stats);
      std::cout << "  [corr debug] invalid_live=" << h_stats[0]
                << " out_proj=" << h_stats[1] << " no_depth=" << h_stats[2]
                << " far=" << h_stats[3] << " angle=" << h_stats[4]
                << " valid=" << h_stats[5] << "\n";
    }
    return h_valid;
  }
};

struct AppConfig {
  DepthSequence::Config seq;
  DynamicFusionPipeline::Params df;
  bool use_vis = false;
};

AppConfig load_config(const std::string &path) {
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
  if (w["max_dx_mean"]) out.df.max_dx_mean = w["max_dx_mean"].as<float>();
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

  if (cfg["sift"]) {
    auto sift = cfg["sift"];
    if (sift["enabled"]) out.df.use_sift = sift["enabled"].as<bool>();
    if (sift["max_features"])
      out.df.sift_max_features = sift["max_features"].as<int>();
    if (sift["max_history"])
      out.df.sift_max_history = sift["max_history"].as<int>();
    if (sift["max_matches_per_frame"])
      out.df.sift_max_matches_per_frame =
          sift["max_matches_per_frame"].as<int>();
    if (sift["octaves"]) out.df.sift_octaves = sift["octaves"].as<int>();
    if (sift["threshold"])
      out.df.sift_threshold = sift["threshold"].as<float>();
    if (sift["max_match_error"])
      out.df.sift_max_match_error = sift["max_match_error"].as<float>();
    if (sift["max_ambiguity"])
      out.df.sift_max_ambiguity = sift["max_ambiguity"].as<float>();
    if (sift["max_3d_dist"])
      out.df.sift_max_3d_dist = sift["max_3d_dist"].as<float>();
    if (sift["weight"]) out.df.sift_weight = sift["weight"].as<float>();
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
