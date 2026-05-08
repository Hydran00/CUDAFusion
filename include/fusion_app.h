#pragma once
#include <cuda_runtime.h>
#include <open3d/Open3D.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cudaImage.h"
#include "cudaSift.h"
#include "depth_sequence.h"
#include "fusion_debug.h"
#include "fusion_math.h"
#include "mesh_raster_kernels.h"
#include "skinning_kernels.h"
#include "solver.h"
#include "tsdf_kernels.h"
#include "tsdf_volume.h"
#include "types.h"
#include "warp_field.h"
// ─────────────────────────────────────────────
//  Pipeline DynamicFusion (self-contained per test)
// ─────────────────────────────────────────────

class DynamicFusionPipeline
{
  struct TrackingProfile
  {
    double raster_total_ms = 0.0;
    double mc_ms = 0.0;
    double skin_ms = 0.0;
    double tri_raster_ms = 0.0;
    double raster_upload_ms = 0.0;
    double corr_ms = 0.0;
    double rigid_ms = 0.0;
    double sift_aug_ms = 0.0;
    double solve_ms = 0.0;
    double validate_ms = 0.0;
    double apply_ms = 0.0;
    double rmse_ms = 0.0;
    double line_search_ms = 0.0;
    int raster_calls = 0;

    bool has_work() const
    {
      return raster_calls > 0 || corr_ms > 0.0 || solve_ms > 0.0 ||
             validate_ms > 0.0 || apply_ms > 0.0 || rigid_ms > 0.0 ||
             rmse_ms > 0.0 || line_search_ms > 0.0;
    }
  };

  static double elapsed_ms(
      const std::chrono::high_resolution_clock::time_point &start)
  {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

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
    int node_update_every_n = 5;
    bool prune_nodes = true;
    bool prune_disconnected = true;
    int prune_min_observed_voxels = 8;
    int prune_min_surface_voxels = 2;
    int prune_min_component_size = 4;
    float prune_support_radius_factor = 1.0f;
    float prune_surface_tsdf_abs = 0.45f;
    float prune_empty_tsdf = 0.75f;
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
    float sift_max_pixel_dist = 0.0f;
    float sift_weight = 1.0f;
    bool debug_vis = false;
    int debug_every_n = 1;
    bool profile_timings = false;
    int profile_every_n = 1;
    bool quiet = false;
  };

  int num_nodes() const { return warp_field_->num_nodes(); }

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
    {
      ScopedStdoutSilence silence(params_.quiet);
      InitCuda(0);
    }
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
    const bool do_profile =
        params_.profile_timings &&
        (frame_count_ % std::max(1, params_.profile_every_n) == 0);
    double profile_upload_ms = 0.0;
    double profile_bbox_ms = 0.0;
    double profile_normals_ms = 0.0;
    double profile_initial_integrate_ms = 0.0;
    double profile_init_graph_ms = 0.0;
    double profile_tracking_ms = 0.0;
    double profile_fusion_ms = 0.0;
    double profile_graph_update_ms = 0.0;
    TrackingProfile tracking_profile;

    // 1. Upload depth
    {
      auto t_upload = std::chrono::high_resolution_clock::now();
      cv::Mat depth32;
      if (depth_m.type() == CV_32FC1)
        depth32 = depth_m;
      else
        depth_m.convertTo(depth32, CV_32FC1);
      if (!depth32.isContinuous())
        depth32 = depth32.clone();
      const float *depth_begin = depth32.ptr<float>(0);
      std::vector<float> h_depth(depth_begin,
                                 depth_begin + depth32.rows * depth32.cols);
      d_depth_.upload(h_depth);
      profile_upload_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_upload)
              .count();
    }

    // 1b. Bounding box filter on depth (before integration)
    if (params_.bbox.enabled)
    {
      auto t_bbox = std::chrono::high_resolution_clock::now();
      apply_depth_bbox_filter();
      profile_bbox_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_bbox)
              .count();
    }

    // 2. Calcola normali dal depth
    auto t_normals = std::chrono::high_resolution_clock::now();
    compute_normals();
    profile_normals_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_normals)
            .count();

    if (do_dbg)
    {
      // cv::imshow("1_input_rgb", dbg_color_input(current_color_gray_, depth_m));
      std::vector<float3> h_n;
      d_depth_normals_.download(h_n);
      cv::imshow("2_depth_normals",
                 dbg_normals(h_n, params_.cam.width, params_.cam.height));
      cv::waitKey(1);
    }

    if (frame_count_ == 0)
    {
      // Primo frame: solo integra
      auto t_integrate = std::chrono::high_resolution_clock::now();
      volume_->integrate(d_depth_, d_depth_normals_, params_.cam, camera_pose_);
      surface_mesh_cache_valid_ = false;
      last_frame_integrated_ = true;
      profile_initial_integrate_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_integrate)
              .count();
      auto t_init_graph = std::chrono::high_resolution_clock::now();
      initialize_node_graph();
      profile_init_graph_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_init_graph)
              .count();
    }
    else
    {
      auto t_tracking = std::chrono::high_resolution_clock::now();
      bool warp_update_ok = false;

      // Warp field: salva trasformazioni precedenti (warm start)
      warp_field_->save_transforms();

      int num_corrs = 0;
      for (int gn = 0; gn < std::max(1, params_.solver.gn_iterations); gn++)
      {
        bool iter_ok = false;

        // 3. VolumeDeform-style prediction: extract canonical mesh, deform
        // vertices, then rasterize the deformed mesh into the current camera.
        rasterize_deformed_mesh_live_surface(do_profile ? &tracking_profile
                                                        : nullptr);

        if (do_dbg && gn == 0)
        {
          std::vector<float3> h_v;
          d_vertices_live_.download(h_v);
          auto h_nodes = warp_field_->download_nodes();
          auto h_transforms = warp_field_->download_transforms();
          cv::imshow(
              "3_live_surface_graph",
              dbg_verts(h_v, params_.cam, inverse_rigid_host(camera_pose_),
                        &h_nodes, &h_transforms));
          cv::waitKey(1);
        }

        // 3b. Bounding box filter on live vertices
        if (params_.bbox.enabled)
        {
          auto t_bbox = std::chrono::high_resolution_clock::now();
          apply_bbox_filter(d_vertices_live_);
          profile_bbox_ms +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t_bbox)
                  .count();
        }

        // 4. Trova corrispondenze ICP. SIFT history is consumed once per frame;
        // later GN iterations refresh only dense projective correspondences.
        {
          auto t_corr = std::chrono::high_resolution_clock::now();
          num_corrs = find_correspondences(do_dbg && gn == 0, gn == 0);
          tracking_profile.corr_ms += elapsed_ms(t_corr);
        }
        if (params_.rigid_tracking && gn == 0 && num_corrs >= 64)
        {
          auto t_rigid = std::chrono::high_resolution_clock::now();
          if (estimate_rigid_tracking_update())
          {
            tracking_profile.rigid_ms += elapsed_ms(t_rigid);
            rasterize_deformed_mesh_live_surface(do_profile ? &tracking_profile
                                                            : nullptr);
            if (params_.bbox.enabled)
            {
              auto t_bbox = std::chrono::high_resolution_clock::now();
              apply_bbox_filter(d_vertices_live_);
              profile_bbox_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t_bbox)
                      .count();
            }
            auto t_corr = std::chrono::high_resolution_clock::now();
            num_corrs = find_correspondences(do_dbg && gn == 0, false);
            tracking_profile.corr_ms += elapsed_ms(t_corr);

            // Re-apply SIFT augmentation after the rigid update so sparse
            // SIFT rows stay present and are consistent with the new pose.
            if (params_.use_sift)
            {
              auto t_sift_aug = std::chrono::high_resolution_clock::now();
              num_corrs = augment_correspondences_with_sift(num_corrs,
                                                            false);
              tracking_profile.sift_aug_ms += elapsed_ms(t_sift_aug);
            }
          }
          else
          {
            tracking_profile.rigid_ms += elapsed_ms(t_rigid);
          }
        }
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
          break;
        auto t_base_rmse = std::chrono::high_resolution_clock::now();
        double base_rmse = correspondence_rmse();
        tracking_profile.rmse_ms += elapsed_ms(t_base_rmse);

        // 5. Ottimizzazione Gauss-Newton
        // Minimal diagnostic instrumentation: when debug_vis is enabled,
        // download a sample of correspondences and compute CPU RMSE before
        // the solver, then download delta_x after the solver.
        if (params_.debug_vis && gn == 0)
        {
          std::vector<Correspondence> h_c;
          d_corrs_.download(h_c);
          size_t valid_cnt = 0;
          double sumw = 0.0;
          double sumsq = 0.0;
          for (size_t i = 0; i < h_c.size(); ++i)
          {
            const auto &c = h_c[i];
            if (!c.valid)
              continue;
            valid_cnt++;
            double r = (double)(c.dst.x - c.src.x) * c.normal.x +
                       (double)(c.dst.y - c.src.y) * c.normal.y +
                       (double)(c.dst.z - c.src.z) * c.normal.z;
            sumw += c.weight;
            sumsq += r * r;
            // if (valid_cnt <= 8)
            // {
            //   printf("[DIAG corr %zu] src=(%.6e,%.6e,%.6e) dst=(%.6e,%.6e,%.6e) n=(%.6e,%.6e,%.6e) r=%.6e w=%.6e\n",
            //          valid_cnt,
            //          (double)c.src.x, (double)c.src.y, (double)c.src.z,
            //          (double)c.dst.x, (double)c.dst.y, (double)c.dst.z,
            //          (double)c.normal.x, (double)c.normal.y, (double)c.normal.z,
            //          r, (double)c.weight);
            // }
          }
          double cpu_rmse = valid_cnt ? std::sqrt(sumsq / (double)valid_cnt)
                                      : 0.0;
          // printf("[DIAG corrs] valid=%zu sumw=%.6e cpu_rmse=%.6e\n",
          //        valid_cnt, sumw, cpu_rmse);
        }

        auto t_solve = std::chrono::high_resolution_clock::now();
        solver_->solve(d_corrs_, (int)d_corrs_.size(),
                       warp_field_->device_nodes(),
                       warp_field_->device_transforms(),
                       warp_field_->num_nodes(), d_delta_x_);
        tracking_profile.solve_ms += elapsed_ms(t_solve);

        if (params_.debug_vis && gn == 0)
        {
          std::vector<float> h_dx;
          d_delta_x_.download(h_dx);
          int n_dx = (int)h_dx.size();
          double dx_norm = 0.0;
          for (int i = 0; i < n_dx; ++i)
            dx_norm += (double)h_dx[i] * (double)h_dx[i];
          dx_norm = std::sqrt(dx_norm);
          printf("[DIAG solver] delta_x_norm=%.6e first3=%.6e %.6e %.6e\n",
                 dx_norm,
                 (double)(n_dx > 0 ? h_dx[0] : 0.0f),
                 (double)(n_dx > 1 ? h_dx[1] : 0.0f),
                 (double)(n_dx > 2 ? h_dx[2] : 0.0f));
        }

        // 6. Valida incremento prima di applicarlo.
        {
          auto t_validate = std::chrono::high_resolution_clock::now();
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
          const float reject_rot =
              params_.max_update_rot > 0.0f
                  ? 5.0f * params_.max_update_rot
                  : std::numeric_limits<float>::infinity();
          const float reject_trans =
              params_.max_update_trans > 0.0f
                  ? 5.0f * params_.max_update_trans
                  : std::numeric_limits<float>::infinity();

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

          iter_ok =
              finite_raw_update && std::isfinite(eff_rot_mean) &&
              std::isfinite(eff_trans_mean) && std::isfinite(eff_rot_max) &&
              std::isfinite(eff_trans_max) && (eff_rot_max < reject_rot) &&
              (eff_trans_max < reject_trans) && mean_ok && !has_bad_spike;
          tracking_profile.validate_ms += elapsed_ms(t_validate);
        }

        // ---- APPLY SOLO SE OK ----
        if (iter_ok)
        {
          auto t_line_search = std::chrono::high_resolution_clock::now();
          if (params_.update_scale <= 0.0f || !std::isfinite(base_rmse))
            break;

          warp_field_->save_transforms();
          float trial_scales[] = {
              params_.update_scale,
              0.5f * params_.update_scale,
              0.25f * params_.update_scale,
              0.125f * params_.update_scale,
              0.0625f * params_.update_scale,
              -0.0625f * params_.update_scale,
              -0.125f * params_.update_scale,
              -0.25f * params_.update_scale,
              -0.5f * params_.update_scale,
              -params_.update_scale,
          };
          float best_scale = 0.0f;
          double best_rmse = base_rmse;

          for (float trial_scale : trial_scales)
          {
            warp_field_->restore_transforms();
            auto t_apply = std::chrono::high_resolution_clock::now();
            upload_scaled_delta_x(trial_scale);
            if (params_.debug_vis)
            {
              auto before_nodes = warp_field_->download_nodes();
              auto before_trans = warp_field_->download_transforms();
              warp_field_->apply_twist_increment(
                  d_delta_x_trial_, params_.max_update_rot,
                  params_.max_update_trans, 1.0f);
              auto after_nodes = warp_field_->download_nodes();
              auto after_trans = warp_field_->download_transforms();
              double sum = 0.0;
              double maxd = 0.0;
              int nn = (int)after_nodes.size();
              for (int ii = 0; ii < nn; ++ii)
              {
                float3 pb = dq_transform_point(dq_centered(before_trans[ii], before_nodes[ii].pos), before_nodes[ii].pos);
                float3 pa = dq_transform_point(dq_centered(after_trans[ii], after_nodes[ii].pos), after_nodes[ii].pos);
                double d = sqrt((pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y) + (pb.z - pa.z) * (pb.z - pa.z));
                sum += d;
                if (d > maxd)
                  maxd = d;
              }
              double mean = nn > 0 ? sum / nn : 0.0;
              std::cout << "[node move trial] frame=" << frame_count_ << " scale=" << trial_scale
                        << " mean_m=" << mean << " max_m=" << maxd << "\n";
            }
            else
            {
              warp_field_->apply_twist_increment(
                  d_delta_x_trial_, params_.max_update_rot,
                  params_.max_update_trans, 1.0f);
            }
            tracking_profile.apply_ms += elapsed_ms(t_apply);

            rasterize_deformed_mesh_live_surface(do_profile ? &tracking_profile
                                                            : nullptr);
            if (params_.bbox.enabled)
            {
              auto t_bbox = std::chrono::high_resolution_clock::now();
              apply_bbox_filter(d_vertices_live_);
              profile_bbox_ms +=
                  std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t_bbox)
                      .count();
            }

            auto t_corr_check = std::chrono::high_resolution_clock::now();
            int checked_corrs = find_correspondences(false, false);
            tracking_profile.corr_ms += elapsed_ms(t_corr_check);
            auto t_rmse = std::chrono::high_resolution_clock::now();
            double checked_rmse = correspondence_rmse();
            tracking_profile.rmse_ms += elapsed_ms(t_rmse);

            if (checked_corrs >= params_.min_valid_corrs &&
                checked_corrs >= (int)(0.75f * (float)num_corrs) &&
                std::isfinite(checked_rmse) && checked_rmse < best_rmse)
            {
              best_rmse = checked_rmse;
              best_scale = trial_scale;
            }
          }

          printf("[GN iter %d] base RMSE=%.6e, best RMSE=%.6e at scale %.6e\n",
                 gn, base_rmse, best_rmse, (double)best_scale);
          warp_field_->restore_transforms();
          if (best_scale == 0.0f)
          {
            tracking_profile.line_search_ms += elapsed_ms(t_line_search);
            break;
          }

          auto t_apply = std::chrono::high_resolution_clock::now();
          upload_scaled_delta_x(best_scale);
          if (params_.debug_vis)
          {
            auto before_nodes = warp_field_->download_nodes();
            auto before_trans = warp_field_->download_transforms();
            warp_field_->apply_twist_increment(
                d_delta_x_trial_, params_.max_update_rot,
                params_.max_update_trans, 1.0f);
            auto after_nodes = warp_field_->download_nodes();
            auto after_trans = warp_field_->download_transforms();
            double sum = 0.0;
            double maxd = 0.0;
            int nn = (int)after_nodes.size();
            for (int ii = 0; ii < nn; ++ii)
            {
              float3 pb = dq_transform_point(dq_centered(before_trans[ii], before_nodes[ii].pos), before_nodes[ii].pos);
              float3 pa = dq_transform_point(dq_centered(after_trans[ii], after_nodes[ii].pos), after_nodes[ii].pos);
              double d = sqrt((pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y) + (pb.z - pa.z) * (pb.z - pa.z));
              sum += d;
              if (d > maxd)
                maxd = d;
            }
            double mean = nn > 0 ? sum / nn : 0.0;
            std::cout << "[node move apply] frame=" << frame_count_ << " mean_m=" << mean << " max_m=" << maxd << "\n";
          }
          else
          {
            warp_field_->apply_twist_increment(
                d_delta_x_trial_, params_.max_update_rot,
                params_.max_update_trans, 1.0f);
          }
          tracking_profile.apply_ms += elapsed_ms(t_apply);

          rasterize_deformed_mesh_live_surface(do_profile ? &tracking_profile
                                                          : nullptr);
          if (params_.bbox.enabled)
          {
            auto t_bbox = std::chrono::high_resolution_clock::now();
            apply_bbox_filter(d_vertices_live_);
            profile_bbox_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_bbox)
                    .count();
          }
          auto t_corr_check = std::chrono::high_resolution_clock::now();
          num_corrs = find_correspondences(false, false);
          tracking_profile.corr_ms += elapsed_ms(t_corr_check);
          warp_update_ok = true;
          tracking_profile.line_search_ms += elapsed_ms(t_line_search);
        }
        else
        {
          break;
        }
      }
      profile_tracking_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_tracking)
              .count();
      // 7. Integra depth. Unwarped fusion after frame 0 is only safe for a
      // static scene/camera; otherwise it smears incompatible live frames into
      // the canonical TSDF.
      auto t_fusion = std::chrono::high_resolution_clock::now();
      if (params_.integrate_warped)
      {
        // if (warp_update_ok)
        if (true)
        {
          mark_optimized_voxels();
          volume_->integrate(
              d_depth_, d_depth_normals_, params_.cam, camera_pose_,
              warp_field_->device_nodes(), warp_field_->device_transforms(),
              warp_field_->num_nodes(), d_voxel_knn_.data, d_voxel_knn_w_.data,
              d_voxel_opt_counts_.data, params_.integrate_min_optimized_count);
          surface_mesh_cache_valid_ = false;
          last_frame_integrated_ = true;
        }
      }
      else if (params_.integrate_unwarped_fallback)
      {
        volume_->integrate(d_depth_, d_depth_normals_, params_.cam,
                           camera_pose_);
        surface_mesh_cache_valid_ = false;
        last_frame_integrated_ = true;
      }
      profile_fusion_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - t_fusion)
              .count();

      // 8. Aggiorna grafo con nuovi nodi
      if (params_.node_update_every_n > 0 &&
          frame_count_ % params_.node_update_every_n == 0)
      {
        auto t_graph_update = std::chrono::high_resolution_clock::now();
        update_node_graph();
        profile_graph_update_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_graph_update)
                .count();
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (do_profile)
    {
      std::cout << "[profiler frame " << std::setw(4) << frame_count_ << "] "
                << std::fixed << std::setprecision(2)
                << "upload=" << profile_upload_ms << " ms"
                << " | bbox=" << profile_bbox_ms << " ms"
                << " | normals=" << profile_normals_ms << " ms"
                << " | first_integrate=" << profile_initial_integrate_ms << " ms"
                << " | init_graph=" << profile_init_graph_ms << " ms"
                << " | tracking=" << profile_tracking_ms << " ms"
                << " | fusion=" << profile_fusion_ms << " ms"
                << " | graph_update=" << profile_graph_update_ms << " ms"
                << " | total=" << ms << " ms\n";
      if (tracking_profile.has_work())
      {
        double tracking_accounted =
            tracking_profile.raster_total_ms + tracking_profile.corr_ms +
            tracking_profile.rigid_ms + tracking_profile.sift_aug_ms +
            tracking_profile.solve_ms + tracking_profile.validate_ms +
            tracking_profile.apply_ms + tracking_profile.rmse_ms;
        double tracking_other =
            std::max(0.0, profile_tracking_ms - tracking_accounted);
        std::cout << "[tracking profile frame " << std::setw(4) << frame_count_
                  << "] " << std::fixed << std::setprecision(2)
                  << "raster_total=" << tracking_profile.raster_total_ms << " ms"
                  << " | mc=" << tracking_profile.mc_ms << " ms"
                  << " | skin=" << tracking_profile.skin_ms << " ms"
                  << " | tri_raster=" << tracking_profile.tri_raster_ms << " ms"
                  << " | raster_upload=" << tracking_profile.raster_upload_ms
                  << " ms"
                  << " | corr=" << tracking_profile.corr_ms << " ms"
                  << " | rigid=" << tracking_profile.rigid_ms << " ms"
                  << " | sift_aug=" << tracking_profile.sift_aug_ms << " ms"
                  << " | solve=" << tracking_profile.solve_ms << " ms"
                  << " | validate=" << tracking_profile.validate_ms << " ms"
                  << " | apply=" << tracking_profile.apply_ms << " ms"
                  << " | rmse=" << tracking_profile.rmse_ms << " ms"
                  << " | line_search=" << tracking_profile.line_search_ms
                  << " ms"
                  << " | other=" << tracking_other << " ms"
                  << " | raster_calls=" << tracking_profile.raster_calls
                  << "\n";
      }
    }

    frame_count_++;
  }

  // Fill existing O3D mesh in-place with the deformed live mesh.
  void update_o3d_mesh(open3d::geometry::TriangleMesh &mesh) const
  {
    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    extract_warped_surface(verts, norms, tris);

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

  // Raycast the TSDF into a pointcloud (vertices + normals) and fill the
  // provided Open3D PointCloud. This performs the per-pixel raycast used for
  // smooth, trilinearly-interpolated normals (the "waxy" look).
  void update_o3d_raycast_pointcloud(open3d::geometry::PointCloud &pc)
  {
    // Call TSDF raycast into device buffers (overwrites d_vertices_live_)
    // Note: volume_->raycast allocates device arrays internally; we call the
    // variant that writes into our preallocated device buffers.
    // Use current camera pose and warp field transforms for warped-raycast.
    // Prepare temporary host containers
    int n_pixels = params_.cam.width * params_.cam.height;

    // Raycast into device arrays
    volume_->raycast(d_vertices_live_, d_normals_live_, params_.cam,
                     camera_pose_, warp_field_->device_nodes(),
                     warp_field_->device_transforms(), warp_field_->num_nodes(),
                     d_voxel_knn_.data, d_voxel_knn_w_.data,
                     d_hit_voxel_idx_.data);

    // Download results
    std::vector<float3> h_v;
    std::vector<float3> h_n;
    d_vertices_live_.download(h_v);
    d_normals_live_.download(h_n);

    pc.Clear();
    pc.points_.reserve(n_pixels);
    pc.normals_.reserve(n_pixels);

    for (int i = 0; i < n_pixels; ++i)
    {
      float3 v = h_v[i];
      float3 n = h_n[i];
      // Skip invalid points (z==INF or default)
      if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
        continue;
      pc.points_.push_back(Eigen::Vector3d(v.x, v.y, v.z));
      pc.normals_.push_back(Eigen::Vector3d(n.x, n.y, n.z));
    }
    // normals were downloaded per-pixel; if any are missing we skip recompute
    // (Open3D C++ uses EstimateNormals with search params; skip here).
    pc.PaintUniformColor(Eigen::Vector3d(0.9, 0.9, 0.9));
  }

  // Fill Open3D PointCloud with current node positions (warped)
  void update_o3d_nodes(open3d::geometry::PointCloud &pc) const
  {
    int n_nodes = warp_field_->num_nodes();
    pc.Clear();
    if (n_nodes == 0)
      return;
    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();
    pc.points_.reserve(n_nodes);

    // Lightweight: display each node using its centered dual-quaternion transform
    for (int i = 0; i < n_nodes; ++i)
    {
      float3 p = dq_transform_point(dq_centered(h_transforms[i], h_nodes[i].pos), h_nodes[i].pos);
      pc.points_.push_back(Eigen::Vector3d(p.x, p.y, p.z));
    }
    pc.PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0)); // green
  }

  // Fill Open3D LineSet with graph edges between nodes (warped positions)
  void update_o3d_edges(open3d::geometry::LineSet &ls) const
  {
    int n_nodes = warp_field_->num_nodes();
    ls.Clear();
    if (n_nodes == 0)
      return;
    auto h_nodes = warp_field_->download_nodes();
    auto h_transforms = warp_field_->download_transforms();
    // points: prefer voxel knn mapping for exact match with raycast
    ls.points_.reserve(n_nodes);
    int total_voxels = params_.tsdf.dims.x * params_.tsdf.dims.y * params_.tsdf.dims.z;
    bool have_voxel_knn = d_voxel_knn_.size() == (size_t)total_voxels * K_NEIGHBORS &&
                          d_voxel_knn_w_.size() == (size_t)total_voxels * K_NEIGHBORS;
    std::vector<int> h_voxel_knn;
    std::vector<float> h_voxel_knn_w;
    if (have_voxel_knn)
    {
      h_voxel_knn.resize((size_t)total_voxels * K_NEIGHBORS);
      h_voxel_knn_w.resize((size_t)total_voxels * K_NEIGHBORS);
      d_voxel_knn_.download(h_voxel_knn);
      d_voxel_knn_w_.download(h_voxel_knn_w);
    }
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
    {
      float3 warped_p;
      if (have_voxel_knn)
      {
        int vidx = canonical_voxel_index(h_nodes[i].pos);
        if (vidx >= 0 && vidx < total_voxels)
        {
          const int *knn = h_voxel_knn.data() + vidx * K_NEIGHBORS;
          const float *knn_w = h_voxel_knn_w.data() + vidx * K_NEIGHBORS;
          float ws_local[K_NEIGHBORS];
          for (int k = 0; k < K_NEIGHBORS; ++k)
            ws_local[k] = knn_w[k];
          warped_p = warp_point_dual_quat_host(h_nodes[i].pos, h_nodes, h_transforms, *(int (*)[K_NEIGHBORS])knn, ws_local);
        }
        else
        {
          float3 p = dq_transform_point(dq_centered(h_transforms[i], h_nodes[i].pos), h_nodes[i].pos);
          warped_p = p;
        }
      }
      else
      {
        int ids[K_NEIGHBORS];
        float ws[K_NEIGHBORS];
        skin_point_host(h_nodes[i].pos, h_nodes, h_transforms, warped_p, ids, ws);
      }
      pts.push_back(Eigen::Vector3d(warped_p.x, warped_p.y, warped_p.z));
      ls.points_.push_back(pts.back());
    }
    // lines (unique undirected edges i<j)
    for (int i = 0; i < n_nodes; ++i)
    {
      for (int k = 0; k < h_nodes[i].num_neighbors; ++k)
      {
        int j = h_nodes[i].neighbors[k];
        if (j <= i || j < 0 || j >= n_nodes)
          continue;
        ls.lines_.push_back(Eigen::Vector2i(i, j));
        ls.colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); // red
      }
    }
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
  DeviceArray<float> d_delta_x_trial_;
  DeviceArray<int> d_voxel_knn_;
  DeviceArray<float> d_voxel_knn_w_;
  DeviceArray<int> d_pixel_knn_;
  DeviceArray<float> d_pixel_knn_w_;
  DeviceArray<int> d_num_valid_;
  DeviceArray<int> d_corr_stats_;
  DeviceArray<int> d_hit_voxel_idx_;
  DeviceArray<int> d_voxel_opt_counts_;
  DeviceArray<int> d_voxel_opt_frame_;
  DeviceArray<float3> d_mesh_vertices_;
  DeviceArray<float3> d_mesh_warped_;
  DeviceArray<int3> d_mesh_triangles_;
  DeviceArray<unsigned long long> d_raster_ztri_;

  Mat4 camera_pose_;
  bool last_frame_integrated_ = false;
  bool surface_mesh_cache_valid_ = false;
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
    volume_->extract_surface(verts, norms, tris);
    int added = warp_field_->add_nodes_from_mesh_geodesic(
        verts, tris, params_.node_min_dist, &norms);
    if (added > 0)
      warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
  }

  void update_node_graph()
  {
    bool need_knn_update = false;
    WarpField::PruneParams prune;
    prune.enabled = params_.prune_nodes;
    prune.remove_disconnected = params_.prune_disconnected;
    prune.min_observed_voxels = params_.prune_min_observed_voxels;
    prune.min_surface_voxels = params_.prune_min_surface_voxels;
    prune.min_component_size = params_.prune_min_component_size;
    prune.support_radius_factor = params_.prune_support_radius_factor;
    prune.surface_tsdf_abs = params_.prune_surface_tsdf_abs;
    prune.empty_tsdf = params_.prune_empty_tsdf;
    int removed = warp_field_->prune_nodes(*volume_, prune);
    need_knn_update = removed > 0;

    std::vector<float3> verts, norms;
    std::vector<int3> tris;
    volume_->extract_surface(verts, norms, tris);
    int added = warp_field_->add_nodes_from_mesh_geodesic(
        verts, tris, params_.node_min_dist, &norms);
    if (added > 0)
      need_knn_update = true;

    // DEBUG: print changes to help diagnose static graph behavior
    if (params_.debug_vis && (removed > 0 || added > 0))
    {
      int n_nodes = warp_field_->num_nodes();
      std::cout << "[graph update] removed=" << removed
                << " added=" << added
                << " nodes=" << n_nodes << " verts=" << verts.size()
                << " tris=" << tris.size() << std::endl;
    }
    if (need_knn_update && warp_field_->num_nodes() > 0)
    {
      warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
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

  void rasterize_deformed_mesh_live_surface(TrackingProfile *profile = nullptr)
  {
    auto t_total = std::chrono::high_resolution_clock::now();
    if (!surface_mesh_cache_valid_)
    {
      auto t_mc = std::chrono::high_resolution_clock::now();
      volume_->extract_surface_device(d_mesh_vertices_, d_mesh_triangles_);
      surface_mesh_cache_valid_ = true;
      if (profile)
        profile->mc_ms += elapsed_ms(t_mc);
    }
    int num_vertices = (int)d_mesh_vertices_.size();
    int num_triangles = (int)d_mesh_triangles_.size();

    int n_pixels = params_.cam.width * params_.cam.height;
    d_vertices_live_.zero();
    d_normals_live_.zero();
    CUDA_CHECK(cudaMemset(d_hit_voxel_idx_.data, 0xff,
                          d_hit_voxel_idx_.bytes()));

    if (num_vertices <= 0 || num_triangles <= 0)
    {
      if (profile)
      {
        profile->raster_total_ms += elapsed_ms(t_total);
        profile->raster_calls++;
      }
      return;
    }

    if (d_mesh_warped_.size() != d_mesh_vertices_.size())
      d_mesh_warped_.allocate(d_mesh_vertices_.size());

    auto t_skin = std::chrono::high_resolution_clock::now();
    int total_voxels =
        params_.tsdf.dims.x * params_.tsdf.dims.y * params_.tsdf.dims.z;
    bool can_skin_on_gpu =
        num_vertices > 0 && warp_field_->num_nodes() > 0 &&
        d_voxel_knn_.size() == (size_t)total_voxels * K_NEIGHBORS &&
        d_voxel_knn_w_.size() == (size_t)total_voxels * K_NEIGHBORS;
    if (can_skin_on_gpu)
    {
      skin_vertices_from_voxel_knn_kernel<<<grid1d(num_vertices), 256>>>(
          d_mesh_vertices_.data, d_mesh_warped_.data, num_vertices,
          params_.tsdf.origin, params_.tsdf.voxel_size, params_.tsdf.dims,
          warp_field_->device_nodes(), warp_field_->device_transforms(),
          warp_field_->num_nodes(), d_voxel_knn_.data, d_voxel_knn_w_.data);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if (warp_field_->num_nodes() == 0)
    {
      CUDA_CHECK(cudaMemcpy(d_mesh_warped_.data, d_mesh_vertices_.data,
                            d_mesh_vertices_.bytes(),
                            cudaMemcpyDeviceToDevice));
    }
    else
    {
      std::vector<float3> verts;
      d_mesh_vertices_.download(verts);
      std::vector<float3> warped(verts.size());
      auto h_nodes = warp_field_->download_nodes();
      auto h_transforms = warp_field_->download_transforms();
      for (size_t i = 0; i < verts.size(); i++)
      {
        int ids[K_NEIGHBORS];
        float ws[K_NEIGHBORS];
        skin_point_host(verts[i], h_nodes, h_transforms, warped[i], ids, ws);
      }
      d_mesh_warped_.upload(warped);
    }
    if (profile)
      profile->skin_ms += elapsed_ms(t_skin);

    Mat4 T_cam_world = inverse_rigid_host(camera_pose_);

    auto t_tri_raster = std::chrono::high_resolution_clock::now();
    if (d_raster_ztri_.size() != (size_t)n_pixels)
      d_raster_ztri_.allocate(n_pixels);
    CUDA_CHECK(cudaMemset(d_raster_ztri_.data, 0xff, d_raster_ztri_.bytes()));

    rasterize_triangles_z_kernel<<<grid1d(num_triangles), 256>>>(
        d_mesh_vertices_.data, d_mesh_warped_.data, d_mesh_triangles_.data,
        num_vertices, num_triangles, params_.cam, T_cam_world,
        d_raster_ztri_.data);
    CUDA_CHECK(cudaGetLastError());

    resolve_rasterized_triangles_kernel<<<grid1d(n_pixels), 256>>>(
        d_mesh_vertices_.data, d_mesh_warped_.data, d_mesh_triangles_.data,
        num_vertices, params_.cam, T_cam_world, params_.tsdf.origin,
        params_.tsdf.voxel_size, params_.tsdf.dims, d_raster_ztri_.data,
        d_vertices_live_.data, d_normals_live_.data, d_hit_voxel_idx_.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profile)
      profile->tri_raster_ms += elapsed_ms(t_tri_raster);

    auto t_upload = std::chrono::high_resolution_clock::now();
    if (profile)
    {
      profile->raster_upload_ms += elapsed_ms(t_upload);
      profile->raster_total_ms += elapsed_ms(t_total);
      profile->raster_calls++;
    }
  }

  void mark_optimized_voxels()
  {
    int n_pixels = params_.cam.width * params_.cam.height;
    mark_optimized_voxels_kernel<<<grid1d(n_pixels), 256>>>(
        d_corrs_.data, d_hit_voxel_idx_.data, n_pixels, params_.tsdf.dims,
        frame_count_, d_voxel_opt_counts_.data, d_voxel_opt_frame_.data);
    cudaDeviceSynchronize();
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
      double r = (double)c.normal.x * diff.x + (double)c.normal.y * diff.y +
                 (double)c.normal.z * diff.z;
      double w = std::max(0.0f, c.weight);
      sum += w * r * r;
      wsum += w;
    }
    return wsum > 0.0 ? std::sqrt(sum / wsum)
                      : std::numeric_limits<double>::infinity();
  }

  void upload_scaled_delta_x(float scale)
  {
    std::vector<float> h_dx;
    d_delta_x_.download(h_dx);
    int n_nodes = warp_field_->num_nodes();
    for (int i = 0; i < n_nodes; ++i)
    {
      float *dx = h_dx.data() + i * 6;
      float rn = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
      float tn = std::sqrt(dx[3] * dx[3] + dx[4] * dx[4] + dx[5] * dx[5]);
      float rs = 1.0f;
      float ts = 1.0f;
      if (params_.max_update_rot > 0.0f && rn > params_.max_update_rot)
        rs = params_.max_update_rot / rn;
      if (params_.max_update_trans > 0.0f && tn > params_.max_update_trans)
        ts = params_.max_update_trans / tn;
      for (int k = 0; k < 3; ++k)
        dx[k] *= rs * scale;
      for (int k = 3; k < 6; ++k)
        dx[k] *= ts * scale;
    }
    d_delta_x_trial_.upload(h_dx);
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

    {
      ScopedStdoutSilence silence(params_.quiet);
      cuda_img.Download();

      InitSiftData(frame->sift, params_.sift_max_features, true, true);
      frame->initialized = true;
      float *tmp = AllocSiftTempMemory(params_.cam.width, params_.cam.height,
                                       params_.sift_octaves, false);
      ExtractSift(frame->sift, cuda_img, params_.sift_octaves, 1.0f,
                  params_.sift_threshold, 0.0f, false, tmp);
      FreeSiftTempMemory(tmp);
    }

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

  int augment_correspondences_with_sift(int current_count,
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
    int sift_matches = 0;
    int reject_match_quality = 0;
    int reject_depth = 0;
    int reject_no_src = 0;
    int reject_far = 0;
    int reject_pixel = 0;
    std::vector<cv::Point2f> dbg_sift_src_pts;
    std::vector<cv::Point2f> dbg_sift_dst_pts;
    std::vector<unsigned char> dbg_sift_accepted;
    for (auto &hist : sift_history_)
    {
      {
        ScopedStdoutSilence silence(params_.quiet);
        MatchSiftData(current->sift, hist->sift);
      }
      int frame_added = 0;
      for (int i = 0; i < current->sift.numPts &&
                      frame_added < params_.sift_max_matches_per_frame;
           i++)
      {
        const SiftPoint &sp = current->sift.h_data[i];
        if (sp.match < 0 || sp.match >= hist->sift.numPts)
          continue;
        sift_matches++;
        if (sp.match_error > params_.sift_max_match_error ||
            sp.ambiguity > params_.sift_max_ambiguity)
        {
          reject_match_quality++;
          continue;
        }

        int u = static_cast<int>(std::lround(sp.xpos));
        int v = static_cast<int>(std::lround(sp.ypos));
        if (u < 0 || u >= params_.cam.width || v < 0 || v >= params_.cam.height)
          continue;

        int dst_px = v * params_.cam.width + u;
        float d = h_depth[dst_px];
        if (d <= 0.01f)
        {
          reject_depth++;
          continue;
        }
        float3 dst = params_.cam.unproject(u, v, d);

        const SparseFeature &src_feat = hist->features[sp.match];
        if (src_feat.canonical_src.z <= 0.01f || src_feat.node_ids[0] < 0)
        {
          reject_no_src++;
          continue;
        }
        float3 warped_src = warp_point_host(src_feat.canonical_src, src_feat,
                                            h_nodes, h_transforms);
        float3 dst_world = camera_pose_.transform_point(dst);

        float3 diff =
            make_float3(warped_src.x - dst_world.x, warped_src.y - dst_world.y,
                        warped_src.z - dst_world.z);
        bool accepted_match = host_norm3(diff) <= params_.sift_max_3d_dist;
        bool has_src_uv = false;
        float2 src_uv = make_float2(0.0f, 0.0f);
        Mat4 T_cam_world = inverse_rigid_host(camera_pose_);
        float3 src_cam = T_cam_world.transform_point(warped_src);
        if (src_cam.z > 0.01f)
        {
          src_uv = params_.cam.project(src_cam);
          has_src_uv = std::isfinite(src_uv.x) && std::isfinite(src_uv.y);
        }
        if (accepted_match && params_.sift_max_pixel_dist > 0.0f &&
            has_src_uv)
        {
          float du = src_uv.x - (float)u;
          float dv = src_uv.y - (float)v;
          if (std::sqrt(du * du + dv * dv) > params_.sift_max_pixel_dist)
          {
            reject_pixel++;
            accepted_match = false;
          }
        }
        if (params_.debug_vis)
        {
          if (has_src_uv)
          {
            dbg_sift_src_pts.push_back(cv::Point2f(src_uv.x, src_uv.y));
            dbg_sift_dst_pts.push_back(cv::Point2f((float)u, (float)v));
            dbg_sift_accepted.push_back(accepted_match ? 1 : 0);
          }
        }
        if (!accepted_match)
        {
          reject_far++;
          continue;
        }

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

    if (params_.debug_vis && !params_.quiet)
    {
      std::cout << "  [sift debug] current_pts=" << current->sift.numPts
                << " history=" << sift_history_.size()
                << " raw_matches=" << sift_matches
                << " reject_quality=" << reject_match_quality
                << " reject_depth=" << reject_depth
                << " reject_no_model_src=" << reject_no_src
                << " reject_far=" << reject_far
                << " reject_pixel=" << reject_pixel
                << " added_constraints=" << added << "\n";
      cv::Mat dbg_gray8;
      gray32.convertTo(dbg_gray8, CV_8UC1);
      // Prefer showing SIFT overlays on the input intensity / grayscale image
      // instead of the depth colormap. Build a depth cv::Mat from h_depth
      // and create a color/intensity visualization, then draw SIFT on it.
      cv::Mat depth_mat(params_.cam.height, params_.cam.width, CV_32FC1,
                        const_cast<float *>(h_depth.data()));
      cv::Mat dbg_color = dbg_color_input(current_color_gray_, depth_mat);
      cv::Mat dbg_sift_vis = dbg_sift_matches(dbg_color, dbg_sift_src_pts,
                                              dbg_sift_dst_pts, dbg_sift_accepted);
      cv::imshow("5_sift_matches", dbg_sift_vis);
      cv::waitKey(1);
    }

    d_corrs_.upload(h_corrs);

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
      h_valid = augment_correspondences_with_sift(h_valid);
    }
    if (debug && !params_.quiet)
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
