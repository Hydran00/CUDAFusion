#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "fusion_app.h"

static std::vector<int> read_vec_i(const YAML::Node &n)
{
  std::vector<int> v;
  for (auto x : n)
    v.push_back(x.as<int>());
  return v;
}

static std::vector<float> read_vec_f(const YAML::Node &n)
{
  std::vector<float> v;
  for (auto x : n)
    v.push_back(x.as<float>());
  return v;
}

static float3 read_float3(const YAML::Node &n)
{
  return make_float3(n[0].as<float>(), n[1].as<float>(), n[2].as<float>());
}

struct AppConfig
{
  DepthSequence::Config seq;
  DynamicFusionPipeline::Params df;
  bool use_vis = false;
  bool quiet = false;
};

static AppConfig load_config(const std::string &path)
{
  YAML::Node cfg = YAML::LoadFile(path);
  AppConfig out;

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

  auto c = cfg["camera"];
  out.seq.cam.fx = c["fx"].as<float>();
  out.seq.cam.fy = c["fy"].as<float>();
  out.seq.cam.cx = c["cx"].as<float>();
  out.seq.cam.cy = c["cy"].as<float>();
  out.seq.cam.width = c["width"].as<int>();
  out.seq.cam.height = c["height"].as<int>();
  out.df.cam = out.seq.cam;

  auto t = cfg["tsdf"];
  auto dims = read_vec_i(t["dims"]);
  out.df.tsdf.dims = {dims[0], dims[1], dims[2]};
  out.df.tsdf.voxel_size = t["voxel_size"].as<float>();
  out.df.tsdf.truncation = t["truncation"].as<float>();
  out.df.tsdf.origin = read_float3(t["origin"]);

  auto sol = cfg["solver"];
  out.df.solver.gn_iterations = sol["gn_iterations"].as<int>();
  out.df.solver.pcg_iterations = sol["pcg_iterations"].as<int>();
  out.df.solver.lambda_smooth = sol["lambda_smooth"].as<float>();
  if (sol["lambda_damping"])
    out.df.solver.lambda_damping = sol["lambda_damping"].as<float>();

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

  if (cfg["profiler"])
  {
    auto profiler = cfg["profiler"];
    if (profiler["enabled"])
      out.df.profile_timings = profiler["enabled"].as<bool>();
    if (profiler["every_n"])
      out.df.profile_every_n = std::max(1, profiler["every_n"].as<int>());
    if (profiler["quiet"])
      out.quiet = profiler["quiet"].as<bool>();
  }

  auto b = cfg["bbox"];
  out.df.bbox.enabled = b["enabled"].as<bool>();
  out.df.bbox.min_pt = read_float3(b["min_pt"]);
  out.df.bbox.max_pt = read_float3(b["max_pt"]);

  out.use_vis = cfg["visualizer"]["enabled"].as<bool>();
  out.df.quiet = out.quiet;

  return out;
}
