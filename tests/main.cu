#include <unistd.h>

#include "fusion_app.h"
#include "yaml_helper.h"

namespace fs = std::filesystem;

static std::string format_ms(double ms, int width = 8, int precision = 2)
{
  std::ostringstream ss;
  ss << std::fixed << std::setw(width) << std::setprecision(precision) << ms;
  return ss.str();
}

static std::string format_int(int value, int width = 0)
{
  std::ostringstream ss;
  if (width > 0)
    ss << std::setw(width);
  ss << value;
  return ss.str();
}

/**
 * @brief Create a wireframe bounding box visualization.
 *
 * Creates a LineSet geometry representing a bounding box from min to max points.
 * The box has 12 edges (12 lines) and is rendered as wireframe.
 */
static std::shared_ptr<open3d::geometry::LineSet> create_bbox_lineset(
    float3 min_pt, float3 max_pt, const Eigen::Vector3d &color = {1.0, 0.0, 0.0})
{
  auto lineset = std::make_shared<open3d::geometry::LineSet>();

  // Define the 8 corners of the bounding box
  std::vector<Eigen::Vector3d> vertices = {
      {min_pt.x, min_pt.y, min_pt.z}, // 0: bottom-left-front
      {max_pt.x, min_pt.y, min_pt.z}, // 1: bottom-right-front
      {max_pt.x, max_pt.y, min_pt.z}, // 2: bottom-right-back
      {min_pt.x, max_pt.y, min_pt.z}, // 3: bottom-left-back
      {min_pt.x, min_pt.y, max_pt.z}, // 4: top-left-front
      {max_pt.x, min_pt.y, max_pt.z}, // 5: top-right-front
      {max_pt.x, max_pt.y, max_pt.z}, // 6: top-right-back
      {min_pt.x, max_pt.y, max_pt.z}, // 7: top-left-back
  };

  // Define the 12 edges of the box
  std::vector<Eigen::Vector2i> lines = {
      // Bottom face (z = min)
      {0, 1},
      {1, 2},
      {2, 3},
      {3, 0},
      // Top face (z = max)
      {4, 5},
      {5, 6},
      {6, 7},
      {7, 4},
      // Vertical edges
      {0, 4},
      {1, 5},
      {2, 6},
      {3, 7}};

  // Create uniform colors for all edges
  std::vector<Eigen::Vector3d> colors(lines.size(), color);

  lineset->points_ = vertices;
  lineset->lines_ = lines;
  lineset->colors_ = colors;

  return lineset;
}

/**
 * @brief Main entry point for the DynamicFusion CUDA-based 3D reconstruction
 * pipeline.
 *
 * This application implements a real-time volumetric fusion system that:
 * 1. Reads depth frames from a sequence (or generates synthetic data)
 * 2. Processes each frame through the Dynamic Fusion pipeline
 * 3. Updates a Truncated Signed Distance Function (TSDF) volume
 * 4. Applies non-rigid warp fields to model dynamic deformations
 * 5. Optionally visualizes the reconstructed mesh in real-time
 * 6. Saves canonical and warped mesh reconstructions at checkpoints
 */
int main(int argc, char **argv)
{
  bool cli_quiet_requested = false;
  for (int i = 1; i < argc; i++)
  {
    std::string a = argv[i];
    if (a == "--quiet" || a == "--profile-only")
      cli_quiet_requested = true;
  }

  if (!cli_quiet_requested)
    std::cout << "[startup] DynamicFusion CUDA implementation\n";

  // ─────────────────────────────────────────────────────────────────
  // CONFIGURATION LOADING
  // Load application settings from YAML config file or use defaults
  // ─────────────────────────────────────────────────────────────────

  AppConfig app;
  bool use_config_file = (argc >= 2) && fs::exists(argv[1]);

  if (use_config_file)
  {
    if (!cli_quiet_requested)
      std::cout << "[config] loading: " << argv[1] << "\n";
    app = load_config(argv[1]);
  }
  else
  {
    if (!cli_quiet_requested)
      std::cout << "[config] no YAML provided, generating a synthetic test sequence\n";

    fs::create_directories("/tmp/df_test/depth");

    int W = 640, H = 480;

    // Generate 30 test frames with a moving sphere
    for (int f = 0; f < 30; f++)
    {
      cv::Mat depth(H, W, CV_16U);

      float cx = W / 2 + f * 2;
      float cy = H / 2;
      float r = 150;

      for (int v = 0; v < H; v++)
      {
        for (int u = 0; u < W; u++)
        {
          float dx = u - cx;
          float dy = v - cy;
          float d2 = dx * dx + dy * dy;

          if (d2 < r * r)
          {
            float z = 1.5f - sqrtf(r * r - d2) / 1000.f;
            depth.at<uint16_t>(v, u) = (uint16_t)(z * 1000);
          }
          else
          {
            depth.at<uint16_t>(v, u) = 0;
          }
        }
      }

      std::string fname = "/tmp/df_test/depth/" + std::to_string(f) + ".png";
      cv::imwrite(fname, depth);
    }

    app.seq.path = "/tmp/df_test/depth";
    app.seq.format = DepthSequence::Format::RAW_PNG;
    app.seq.depth_scale = 0.001f;
    app.seq.start_frame = 0;
    app.seq.max_frames = -1;

    app.seq.cam.fx = 570.342f;
    app.seq.cam.fy = 570.342f;
    app.seq.cam.cx = 320.0f;
    app.seq.cam.cy = 240.0f;
    app.seq.cam.width = 640;
    app.seq.cam.height = 480;

    app.use_vis = false;
  }

  // ─────────────────────────────────────────────────────────────────
  // COMMAND-LINE ARGUMENT PROCESSING
  // Override configuration with CLI flags (visualization, debugging, format)
  // ─────────────────────────────────────────────────────────────────

  const int first_option_arg = use_config_file ? 2 : 1;
  for (int i = first_option_arg; i < argc; i++)
  {
    std::string a = argv[i];

    if (a == "--vis")
    {
      app.use_vis = true;
    }
    else if (a == "--no-vis")
    {
      app.use_vis = false;
    }
    else if (a == "--debug-vis")
    {
      app.df.debug_vis = true;
    }
    else if (a == "--debug-every" && i + 1 < argc)
    {
      app.df.debug_every_n = std::stoi(argv[++i]);
    }
    else if (a == "--profile" || a == "--profiler")
    {
      app.df.profile_timings = true;
    }
    else if (a == "--no-profile" || a == "--no-profiler")
    {
      app.df.profile_timings = false;
    }
    else if ((a == "--profile-every" || a == "--profiler-every") && i + 1 < argc)
    {
      app.df.profile_every_n = std::max(1, std::stoi(argv[++i]));
    }
    else if (a == "--quiet")
    {
      app.quiet = true;
    }
    else if (a == "--profile-only")
    {
      app.quiet = true;
      app.df.profile_timings = true;
    }
    else if (a == "--verbose")
    {
      app.quiet = false;
    }
    else if (a == "--max-frames" && i + 1 < argc)
    {
      app.seq.max_frames = std::stoi(argv[++i]);
    }
    else if (a == "tum")
    {
      app.seq.format = DepthSequence::Format::TUM;
    }
    else if (a == "icl")
    {
      app.seq.format = DepthSequence::Format::ICL;
    }
    else if (a == "raw")
    {
      app.seq.format = DepthSequence::Format::RAW_PNG;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // INITIALIZE PIPELINE PARAMETERS
  // Set up the Dynamic Fusion pipeline with camera intrinsics
  // ─────────────────────────────────────────────────────────────────

  DynamicFusionPipeline::Params df_params = app.df;
  app.df.quiet = app.quiet;
  df_params.quiet = app.quiet;
  df_params.cam = app.seq.cam;
  if (df_params.profile_timings)
  {
    std::cout << "[profiler] enabled | every=" << df_params.profile_every_n
              << " frame(s)";
    if (df_params.quiet)
      std::cout << " | quiet=yes";
    std::cout << "\n";
  }

  // ─────────────────────────────────────────────────────────────────
  // CREATE DEPTH SEQUENCE AND FUSION PIPELINE
  // Initialize the frame loader and the main processing pipeline
  // ─────────────────────────────────────────────────────────────────

  DepthSequence sequence(app.seq);
  DynamicFusionPipeline pipeline(df_params);

  // ─────────────────────────────────────────────────────────────────
  // INITIALIZE VISUALIZATION (OPTIONAL)
  // Set up Open3D visualizer for real-time mesh display
  // ─────────────────────────────────────────────────────────────────

  std::shared_ptr<open3d::visualization::Visualizer> vis;
  std::shared_ptr<open3d::geometry::TriangleMesh> vis_mesh;
  std::shared_ptr<open3d::geometry::PointCloud> vis_pc;
  std::shared_ptr<open3d::geometry::PointCloud> vis_nodes;
  std::shared_ptr<open3d::geometry::LineSet> vis_edges;

  if (app.use_vis)
  {
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("DynamicFusion", 2500, 2000);

    vis->GetRenderOption().mesh_show_back_face_ = true;

    vis_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    vis_pc = std::make_shared<open3d::geometry::PointCloud>();
    vis_nodes = std::make_shared<open3d::geometry::PointCloud>();
    vis_edges = std::make_shared<open3d::geometry::LineSet>();

    // auto ref_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.3);
    // vis->AddGeometry(ref_frame);

    float3 tsdf_min = df_params.tsdf.origin;
    float3 tsdf_max = make_float3(
        df_params.tsdf.origin.x + df_params.tsdf.dims.x * df_params.tsdf.voxel_size,
        df_params.tsdf.origin.y + df_params.tsdf.dims.y * df_params.tsdf.voxel_size,
        df_params.tsdf.origin.z + df_params.tsdf.dims.z * df_params.tsdf.voxel_size);
    auto tsdf_bbox = create_bbox_lineset(tsdf_min, tsdf_max, {1.0, 1.0, 0.0});
    vis->AddGeometry(tsdf_bbox);
    if (df_params.debug_vis)
    {
      vis->AddGeometry(vis_nodes);
      vis->AddGeometry(vis_edges);
    }
    vis->GetViewControl().SetFront({0.14, -0.25, -0.96});
    vis->GetViewControl().SetLookat({-0.13, -0.04, 0.68});
    vis->GetViewControl().SetUp({0.13, -0.96, 0.26});
    vis->GetViewControl().SetZoom(0.22);
    vis->PollEvents();
    vis->UpdateRender();
  }

  // ─────────────────────────────────────────────────────────────────
  // MAIN PROCESSING LOOP
  // Process each depth frame through the fusion pipeline
  // ─────────────────────────────────────────────────────────────────

  int frame_idx = 0;
  auto t_start = std::chrono::high_resolution_clock::now();

  while (sequence.has_next())
  {
    auto frame_t0 = std::chrono::high_resolution_clock::now();

    DepthFrame frame = sequence.next();

    if (frame.depth_m.empty())
    {
      std::cerr << "[warn] frame=" << format_int(frame_idx, 4)
                << " empty, skipped\n";
      frame_idx++;
      continue;
    }

    double process_ms = 0.0;
    double vis_ms = 0.0;
    double checkpoint_ms = 0.0;

    auto process_t0 = std::chrono::high_resolution_clock::now();
    pipeline.process_frame(frame.depth_m, frame.color_gray);
    process_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - process_t0)
                     .count();
    frame_idx++;

    if (app.use_vis)
    {
      auto vis_t0 = std::chrono::high_resolution_clock::now();
      if (frame_idx == 1)
      {
        // Use raycast pointcloud visualization by default (faster, smooth normals)
        pipeline.update_o3d_raycast_pointcloud(*vis_pc);
        vis->AddGeometry(vis_pc);
        // if (df_params.debug_vis)
        // {
        // pipeline.update_o3d_nodes(*vis_nodes);
        // vis->AddGeometry(vis_nodes);
        // pipeline.update_o3d_edges(*vis_edges);
        // vis->AddGeometry(vis_edges);
        // }
        if (!app.quiet)
          std::cout << "[vis] initial raycast pointcloud | points=" << vis_pc->points_.size() << "\n";
      }

      if (frame_idx > 1)
      {
        pipeline.update_o3d_raycast_pointcloud(*vis_pc);
        vis->UpdateGeometry(vis_pc);
        // if (df_params.debug_vis)
        // {
        //   pipeline.update_o3d_nodes(*vis_nodes);
        //   vis->UpdateGeometry(vis_nodes);
        //   pipeline.update_o3d_edges(*vis_edges);
        //   vis->UpdateGeometry(vis_edges);
        // }
      }

      // vis->GetViewControl().SetFront({0.14, -0.25, -0.96});
      // vis->GetViewControl().SetLookat({-0.13, -0.04, 0.68});
      // vis->GetViewControl().SetUp({0.13, -0.96, 0.26});
      // vis->GetViewControl().SetZoom(0.22);

      vis->PollEvents();
      vis->UpdateRender();
      vis_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - vis_t0)
                   .count();
    }

    // if (frame_idx % 30 == 0)
    // {
    //   auto checkpoint_t0 = std::chrono::high_resolution_clock::now();
    //   std::string path = "mesh_frame_" + std::to_string(frame_idx);
    //   pipeline.save_mesh_ply(path + "_canonical.ply");
    //   pipeline.save_warped_mesh_ply(path + "_warped.ply");
    //   checkpoint_ms = std::chrono::duration<double, std::milli>(
    //                       std::chrono::high_resolution_clock::now() - checkpoint_t0)
    //                       .count();
    //   if (!app.quiet)
    //   {
    //     std::cout << "[checkpoint] frame=" << format_int(frame_idx, 4)
    //               << " saved canonical + warped meshes"
    //               << " | time=" << format_ms(checkpoint_ms) << " ms\n";
    //   }
    // }

    double frame_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - frame_t0)
                          .count();
    if (!app.quiet)
    {
      std::cout << "[frame " << format_int(frame_idx, 4) << "]"
                << " num nodes=" << pipeline.num_nodes()
                << " total=" << format_ms(frame_ms) << " ms"
                << " | process=" << format_ms(process_ms) << " ms"
                << " | vis=" << format_ms(vis_ms) << " ms"
                << " | checkpoint=" << format_ms(checkpoint_ms) << " ms"
                << " | integrated=" << (pipeline.last_frame_integrated() ? "yes" : "no")
                << "\n--------------------------\n";
    }
    if (frame_idx == 2)
    {
      std::string a;
      std::cout << "Press enter to start processing frames...\n";
      std::getline(std::cin, a);
    }
  }

  if (app.use_vis)
  {
    vis->DestroyVisualizerWindow();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  double avg_fps = total_ms > 0.0 ? (frame_idx * 1000.0 / total_ms) : 0.0;

  if (!app.quiet || df_params.profile_timings)
  {
    std::cout << "[done] frames=" << format_int(frame_idx, 4)
              << " total=" << format_ms(total_ms, 9, 1) << " ms"
              << " | avg_fps=" << format_ms(avg_fps, 7, 1)
              << "\n";
  }

  pipeline.save_mesh_ply("mesh_final_canonical.ply");
  pipeline.save_warped_mesh_ply("mesh_final_warped.ply");

  return 0;
}
