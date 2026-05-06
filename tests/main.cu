#include <unistd.h>

#include "fusion_app.h"

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
int main(int argc, char **argv) {
  std::cout << "=== DynamicFusion CUDA Implementation ===\n\n";

  // ─────────────────────────────────────────────────────────────────
  // CONFIGURATION LOADING
  // Load application settings from YAML config file or use defaults
  // ─────────────────────────────────────────────────────────────────

  AppConfig app;
  bool use_config_file = (argc >= 2);

  if (use_config_file && fs::exists(argv[1])) {
    // User provided a valid config file path
    std::cout << "[Info] Loading config: " << argv[1] << "\n";
    app = load_config(argv[1]);
  } else {
    // No valid config provided; generate synthetic depth sequence for testing
    std::cout << "Usage: " << argv[0] << " [config.yaml] [options]\n";
    std::cout << "[Info] No valid config YAML file provided.\n"
              << "Generating synthetic depth sequence for testing...\n\n";

    // ── Generate Synthetic Depth Sequence ──────────────────────────
    // Creates a sequence of 30 depth frames with a moving sphere object
    fs::create_directories("/tmp/df_test/depth");

    int W = 640, H = 480;

    // Generate 30 test frames with a moving sphere
    for (int f = 0; f < 30; f++) {
      cv::Mat depth(H, W, CV_16U);

      // Compute sphere center position (moves from left to right)
      float cx = W / 2 + f * 2;
      float cy = H / 2;
      float r = 150;  // Sphere radius in pixels

      // Render a synthetic sphere by computing depth at each pixel
      for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
          float dx = u - cx;
          float dy = v - cy;
          float d2 = dx * dx + dy * dy;

          if (d2 < r * r) {
            // Inside the sphere: compute depth using sphere equation
            // z = 1.5 - sqrt(r^2 - (dx^2 + dy^2)) / 1000
            float z = 1.5f - sqrtf(r * r - d2) / 1000.f;
            depth.at<uint16_t>(v, u) = (uint16_t)(z * 1000);
          } else {
            // Outside the sphere: no depth (0 = invalid/background)
            depth.at<uint16_t>(v, u) = 0;
          }
        }
      }

      // Save the depth frame as PNG (uint16 format)
      std::string fname = "/tmp/df_test/depth/" + std::to_string(f) + ".png";
      cv::imwrite(fname, depth);
    }

    // Configure the application to use the synthetic depth sequence
    app.seq.path = "/tmp/df_test/depth";
    app.seq.format = DepthSequence::Format::RAW_PNG;
    app.seq.depth_scale =
        0.001f;  // Convert uint16 values to meters (1mm per unit)
    app.seq.start_frame = 0;
    app.seq.max_frames = -1;  // Process all frames

    // Camera intrinsic parameters (simulated camera)
    app.seq.cam.fx = 570.342f;  // Focal length X (pixels)
    app.seq.cam.fy = 570.342f;  // Focal length Y (pixels)
    app.seq.cam.cx = 320.0f;    // Principal point X (pixels)
    app.seq.cam.cy = 240.0f;    // Principal point Y (pixels)
    app.seq.cam.width = 640;
    app.seq.cam.height = 480;

    app.use_vis = false;
  }

  // ─────────────────────────────────────────────────────────────────
  // COMMAND-LINE ARGUMENT PROCESSING
  // Override configuration with CLI flags (visualization, debugging, format)
  // ─────────────────────────────────────────────────────────────────

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];

    if (a == "--vis") {
      // Enable real-time visualization using Open3D
      app.use_vis = true;
    } else if (a == "--no-vis") {
      // Disable real-time visualization (faster processing)
      app.use_vis = false;
    } else if (a == "--debug-vis") {
      // Enable debug visualization for intermediate results
      app.df.debug_vis = true;
    } else if (a == "--debug-every" && i + 1 < argc) {
      // Save debug data every N frames
      app.df.debug_every_n = std::stoi(argv[++i]);
    } else if (a == "--max-frames" && i + 1 < argc) {
      // Limit number of frames to process (useful for quick tests)
      app.seq.max_frames = std::stoi(argv[++i]);
    } else if (a == "tum") {
      // TUM dataset format (RGB-D benchmark dataset)
      app.seq.format = DepthSequence::Format::TUM;
    } else if (a == "icl") {
      // ICL dataset format (Indoor scene dataset)
      app.seq.format = DepthSequence::Format::ICL;
    } else if (a == "raw") {
      // Raw PNG format (single channel uint16 depth images)
      app.seq.format = DepthSequence::Format::RAW_PNG;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // INITIALIZE PIPELINE PARAMETERS
  // Set up the Dynamic Fusion pipeline with camera intrinsics
  // ─────────────────────────────────────────────────────────────────

  DynamicFusionPipeline::Params df_params = app.df;
  df_params.cam = app.seq.cam;

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

  if (app.use_vis) {
    // Create visualizer window (1920x1080 resolution)
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("DynamicFusion", 1920, 1080);

    // Enable back-face culling for better visualization
    vis->GetRenderOption().mesh_show_back_face_ = true;

    // Initialize triangle mesh object to hold the fused reconstruction
    vis_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // Add a reference coordinate frame to the visualization
    auto ref_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.3);
    vis->AddGeometry(ref_frame);
  }

  // ─────────────────────────────────────────────────────────────────
  // MAIN PROCESSING LOOP
  // Process each depth frame through the fusion pipeline
  // ─────────────────────────────────────────────────────────────────

  int frame_idx = 0;
  auto t_start = std::chrono::high_resolution_clock::now();

  while (sequence.has_next()) {
    // Measure processing time for this frame
    auto t0 = std::chrono::high_resolution_clock::now();

    // Load the next depth frame from the sequence
    DepthFrame frame = sequence.next();

    // Skip empty frames (invalid or missing data)
    if (frame.depth_m.empty()) {
      std::cerr << "[Warning] Frame " << frame_idx << " is empty, skipping.\n";
      frame_idx++;
      continue;
    }

    // Process the depth frame:
    // 1. Align the depth map with camera pose estimation
    // 2. Update TSDF volume with the new depth data
    // 3. Extract mesh from the updated TSDF
    // 4. Apply non-rigid warp fields if motion detected
    pipeline.process_frame(frame.depth_m, frame.color_gray);
    frame_idx++;

    // ── Real-time Visualization Update ────────────────────────────
    // Update the displayed mesh only when fusion changes, but keep
    // the window responsive by polling events every frame
    if (app.use_vis) {
      // On first frame: extract initial mesh and add to visualization
      if (frame_idx == 1) {
        pipeline.update_o3d_mesh(*vis_mesh);
        std::cout << "Initial mesh: " << vis_mesh->vertices_.size()
                  << " vertices, " << vis_mesh->triangles_.size()
                  << " triangles\n";
        vis->AddGeometry(vis_mesh);
      }

      // On subsequent frames: update every frame so accepted warp updates are
      // visible even when TSDF integration is skipped.
      if (frame_idx > 1) {
        pipeline.update_o3d_mesh(*vis_mesh);
        vis->UpdateGeometry(vis_mesh);
      }

      // Keep the visualization window responsive
      vis->PollEvents();
      vis->UpdateRender();
      // sleep_for(10ms) can be added here if the loop is too fast and CPU usage
      // is high

      sleep(0.050);
    }

    // ── Checkpoint: Save mesh every 30 frames ──────────────────────
    // Save both canonical (original) and warped (deformed) meshes
    if (frame_idx % 30 == 0) {
      std::string path = "mesh_frame_" + std::to_string(frame_idx);
      pipeline.save_mesh_ply(path + "_canonical.ply");
      pipeline.save_warped_mesh_ply(path + "_warped.ply");
    }

    // Print timing statistics for this frame
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Frame " << std::setw(4) << frame_idx << " processed in "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "------------------" << std::endl;
  }

  // ─────────────────────────────────────────────────────────────────
  // CLEANUP
  // Close visualization window if active
  // ─────────────────────────────────────────────────────────────────

  if (app.use_vis) {
    vis->DestroyVisualizerWindow();
  }

  // ─────────────────────────────────────────────────────────────────
  // FINAL STATISTICS AND OUTPUT
  // Print processing summary and save final reconstructions
  // ─────────────────────────────────────────────────────────────────

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_s = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "\n=== Processing Complete ===\n";
  std::cout << "Frames processed: " << frame_idx << "\n";
  std::cout << "Total time:       " << std::fixed << std::setprecision(2)
            << total_s << " s\n";
  std::cout << "Average FPS:      " << std::setprecision(1)
            << frame_idx / total_s << "\n";

  // Save final reconstructed meshes (both canonical and warped versions)
  // - canonical.ply: mesh in the rest frame (undeformed)
  // - warped.ply: mesh with non-rigid deformations applied
  pipeline.save_mesh_ply("mesh_final_canonical.ply");
  pipeline.save_warped_mesh_ply("mesh_final_warped.ply");

  return 0;
}
