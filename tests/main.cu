#include "fusion_app.h"
int main(int argc, char **argv) {
  std::cout << "=== DynamicFusion CUDA Implementation ===\n\n";

  // ─────────────────────────────────────────────
  // CONFIG
  // ─────────────────────────────────────────────

  AppConfig app;
  bool use_config_file = (argc >= 2);

  if (use_config_file && fs::exists(argv[1])) {
    std::cout << "[Info] Loading config: " << argv[1] << "\n";
    app = load_config(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " [config.yaml] [options]\n";
    std::cout << "[Info] Nessun config YAML valido.\n"
              << "Uso sequenza sintetica...\n\n";

    // ── Sequenza sintetica ─────────────────────
    fs::create_directories("/tmp/df_test/depth");

    int W = 640, H = 480;

    for (int f = 0; f < 30; f++) {
      cv::Mat depth(H, W, CV_16U);

      float cx = W / 2 + f * 2;
      float cy = H / 2;
      float r = 150;

      for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
          float dx = u - cx;
          float dy = v - cy;
          float d2 = dx * dx + dy * dy;

          if (d2 < r * r) {
            float z = 1.5f - sqrtf(r * r - d2) / 1000.f;
            depth.at<uint16_t>(v, u) = (uint16_t)(z * 1000);
          } else {
            depth.at<uint16_t>(v, u) = 0;
          }
        }
      }

      std::string fname = "/tmp/df_test/depth/" + std::to_string(f) + ".png";

      cv::imwrite(fname, depth);
    }

    // fallback config
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

  // ─────────────────────────────────────────────
  // CLI overrides (solo vis + format override)
  // ─────────────────────────────────────────────

  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];

    if (a == "--vis") {
      app.use_vis = true;
    } else if (a == "--no-vis") {
      app.use_vis = false;
    } else if (a == "--debug-vis") {
      app.df.debug_vis = true;
    } else if (a == "--debug-every" && i + 1 < argc) {
      app.df.debug_every_n = std::stoi(argv[++i]);
    } else if (a == "--max-frames" && i + 1 < argc) {
      app.seq.max_frames = std::stoi(argv[++i]);
    } else if (a == "tum") {
      app.seq.format = DepthSequence::Format::TUM;
    } else if (a == "icl") {
      app.seq.format = DepthSequence::Format::ICL;
    } else if (a == "raw") {
      app.seq.format = DepthSequence::Format::RAW_PNG;
    }
  }

  // ─────────────────────────────────────────────
  // PIPELINE PARAMS
  // ─────────────────────────────────────────────

  DynamicFusionPipeline::Params df_params = app.df;
  df_params.cam = app.seq.cam;

  // ─────────────────────────────────────────────
  // SEQUENCE + PIPELINE
  // ─────────────────────────────────────────────

  DepthSequence sequence(app.seq);
  DynamicFusionPipeline pipeline(df_params);

  // ─────────────────────────────────────────────
  // VISUALIZER
  // ─────────────────────────────────────────────

  std::shared_ptr<open3d::visualization::Visualizer> vis;
  std::shared_ptr<open3d::geometry::TriangleMesh> vis_mesh;

  if (app.use_vis) {
    vis = std::make_shared<open3d::visualization::Visualizer>();
    vis->CreateVisualizerWindow("DynamicFusion", 1920, 1080);

    vis->GetRenderOption().mesh_show_back_face_ = true;

    vis_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

    auto ref_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.3);

    vis->AddGeometry(ref_frame);
  }

  // ── Loop principale ───────────────────────

  int frame_idx = 0;
  auto t_start = std::chrono::high_resolution_clock::now();

  while (sequence.has_next()) {
    auto t0 = std::chrono::high_resolution_clock::now();
    DepthFrame frame = sequence.next();

    if (frame.depth_m.empty()) {
      std::cerr << "[Warning] Frame " << frame_idx << " vuoto, skip.\n";
      frame_idx++;
      continue;
    }

    pipeline.process_frame(frame.depth_m, frame.color_gray);
    frame_idx++;

    // Vis update: rebuild the TSDF mesh only when fusion actually changed it,
    // but keep the window event loop alive every frame.
    if (app.use_vis) {
      if (frame_idx == 1) {
        pipeline.update_o3d_mesh(*vis_mesh);
        std::cout << "Mesh iniziale: " << vis_mesh->vertices_.size()
                  << " vertici, " << vis_mesh->triangles_.size()
                  << " triangoli\n";
        vis->AddGeometry(vis_mesh);
      }
      if (frame_idx > 1 && pipeline.last_frame_integrated()) {
        pipeline.update_o3d_mesh(*vis_mesh);
        vis->UpdateGeometry(vis_mesh);
      }
      // vis->GetRenderOption().mesh_show_wireframe_ = true;
      vis->PollEvents();
      vis->UpdateRender();
    }

    // Salva mesh ogni 30 frame
    if (frame_idx % 30 == 0) {
      std::string path = "mesh_frame_" + std::to_string(frame_idx);
      pipeline.save_mesh_ply(path + "_canonical.ply");
      pipeline.save_warped_mesh_ply(path + "_warped.ply");
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Frame " << std::setw(4) << frame_idx << " processato in "
              << std::fixed << std::setprecision(1) << ms << " ms\n";
    std::cout << "------------------" << std::endl;
  }

  if (app.use_vis) {
    vis->DestroyVisualizerWindow();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_s = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "\n=== Completato ===\n";
  std::cout << "Frame processati: " << frame_idx << "\n";
  std::cout << "Tempo totale:     " << std::fixed << std::setprecision(2)
            << total_s << " s\n";
  std::cout << "FPS medio:        " << std::setprecision(1)
            << frame_idx / total_s << "\n";

  // Mesh finale
  pipeline.save_mesh_ply("mesh_final_canonical.ply");
  pipeline.save_warped_mesh_ply("mesh_final_warped.ply");

  return 0;
}
