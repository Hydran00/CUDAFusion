#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "types.h"
#include "tsdf_volume.h"
#include "tsdf_kernels.h"
#include "warp_field.h"
#include "solver.h"

#include <open3d/Open3D.h>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  Depth sequence loader
//  Supporta: TUM RGB-D, ICL-NUIM, cartella raw
// ─────────────────────────────────────────────

struct DepthFrame
{
    cv::Mat depth_m; // float32, metri
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
        // depth images uint16, divide per 5000 → metres
        std::string assoc_path = cfg_.path + "/associations.txt";
        std::ifstream assoc(assoc_path);
        if (!assoc.is_open())
        {
            // Fallback: scansiona cartella depth/
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
            raw.convertTo(depth_m, CV_32F, 1.0f / 5000.0f); // TUM scale

            frames_.push_back({depth_m, ts_depth, count});
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

    void load_raw_dir(const std::string &dir, const std::string &ext)
    {
        if (!fs::exists(dir))
        {
            std::cerr << "[DepthSequence] Directory not found: " << dir << "\n";
            return;
        }

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
                raw.convertTo(depth_m, CV_32F, cfg_.depth_scale);
            }
            else
            {
                raw.convertTo(depth_m, CV_32F);
            }

            // Filtra valori invalidi
            cv::threshold(depth_m, depth_m, 0.1f, 0, cv::THRESH_TOZERO);
            cv::threshold(depth_m, depth_m, 6.0f, 0, cv::THRESH_TOZERO_INV);

            frames_.push_back({depth_m, (double)count, count});
            count++;
        }
    }
};

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
        float node_radius = 0.05f;
        float node_min_dist = 0.025f;
        float dist_threshold = 0.05f; // ICP distanza max
        float angle_threshold = 0.7f; // cos(angolo) min normali
        int max_nodes = 4096;
        int max_corrs = 300000;
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
        d_corrs_.allocate(p.max_corrs);
        d_delta_x_.allocate(p.max_nodes * 6);

        camera_pose_ = Mat4::identity();
    }

    void process_frame(const cv::Mat &depth_m)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Upload depth
        {
            std::vector<float> h_depth(
                (float *)depth_m.datastart,
                (float *)depth_m.dataend);
            d_depth_.upload(h_depth);
        }

        // 2. Calcola normali dal depth
        compute_normals();

        if (frame_count_ == 0)
        {
            // Primo frame: solo integra
            volume_->integrate(d_depth_, d_depth_normals_,
                               params_.cam, camera_pose_);
            initialize_node_graph();
        }
        else
        {
            // Warp field: salva trasformazioni precedenti (warm start)
            warp_field_->save_transforms();

            // 3. Raycasting modello canonico → live surface
            volume_->raycast(
                d_vertices_live_, d_normals_live_,
                params_.cam, camera_pose_,
                warp_field_->device_nodes(),
                warp_field_->device_transforms(),
                warp_field_->num_nodes(),
                d_voxel_knn_.data, d_voxel_knn_w_.data);

            // 4. Trova corrispondenze ICP
            int num_corrs = find_correspondences();
            std::cout << "  Corrispondenze valide: " << num_corrs << "\n";

            if (num_corrs > 100)
            {
                // 5. Ottimizzazione Gauss-Newton
                solver_->solve(
                    d_corrs_, num_corrs,
                    warp_field_->device_nodes(),
                    warp_field_->device_transforms(),
                    warp_field_->num_nodes(),
                    d_delta_x_);

                // 6. Applica incremento
                warp_field_->apply_twist_increment(d_delta_x_);
            }

            // 7. Integra depth con warp corrente
            volume_->integrate(
                d_depth_, d_depth_normals_,
                params_.cam, camera_pose_,
                warp_field_->device_nodes(),
                warp_field_->device_transforms(),
                warp_field_->num_nodes(),
                d_voxel_knn_.data, d_voxel_knn_w_.data);

            // 8. Aggiorna grafo con nuovi nodi
            if (frame_count_ % 5 == 0) // ogni 5 frame
                update_node_graph();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[Frame " << std::setw(4) << frame_count_
                  << "] " << std::fixed << std::setprecision(1)
                  << ms << " ms | nodes: " << warp_field_->num_nodes() << "\n";

        frame_count_++;
    }

    // Fill existing O3D mesh in-place (for vis update)
    void update_o3d_mesh(open3d::geometry::TriangleMesh &mesh) const
    {
        std::vector<float3> verts, norms;
        std::vector<int3> tris;
        volume_->extract_surface(verts, norms, tris);
        std::cout << "verts: " << verts.size()
                  << " tris: " << tris.size() << std::endl;

        mesh.vertices_.clear();
        mesh.triangles_.clear();
        mesh.vertices_.reserve(verts.size());
        mesh.triangles_.reserve(tris.size());
        for (auto &v : verts)
            mesh.vertices_.push_back({v.x, v.y, v.z});
        for (auto &t : tris)
            mesh.triangles_.push_back({t.x, t.y, t.z});
        mesh.ComputeVertexNormals();
        mesh.PaintUniformColor({1.0, 1.0, 0.0});
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

        std::cout << "[PLY] Salvato: " << path
                  << " (" << verts.size() << " vertici)\n";
    }

private:
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

    Mat4 camera_pose_;

    void compute_normals()
    {
        dim3 block(16, 16);
        dim3 grid(
            (params_.cam.width + block.x - 1) / block.x,
            (params_.cam.height + block.y - 1) / block.y);

        compute_depth_normals_kernel<<<grid, block>>>(
            d_depth_.data, d_depth_normals_.data,
            params_.cam.width, params_.cam.height, params_.cam);
        cudaDeviceSynchronize();
    }

    void initialize_node_graph()
    {
        std::vector<float3> verts, norms;
        std::vector<int3> tris;
        volume_->extract_surface(verts, norms, tris);

        int added = warp_field_->add_nodes_from_surface(verts, params_.node_min_dist);
        warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);

        std::cout << "[Init] Nodi aggiunti: " << added << "\n";
    }

    void update_node_graph()
    {
        std::vector<float3> verts, norms;
        std::vector<int3> tris;
        volume_->extract_surface(verts, norms, tris);

        int added = warp_field_->add_nodes_from_surface(verts, params_.node_min_dist);
        if (added > 0)
        {
            warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
            std::cout << "[Update] Nuovi nodi: " << added << "\n";
        }
    }

    int find_correspondences()
    {
        // Versione semplificata: usa un kernel custom
        // (omesso per brevità — in produzione: find_correspondences_kernel)
        // Restituisce numero di corrispondenze valide
        // Per ora: placeholder che conta i pixel live validi
        std::vector<float3> h_verts(d_vertices_live_.size());
        cudaMemcpy(h_verts.data(), d_vertices_live_.data,
                   d_vertices_live_.bytes(), cudaMemcpyDeviceToHost);

        int count = 0;
        for (const auto &v : h_verts)
            if (v.x * v.x + v.y * v.y + v.z * v.z > 1e-6f)
                count++;
        return count;
    }
};

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────

int main(int argc, char **argv)
{
    std::cout << "=== DynamicFusion CUDA Implementation ===\n\n";

    // ── Configurazione ────────────────────────

    DepthSequence::Config seq_cfg;

    if (argc >= 2)
    {
        seq_cfg.path = argv[1];
    }
    else
    {
        // Dataset di test sintetico: genera depth maps procedurali
        std::cout << "[Info] Nessun dataset specificato.\n"
                  << "  Uso: " << argv[0] << " <path_dataset> [format]\n"
                  << "  Formati: tum (default), icl, raw\n"
                  << "  Generando sequenza sintetica...\n\n";

        // Crea cartella temporanea con depth maps sintetiche
        fs::create_directories("/tmp/df_test/depth");
        int W = 640, H = 480;
        for (int f = 0; f < 30; f++)
        {
            cv::Mat depth(H, W, CV_16U);
            // Sfera che si muove lentamente
            float cx = W / 2 + f * 2, cy = H / 2, r = 150;
            for (int v = 0; v < H; v++)
                for (int u = 0; u < W; u++)
                {
                    float dx = u - cx, dy = v - cy;
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
            std::string fname = "/tmp/df_test/depth/" +
                                std::to_string(f) + ".png";
            cv::imwrite(fname, depth);
        }
        seq_cfg.path = "/tmp/df_test/depth";
        seq_cfg.format = DepthSequence::Format::RAW_PNG;
    }

    // Parse remaining args: format + --vis
    bool use_vis = false;
    for (int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--vis")
        {
            use_vis = true;
        }
        else if (a == "tum")
        {
            seq_cfg.format = DepthSequence::Format::TUM;
        }
        else if (a == "icl")
        {
            seq_cfg.format = DepthSequence::Format::ICL;
        }
        else if (a == "raw")
        {
            seq_cfg.format = DepthSequence::Format::RAW_PNG;
        }
    }

    // Intrinseci camera (default: TUM fr1)
    seq_cfg.cam.fx = 570.342f;
    seq_cfg.cam.fy = 570.342f;
    seq_cfg.cam.cx = 320.0f;
    seq_cfg.cam.cy = 240.0f;
    seq_cfg.cam.width = 640;
    seq_cfg.cam.height = 480;

    // ── Pipeline params ───────────────────────

    DynamicFusionPipeline::Params df_params;
    df_params.cam = seq_cfg.cam;

    df_params.tsdf.dims = {256, 256, 256};
    df_params.tsdf.voxel_size = 0.01f;
    df_params.tsdf.truncation = 0.03f;
    df_params.tsdf.origin = {-0.768f, -0.768f, 0.3f};

    df_params.solver.gn_iterations = 3;
    df_params.solver.pcg_iterations = 10;
    df_params.solver.lambda_smooth = 5.0f;

    df_params.node_radius = 0.05f;
    df_params.node_min_dist = 0.025f;

    // ── Carica sequenza ───────────────────────

    DepthSequence sequence(seq_cfg);

    // ── Inizializza pipeline ──────────────────

    DynamicFusionPipeline pipeline(df_params);

    // ── Visualizer setup ──────────────────────

    std::shared_ptr<open3d::visualization::Visualizer> vis;
    std::shared_ptr<open3d::geometry::TriangleMesh> vis_mesh;

    if (use_vis)
    {
        vis = std::make_shared<open3d::visualization::Visualizer>();
        vis->CreateVisualizerWindow("DynamicFusion", 1920, 1080);
        vis->GetRenderOption().mesh_show_back_face_ = true;
        vis_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    }

    // ── Loop principale ───────────────────────

    int frame_idx = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (sequence.has_next())
    {
        DepthFrame frame = sequence.next();

        if (frame.depth_m.empty())
        {
            std::cerr << "[Warning] Frame " << frame_idx << " vuoto, skip.\n";
            frame_idx++;
            continue;
        }

        pipeline.process_frame(frame.depth_m);
        frame_idx++;

        // Vis update every frame
        if (use_vis)
        {
            if (frame_idx == 1)
            {
                pipeline.update_o3d_mesh(*vis_mesh);
                std::cout << "Mesh iniziale: " << vis_mesh->vertices_.size()
                          << " vertici, " << vis_mesh->triangles_.size() << " triangoli\n";
                vis->AddGeometry(vis_mesh);
            }
            pipeline.update_o3d_mesh(*vis_mesh);
            vis->UpdateGeometry(vis_mesh);
            // vis->GetRenderOption().mesh_show_wireframe_ = true;
            vis->PollEvents();
            vis->UpdateRender();
        }

        // Salva mesh ogni 30 frame
        if (frame_idx % 30 == 0)
        {
            std::string path = "mesh_frame_" +
                               std::to_string(frame_idx) + ".ply";
            pipeline.save_mesh_ply(path);
        }
    }

    if (use_vis)
        vis->DestroyVisualizerWindow();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== Completato ===\n";
    std::cout << "Frame processati: " << frame_idx << "\n";
    std::cout << "Tempo totale:     " << std::fixed << std::setprecision(2)
              << total_s << " s\n";
    std::cout << "FPS medio:        " << std::setprecision(1)
              << frame_idx / total_s << "\n";

    // Mesh finale
    pipeline.save_mesh_ply("mesh_final.ply");

    return 0;
}
