#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <limits>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "types.h"
#include "tsdf_volume.h"
#include "tsdf_kernels.h"
#include "warp_field.h"
#include "solver.h"

#include <open3d/Open3D.h>
#include "yaml_helper.h"
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

            frames_.push_back({depth_m, (double)count, count});
            count++;
        }
    }
};

// ─────────────────────────────────────────────
//  Debug visualization helpers
// ─────────────────────────────────────────────

static cv::Mat dbg_depth(const std::vector<float> &d, int W, int H, float max_m = 4.0f)
{
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

static cv::Mat dbg_normals(const std::vector<float3> &n, int W, int H)
{
    cv::Mat img(H, W, CV_8UC3);
    for (int i = 0; i < H * W; i++) {
        float len = sqrtf(n[i].x * n[i].x + n[i].y * n[i].y + n[i].z * n[i].z);
        if (len < 0.5f) {
            img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] = 0;
        } else {
            img.data[i * 3 + 0] = (uint8_t)((n[i].z * 0.5f + 0.5f) * 255); // B=z
            img.data[i * 3 + 1] = (uint8_t)((n[i].y * 0.5f + 0.5f) * 255); // G=y
            img.data[i * 3 + 2] = (uint8_t)((n[i].x * 0.5f + 0.5f) * 255); // R=x
        }
    }
    return img;
}

static cv::Mat dbg_verts(const std::vector<float3> &verts, const CameraIntrinsics &cam)
{
    cv::Mat depth(cam.height, cam.width, CV_32F, cv::Scalar(0.f));
    for (const auto &v : verts) {
        if (v.z <= 0.01f) continue;
        int u  = (int)(cam.fx * v.x / v.z + cam.cx);
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
    cv::putText(color, "live surface (raycasted)", {8, 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
    return color;
}

static cv::Mat dbg_corrs(const std::vector<Correspondence> &corrs,
                         const CameraIntrinsics &cam, int n_valid)
{
    cv::Mat img(cam.height, cam.width, CV_8UC3, cv::Scalar(15, 15, 15));
    int drawn = 0;
    for (const auto &c : corrs) {
        if (!c.valid) continue;
        if (c.dst.z > 0.01f) {
            int u = (int)(cam.fx * c.dst.x / c.dst.z + cam.cx);
            int v = (int)(cam.fy * c.dst.y / c.dst.z + cam.cy);
            if (u >= 0 && u < cam.width && v >= 0 && v < cam.height)
                cv::circle(img, {u, v}, 1, {0, 0, 200}, -1);
        }
        if (c.src.z > 0.01f) {
            int u = (int)(cam.fx * c.src.x / c.src.z + cam.cx);
            int v = (int)(cam.fy * c.src.y / c.src.z + cam.cy);
            if (u >= 0 && u < cam.width && v >= 0 && v < cam.height)
                cv::circle(img, {u, v}, 1, {0, 200, 0}, -1);
        }
        if (++drawn >= 10000) break;
    }
    cv::putText(img, "G=live  R=depth  valid=" + std::to_string(n_valid),
                {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
    return img;
}

static cv::Mat dbg_delta_x(const std::vector<float> &dx, int n_nodes)
{
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
        int h  = (int)(norms[i] / max_norm * bar_h);
        int x0 = pad + i * (bar_w + pad);
        cv::rectangle(img, {x0, bar_h - h}, {x0 + bar_w, bar_h}, {0, 200, 200}, -1);
    }
    cv::putText(img, "delta_x norm/node  max=" + std::to_string(max_norm),
                {4, bar_h + 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200, 200, 200}, 1);
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
        int min_valid_corrs = 2000;
        int max_nodes = 4096;
        int max_corrs = 300000;
        int node_update_every_n = 5;
        float max_dx_mean = 0.03f;
        BBoxFilter bbox;
        bool integrate_warped = false;
        bool debug_vis = false;
        int  debug_every_n = 1;
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
        d_corrs_.allocate(n_pixels);
        d_delta_x_.allocate(p.max_nodes * 6);
        d_hit_voxel_idx_.allocate(n_pixels);
        d_num_valid_.allocate(1);
        d_corr_stats_.allocate(6);

        camera_pose_ = Mat4::identity();
    }

    void process_frame(const cv::Mat &depth_m)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        const bool do_dbg = params_.debug_vis && (frame_count_ % params_.debug_every_n == 0);

        if (frame_count_ == 0)
            print_cpu_depth_stats(depth_m, "cpu depth input");

        // 1. Upload depth
        {
            std::vector<float> h_depth(
                (float *)depth_m.datastart,
                (float *)depth_m.dataend);
            if (frame_count_ == 0)
                print_host_depth_stats(h_depth, "host upload vector");
            d_depth_.upload(h_depth);
        }

        if (do_dbg)
        {
            std::vector<float> h_d;
            d_depth_.download(h_d);
            cv::Mat vis = dbg_depth(h_d, params_.cam.width, params_.cam.height);
            cv::putText(vis, "frame " + std::to_string(frame_count_),
                        {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {220, 220, 220}, 1);
            cv::imshow("1_depth_raw", vis);
            cv::waitKey(1);
        }

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
            cv::imshow("2_depth_normals", dbg_normals(h_n, params_.cam.width, params_.cam.height));
            cv::waitKey(1);
        }

        if (frame_count_ == 0)
        {
            // Primo frame: solo integra
            volume_->integrate(d_depth_, d_depth_normals_,
                               params_.cam, camera_pose_);
            print_tsdf_stats("tsdf after first integrate");
            initialize_node_graph();
        }
        else
        {
            bool warp_update_ok = false;

            // Warp field: salva trasformazioni precedenti (warm start)
            warp_field_->save_transforms();

            // 3. Raycasting modello canonico → live surface
            volume_->raycast(
                d_vertices_live_, d_normals_live_,
                params_.cam, camera_pose_,
                warp_field_->device_nodes(),
                warp_field_->device_transforms(),
                warp_field_->num_nodes(),
                d_voxel_knn_.data, d_voxel_knn_w_.data,
                d_hit_voxel_idx_.data);

            if (do_dbg)
            {
                std::vector<float3> h_v;
                d_vertices_live_.download(h_v);
                cv::imshow("3_live_surface", dbg_verts(h_v, params_.cam));
                cv::waitKey(1);
            }

            // 3b. Bounding box filter on live vertices
            if (params_.bbox.enabled)
                apply_bbox_filter(d_vertices_live_);

            // 4. Trova corrispondenze ICP
            int num_corrs = find_correspondences(do_dbg);
            std::cout << "  Corrispondenze valide: " << num_corrs << "\n";

            if (do_dbg)
            {
                std::vector<Correspondence> h_c;
                d_corrs_.download(h_c);
                cv::imshow("4_correspondences", dbg_corrs(h_c, params_.cam, num_corrs));
                cv::waitKey(1);
            }

            if (num_corrs >= params_.min_valid_corrs)
            {
                // 5. Ottimizzazione Gauss-Newton
                // Pass full pixel array size; solver skips invalid entries via corr.valid
                int n_pixels = params_.cam.width * params_.cam.height;
                solver_->solve(
                    d_corrs_, n_pixels,
                    warp_field_->device_nodes(),
                    warp_field_->device_transforms(),
                    warp_field_->num_nodes(),
                    d_delta_x_);

                // 6. Valida incremento prima di applicarlo.
                {
                    std::vector<float> h_dx;
                    d_delta_x_.download(h_dx);
                    float norm2 = 0;
                    int n6 = warp_field_->num_nodes() * 6;
                    for (int i = 0; i < n6; i++)
                        norm2 += h_dx[i] * h_dx[i];
                    float dx_norm = sqrtf(norm2);
                    float dx_mean = dx_norm / std::max(1, warp_field_->num_nodes());
                    warp_update_ok = dx_mean < params_.max_dx_mean;
                    std::cout << "  [dx] ||delta_x||=" << dx_norm
                              << " mean/node=" << dx_mean
                              << " nodes=" << warp_field_->num_nodes()
                              << " max_mean=" << params_.max_dx_mean << "\n";
                    if (do_dbg)
                    {
                        cv::imshow("5_delta_x", dbg_delta_x(h_dx, warp_field_->num_nodes()));
                        cv::waitKey(1);
                    }
                }
                if (warp_update_ok)
                {
                    warp_field_->apply_twist_increment(d_delta_x_);
                }
                else
                {
                    std::cout << "  [warp] skip applying unstable update\n";
                }
            }
            else
            {
                std::cout << "  [solve] skip warp update (too few correspondences, min="
                          << params_.min_valid_corrs << ")\n";
            }

            // 7. Integra depth con warp corrente
            if (warp_update_ok && params_.integrate_warped)
            {
                volume_->integrate(
                    d_depth_, d_depth_normals_,
                    params_.cam, camera_pose_,
                    warp_field_->device_nodes(),
                    warp_field_->device_transforms(),
                    warp_field_->num_nodes(),
                    d_voxel_knn_.data, d_voxel_knn_w_.data);
            }
            else
            {
                std::cout << "  [integrate] skip warped fusion"
                          << (warp_update_ok ? " (disabled)" : " (warp update not stable)")
                          << "\n";
            }

            // 8. Aggiorna grafo con nuovi nodi
            if (params_.node_update_every_n > 0 &&
                frame_count_ % params_.node_update_every_n == 0)
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
        mesh.PaintUniformColor({1.0, 1.0, 1.0});
    }

    // Salva la mesh warped (live frame) come PLY
    // CPU skinning: per ogni vertice canonico, applica blend delle trasformazioni nodi più vicini
    void save_warped_mesh_ply(const std::string &path) const
    {
        std::vector<float3> verts, norms;
        std::vector<int3> tris;
        volume_->extract_surface(verts, norms, tris);

        int num_nodes = warp_field_->num_nodes();
        if (num_nodes == 0)
        {
            save_mesh_ply(path);
            return;
        }

        auto h_nodes = warp_field_->download_nodes();
        auto h_transforms = warp_field_->download_transforms();

        std::vector<float3> warped(verts.size());
        for (size_t vi = 0; vi < verts.size(); vi++)
        {
            float3 p = verts[vi];

            // Brute-force K-NN among nodes
            float best_d2[K_NEIGHBORS];
            int best_id[K_NEIGHBORS];
            for (int k = 0; k < K_NEIGHBORS; k++)
            {
                best_d2[k] = 1e30f;
                best_id[k] = -1;
            }
            for (int ni = 0; ni < num_nodes; ni++)
            {
                float3 d = {p.x - h_nodes[ni].pos.x, p.y - h_nodes[ni].pos.y, p.z - h_nodes[ni].pos.z};
                float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if (d2 < best_d2[K_NEIGHBORS - 1])
                {
                    best_d2[K_NEIGHBORS - 1] = d2;
                    best_id[K_NEIGHBORS - 1] = ni;
                    for (int k = K_NEIGHBORS - 2; k >= 0; k--)
                    {
                        if (best_d2[k + 1] < best_d2[k])
                        {
                            std::swap(best_d2[k], best_d2[k + 1]);
                            std::swap(best_id[k], best_id[k + 1]);
                        }
                    }
                }
            }

            float w_sum = 0;
            float weights[K_NEIGHBORS] = {};
            for (int k = 0; k < K_NEIGHBORS; k++)
            {
                if (best_id[k] < 0)
                    continue;
                float r = h_nodes[best_id[k]].radius;
                weights[k] = expf(-best_d2[k] / (2.f * r * r));
                w_sum += weights[k];
            }

            float3 wp = {0, 0, 0};
            for (int k = 0; k < K_NEIGHBORS; k++)
            {
                if (best_id[k] < 0 || w_sum < 1e-8f)
                    continue;
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
    DeviceArray<int> d_pixel_knn_;
    DeviceArray<float> d_pixel_knn_w_;
    DeviceArray<int> d_num_valid_;
    DeviceArray<int> d_corr_stats_;
    DeviceArray<int> d_hit_voxel_idx_;

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
        auto t0_extract = std::chrono::high_resolution_clock::now();
        volume_->extract_surface(verts, norms, tris);
        auto t1_extract = std::chrono::high_resolution_clock::now();
        double ms_extract = std::chrono::duration<double, std::milli>(t1_extract - t0_extract).count();
        std::cout << "  Surface extracted: " << verts.size() << " vertices in " << ms_extract << " ms\n";
        auto t0_add = std::chrono::high_resolution_clock::now();
        int added = warp_field_->add_nodes_from_surface(verts, params_.node_min_dist);
        auto t1_add = std::chrono::high_resolution_clock::now();
        double ms_add = std::chrono::duration<double, std::milli>(t1_add - t0_add).count();
        std::cout << "  Nodes initialized: " << added << " in " << ms_add << " ms\n";
        auto t0_knn = std::chrono::high_resolution_clock::now();
        warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
        auto t1_knn = std::chrono::high_resolution_clock::now();
        double ms_knn = std::chrono::duration<double, std::milli>(t1_knn - t0_knn).count();
        std::cout << "  Voxel k-NN computed in " << ms_knn << " ms\n";

        std::cout << "[Init] Nodi aggiunti: " << added << "\n";
    }

    void update_node_graph()
    {
        // Download canonical voxel hit-map from raycast → convert to canonical world pos
        // Avoids MC; gives correct canonical (not warped) positions for new nodes
        int n_pixels = params_.cam.width * params_.cam.height;
        std::vector<int> h_vidx(n_pixels);
        cudaMemcpy(h_vidx.data(), d_hit_voxel_idx_.data,
                   n_pixels * sizeof(int), cudaMemcpyDeviceToHost);

        const auto &tp = params_.tsdf;
        std::vector<float3> canonical_pts;
        canonical_pts.reserve(n_pixels / 4);
        for (int px = 0; px < n_pixels; px++)
        {
            int vidx = h_vidx[px];
            if (vidx < 0)
                continue;
            int vx = vidx % tp.dims.x;
            int vy = (vidx / tp.dims.x) % tp.dims.y;
            int vz = vidx / (tp.dims.x * tp.dims.y);
            canonical_pts.push_back(make_float3(
                tp.origin.x + vx * tp.voxel_size,
                tp.origin.y + vy * tp.voxel_size,
                tp.origin.z + vz * tp.voxel_size));
        }

        int added = warp_field_->add_nodes_from_surface(canonical_pts, params_.node_min_dist);
        if (added > 0)
        {
            warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
            std::cout << "[Update] Nuovi nodi: " << added << "\n";
        }
    }

    void apply_depth_bbox_filter()
    {
        dim3 block(16, 16);
        dim3 grid(
            (params_.cam.width + block.x - 1) / block.x,
            (params_.cam.height + block.y - 1) / block.y);
        depth_bbox_filter_kernel<<<grid, block>>>(
            d_depth_.data,
            params_.cam.width, params_.cam.height,
            params_.cam,
            params_.bbox.min_pt, params_.bbox.max_pt);
        cudaDeviceSynchronize();
    }

    void apply_bbox_filter(DeviceArray<float3> &verts)
    {
        int n = (int)verts.size();
        bbox_filter_kernel<<<grid1d(n), 256>>>(
            verts.data, n,
            params_.bbox.min_pt, params_.bbox.max_pt);
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
                  << " continuous=" << depth.isContinuous()
                  << " valid=" << valid
                  << " min=" << (valid ? min_v : 0.0f)
                  << " max=" << (valid ? max_v : 0.0f)
                  << " mean=" << (valid ? (sum / valid) : 0.0) << "\n";
    }

    void print_host_depth_stats(const std::vector<float> &depth, const char *label)
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
                  << " valid=" << valid
                  << " min=" << (valid ? min_v : 0.0f)
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
        std::cout << "  [" << label << "] observed=" << observed
                  << " neg=" << neg
                  << " near_zero=" << near_zero
                  << " min=" << (observed ? min_tsdf : 0.0f)
                  << " max=" << (observed ? max_tsdf : 0.0f) << "\n";
    }

    int find_correspondences(bool debug)
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

        dim3 block(16, 16);
        dim3 grid(
            (params_.cam.width + block.x - 1) / block.x,
            (params_.cam.height + block.y - 1) / block.y);

        // Step 1: map canonical voxel idx → pixel k-NN (exact, no live≈canonical approx)
        compute_pixel_knn_kernel<<<grid, block>>>(
            d_hit_voxel_idx_.data,
            params_.cam.width, params_.cam.height,
            d_voxel_knn_.data, d_voxel_knn_w_.data,
            d_pixel_knn_.data, d_pixel_knn_w_.data);

        // Step 2: projective ICP — fill d_corrs_
        Mat4 T_cam_world = Mat4::identity(); // camera_pose_ = identity
        find_correspondences_kernel<<<grid, block>>>(
            d_vertices_live_.data, d_normals_live_.data,
            d_depth_.data, d_depth_normals_.data,
            d_corrs_.data, d_num_valid_.data,
            params_.cam.width, params_.cam.height,
            params_.cam, T_cam_world,
            d_pixel_knn_.data, d_pixel_knn_w_.data,
            params_.dist_threshold, params_.angle_threshold,
            debug ? d_corr_stats_.data : nullptr);
        cudaDeviceSynchronize();

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_num_valid_.data, sizeof(int), cudaMemcpyDeviceToHost);
        if (debug)
        {
            std::vector<int> h_stats;
            d_corr_stats_.download(h_stats);
            std::cout << "  [corr debug] invalid_live=" << h_stats[0]
                      << " out_proj=" << h_stats[1]
                      << " no_depth=" << h_stats[2]
                      << " far=" << h_stats[3]
                      << " angle=" << h_stats[4]
                      << " valid=" << h_stats[5] << "\n";
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
    if (w["integrate_warped"])
        out.df.integrate_warped = w["integrate_warped"].as<bool>();

    // ───── icp ─────
    auto icp = cfg["icp"];
    out.df.dist_threshold = icp["dist_threshold"].as<float>();
    out.df.angle_threshold = icp["angle_threshold"].as<float>();
    if (icp["min_valid_corrs"])
        out.df.min_valid_corrs = icp["min_valid_corrs"].as<int>();

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

int main(int argc, char **argv)
{
    std::cout << "=== DynamicFusion CUDA Implementation ===\n\n";

    // ─────────────────────────────────────────────
    // CONFIG
    // ─────────────────────────────────────────────

    AppConfig app;
    bool use_config_file = (argc >= 2);

    if (use_config_file && fs::exists(argv[1]))
    {
        std::cout << "[Info] Loading config: " << argv[1] << "\n";
        app = load_config(argv[1]);
    }
    else
    {
        std::cout << "[Info] Nessun config YAML valido.\n"
                  << "Uso sequenza sintetica...\n\n";

        // ── Sequenza sintetica ─────────────────────
        fs::create_directories("/tmp/df_test/depth");

        int W = 640, H = 480;

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

            std::string fname = "/tmp/df_test/depth/" +
                                std::to_string(f) + ".png";

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

    for (int i = 2; i < argc; i++)
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

    if (app.use_vis)
    {
        vis = std::make_shared<open3d::visualization::Visualizer>();
        vis->CreateVisualizerWindow("DynamicFusion", 1920, 1080);

        vis->GetRenderOption().mesh_show_back_face_ = true;

        vis_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

        auto ref_frame =
            open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.3);

        vis->AddGeometry(ref_frame);
    }

    // ── Loop principale ───────────────────────

    int frame_idx = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (sequence.has_next())
    {
        auto t0 = std::chrono::high_resolution_clock::now();
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
        if (app.use_vis)
        {
            if (frame_idx == 1)
            {
                pipeline.update_o3d_mesh(*vis_mesh);
                std::cout << "Mesh iniziale: " << vis_mesh->vertices_.size()
                          << " vertici, " << vis_mesh->triangles_.size() << " triangoli\n";
                vis->AddGeometry(vis_mesh);
            }
            if (frame_idx % 5 == 0)
            {
                pipeline.update_o3d_mesh(*vis_mesh);
                vis->UpdateGeometry(vis_mesh);
                // vis->GetRenderOption().mesh_show_wireframe_ = true;
                vis->PollEvents();
                vis->UpdateRender();
            }
        }

        // Salva mesh ogni 30 frame
        if (frame_idx % 30 == 0)
        {
            std::string path = "mesh_frame_" + std::to_string(frame_idx);
            pipeline.save_mesh_ply(path + "_canonical.ply");
            pipeline.save_warped_mesh_ply(path + "_warped.ply");
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Frame " << std::setw(4) << frame_idx
                  << " processato in " << std::fixed << std::setprecision(1)
                  << ms << " ms\n";
        std::cout << "------------------" << std::endl;
    }

    if (app.use_vis)
    {
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
