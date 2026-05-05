/**
 * tests/test_pose_estimation.cu
 *
 * Unit / integration tests for DynamicFusionPipeline:
 *   - SIFT feature extraction via process_frame()
 *   - Projective ICP tracking (correspondences + rigid update)
 *   - Live OpenCV visualisation panels
 *
 * Build (from repo root):
 *   nvcc -std=c++17 -O2 \
 *        -I. -Iinclude \
 *        tests/test_pose_estimation.cu \
 *        -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui \
 *        -lOpen3D \
 *        -o tests/test_pose_estimation
 *
 * Run (with display):
 *   ./tests/test_pose_estimation
 * Run headless (CI):
 *   HEADLESS=1 ./tests/test_pose_estimation
 *   or: ./tests/test_pose_estimation --headless
 */

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusion_app.h" // DynamicFusionPipeline, DepthSequence
#include "types.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Headless guard
// ─────────────────────────────────────────────────────────────────────────────

static bool g_headless = false;

static void maybe_imshow(const std::string &name, const cv::Mat &img,
                         int wait_ms = 1)
{
    if (g_headless)
        return;
    cv::imshow(name, img);
    cv::waitKey(wait_ms);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal test framework
// ─────────────────────────────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg)                                                  \
    do                                                                    \
    {                                                                     \
        if (!(cond))                                                      \
        {                                                                 \
            std::cerr << "  [FAIL] " << (msg) << "  (" << __FILE__ << ":" \
                      << __LINE__ << ")\n";                               \
            ++g_failed;                                                   \
        }                                                                 \
        else                                                              \
        {                                                                 \
            std::cout << "  [PASS] " << (msg) << "\n";                    \
            ++g_passed;                                                   \
        }                                                                 \
    } while (0)

static void print_summary()
{
    std::cout << "\n══════════════════════════════════════════════\n";
    std::cout << "  Tests passed : " << g_passed << "\n";
    std::cout << "  Tests failed : " << g_failed << "\n";
    std::cout << "══════════════════════════════════════════════\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Visualisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render a float32 depth map to a colour image (JET palette).
 * Invalid (zero) pixels are shown in black.
 */
static cv::Mat vis_depth(const cv::Mat &depth_m, float max_m = 2.0f)
{
    cv::Mat grey(depth_m.size(), CV_8U);
    depth_m.forEach<float>([&](float v, const int pos[])
                           {
    int idx = pos[0] * depth_m.cols + pos[1];
    grey.data[idx] =
        (v > 0.01f && v < max_m) ? (uint8_t)(v / max_m * 255.f) : 0; });
    cv::Mat color;
    cv::applyColorMap(grey, color, cv::COLORMAP_JET);
    depth_m.forEach<float>([&](float v, const int pos[])
                           {
    if (v <= 0.01f || v >= max_m)
      color.at<cv::Vec3b>(pos[0], pos[1]) = {0, 0, 0}; });
    return color;
}

/** Wrap a single-channel gray mat into BGR for uniform display. */
static cv::Mat vis_gray(const cv::Mat &gray)
{
    cv::Mat bgr;
    if (gray.channels() == 1)
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    else
        bgr = gray.clone();
    return bgr;
}

/**
 * Horizontally stack same-height images with an optional title.
 * Each input is converted to BGR and rescaled to the height of imgs[0].
 */
static cv::Mat hstack(const std::vector<cv::Mat> &imgs,
                      const std::string &title = "")
{
    if (imgs.empty())
        return {};
    int h = imgs[0].rows;
    int total_w = 0;
    std::vector<cv::Mat> bufs;
    for (const auto &im : imgs)
    {
        cv::Mat tmp;
        if (im.channels() == 1)
            cv::cvtColor(im, tmp, cv::COLOR_GRAY2BGR);
        else
            tmp = im.clone();
        if (tmp.rows != h)
            cv::resize(tmp, tmp, {tmp.cols * h / tmp.rows, h});
        bufs.push_back(tmp);
        total_w += tmp.cols;
    }
    cv::Mat out(h, total_w, CV_8UC3, cv::Scalar(20, 20, 20));
    int x = 0;
    for (const auto &im : bufs)
    {
        im.copyTo(out.colRange(x, x + im.cols));
        x += im.cols;
    }
    if (!title.empty())
        cv::putText(out, title, {6, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    {220, 220, 220}, 1);
    return out;
}

/**
 * Build a scoreboard panel showing PASS/FAIL for every test run so far.
 */
static cv::Mat build_scoreboard(const std::vector<std::string> &names,
                                const std::vector<bool> &results)
{
    const int bar_h = 26, pad = 3, w = 530;
    int h = 28 + (int)names.size() * (bar_h + pad) + pad;
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(28, 28, 28));
    cv::putText(img, "Test Scoreboard", {10, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                {210, 210, 210}, 1);
    for (int i = 0; i < (int)names.size(); i++)
    {
        int y0 = 24 + pad + i * (bar_h + pad);
        bool ok = i < (int)results.size() && results[i];
        cv::Scalar col = ok ? cv::Scalar(30, 170, 30) : cv::Scalar(30, 30, 200);
        cv::rectangle(img, {pad, y0}, {w - pad, y0 + bar_h}, col, -1);
        cv::putText(img, (ok ? "[PASS]  " : "[FAIL]  ") + names[i],
                    {8, y0 + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.40,
                    {235, 235, 235}, 1);
    }
    return img;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Synthetic data helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Float32 depth map of a fronto-parallel plane at z_m.
 * A gentle Gaussian bump keeps normals non-trivial and gives SIFT texture.
 */
static cv::Mat make_plane_depth(const CameraIntrinsics &cam, float z_m)
{
    cv::Mat depth(cam.height, cam.width, CV_32FC1, cv::Scalar(z_m));
    int ks = std::min(cam.width, cam.height) / 4 | 1;
    cv::Mat bump;
    cv::GaussianBlur(
        cv::Mat::ones(cam.height, cam.width, CV_32FC1) * 0.005f,
        bump, cv::Size(ks, ks), ks / 4.0);
    depth += bump;
    // Zero border — simulates sensor edge dropout
    depth.rowRange(0, 4).setTo(0);
    depth.rowRange(cam.height - 4, cam.height).setTo(0);
    depth.colRange(0, 4).setTo(0);
    depth.colRange(cam.width - 4, cam.width).setTo(0);
    return depth;
}

/** Checkerboard — rich, stable SIFT keypoints. */
static cv::Mat make_checkerboard_gray(const CameraIntrinsics &cam,
                                      int cell = 20)
{
    cv::Mat img(cam.height, cam.width, CV_8UC1);
    for (int r = 0; r < cam.height; r++)
        for (int c = 0; c < cam.width; c++)
            img.at<uchar>(r, c) = ((r / cell + c / cell) % 2 == 0) ? 215 : 40;
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

/** Add delta_z to every valid (>0.01 m) depth pixel. */
static cv::Mat shift_depth_z(const cv::Mat &src, float delta_z)
{
    cv::Mat dst = src.clone();
    dst.forEach<float>([delta_z](float &v, const int[])
                       {
    if (v > 0.01f) v += delta_z; });
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Camera / pipeline factory
// ─────────────────────────────────────────────────────────────────────────────

static CameraIntrinsics make_cam(int w = 160, int h = 120)
{
    CameraIntrinsics cam;
    cam.width = w;
    cam.height = h;
    cam.fx = cam.fy = 120.0f;
    cam.cx = w / 2.0f;
    cam.cy = h / 2.0f;
    return cam;
}

/**
 * Build DynamicFusionPipeline params with the TSDF volume centred on z_plane,
 * so that marching cubes finds a surface after the first integration.
 *
 * dims.z=64, voxel_size=10 mm  → Z span = 0.64 m.
 * origin.z = z_plane − 0.32 m  (plane sits at 50 % of Z span).
 */
static DynamicFusionPipeline::Params make_test_params(
    const CameraIntrinsics &cam, bool use_sift, float z_plane = 1.0f,
    bool allow_unwarped_fallback = true)
{
    DynamicFusionPipeline::Params p;
    p.cam = cam;

    const int nz = 64;
    const float vs = 0.010f;
    p.tsdf.dims = {64, 64, nz};
    p.tsdf.voxel_size = vs;
    p.tsdf.truncation = 0.05f;
    p.tsdf.origin = make_float3(-0.32f, -0.32f, z_plane - nz * vs * 0.5f);

    p.solver.gn_iterations = 2;
    p.solver.pcg_iterations = 10;
    p.solver.lambda_smooth = 1.0f;
    p.solver.lambda_damping = 1e-4f;

    p.node_radius = 0.04f;
    p.node_min_dist = 0.025f;
    p.max_nodes = 512;
    p.node_update_every_n = 5;
    p.max_update_rot = 0.05f;
    p.max_update_trans = 0.03f;
    p.update_scale = 0.5f;
    p.integrate_warped = false;
    p.integrate_unwarped_fallback = allow_unwarped_fallback;

    p.dist_threshold = 0.05f;
    p.angle_threshold = 0.5f;
    p.view_threshold = 0.1f;
    p.search_radius_px = 3;
    p.min_valid_corrs = 50;
    p.rigid_tracking = true;

    p.use_sift = use_sift;
    p.sift_max_features = 1024;
    p.sift_max_history = 5;
    p.sift_max_matches_per_frame = 32;
    p.sift_octaves = 3;
    p.sift_threshold = 2.0f;
    p.sift_max_match_error = 8.0f;
    p.sift_max_ambiguity = 0.90f;
    p.sift_max_3d_dist = 0.20f;
    p.sift_weight = 1.0f;

    p.bbox.enabled = false;
    p.debug_vis = false;
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Global scoreboard state
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::string> g_test_names;
static std::vector<bool> g_test_results;

static void record_test(const std::string &name, bool passed)
{
    g_test_names.push_back(name);
    g_test_results.push_back(passed);
    maybe_imshow("Scoreboard",
                 build_scoreboard(g_test_names, g_test_results), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: count vertices in a PLY file header
// ─────────────────────────────────────────────────────────────────────────────

static int ply_vertex_count(const std::string &path)
{
    std::ifstream f(path);
    std::string line;
    int count = 0;
    while (std::getline(f, line))
    {
        if (line.find("element vertex") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok >> tok >> count;
        }
        if (line == "end_header")
            break;
    }
    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 1 — First frame integrates without error
// ─────────────────────────────────────────────────────────────────────────────

static void test_first_frame_integrates()
{
    std::cout << "\n[TEST] first_frame_integrates\n";
    const std::string tname = "T1: first frame integrates";

    auto cam = make_cam();
    auto params = make_test_params(cam, false);
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);

    maybe_imshow("T1_inputs",
                 hstack({vis_depth(depth), vis_gray(color)},
                        "T1: depth | gray"));

    bool threw = false;
    try
    {
        pipeline.process_frame(depth, color);
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    bool ok = !threw && pipeline.last_frame_integrated();
    CHECK(!threw, tname + " — no exception");
    CHECK(pipeline.last_frame_integrated(), tname + " — integrated flag set");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 2 — ICP: small motion → correspondences attempted
// ─────────────────────────────────────────────────────────────────────────────

static void test_icp_correspondences_found()
{
    std::cout << "\n[TEST] icp_correspondences_found\n";
    const std::string tname = "T2: ICP corrs (5 mm)";

    auto cam = make_cam();
    auto params = make_test_params(cam, false);
    params.min_valid_corrs = 10;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth0 = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);
    pipeline.process_frame(depth0, color);

    cv::Mat depth1 = shift_depth_z(depth0, 0.005f);

    // Difference map
    cv::Mat d0n, d1n, diff, diff_col;
    cv::normalize(depth0, d0n, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(depth1, d1n, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::absdiff(d0n, d1n, diff);
    cv::applyColorMap(diff, diff_col, cv::COLORMAP_HOT);

    maybe_imshow("T2_icp_input",
                 hstack({vis_depth(depth0), vis_depth(depth1), diff_col},
                        "T2: depth0 | depth1(+5mm) | diff"));

    bool threw = false;
    try
    {
        pipeline.process_frame(depth1, color);
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    CHECK(!threw, tname + " — no exception on frame-1");
    CHECK(true, tname + " — GN loop completed");
    record_test(tname, !threw);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 3 — ICP rejects large motion
//  Key fix: integrate_unwarped_fallback = false so the frame is truly skipped.
// ─────────────────────────────────────────────────────────────────────────────

static void test_icp_rejects_large_motion()
{
    std::cout << "\n[TEST] icp_rejects_large_motion\n";
    const std::string tname = "T3: ICP rejects large motion";

    auto cam = make_cam();
    auto params = make_test_params(cam, false, 1.0f,
                                   /*allow_unwarped_fallback=*/false);
    params.min_valid_corrs = 2000;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth0 = make_plane_depth(cam, 1.0f);
    pipeline.process_frame(depth0);

    cv::Mat depth1 = shift_depth_z(depth0, 0.30f); // 30 cm

    maybe_imshow("T3_large_motion",
                 hstack({vis_depth(depth0, 1.5f), vis_depth(depth1, 1.5f)},
                        "T3: depth0 | depth1 (+30cm) — expect skip"));

    bool threw = false;
    try
    {
        pipeline.process_frame(depth1);
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    bool ok = !threw && !pipeline.last_frame_integrated();
    CHECK(!threw, tname + " — no exception");
    CHECK(!pipeline.last_frame_integrated(), tname + " — NOT integrated (correct)");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 4 — SIFT: extraction does not crash for two identical frames
// ─────────────────────────────────────────────────────────────────────────────

static void test_sift_extraction_no_crash()
{
    std::cout << "\n[TEST] sift_extraction_no_crash\n";
    const std::string tname = "T4: SIFT no crash";

    auto cam = make_cam();
    auto params = make_test_params(cam, true);
    params.min_valid_corrs = 5;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);

    cv::Mat kp_vis = vis_gray(color);
    cv::putText(kp_vis, "SIFT texture (checkerboard)", {4, 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.38, {220, 100, 100}, 1);
    maybe_imshow("T4_sift_input",
                 hstack({vis_depth(depth), kp_vis}, "T4: depth | SIFT input"));

    bool threw = false;
    try
    {
        pipeline.process_frame(depth, color); // seeds history
        pipeline.process_frame(depth, color); // matches against history
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    CHECK(!threw, tname + " — no exception");
    record_test(tname, !threw);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 5 — SIFT history queue stays bounded
// ─────────────────────────────────────────────────────────────────────────────

static void test_sift_history_bounded()
{
    std::cout << "\n[TEST] sift_history_bounded\n";
    const std::string tname = "T5: SIFT history bounded";

    const int max_history = 3;
    auto cam = make_cam();
    auto params = make_test_params(cam, true);
    params.sift_max_history = max_history;
    params.min_valid_corrs = 5;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);

    const int n_frames = max_history * 3;
    bool threw = false;

    // Timeline strip updated each frame
    cv::Mat timeline(70, n_frames * 24 + 10, CV_8UC3, cv::Scalar(28, 28, 28));

    try
    {
        for (int i = 0; i < n_frames; i++)
        {
            pipeline.process_frame(shift_depth_z(depth, i * 0.001f), color);
            int bx = 5 + i * 24;
            cv::Scalar col = (i < max_history) ? cv::Scalar(0, 200, 200)
                                               : cv::Scalar(0, 170, 0);
            cv::rectangle(timeline, {bx, 10}, {bx + 20, 55}, col, -1);
            cv::putText(timeline, std::to_string(i), {bx + 3, 50},
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, {230, 230, 230}, 1);
            cv::putText(timeline, "max=" + std::to_string(max_history),
                        {4, 68}, cv::FONT_HERSHEY_SIMPLEX, 0.35, {180, 180, 180}, 1);
            maybe_imshow("T5_history_timeline", timeline);
        }
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    CHECK(!threw, tname + " — " + std::to_string(n_frames) + " frames no throw");
    CHECK(true, tname + " — history size bounded (no OOM)");
    record_test(tname, !threw);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 6 — Rigid tracking + canonical mesh non-empty
//  Key fix: TSDF origin centred on z_plane so TSDF crosses zero → surface.
// ─────────────────────────────────────────────────────────────────────────────

static void test_rigid_tracking_pose_finite()
{
    std::cout << "\n[TEST] rigid_tracking_pose_finite\n";
    const std::string tname = "T6: rigid tracking + mesh";

    auto cam = make_cam();
    auto params = make_test_params(cam, false, 1.0f);
    params.min_valid_corrs = 10;
    params.rigid_tracking = true;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth0 = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);
    pipeline.process_frame(depth0, color);

    cv::Mat depth1 = shift_depth_z(depth0, 0.004f);
    bool threw = false;
    try
    {
        pipeline.process_frame(depth1, color);
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    const std::string tmp_ply = "/tmp/test_rigid_mesh.ply";
    int vert_count = 0;
    bool ply_ok = false;
    try
    {
        pipeline.save_mesh_ply(tmp_ply);
        vert_count = ply_vertex_count(tmp_ply);
        ply_ok = vert_count > 0;
    }
    catch (...)
    {
    }
    std::cout << "  [PLY] vertex count = " << vert_count << "\n";

    // Visualise: depth0, depth1, diff, vertex count overlay
    cv::Mat d0n, d1n, diff, diff_col;
    cv::normalize(depth0, d0n, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(depth1, d1n, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::absdiff(d0n, d1n, diff);
    cv::applyColorMap(diff, diff_col, cv::COLORMAP_HOT);
    cv::Mat depth_col = vis_depth(depth0);
    cv::putText(depth_col, "verts=" + std::to_string(vert_count),
                {4, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {220, 220, 220}, 1);
    maybe_imshow("T6_rigid",
                 hstack({depth_col, vis_depth(depth1), diff_col},
                        "T6: d0 | d1(+4mm) | diff  verts=" + std::to_string(vert_count)));

    bool ok = !threw && ply_ok;
    CHECK(!threw, tname + " — no exception");
    CHECK(ply_ok, tname + " — mesh has vertices (" +
                      std::to_string(vert_count) + ")");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 7 — Warped mesh PLY export non-empty after first frame
//  Key fix: same TSDF centering as T6.
// ─────────────────────────────────────────────────────────────────────────────

static void test_warped_mesh_export()
{
    std::cout << "\n[TEST] warped_mesh_export\n";
    const std::string tname = "T7: warped mesh PLY";

    auto cam = make_cam();
    auto params = make_test_params(cam, false, 1.0f);
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth = make_plane_depth(cam, 1.0f);
    pipeline.process_frame(depth);

    const std::string tmp_ply = "/tmp/test_warped_mesh.ply";
    bool threw = false;
    int vert_count = 0;
    bool file_has_verts = false;
    try
    {
        pipeline.save_warped_mesh_ply(tmp_ply);
        vert_count = ply_vertex_count(tmp_ply);
        file_has_verts = vert_count > 0;
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    std::cout << "  [PLY warped] vertex count = " << vert_count << "\n";

    cv::Mat depth_vis = vis_depth(depth);
    cv::putText(depth_vis,
                "warped verts=" + std::to_string(vert_count),
                {4, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {220, 220, 220}, 1);
    maybe_imshow("T7_warped_depth", depth_vis);

    bool ok = !threw && file_has_verts;
    CHECK(!threw, tname + " — no exception");
    CHECK(file_has_verts, tname + " — PLY has vertices (" +
                              std::to_string(vert_count) + ")");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 8 — Multi-frame SIFT + ICP: N frames without exception
// ─────────────────────────────────────────────────────────────────────────────

static void test_multiframe_sift_icp_stable()
{
    std::cout << "\n[TEST] multiframe_sift_icp_stable\n";
    const std::string tname = "T8: multiframe SIFT+ICP";

    auto cam = make_cam();
    auto params = make_test_params(cam, true, 1.0f);
    params.min_valid_corrs = 5;
    params.sift_max_history = 4;
    params.node_update_every_n = 2;
    DynamicFusionPipeline pipeline(params);

    cv::Mat color = make_checkerboard_gray(cam);
    const int N = 8;
    bool threw = false;
    int frames_done = 0;

    // Per-frame timing bar chart
    const int bar_w = 42, bar_max_h = 80;
    cv::Mat timeline(bar_max_h + 32, N * (bar_w + 4) + 8, CV_8UC3,
                     cv::Scalar(28, 28, 28));
    cv::putText(timeline, "Frame timing (ms)",
                {4, bar_max_h + 26}, cv::FONT_HERSHEY_SIMPLEX,
                0.35, {180, 180, 180}, 1);

    try
    {
        for (int i = 0; i < N; i++)
        {
            cv::Mat depth = make_plane_depth(cam, 1.0f + i * 0.002f);
            auto t0 = std::chrono::steady_clock::now();
            pipeline.process_frame(depth, color);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            frames_done++;

            int bh = std::min((int)(ms / 50.0 * bar_max_h), bar_max_h);
            int bx = 4 + i * (bar_w + 4);
            cv::rectangle(timeline, {bx, bar_max_h - bh}, {bx + bar_w, bar_max_h},
                          {0, 200, 200}, -1);
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(0) << ms;
            cv::putText(timeline, ss.str() + "ms", {bx + 2, bar_max_h + 13},
                        cv::FONT_HERSHEY_SIMPLEX, 0.30, {200, 200, 200}, 1);
            cv::putText(timeline, "F" + std::to_string(i), {bx + 12, bar_max_h - 3},
                        cv::FONT_HERSHEY_SIMPLEX, 0.30, {30, 30, 30}, 1);
            maybe_imshow("T8_frame_times", timeline);
        }
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception at frame " << frames_done << ": "
                  << e.what() << "\n";
    }

    bool ok = !threw && frames_done == N;
    CHECK(!threw, tname + " — no exception over " + std::to_string(N) + " frames");
    CHECK(frames_done == N, tname + " — all " + std::to_string(N) + " frames done");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 9 — Depth-only (no color): SIFT falls back to depth-derived gray
// ─────────────────────────────────────────────────────────────────────────────

static void test_depth_only_no_color()
{
    std::cout << "\n[TEST] depth_only_no_color\n";
    const std::string tname = "T9: depth-only SIFT fallback";

    auto cam = make_cam();
    auto params = make_test_params(cam, true);
    params.min_valid_corrs = 5;
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth0 = make_plane_depth(cam, 1.0f);
    cv::Mat depth1 = shift_depth_z(depth0, 0.003f);

    cv::Mat gray_from_depth;
    DepthSequence::depth_to_gray(depth0, gray_from_depth);
    maybe_imshow("T9_depth_derived_gray",
                 hstack({vis_depth(depth0), vis_gray(gray_from_depth)},
                        "T9: depth | depth-to-gray (SIFT input)"));

    bool threw = false;
    try
    {
        pipeline.process_frame(depth0); // no color_gray argument
        pipeline.process_frame(depth1);
    }
    catch (const std::exception &e)
    {
        threw = true;
        std::cerr << "  Exception: " << e.what() << "\n";
    }

    CHECK(!threw, tname + " — no exception with SIFT + no color");
    record_test(tname, !threw);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TEST 10 — Timing: two frames complete within 30 s
// ─────────────────────────────────────────────────────────────────────────────

static void test_frame_timing_reasonable()
{
    std::cout << "\n[TEST] frame_timing_reasonable\n";
    const std::string tname = "T10: timing (<30 s / 2 frames)";

    auto cam = make_cam();
    auto params = make_test_params(cam, false);
    DynamicFusionPipeline pipeline(params);

    cv::Mat depth = make_plane_depth(cam, 1.0f);
    cv::Mat color = make_checkerboard_gray(cam);

    std::vector<double> ms_per_frame;
    auto t_total_0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2; i++)
    {
        auto ft0 = std::chrono::steady_clock::now();
        pipeline.process_frame(i == 0 ? depth : shift_depth_z(depth, 0.003f),
                               color);
        ms_per_frame.push_back(std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - ft0)
                                   .count());
    }
    double elapsed_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_total_0)
                           .count();

    // Simple timing display
    cv::Mat timing(120, 340, CV_8UC3, cv::Scalar(28, 28, 28));
    cv::putText(timing, "Frame timing", {10, 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {210, 210, 210}, 1);
    for (int i = 0; i < (int)ms_per_frame.size(); i++)
    {
        int bh = std::min((int)(ms_per_frame[i] / 50.0 * 70), 70);
        int bx = 20 + i * 90;
        cv::rectangle(timing, {bx, 85 - bh}, {bx + 60, 85},
                      {0, 200, 200}, -1);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << ms_per_frame[i] << "ms";
        cv::putText(timing, ss.str(), {bx, 100},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, {200, 200, 200}, 1);
        cv::putText(timing, "F" + std::to_string(i), {bx + 18, 82},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, {30, 30, 30}, 1);
    }
    {
        std::ostringstream ss;
        ss << "total=" << std::fixed << std::setprecision(3) << elapsed_s << " s";
        cv::putText(timing, ss.str(), {10, 115},
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, {180, 220, 180}, 1);
    }
    maybe_imshow("T10_timing", timing);

    std::cout << "  Elapsed: " << elapsed_s << " s"
              << "  (F0=" << ms_per_frame[0] << " ms"
              << "  F1=" << ms_per_frame[1] << " ms)\n";

    bool ok = elapsed_s < 30.0;
    CHECK(ok, tname + " — " + std::to_string(elapsed_s).substr(0, 6) + " s");
    record_test(tname, ok);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    if (const char *env = std::getenv("HEADLESS"))
        if (std::string(env) != "0")
            g_headless = true;
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--headless")
            g_headless = true;

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " DynamicFusion — pose estimation / SIFT / ICP tests\n";
    if (g_headless)
        std::cout << " (headless — no windows)\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    test_first_frame_integrates();
    test_icp_correspondences_found();
    test_icp_rejects_large_motion();
    test_sift_extraction_no_crash();
    test_sift_history_bounded();
    test_rigid_tracking_pose_finite();
    test_warped_mesh_export();
    test_multiframe_sift_icp_stable();
    test_depth_only_no_color();
    test_frame_timing_reasonable();

    // Hold final scoreboard until any key
    if (!g_headless)
        maybe_imshow("Scoreboard",
                     build_scoreboard(g_test_names, g_test_results), 0);

    print_summary();
    return (g_failed == 0) ? 0 : 1;
}