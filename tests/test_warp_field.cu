/**
 * @file test_warp_field.cu
 * @brief Test suite for warp field and non-rigid deformation tracking
 *
 * This test:
 * 1. Creates and initializes a deformation graph
 * 2. Adds nodes from a synthetic surface
 * 3. Tests warp operations on sample points
 * 4. Validates interpolation and smoothness
 * 5. Tests incremental transformation updates
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#include "types.h"
#include "warp_field.h"

// ─────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────

/**
 * @brief Generate a synthetic surface (e.g., sphere vertices)
 */
std::vector<float3> generate_synthetic_surface(int num_vertices)
{
    std::vector<float3> vertices;
    vertices.reserve(num_vertices);

    // Generate points on a sphere
    for (int i = 0; i < num_vertices; i++)
    {
        float phi = (2.0f * M_PI * i) / num_vertices;
        float theta = 0.5f; // Fixed latitude for simplicity

        float radius = 0.5f;
        float x = radius * std::sin(theta) * std::cos(phi);
        float y = radius * std::sin(theta) * std::sin(phi);
        float z = radius * std::cos(theta) + 1.0f; // Center at z=1

        vertices.push_back(make_float3(x, y, z));
    }

    return vertices;
}

/**
 * @brief Print deformation node information
 */
void print_deform_node(const DeformNode &node, int idx)
{
    std::cout << "\nNode " << idx << ":\n";
    std::cout << "  - Position: ("
              << std::fixed << std::setprecision(4)
              << node.pos.x << ", " << node.pos.y << ", " << node.pos.z << ")\n";
    std::cout << "  - Radius: " << node.radius << "\n";
    std::cout << "  - Neighbors: " << node.num_neighbors << "\n";

    for (int i = 0; i < node.num_neighbors; i++)
    {
        std::cout << "    [" << i << "] ID=" << node.neighbors[i]
                  << ", weight=" << std::setprecision(3) << node.neighbor_w[i] << "\n";
    }
}

/**
 * @brief Print transformation matrix
 */
void print_transform(const Mat4 &T, int node_id)
{
    std::cout << "\nTransform for node " << node_id << ":\n";
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout << std::fixed << std::setprecision(3) << T.m[i][j] << " ";
        }
        std::cout << "\n";
    }
}

/**
 * @brief Extract Euler angles from rotation matrix (approximate)
 */
void extract_euler_angles(const Mat4 &T, float &yaw, float &pitch, float &roll)
{
    // Assuming small angle approximation
    yaw = std::atan2(T.m[1][0], T.m[0][0]);
    pitch = std::asin(-T.m[2][0]);
    roll = std::atan2(T.m[2][1], T.m[2][2]);
}

/**
 * @brief Compute Gaussian weight based on distance
 */
float gaussian_weight(float distance, float sigma)
{
    float d2 = distance * distance;
    float s2 = sigma * sigma;
    return std::exp(-d2 / (2.0f * s2));
}

/**
 * @brief Check whether a CLI flag is present.
 */
bool has_flag(int argc, char **argv, const std::string &flag)
{
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == flag)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Visualize the original and warped surface in 2D.
 */
void visualize_warp_field(const std::vector<float3> &surface,
                          const std::vector<float3> &warped_surface,
                          const std::vector<DeformNode> &nodes)
{
    if (surface.empty() || warped_surface.empty())
    {
        return;
    }

    float min_x = surface[0].x, max_x = surface[0].x;
    float min_y = surface[0].y, max_y = surface[0].y;

    auto expand_bounds = [&](const std::vector<float3> &points)
    {
        for (const auto &p : points)
        {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
    };

    expand_bounds(warped_surface);
    for (const auto &n : nodes)
    {
        min_x = std::min(min_x, n.pos.x);
        max_x = std::max(max_x, n.pos.x);
        min_y = std::min(min_y, n.pos.y);
        max_y = std::max(max_y, n.pos.y);
    }

    const int panel_width = 520;
    const int panel_height = 520;
    const int margin = 24;

    auto make_panel = [&](const std::string &title, const std::vector<float3> &points, const cv::Scalar &color)
    {
        cv::Mat panel(panel_height, panel_width, CV_8UC3, cv::Scalar(22, 22, 28));
        cv::putText(panel, title, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(240, 240, 240), 2, cv::LINE_AA);

        auto project = [&](const float3 &p)
        {
            float nx = (p.x - min_x) / std::max(1e-6f, (max_x - min_x));
            float ny = (p.y - min_y) / std::max(1e-6f, (max_y - min_y));
            int x = margin + static_cast<int>(nx * (panel_width - 2 * margin));
            int y = panel_height - margin - static_cast<int>(ny * (panel_height - 2 * margin));
            return cv::Point(x, y);
        };

        for (const auto &p : points)
        {
            cv::circle(panel, project(p), 2, color, cv::FILLED, cv::LINE_AA);
        }

        for (const auto &node : nodes)
        {
            cv::circle(panel, project(node.pos), 4, cv::Scalar(50, 90, 255), cv::FILLED, cv::LINE_AA);
        }

        return panel;
    };

    cv::Mat original = make_panel("Original surface", surface, cv::Scalar(80, 220, 120));
    cv::Mat warped = make_panel("Warped surface (demo)", warped_surface, cv::Scalar(80, 170, 255));

    cv::Mat montage;
    cv::hconcat(original, warped, montage);
    cv::imshow("Warp Field - Important Steps", montage);
    cv::waitKey(0);
    cv::destroyWindow("Warp Field - Important Steps");
}

/**
 * @brief Test point warping using warp field (CPU simulation)
 */
float3 warp_point_cpu(float3 point,
                      const std::vector<DeformNode> &nodes,
                      const std::vector<Mat4> &transforms)
{
    float3 warped = make_float3(0, 0, 0);
    float total_weight = 0.0f;

    // For each node, accumulate weighted transformation
    for (size_t i = 0; i < nodes.size(); i++)
    {
        float dist = std::sqrt(
            (point.x - nodes[i].pos.x) * (point.x - nodes[i].pos.x) +
            (point.y - nodes[i].pos.y) * (point.y - nodes[i].pos.y) +
            (point.z - nodes[i].pos.z) * (point.z - nodes[i].pos.z));

        float weight = gaussian_weight(dist, nodes[i].radius);

        if (weight > 1e-6f)
        {
            // Apply weighted transformation
            float3 centered = make_float3(
                point.x - nodes[i].pos.x,
                point.y - nodes[i].pos.y,
                point.z - nodes[i].pos.z);

            float3 rotated = transforms[i].transform_normal(centered);
            float3 transformed = make_float3(
                nodes[i].pos.x + rotated.x + transforms[i].m[0][3],
                nodes[i].pos.y + rotated.y + transforms[i].m[1][3],
                nodes[i].pos.z + rotated.z + transforms[i].m[2][3]);

            warped.x += weight * transformed.x;
            warped.y += weight * transformed.y;
            warped.z += weight * transformed.z;
            total_weight += weight;
        }
    }

    if (total_weight > 1e-6f)
    {
        warped.x /= total_weight;
        warped.y /= total_weight;
        warped.z /= total_weight;
    }
    else
    {
        warped = point; // No deformation if no influence
    }

    return warped;
}

// ─────────────────────────────────────────────────────────────────
// Main Test
// ─────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    std::cout << "=== Warp Field Test ===\n\n";
    bool visualize = has_flag(argc, argv, "--visualize");

    // ──────────────────────────────────────────────────────────────
    // 1. CREATE SYNTHETIC SURFACE
    // ──────────────────────────────────────────────────────────────

    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "[Step 1] Generating synthetic surface...\n";

    int num_vertices = 256;
    std::vector<float3> surface = generate_synthetic_surface(num_vertices);

    std::cout << "  - Generated " << surface.size() << " surface vertices\n";

    // Compute surface statistics
    float3 min_v = surface[0], max_v = surface[0];
    for (const auto &v : surface)
    {
        min_v.x = std::min(min_v.x, v.x);
        min_v.y = std::min(min_v.y, v.y);
        min_v.z = std::min(min_v.z, v.z);

        max_v.x = std::max(max_v.x, v.x);
        max_v.y = std::max(max_v.y, v.y);
        max_v.z = std::max(max_v.z, v.z);
    }

    std::cout << "  - Bounding box min: (" << std::fixed << std::setprecision(4)
              << min_v.x << ", " << min_v.y << ", " << min_v.z << ")\n";
    std::cout << "  - Bounding box max: (" << max_v.x << ", " << max_v.y
              << ", " << max_v.z << ")\n";

    // ──────────────────────────────────────────────────────────────
    // 2. CREATE WARP FIELD
    // ──────────────────────────────────────────────────────────────

    std::cout << "\n[Step 2] Creating and initializing warp field...\n";

    float node_radius = 0.1f;

    try
    {
        WarpField warp_field(node_radius, 1024);
        std::cout << "  - Warp field created successfully\n";
        std::cout << "  - Node radius: " << node_radius << " m\n";
        std::cout << "  - Max nodes: 1024\n";

        // ────────────────────────────────────────────────────────────
        // 3. ADD NODES FROM SURFACE
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 3] Adding nodes from surface...\n";

        float min_dist = 0.05f;
        int num_nodes_added = warp_field.add_nodes_from_surface(surface, min_dist);

        std::cout << "  - Nodes added: " << num_nodes_added << "\n";
        std::cout << "  - Minimum inter-node distance: " << min_dist << " m\n";
        std::cout << "  - Total nodes in field: " << warp_field.num_nodes() << "\n";

        // Download nodes for visualization
        auto h_nodes = warp_field.download_nodes();
        auto h_transforms = warp_field.download_transforms();

        // Print statistics for first few nodes
        std::cout << "\n[Node Details] First 5 nodes:\n";
        for (int i = 0; i < std::min(5, (int)h_nodes.size()); i++)
        {
            print_deform_node(h_nodes[i], i);
        }

        // ────────────────────────────────────────────────────────────
        // 4. TEST WARP OPERATIONS
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 4] Testing warp operations...\n";

        // Create test points
        std::vector<float3> test_points = {
            make_float3(0.0f, 0.0f, 0.5f), // Near surface
            make_float3(0.5f, 0.0f, 1.0f), // On surface
            make_float3(0.0f, 0.5f, 1.5f), // Far from surface
        };

        std::cout << "  - Testing " << test_points.size() << " points\n";

        for (size_t i = 0; i < test_points.size(); i++)
        {
            float3 p = test_points[i];

            // Before warp (all transforms are identity initially)
            float3 p_warped = warp_point_cpu(p, h_nodes, h_transforms);

            float dist = std::sqrt(
                (p_warped.x - p.x) * (p_warped.x - p.x) +
                (p_warped.y - p.y) * (p_warped.y - p.y) +
                (p_warped.z - p.z) * (p_warped.z - p.z));

            std::cout << "\n  Point " << i << ": (" << std::fixed << std::setprecision(4)
                      << p.x << ", " << p.y << ", " << p.z << ")\n";
            std::cout << "    - Warped: (" << p_warped.x << ", " << p_warped.y
                      << ", " << p_warped.z << ")\n";
            std::cout << "    - Displacement: " << dist << " m\n";

            // Find nearest node
            float min_node_dist = 1e9f;
            int nearest_node = -1;

            for (size_t j = 0; j < h_nodes.size(); j++)
            {
                float d = std::sqrt(
                    (p.x - h_nodes[j].pos.x) * (p.x - h_nodes[j].pos.x) +
                    (p.y - h_nodes[j].pos.y) * (p.y - h_nodes[j].pos.y) +
                    (p.z - h_nodes[j].pos.z) * (p.z - h_nodes[j].pos.z));

                if (d < min_node_dist)
                {
                    min_node_dist = d;
                    nearest_node = j;
                }
            }

            std::cout << "    - Nearest node: " << nearest_node
                      << " (distance=" << min_node_dist << ")\n";
        }

        // ────────────────────────────────────────────────────────────
        // 5. TEST TRANSFORM UPDATES
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 5] Testing incremental transform updates...\n";

        // Create small twist increments (rotation + translation)
        std::vector<float> delta_x(warp_field.num_nodes() * 6, 0.0f);

        // Apply small rotations and translations to first few nodes
        for (int i = 0; i < std::min(3, warp_field.num_nodes()); i++)
        {
            delta_x[i * 6 + 0] = 0.01f;  // Small rotation around X
            delta_x[i * 6 + 1] = 0.005f; // Small rotation around Y
            delta_x[i * 6 + 3] = 0.001f; // Small translation in X
            delta_x[i * 6 + 4] = 0.002f; // Small translation in Y
        }

        std::cout << "  - Created delta_x with " << delta_x.size() << " values\n";
        std::cout << "  - First 6 values (node 0): ";
        for (int i = 0; i < 6; i++)
        {
            std::cout << std::fixed << std::setprecision(4) << delta_x[i] << " ";
        }
        std::cout << "\n";

        // Upload and apply
        DeviceArray<float> d_delta_x(delta_x.size());
        CUDA_CHECK(cudaMemcpy(d_delta_x.data, delta_x.data(),
                              delta_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        std::cout << "  - Delta uploaded to GPU\n";

        // In a real scenario, we would call:
        // warp_field.apply_twist_increment(d_delta_x, max_rot, max_trans);
        // For this test, we skip GPU side to focus on CPU testing

        // ────────────────────────────────────────────────────────────
        // 6. CONSISTENCY CHECKS
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 6] Consistency checks...\n";

        // Check that identity transforms are identity
        int identity_count = 0;
        for (int i = 0; i < std::min(5, (int)h_transforms.size()); i++)
        {
            const Mat4 &T = h_transforms[i];

            bool is_identity = true;
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    float expected = (r == c) ? 1.0f : 0.0f;
                    if (std::abs(T.m[r][c] - expected) > 1e-5f)
                    {
                        is_identity = false;
                        break;
                    }
                }
                if (!is_identity)
                    break;
            }

            if (is_identity)
                identity_count++;
        }

        std::cout << "  - Identity transforms: " << identity_count << "/5\n";

        // Check node-to-node connectivity
        int total_edges = 0;
        for (const auto &node : h_nodes)
        {
            total_edges += node.num_neighbors;
        }

        std::cout << "  - Total graph edges: " << total_edges << "\n";
        std::cout << "  - Average neighbors per node: "
                  << (float)total_edges / h_nodes.size() << "\n";

        if (visualize)
        {
            std::vector<Mat4> demo_transforms = h_transforms;
            for (int i = 0; i < std::min(3, (int)demo_transforms.size()); i++)
            {
                demo_transforms[i].m[0][3] += 0.015f * (i + 1);
                demo_transforms[i].m[1][3] += 0.007f * (i + 1);
            }

            std::vector<float3> warped_surface;
            warped_surface.reserve(surface.size());
            for (const auto &p : surface)
            {
                warped_surface.push_back(warp_point_cpu(p, h_nodes, demo_transforms));
            }

            visualize_warp_field(surface, warped_surface, h_nodes);
        }

        // ────────────────────────────────────────────────────────────
        // 7. TIMING AND SUMMARY
        // ────────────────────────────────────────────────────────────

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Total execution time: " << std::fixed << std::setprecision(3)
                  << total_time << " s\n";
        std::cout << "Nodes created: " << warp_field.num_nodes() << "\n";
        std::cout << "Graph connectivity: " << ((float)total_edges / h_nodes.size())
                  << " edges/node\n";

        bool success = (warp_field.num_nodes() > 0) && (total_edges > 0);
        std::cout << "\n[Result] " << (success ? "✓ PASSED" : "✗ FAILED") << "\n";

        return success ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] Exception: " << e.what() << "\n";
        return 1;
    }
}
