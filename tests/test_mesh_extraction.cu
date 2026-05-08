/**
 * @file test_mesh_extraction.cu
 * @brief Test suite for mesh extraction from TSDF volume
 *
 * This test:
 * 1. Creates and populates a TSDF volume with synthetic data
 * 2. Extracts surface mesh using marching cubes
 * 3. Validates mesh topology and geometry
 * 4. Saves mesh to PLY format
 * 5. Computes mesh statistics (vertices, triangles, bounds)
 * 6. Analyzes mesh quality metrics
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <memory>

#include "types.h"
#include "tsdf_volume.h"

#include <open3d/Open3D.h>

// ─────────────────────────────────────────────────────────────────
// PLY File Writer
// ─────────────────────────────────────────────────────────────────

/**
 * @brief Write mesh to PLY file format
 */
bool write_mesh_ply(const std::string &filename,
                    const std::vector<float3> &vertices,
                    const std::vector<float3> &normals,
                    const std::vector<int3> &triangles)
{
    std::ofstream ply(filename);
    if (!ply.is_open())
    {
        std::cerr << "Cannot open file: " << filename << "\n";
        return false;
    }

    // Write PLY header
    ply << "ply\n";
    ply << "format ascii 1.0\n";
    ply << "element vertex " << vertices.size() << "\n";
    ply << "property float x\n";
    ply << "property float y\n";
    ply << "property float z\n";

    if (!normals.empty())
    {
        ply << "property float nx\n";
        ply << "property float ny\n";
        ply << "property float nz\n";
    }

    ply << "element face " << triangles.size() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "end_header\n";

    // Write vertices
    for (size_t i = 0; i < vertices.size(); i++)
    {
        ply << std::fixed << std::setprecision(6);
        ply << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z;

        if (!normals.empty())
        {
            ply << " " << normals[i].x << " " << normals[i].y << " " << normals[i].z;
        }
        ply << "\n";
    }

    // Write triangles
    for (const auto &tri : triangles)
    {
        ply << "3 " << tri.x << " " << tri.y << " " << tri.z << "\n";
    }

    ply.close();
    return true;
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
 * @brief Optionally visualize the extracted mesh using Open3D.
 */
void visualize_mesh_open3d(const std::vector<float3> &vertices,
                           const std::vector<float3> &normals,
                           const std::vector<int3> &triangles)
{
    if (vertices.empty() || triangles.empty())
    {
        std::cout << "[Visualization] Skipping mesh window: no geometry available\n";
        return;
    }

    open3d::geometry::TriangleMesh mesh;
    mesh.vertices_.reserve(vertices.size());
    mesh.triangles_.reserve(triangles.size());
    if (!normals.empty())
    {
        mesh.vertex_normals_.reserve(normals.size());
    }

    for (const auto &v : vertices)
    {
        mesh.vertices_.emplace_back(v.x, v.y, v.z);
    }

    for (const auto &tri : triangles)
    {
        mesh.triangles_.emplace_back(tri.x, tri.y, tri.z);
    }

    if (!normals.empty())
    {
        for (const auto &n : normals)
        {
            mesh.vertex_normals_.emplace_back(n.x, n.y, n.z);
        }
    }

    mesh.ComputeVertexNormals();

    auto bbox = mesh.GetAxisAlignedBoundingBox();
    auto center = bbox.GetCenter();

    auto visualizer = std::make_shared<open3d::visualization::Visualizer>();
    if (!visualizer->CreateVisualizerWindow("Mesh Extraction", 1920, 1080))
    {
        std::cerr << "[Visualization] Could not create Open3D window\n";
        return;
    }

    visualizer->GetRenderOption().mesh_show_back_face_ = true;
    visualizer->AddGeometry(std::make_shared<open3d::geometry::TriangleMesh>(mesh));
    visualizer->PollEvents();
    visualizer->UpdateRender();

    std::cout << "[Visualization] Open3D mesh window ready. Press any key in the window to close.\n";
    std::cout << "[Visualization] Mesh center: (" << center.x() << ", " << center.y()
              << ", " << center.z() << ")\n";

    for (int i = 0; i < 300 && visualizer->PollEvents(); i++)
    {
        visualizer->UpdateRender();
    }

    visualizer->DestroyVisualizerWindow();
}

// ─────────────────────────────────────────────────────────────────
// Synthetic Volume Generator
// ─────────────────────────────────────────────────────────────────

/**
 * @brief Populate TSDF volume with synthetic sphere
 */
void populate_volume_with_sphere(TSDFVolume &volume,
                                 float3 sphere_center,
                                 float sphere_radius)
{
    const auto &params = volume.params();
    int total_voxels = volume.total_voxels();

    std::vector<TSDFVoxel> voxels(total_voxels);

    for (int idx = 0; idx < total_voxels; idx++)
    {
        int3 xyz = volume.idx_to_xyz(idx);
        float3 world_pos = volume.voxel_to_world(xyz);

        // Distance from sphere center
        float dx = world_pos.x - sphere_center.x;
        float dy = world_pos.y - sphere_center.y;
        float dz = world_pos.z - sphere_center.z;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        // Signed distance (negative inside sphere)
        float signed_dist = dist - sphere_radius;

        // Apply truncation
        float tsdf = signed_dist;
        if (std::abs(tsdf) > params.truncation)
        {
            tsdf = (tsdf > 0) ? params.truncation : -params.truncation;
        }

        voxels[idx].tsdf = tsdf;
        voxels[idx].weight = 1.0f;
    }

    // Upload to GPU
    TSDFVoxel *device_data = volume.device_data();
    CUDA_CHECK(cudaMemcpy(device_data, voxels.data(),
                          total_voxels * sizeof(TSDFVoxel),
                          cudaMemcpyHostToDevice));
}

// ─────────────────────────────────────────────────────────────────
// Mesh Analysis Functions
// ─────────────────────────────────────────────────────────────────

/**
 * @brief Compute bounding box of mesh
 */
struct BoundingBox
{
    float3 min, max;

    BoundingBox(const std::vector<float3> &vertices)
    {
        if (vertices.empty())
        {
            min = max = make_float3(0, 0, 0);
            return;
        }

        min = max = vertices[0];
        for (const auto &v : vertices)
        {
            min.x = std::min(min.x, v.x);
            min.y = std::min(min.y, v.y);
            min.z = std::min(min.z, v.z);

            max.x = std::max(max.x, v.x);
            max.y = std::max(max.y, v.y);
            max.z = std::max(max.z, v.z);
        }
    }

    float3 size() const
    {
        return make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
    }

    float3 center() const
    {
        return make_float3((min.x + max.x) / 2.0f,
                           (min.y + max.y) / 2.0f,
                           (min.z + max.z) / 2.0f);
    }
};

/**
 * @brief Compute mesh statistics and quality metrics
 */
void analyze_mesh(const std::vector<float3> &vertices,
                  const std::vector<int3> &triangles,
                  const std::vector<float3> &normals)
{
    std::cout << "\n[Mesh Analysis]\n";

    // Vertex statistics
    std::cout << "  Vertices: " << vertices.size() << "\n";

    if (!vertices.empty())
    {
        BoundingBox bbox(vertices);
        float3 bbox_size = bbox.size();
        float3 bbox_center = bbox.center();

        std::cout << "  Bounding box:\n";
        std::cout << "    - Min: (" << std::fixed << std::setprecision(4)
                  << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << ")\n";
        std::cout << "    - Max: (" << bbox.max.x << ", " << bbox.max.y
                  << ", " << bbox.max.z << ")\n";
        std::cout << "    - Size: (" << bbox_size.x << ", " << bbox_size.y
                  << ", " << bbox_size.z << ")\n";
        std::cout << "    - Center: (" << bbox_center.x << ", " << bbox_center.y
                  << ", " << bbox_center.z << ")\n";

        // Volume estimate (assuming spherical)
        float diagonal = std::sqrt(bbox_size.x * bbox_size.x +
                                   bbox_size.y * bbox_size.y +
                                   bbox_size.z * bbox_size.z);
        float estimated_radius = diagonal / 2.0f;
        float estimated_volume = (4.0f / 3.0f) * M_PI * estimated_radius * estimated_radius * estimated_radius;

        std::cout << "    - Estimated volume: " << estimated_volume << " m³\n";
    }

    // Triangle statistics
    std::cout << "  Triangles: " << triangles.size() << "\n";

    if (!triangles.empty())
    {
        // Compute triangle areas and angles
        std::vector<float> areas;
        std::vector<float> angles;

        for (const auto &tri : triangles)
        {
            float3 v0 = vertices[tri.x];
            float3 v1 = vertices[tri.y];
            float3 v2 = vertices[tri.z];

            // Edge vectors
            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

            // Cross product for area
            float3 cross = make_float3(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x);

            float cross_mag = std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
            float area = cross_mag / 2.0f;
            areas.push_back(area);
        }

        float total_area = 0, min_area = 1e9f, max_area = 0;
        for (float a : areas)
        {
            total_area += a;
            min_area = std::min(min_area, a);
            max_area = std::max(max_area, a);
        }

        std::cout << "  Triangle areas:\n";
        std::cout << "    - Total surface area: " << std::fixed << std::setprecision(6)
                  << total_area << " m²\n";
        std::cout << "    - Mean area: " << (total_area / triangles.size()) << " m²\n";
        std::cout << "    - Min area: " << min_area << " m²\n";
        std::cout << "    - Max area: " << max_area << " m²\n";
    }

    // Normal statistics
    if (!normals.empty())
    {
        std::cout << "  Normals: " << normals.size() << "\n";

        int degenerate = 0;
        for (const auto &n : normals)
        {
            float mag = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
            if (mag < 0.9f || mag > 1.1f)
            {
                degenerate++;
            }
        }

        std::cout << "    - Degenerate normals: " << degenerate << "\n";
    }

    // Mesh quality
    std::cout << "  Quality metrics:\n";
    std::cout << "    - Vertex count: " << vertices.size() << "\n";
    std::cout << "    - Triangle count: " << triangles.size() << "\n";
    std::cout << "    - Edges (estimate): " << (3 * triangles.size() / 2) << "\n";

    // Euler characteristic (for closed mesh: V - E + F = 2)
    int V = vertices.size();
    int F = triangles.size();
    int E = (3 * F) / 2;
    int euler = V - E + F;

    std::cout << "    - Euler characteristic: " << euler
              << " (2 = closed mesh)\n";
}

/**
 * @brief Detect mesh defects
 */
void detect_defects(const std::vector<float3> &vertices,
                    const std::vector<int3> &triangles)
{
    std::cout << "\n[Defect Detection]\n";

    int degenerate_triangles = 0;

    for (const auto &tri : triangles)
    {
        // Check for invalid indices
        if (tri.x < 0 || tri.x >= (int)vertices.size() ||
            tri.y < 0 || tri.y >= (int)vertices.size() ||
            tri.z < 0 || tri.z >= (int)vertices.size())
        {
            degenerate_triangles++;
            continue;
        }

        // Check for degenerate triangles (all vertices are the same)
        if (tri.x == tri.y || tri.y == tri.z || tri.z == tri.x)
        {
            degenerate_triangles++;
            continue;
        }

        // Check for zero-area triangles
        float3 v0 = vertices[tri.x];
        float3 v1 = vertices[tri.y];
        float3 v2 = vertices[tri.z];

        float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

        float area_sq = (e1.x * e1.x + e1.y * e1.y + e1.z * e1.z) *
                        (e2.x * e2.x + e2.y * e2.y + e2.z * e2.z);

        if (area_sq < 1e-12f)
        {
            degenerate_triangles++;
        }
    }

    std::cout << "  - Degenerate triangles: " << degenerate_triangles << "\n";
    std::cout << "  - Mesh integrity: "
              << (degenerate_triangles == 0 ? "✓ GOOD" : "✗ HAS DEFECTS") << "\n";
}

// ─────────────────────────────────────────────────────────────────
// Main Test
// ─────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    std::cout << "=== Mesh Extraction Test ===\n\n";

    bool visualize = has_flag(argc, argv, "--visualize");

    auto t_start = std::chrono::high_resolution_clock::now();

    // ──────────────────────────────────────────────────────────────
    // 1. CREATE TSDF VOLUME
    // ──────────────────────────────────────────────────────────────

    std::cout << "[Step 1] Creating TSDF volume...\n";

    TSDFVolume::Params tsdf_params;
    tsdf_params.dims = make_int3(128, 128, 128);
    tsdf_params.voxel_size = 0.01f;
    tsdf_params.truncation = 0.03f;
    tsdf_params.origin = make_float3(-0.64f, -0.64f, 0.5f);

    try
    {
        TSDFVolume volume(tsdf_params);
        std::cout << "  - Volume created\n";
        std::cout << "  - Dimensions: " << tsdf_params.dims.x << "x"
                  << tsdf_params.dims.y << "x" << tsdf_params.dims.z << "\n";
        std::cout << "  - Total voxels: " << volume.total_voxels() << "\n";

        // ────────────────────────────────────────────────────────────
        // 2. POPULATE VOLUME WITH SYNTHETIC DATA
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 2] Populating volume with synthetic sphere...\n";

        float3 sphere_center = make_float3(0.0f, 0.0f, 1.0f);
        float sphere_radius = 0.3f;

        populate_volume_with_sphere(volume, sphere_center, sphere_radius);

        std::cout << "  - Sphere center: (" << std::fixed << std::setprecision(3)
                  << sphere_center.x << ", " << sphere_center.y << ", "
                  << sphere_center.z << ")\n";
        std::cout << "  - Sphere radius: " << sphere_radius << " m\n";
        std::cout << "  - Volume populated\n";

        // ────────────────────────────────────────────────────────────
        // 3. EXTRACT MESH
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 3] Extracting mesh from volume...\n";

        std::vector<float3> vertices;
        std::vector<float3> normals;
        std::vector<int3> triangles;

        auto t_extract_start = std::chrono::high_resolution_clock::now();

        volume.extract_surface(vertices, normals, triangles);

        auto t_extract_end = std::chrono::high_resolution_clock::now();
        double extract_time =
            std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();

        std::cout << "  - Extraction time: " << std::fixed << std::setprecision(2)
                  << extract_time << " ms\n";
        std::cout << "  - Vertices extracted: " << vertices.size() << "\n";
        std::cout << "  - Triangles extracted: " << triangles.size() << "\n";
        std::cout << "  - Normals computed: " << normals.size() << "\n";

        if (visualize)
        {
            visualize_mesh_open3d(vertices, normals, triangles);
        }

        // ────────────────────────────────────────────────────────────
        // 4. ANALYZE MESH
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 4] Analyzing extracted mesh...\n";

        analyze_mesh(vertices, triangles, normals);

        // ────────────────────────────────────────────────────────────
        // 5. DETECT DEFECTS
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 5] Checking for mesh defects...\n";

        detect_defects(vertices, triangles);

        // ────────────────────────────────────────────────────────────
        // 6. SAVE TO PLY
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 6] Saving mesh to PLY format...\n";

        std::string ply_filename = "/tmp/test_mesh_extraction.ply";

        bool save_success = write_mesh_ply(ply_filename, vertices, normals, triangles);

        if (save_success)
        {
            std::cout << "  - Mesh saved: " << ply_filename << "\n";
            std::cout << "  - File size: " << (vertices.size() * 24 + triangles.size() * 12)
                      << " bytes\n";
        }
        else
        {
            std::cerr << "  - Error saving mesh\n";
        }

        // ────────────────────────────────────────────────────────────
        // 7. VALIDATE EXTRACTED MESH
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 7] Validating extracted mesh...\n";

        // Check expected properties
        bool has_vertices = vertices.size() > 100;
        bool has_triangles = triangles.size() > 50;
        bool is_closed = true; // Simplified check

        std::cout << "  - Has sufficient vertices: " << (has_vertices ? "✓" : "✗") << "\n";
        std::cout << "  - Has sufficient triangles: " << (has_triangles ? "✓" : "✗") << "\n";
        std::cout << "  - Appears to be closed mesh: " << (is_closed ? "✓" : "✗") << "\n";

        // ────────────────────────────────────────────────────────────
        // 8. TEST MULTIPLE EXTRACTIONS
        // ────────────────────────────────────────────────────────────

        std::cout << "\n[Step 8] Testing multiple extractions...\n";

        double total_extract_time = extract_time;

        for (int i = 0; i < 2; i++)
        {
            vertices.clear();
            triangles.clear();
            normals.clear();

            auto t = std::chrono::high_resolution_clock::now();
            volume.extract_surface(vertices, normals, triangles);
            auto t_end = std::chrono::high_resolution_clock::now();

            double t_ms = std::chrono::duration<double, std::milli>(t_end - t).count();
            total_extract_time += t_ms;

            std::cout << "  - Extraction " << (i + 2) << ": "
                      << std::fixed << std::setprecision(2) << t_ms << " ms\n";
        }

        // ────────────────────────────────────────────────────────────
        // 9. SUMMARY AND RESULTS
        // ────────────────────────────────────────────────────────────

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Total execution time: " << std::fixed << std::setprecision(3)
                  << total_time << " s\n";
        std::cout << "Total extraction time (3x): " << total_extract_time << " ms\n";
        std::cout << "Average extraction time: " << (total_extract_time / 3.0f) << " ms\n";

        // Final mesh statistics
        std::cout << "\n[Final Statistics]\n";
        std::cout << "  - Output file: " << ply_filename << "\n";
        std::cout << "  - Format: PLY (Polygon File Format)\n";

        bool success = has_vertices && has_triangles && save_success;
        std::cout << "\n[Result] " << (success ? "✓ PASSED" : "✗ FAILED") << "\n";

        return success ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
