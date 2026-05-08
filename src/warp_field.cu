#include "warp_field.h"
#include "tsdf_volume.h"
#include "se3_math.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────
//  CPU spatial hash for O(1) proximity queries
// ─────────────────────────────────────────────

namespace
{

    using CellKey = std::tuple<int, int, int>;

    struct CellKeyHash
    {
        size_t operator()(const CellKey &k) const
        {
            auto [x, y, z] = k;
            return (size_t)x * 73856093u ^ (size_t)y * 19349663u ^ (size_t)z * 83492791u;
        }
    };

    // Maps each cell → list of node indices whose position falls in that cell
    using SpatialGrid = std::unordered_map<CellKey, std::vector<int>, CellKeyHash>;

    inline float dot3h(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline float norm3h(float3 v)
    {
        return sqrtf(dot3h(v, v));
    }

    inline float3 normalize3h(float3 v)
    {
        float n = norm3h(v);
        return n > 1e-6f ? make_float3(v.x / n, v.y / n, v.z / n) : make_float3(0, 0, 0);
    }

    inline CellKey cell_of(float3 p, float inv_cell)
    {
        return {(int)floorf(p.x * inv_cell),
                (int)floorf(p.y * inv_cell),
                (int)floorf(p.z * inv_cell)};
    }

    // True if any node in grid is within min_dist of v.
    // With cell_size = min_dist, only 3³=27 neighbors need checking.
    bool is_too_close(const SpatialGrid &grid, float3 v,
                      float inv_cell, float min_dist2,
                      const std::vector<DeformNode> &nodes)
    {
        auto [cx, cy, cz] = cell_of(v, inv_cell);
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    auto it = grid.find({cx + dx, cy + dy, cz + dz});
                    if (it == grid.end())
                        continue;
                    for (int idx : it->second)
                    {
                        float3 d = {v.x - nodes[idx].pos.x,
                                    v.y - nodes[idx].pos.y,
                                    v.z - nodes[idx].pos.z};
                        if (d.x * d.x + d.y * d.y + d.z * d.z < min_dist2)
                            return true;
                    }
                }
        return false;
    }

    // Collect node indices within search_dist of v using grid.
    // search_radius in cells = ceil(search_dist / cell_size).
    void collect_neighbors(const SpatialGrid &grid, float3 v,
                           float inv_cell, float search_dist2,
                           const std::vector<DeformNode> &nodes,
                           std::vector<std::pair<float, int>> &out)
    {
        int R = 3; // Covers the local graph radius used below.
        auto [cx, cy, cz] = cell_of(v, inv_cell);
        for (int dx = -R; dx <= R; dx++)
            for (int dy = -R; dy <= R; dy++)
                for (int dz = -R; dz <= R; dz++)
                {
                    auto it = grid.find({cx + dx, cy + dy, cz + dz});
                    if (it == grid.end())
                        continue;
                    for (int idx : it->second)
                    {
                        float3 d = {v.x - nodes[idx].pos.x,
                                    v.y - nodes[idx].pos.y,
                                    v.z - nodes[idx].pos.z};
                        float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;
                        if (dist2 < search_dist2)
                            out.push_back({dist2, idx});
                    }
                }
    }

    float dist3h(float3 a, float3 b)
    {
        float3 d = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        return sqrtf(dot3h(d, d));
    }

    struct SurfaceGraph
    {
        std::vector<float3> vertices;
        std::vector<std::vector<std::pair<int, float>>> adjacency;
        std::vector<int> vertex_rep;
        SpatialGrid spatial;
        float inv_cell = 1.0f;
    };

    CellKey quantized_cell(float3 p, float inv_cell)
    {
        return {(int)floorf(p.x * inv_cell + 0.5f),
                (int)floorf(p.y * inv_cell + 0.5f),
                (int)floorf(p.z * inv_cell + 0.5f)};
    }

    void add_graph_edge(SurfaceGraph &graph, int a, int b)
    {
        if (a == b)
            return;
        float w = dist3h(graph.vertices[a], graph.vertices[b]);
        if (w <= 1e-8f)
            return;
        graph.adjacency[a].push_back({b, w});
        graph.adjacency[b].push_back({a, w});
    }

    SurfaceGraph build_surface_graph(const std::vector<float3> &surface_vertices,
                                     const std::vector<int3> &surface_triangles,
                                     float min_dist)
    {
        SurfaceGraph graph;
        graph.vertex_rep.resize(surface_vertices.size(), -1);
        const float weld_cell = fmaxf(1e-5f, 0.05f * min_dist);
        const float inv_weld = 1.0f / weld_cell;
        std::unordered_map<CellKey, int, CellKeyHash> reps;
        reps.reserve(surface_vertices.size());

        for (size_t i = 0; i < surface_vertices.size(); ++i)
        {
            CellKey key = quantized_cell(surface_vertices[i], inv_weld);
            auto it = reps.find(key);
            if (it == reps.end())
            {
                int rep = (int)graph.vertices.size();
                reps.emplace(key, rep);
                graph.vertices.push_back(surface_vertices[i]);
                graph.adjacency.emplace_back();
                graph.vertex_rep[i] = rep;
            }
            else
            {
                graph.vertex_rep[i] = it->second;
            }
        }

        for (const int3 &tri : surface_triangles)
        {
            if (tri.x < 0 || tri.y < 0 || tri.z < 0 ||
                tri.x >= (int)graph.vertex_rep.size() ||
                tri.y >= (int)graph.vertex_rep.size() ||
                tri.z >= (int)graph.vertex_rep.size())
                continue;
            int a = graph.vertex_rep[tri.x];
            int b = graph.vertex_rep[tri.y];
            int c = graph.vertex_rep[tri.z];
            add_graph_edge(graph, a, b);
            add_graph_edge(graph, b, c);
            add_graph_edge(graph, c, a);
        }

        graph.inv_cell = 1.0f / min_dist;
        graph.spatial.reserve(graph.vertices.size() * 2);
        for (int i = 0; i < (int)graph.vertices.size(); ++i)
            graph.spatial[cell_of(graph.vertices[i], graph.inv_cell)].push_back(i);

        return graph;
    }

    int nearest_graph_vertex(const SurfaceGraph &graph, float3 p)
    {
        if (graph.vertices.empty())
            return -1;

        auto [cx, cy, cz] = cell_of(p, graph.inv_cell);
        float best_d2 = std::numeric_limits<float>::infinity();
        int best = -1;
        for (int radius = 1; radius <= 4 && best < 0; ++radius)
        {
            for (int dx = -radius; dx <= radius; dx++)
                for (int dy = -radius; dy <= radius; dy++)
                    for (int dz = -radius; dz <= radius; dz++)
                    {
                        auto it = graph.spatial.find({cx + dx, cy + dy, cz + dz});
                        if (it == graph.spatial.end())
                            continue;
                        for (int idx : it->second)
                        {
                            float3 d = make_float3(p.x - graph.vertices[idx].x,
                                                   p.y - graph.vertices[idx].y,
                                                   p.z - graph.vertices[idx].z);
                            float d2 = dot3h(d, d);
                            if (d2 < best_d2)
                            {
                                best_d2 = d2;
                                best = idx;
                            }
                        }
                    }
        }
        if (best >= 0)
            return best;

        for (int i = 0; i < (int)graph.vertices.size(); ++i)
        {
            float3 d = make_float3(p.x - graph.vertices[i].x,
                                   p.y - graph.vertices[i].y,
                                   p.z - graph.vertices[i].z);
            float d2 = dot3h(d, d);
            if (d2 < best_d2)
            {
                best_d2 = d2;
                best = i;
            }
        }
        return best;
    }

} // namespace

// ─────────────────────────────────────────────
//  Kernel: aggiorna trasformazioni con exp(Δx)
// ─────────────────────────────────────────────

__global__ void apply_twist_kernel(
    const float *delta_x,
    DualQuat *transforms,
    int num_nodes,
    float max_rot,
    float max_trans,
    float update_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    float twist[6];
#pragma unroll
    for (int k = 0; k < 6; k++)
        twist[k] = update_scale * delta_x[i * 6 + k];

    float rot_norm = sqrtf(twist[0] * twist[0] +
                           twist[1] * twist[1] +
                           twist[2] * twist[2]);
    if (max_rot > 0.0f && rot_norm > max_rot)
    {
        float s = max_rot / rot_norm;
        twist[0] *= s;
        twist[1] *= s;
        twist[2] *= s;
    }

    float trans_norm = sqrtf(twist[3] * twist[3] +
                             twist[4] * twist[4] +
                             twist[5] * twist[5]);
    if (max_trans > 0.0f && trans_norm > max_trans)
    {
        float s = max_trans / trans_norm;
        twist[3] *= s;
        twist[4] *= s;
        twist[5] *= s;
    }

    DualQuat dT = dq_from_twist(twist);
    transforms[i] = dq_mul(dT, transforms[i]);
}

// ─────────────────────────────────────────────
//  Kernel: calcola k-NN field nel volume
//  Per ogni voxel nella narrow band,
//  trova i K_NEIGHBORS nodi più vicini
// ─────────────────────────────────────────────

__global__ void compute_voxel_knn_kernel(
    const TSDFVoxel *voxels,
    int3 dims,
    float3 origin,
    float voxel_size,
    const DeformNode *nodes,
    int num_nodes,
    int *knn_ids,
    float *knn_ws)
{
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;

    if (vx >= dims.x || vy >= dims.y || vz >= dims.z)
        return;

    int vidx = vz * dims.x * dims.y + vy * dims.x + vx;

    // Solo voxel non vuoti
    if (voxels[vidx].weight <= 0 || num_nodes == 0)
    {
        for (int k = 0; k < K_NEIGHBORS; k++)
        {
            knn_ids[vidx * K_NEIGHBORS + k] = -1;
            knn_ws[vidx * K_NEIGHBORS + k] = 0.0f;
        }
        return;
    }

    float3 p = make_float3(
        origin.x + (vx + 0.5f) * voxel_size,
        origin.y + (vy + 0.5f) * voxel_size,
        origin.z + (vz + 0.5f) * voxel_size);

    float best_dist[K_NEIGHBORS];
    int best_id[K_NEIGHBORS];

    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        best_dist[k] = 1e30f;
        best_id[k] = -1;
    }

    // Match the host skinning path: pure geometric KNN in canonical space.
    // The previous normal-orientation filter can reject valid influences when
    // TSDF gradients are noisy, leaving vertices/correspondences underconstrained.
    for (int ni = 0; ni < num_nodes; ni++)
    {

        float3 d = make_float3(
            p.x - nodes[ni].pos.x,
            p.y - nodes[ni].pos.y,
            p.z - nodes[ni].pos.z);

        float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;

        if (dist2 < best_dist[K_NEIGHBORS - 1])
        {

            best_dist[K_NEIGHBORS - 1] = dist2;
            best_id[K_NEIGHBORS - 1] = ni;

            // bubble-up
            for (int k = K_NEIGHBORS - 2; k >= 0; k--)
            {
                if (best_dist[k + 1] < best_dist[k])
                {
                    float td = best_dist[k];
                    best_dist[k] = best_dist[k + 1];
                    best_dist[k + 1] = td;

                    int ti = best_id[k];
                    best_id[k] = best_id[k + 1];
                    best_id[k + 1] = ti;
                }
            }
        }
    }

    float w_sum = 0.0f;
    float weights[K_NEIGHBORS];

    // pesi robusti (evita NaN/instabilità)
    for (int k = 0; k < K_NEIGHBORS; k++)
    {

        if (best_id[k] < 0)
        {
            weights[k] = 0.0f;
            continue;
        }

        float r = fmaxf(nodes[best_id[k]].radius, 1e-6f);
        weights[k] = expf(-best_dist[k] / (2.0f * r * r));
        w_sum += weights[k];
    }

    float inv_sum = (w_sum > 1e-8f) ? 1.0f / w_sum : 0.0f;

    for (int k = 0; k < K_NEIGHBORS; k++)
    {

        knn_ids[vidx * K_NEIGHBORS + k] = best_id[k];
        knn_ws[vidx * K_NEIGHBORS + k] = weights[k] * inv_sum;
    }
}

__global__ void score_node_tsdf_support_kernel(
    const TSDFVoxel *voxels,
    int3 dims,
    float3 origin,
    float voxel_size,
    const DeformNode *nodes,
    int num_nodes,
    float support_radius_factor,
    float surface_tsdf_abs,
    float empty_tsdf,
    int min_observed_voxels,
    int min_surface_voxels,
    unsigned char *keep_flags)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes)
        return;

    DeformNode node = nodes[ni];
    float radius = fmaxf(node.radius * support_radius_factor, voxel_size);
    float radius2 = radius * radius;
    int3 c = make_int3(
        (int)floorf((node.pos.x - origin.x) / voxel_size),
        (int)floorf((node.pos.y - origin.y) / voxel_size),
        (int)floorf((node.pos.z - origin.z) / voxel_size));
    int r = (int)ceilf(radius / voxel_size);

    int observed = 0;
    int surface = 0;
    int empty = 0;
    int x0 = c.x - r < 0 ? 0 : c.x - r;
    int y0 = c.y - r < 0 ? 0 : c.y - r;
    int z0 = c.z - r < 0 ? 0 : c.z - r;
    int x1 = c.x + r >= dims.x ? dims.x - 1 : c.x + r;
    int y1 = c.y + r >= dims.y ? dims.y - 1 : c.y + r;
    int z1 = c.z + r >= dims.z ? dims.z - 1 : c.z + r;

    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                float3 p = make_float3(
                    origin.x + (x + 0.5f) * voxel_size,
                    origin.y + (y + 0.5f) * voxel_size,
                    origin.z + (z + 0.5f) * voxel_size);
                float3 d = make_float3(p.x - node.pos.x,
                                       p.y - node.pos.y,
                                       p.z - node.pos.z);
                if (d.x * d.x + d.y * d.y + d.z * d.z > radius2)
                    continue;

                const TSDFVoxel &v = voxels[z * dims.x * dims.y + y * dims.x + x];
                if (v.weight <= 0.0f)
                    continue;
                observed++;
                if (fabsf(v.tsdf) <= surface_tsdf_abs)
                    surface++;
                if (v.tsdf >= empty_tsdf)
                    empty++;
            }

    bool unknown = observed < min_observed_voxels;
    bool supported = surface >= min_surface_voxels;
    bool confidently_empty = empty >= min_observed_voxels && surface == 0;
    keep_flags[ni] = (unknown || supported || !confidently_empty) ? 1 : 0;
}

__device__ __forceinline__ bool atomic_min_float(float *addr, float value)
{
    int *addr_i = reinterpret_cast<int *>(addr);
    int old = *addr_i;
    while (value < __int_as_float(old))
    {
        int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed)
            return true;
    }
    return false;
}

__global__ void init_geodesic_dist_kernel(float *dist,
                                          const int *node_reps,
                                          int source_start,
                                          int batch_count,
                                          int n_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * n_vertices;
    if (idx >= total)
        return;

    int s = idx / n_vertices;
    int v = idx - s * n_vertices;
    int source = node_reps[source_start + s];
    dist[idx] = (v == source) ? 0.0f : 1e30f;
}

__global__ void relax_geodesic_edges_kernel(const int *row_ptr,
                                            const int *col_idx,
                                            const float *edge_w,
                                            int n_vertices,
                                            int batch_count,
                                            float max_dist,
                                            float *dist,
                                            int *changed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * n_vertices;
    if (idx >= total)
        return;

    int s = idx / n_vertices;
    int u = idx - s * n_vertices;
    float du = dist[idx];
    if (!isfinite(du) || du > max_dist)
        return;

    int base = s * n_vertices;
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e)
    {
        int v = col_idx[e];
        float nd = du + edge_w[e];
        if (nd <= max_dist && atomic_min_float(&dist[base + v], nd))
            *changed = 1;
    }
}

__global__ void select_geodesic_neighbors_kernel(const float *dist,
                                                 const int *node_reps,
                                                 int source_start,
                                                 int batch_count,
                                                 int n_vertices,
                                                 int n_nodes,
                                                 float max_dist,
                                                 int *out_ids,
                                                 float *out_dist)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= batch_count)
        return;

    int source_node = source_start + s;
    float best_dist[K_GRAPH];
    int best_id[K_GRAPH];
#pragma unroll
    for (int k = 0; k < K_GRAPH; ++k)
    {
        best_dist[k] = 1e30f;
        best_id[k] = -1;
    }

    const float *src_dist = dist + s * n_vertices;
    for (int node_id = 0; node_id < n_nodes; ++node_id)
    {
        if (node_id == source_node)
            continue;
        int rep = node_reps[node_id];
        if (rep < 0 || rep >= n_vertices)
            continue;
        float gd = src_dist[rep];
        if (!isfinite(gd) || gd <= 0.0f || gd > max_dist ||
            gd >= best_dist[K_GRAPH - 1])
            continue;

        best_dist[K_GRAPH - 1] = gd;
        best_id[K_GRAPH - 1] = node_id;
        for (int k = K_GRAPH - 2; k >= 0; --k)
        {
            if (best_dist[k + 1] >= best_dist[k])
                break;
            float td = best_dist[k];
            best_dist[k] = best_dist[k + 1];
            best_dist[k + 1] = td;
            int ti = best_id[k];
            best_id[k] = best_id[k + 1];
            best_id[k + 1] = ti;
        }
    }

#pragma unroll
    for (int k = 0; k < K_GRAPH; ++k)
    {
        int out = s * K_GRAPH + k;
        out_ids[out] = best_id[k];
        out_dist[out] = best_dist[k];
    }
}

std::vector<std::vector<std::pair<float, int>>> geodesic_neighbors_for_nodes_gpu(
    const SurfaceGraph &graph,
    const std::vector<int> &node_reps,
    float max_geodesic_dist)
{
    const int n_nodes = (int)node_reps.size();
    const int n_vertices = (int)graph.vertices.size();
    std::vector<std::vector<std::pair<float, int>>> result(n_nodes);
    if (n_nodes <= 0 || n_vertices <= 0)
        return result;

    std::vector<int> row_ptr(n_vertices + 1, 0);
    int edge_count = 0;
    for (int v = 0; v < n_vertices; ++v)
    {
        row_ptr[v] = edge_count;
        edge_count += (int)graph.adjacency[v].size();
    }
    row_ptr[n_vertices] = edge_count;

    std::vector<int> col_idx(edge_count);
    std::vector<float> edge_w(edge_count);
    for (int v = 0; v < n_vertices; ++v)
    {
        int out = row_ptr[v];
        for (const auto &e : graph.adjacency[v])
        {
            col_idx[out] = e.first;
            edge_w[out] = e.second;
            out++;
        }
    }

    DeviceArray<int> d_row_ptr, d_col_idx, d_node_reps, d_out_ids, d_changed;
    DeviceArray<float> d_edge_w, d_dist, d_out_dist;
    d_row_ptr.upload(row_ptr);
    d_col_idx.upload(col_idx);
    d_edge_w.upload(edge_w);
    d_node_reps.upload(node_reps);
    d_changed.allocate(1);

    constexpr int kBatchSources = 32;
    const int block = 256;
    const int max_iters = 96;

    for (int source_start = 0; source_start < n_nodes; source_start += kBatchSources)
    {
        int batch_count = std::min(kBatchSources, n_nodes - source_start);
        d_dist.allocate((size_t)batch_count * n_vertices);
        d_out_ids.allocate((size_t)batch_count * K_GRAPH);
        d_out_dist.allocate((size_t)batch_count * K_GRAPH);

        int init_total = batch_count * n_vertices;
        init_geodesic_dist_kernel<<<grid1d(init_total, block), block>>>(
            d_dist.data, d_node_reps.data, source_start, batch_count, n_vertices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iter = 0; iter < max_iters; ++iter)
        {
            d_changed.zero();
            relax_geodesic_edges_kernel<<<grid1d(init_total, block), block>>>(
                d_row_ptr.data, d_col_idx.data, d_edge_w.data, n_vertices,
                batch_count, max_geodesic_dist, d_dist.data, d_changed.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_changed = 0;
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed.data, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (!h_changed)
                break;
        }

        select_geodesic_neighbors_kernel<<<grid1d(batch_count, 128), 128>>>(
            d_dist.data, d_node_reps.data, source_start, batch_count, n_vertices,
            n_nodes, max_geodesic_dist, d_out_ids.data, d_out_dist.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_ids;
        std::vector<float> h_dist;
        d_out_ids.download(h_ids);
        d_out_dist.download(h_dist);
        for (int s = 0; s < batch_count; ++s)
        {
            int node_id = source_start + s;
            for (int k = 0; k < K_GRAPH; ++k)
            {
                int nb = h_ids[s * K_GRAPH + k];
                float gd = h_dist[s * K_GRAPH + k];
                if (nb >= 0 && std::isfinite(gd))
                    result[node_id].push_back({gd, nb});
            }
        }
    }

    return result;
}
// ─────────────────────────────────────────────
//  WarpField implementazione
// ─────────────────────────────────────────────

WarpField::WarpField(float node_radius, int max_nodes)
    : node_radius_(node_radius), num_nodes_(0), max_nodes_(max_nodes)
{
    d_nodes_.allocate(max_nodes);
    d_transforms_.allocate(max_nodes);
    d_transforms_prev_.allocate(max_nodes);
    h_nodes_.reserve(max_nodes);
    h_transforms_.reserve(max_nodes);
}

int WarpField::add_nodes_from_surface(
    const std::vector<float3> &surface_vertices,
    float min_dist,
    const std::vector<float3> *surface_normals)
{
    const float inv_cell = 1.f / min_dist;
    const float min_dist2 = min_dist * min_dist;
    // Local graph only: avoid edges jumping across separate objects/surfaces.
    const float graph_dist = fmaxf(2.5f * min_dist, node_radius_);
    const float graph_dist2 = graph_dist * graph_dist;

    // Build spatial grid from existing nodes
    SpatialGrid grid;
    grid.reserve(num_nodes_ * 2);
    for (int i = 0; i < num_nodes_; i++)
        grid[cell_of(h_nodes_[i].pos, inv_cell)].push_back(i);

    int added = 0;

    // Increased threshold: 0.70 = cos(45°) vs 0.50 = cos(60°)
    // Prevents long edges across differently-oriented surfaces (e.g. arm-to-torso)
    constexpr float kMinGraphNormalDot = 0.70f;
    for (size_t vi = 0; vi < surface_vertices.size(); vi++)
    {
        const float3 &v = surface_vertices[vi];
        if (num_nodes_ >= max_nodes_)
            break;
        if (is_too_close(grid, v, inv_cell, min_dist2, h_nodes_))
            continue;

        DeformNode node;
        node.pos = v;
        node.normal = make_float3(0, 0, 0);
        if (surface_normals && vi < surface_normals->size() &&
            norm3h((*surface_normals)[vi]) > 1e-6f)
            node.normal = normalize3h((*surface_normals)[vi]);
        node.radius = node_radius_;
        node.num_neighbors = 0;
        memset(node.neighbors, -1, sizeof(node.neighbors));
        memset(node.neighbor_w, 0, sizeof(node.neighbor_w));

        // Find graph neighbors via spatial hash (avoids O(N_nodes) scan)
        std::vector<std::pair<float, int>> near;
        near.reserve(K_GRAPH * 4);
        collect_neighbors(grid, v, inv_cell, graph_dist2, h_nodes_, near);

        // Keep the graph local in topology, not only in Euclidean distance.
        // Close but differently oriented surfaces (e.g. arm near torso) must
        // not be tied by ARAP edges, otherwise the optimizer glues them.
        if (norm3h(node.normal) > 1e-6f)
        {
            near.erase(std::remove_if(near.begin(), near.end(),
                                      [&](const auto &e)
                                      {
                                          const float3 nb = h_nodes_[e.second].normal;
                                          return norm3h(nb) > 1e-6f &&
                                                 dot3h(node.normal, nb) < kMinGraphNormalDot;
                                      }),
                       near.end());
        }

        std::sort(near.begin(), near.end());
        int k = std::min((int)near.size(), K_GRAPH);
        for (int i = 0; i < k; i++)
        {
            node.neighbors[i] = near[i].second;
            node.neighbor_w[i] = 1.f;
        }
        node.num_neighbors = k;

        // Init transform: distance-weighted avg of k nearest neighbors
        DualQuat T_init = DualQuat::identity();
        if (k > 0)
        {
            float w_sum = 0;
            DualQuat T_acc;
            T_acc.real = make_float4(0, 0, 0, 0);
            T_acc.dual = make_float4(0, 0, 0, 0);
            for (int i = 0; i < k; i++)
            {
                float w = 1.f / (sqrtf(near[i].first) + 1e-6f);
                DualQuat Tn = h_transforms_[near[i].second];
                if (quat_dot(Tn.real, T_acc.real) < 0.0f)
                {
                    Tn.real.x *= -1;
                    Tn.real.y *= -1;
                    Tn.real.z *= -1;
                    Tn.real.w *= -1;
                    Tn.dual.x *= -1;
                    Tn.dual.y *= -1;
                    Tn.dual.z *= -1;
                    Tn.dual.w *= -1;
                }
                T_acc.real.x += w * Tn.real.x;
                T_acc.real.y += w * Tn.real.y;
                T_acc.real.z += w * Tn.real.z;
                T_acc.real.w += w * Tn.real.w;
                T_acc.dual.x += w * Tn.dual.x;
                T_acc.dual.y += w * Tn.dual.y;
                T_acc.dual.z += w * Tn.dual.z;
                T_acc.dual.w += w * Tn.dual.w;
                w_sum += w;
            }
            if (w_sum > 0)
                T_init = dq_normalize(T_acc);
        }

        h_nodes_.push_back(node);
        h_transforms_.push_back(T_init);
        // Insert new node into grid so subsequent verts see it
        grid[cell_of(v, inv_cell)].push_back(num_nodes_);
        num_nodes_++;
        added++;
    }

    if (added > 0)
    {
        cudaMemcpy(d_nodes_.data, h_nodes_.data(),
                   num_nodes_ * sizeof(DeformNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transforms_.data, h_transforms_.data(),
                   num_nodes_ * sizeof(DualQuat), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transforms_prev_.data, h_transforms_.data(),
                   num_nodes_ * sizeof(DualQuat), cudaMemcpyHostToDevice);
    }

    return added;
}

int WarpField::add_nodes_from_mesh_geodesic(
    const std::vector<float3> &surface_vertices,
    const std::vector<int3> &surface_triangles,
    float min_dist,
    const std::vector<float3> *surface_normals)
{
    int added = add_nodes_from_surface(surface_vertices, min_dist, surface_normals);
    if (added <= 0 || num_nodes_ == 0 || surface_vertices.empty() ||
        surface_triangles.empty())
        return added;

    SurfaceGraph graph = build_surface_graph(surface_vertices, surface_triangles, min_dist);
    if (graph.vertices.empty())
        return added;

    std::vector<int> node_reps(num_nodes_, -1);
    for (int ni = 0; ni < num_nodes_; ++ni)
        node_reps[ni] = nearest_graph_vertex(graph, h_nodes_[ni].pos);

    const float max_geodesic_dist = fmaxf(3.5f * min_dist, 1.5f * node_radius_);
    auto geodesic_neighbors =
        geodesic_neighbors_for_nodes_gpu(graph, node_reps, max_geodesic_dist);

    constexpr float kMinGraphNormalDot = 0.70f;
    for (int ni = 0; ni < num_nodes_; ++ni)
    {
        DeformNode &node = h_nodes_[ni];
        node.num_neighbors = 0;
        memset(node.neighbors, -1, sizeof(node.neighbors));
        memset(node.neighbor_w, 0, sizeof(node.neighbor_w));

        for (const auto &entry : geodesic_neighbors[ni])
        {
            int nb = entry.second;
            float gd = entry.first;
            if (nb < 0 || nb >= num_nodes_ || nb == ni)
                continue;
            if (norm3h(node.normal) > 1e-6f && norm3h(h_nodes_[nb].normal) > 1e-6f &&
                dot3h(node.normal, h_nodes_[nb].normal) < kMinGraphNormalDot)
                continue;

            int k = node.num_neighbors;
            if (k >= K_GRAPH)
                break;
            node.neighbors[k] = nb;
            float r = fmaxf(node.radius, 1e-6f);
            node.neighbor_w[k] = expf(-(gd * gd) / (2.0f * r * r));
            node.num_neighbors++;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_nodes_.data, h_nodes_.data(),
                          num_nodes_ * sizeof(DeformNode),
                          cudaMemcpyHostToDevice));
    return added;
}

void WarpField::compute_voxel_knn(
    const TSDFVolume &volume,
    DeviceArray<int> &d_voxel_knn,
    DeviceArray<float> &d_voxel_knn_w)
{
    if (num_nodes_ == 0)
        return;

    const auto &p = volume.params();
    int total = p.dims.x * p.dims.y * p.dims.z;

    d_voxel_knn.allocate(total * K_NEIGHBORS);
    d_voxel_knn_w.allocate(total * K_NEIGHBORS);

    dim3 block(8, 8, 1);
    dim3 grid(
        (p.dims.x + block.x - 1) / block.x,
        (p.dims.y + block.y - 1) / block.y,
        p.dims.z);

    compute_voxel_knn_kernel<<<grid, block>>>(
        volume.device_data(),
        p.dims, p.origin, p.voxel_size,
        d_nodes_.data, num_nodes_,
        d_voxel_knn.data, d_voxel_knn_w.data);

    cudaDeviceSynchronize();
}

int WarpField::prune_nodes(const TSDFVolume &volume,
                           const WarpField::PruneParams &params)
{
    if (!params.enabled || num_nodes_ == 0)
        return 0;

    DeviceArray<unsigned char> d_keep(num_nodes_);
    const auto &p = volume.params();
    score_node_tsdf_support_kernel<<<grid1d(num_nodes_, 128), 128>>>(
        volume.device_data(), p.dims, p.origin, p.voxel_size,
        d_nodes_.data, num_nodes_, params.support_radius_factor,
        params.surface_tsdf_abs, params.empty_tsdf,
        params.min_observed_voxels, params.min_surface_voxels, d_keep.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> keep;
    d_keep.download(keep);

    if (params.remove_disconnected)
    {
        std::vector<std::vector<int>> adj(num_nodes_);
        for (int i = 0; i < num_nodes_; ++i)
        {
            if (!keep[i])
                continue;
            for (int k = 0; k < h_nodes_[i].num_neighbors; ++k)
            {
                int nb = h_nodes_[i].neighbors[k];
                if (nb < 0 || nb >= num_nodes_ || !keep[nb])
                    continue;
                adj[i].push_back(nb);
                adj[nb].push_back(i);
            }
        }

        std::vector<unsigned char> seen(num_nodes_, 0);
        for (int start = 0; start < num_nodes_; ++start)
        {
            if (!keep[start] || seen[start])
                continue;
            std::vector<int> stack{start};
            std::vector<int> comp;
            seen[start] = 1;
            while (!stack.empty())
            {
                int u = stack.back();
                stack.pop_back();
                comp.push_back(u);
                for (int v : adj[u])
                {
                    if (!seen[v])
                    {
                        seen[v] = 1;
                        stack.push_back(v);
                    }
                }
            }
            if ((int)comp.size() < params.min_component_size)
            {
                for (int idx : comp)
                    keep[idx] = 0;
            }
        }
    }

    std::vector<int> remap(num_nodes_, -1);
    std::vector<DeformNode> new_nodes;
    std::vector<DualQuat> new_transforms;
    new_nodes.reserve(num_nodes_);
    new_transforms.reserve(num_nodes_);
    for (int i = 0; i < num_nodes_; ++i)
    {
        if (!keep[i])
            continue;
        remap[i] = (int)new_nodes.size();
        new_nodes.push_back(h_nodes_[i]);
        new_transforms.push_back(h_transforms_[i]);
    }

    int removed = num_nodes_ - (int)new_nodes.size();
    if (removed <= 0)
        return 0;

    for (DeformNode &node : new_nodes)
    {
        int out = 0;
        int old_neighbors[K_GRAPH];
        float old_weights[K_GRAPH];
        memcpy(old_neighbors, node.neighbors, sizeof(old_neighbors));
        memcpy(old_weights, node.neighbor_w, sizeof(old_weights));
        memset(node.neighbors, -1, sizeof(node.neighbors));
        memset(node.neighbor_w, 0, sizeof(node.neighbor_w));
        for (int k = 0; k < node.num_neighbors && out < K_GRAPH; ++k)
        {
            int nb = old_neighbors[k];
            if (nb < 0 || nb >= (int)remap.size() || remap[nb] < 0)
                continue;
            node.neighbors[out] = remap[nb];
            node.neighbor_w[out] = old_weights[k];
            out++;
        }
        node.num_neighbors = out;
    }

    num_nodes_ = (int)new_nodes.size();
    h_nodes_.swap(new_nodes);
    h_transforms_.swap(new_transforms);
    CUDA_CHECK(cudaMemcpy(d_nodes_.data, h_nodes_.data(),
                          num_nodes_ * sizeof(DeformNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transforms_.data, h_transforms_.data(),
                          num_nodes_ * sizeof(DualQuat),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transforms_prev_.data, h_transforms_.data(),
                          num_nodes_ * sizeof(DualQuat),
                          cudaMemcpyHostToDevice));
    return removed;
}

void WarpField::save_transforms()
{
    cudaMemcpy(d_transforms_prev_.data, d_transforms_.data,
               num_nodes_ * sizeof(DualQuat), cudaMemcpyDeviceToDevice);
}

void WarpField::restore_transforms()
{
    cudaMemcpy(d_transforms_.data, d_transforms_prev_.data,
               num_nodes_ * sizeof(DualQuat), cudaMemcpyDeviceToDevice);
    h_transforms_.resize(num_nodes_);
    cudaMemcpy(h_transforms_.data(), d_transforms_.data,
               num_nodes_ * sizeof(DualQuat), cudaMemcpyDeviceToHost);
}

void WarpField::apply_twist_increment(const DeviceArray<float> &delta_x,
                                      float max_rot, float max_trans,
                                      float update_scale)
{
    if (num_nodes_ == 0)
        return;

    int block = 256;
    int grid = (num_nodes_ + block - 1) / block;

    apply_twist_kernel<<<grid, block>>>(
        delta_x.data, d_transforms_.data, num_nodes_, max_rot, max_trans,
        update_scale);

    cudaDeviceSynchronize();

    // Sincronizza su CPU
    h_transforms_.resize(num_nodes_);
    cudaMemcpy(h_transforms_.data(), d_transforms_.data,
               num_nodes_ * sizeof(DualQuat), cudaMemcpyDeviceToHost);
}

void WarpField::reset_transforms()
{
    h_transforms_.assign(num_nodes_, DualQuat::identity());
    cudaMemcpy(d_transforms_.data, h_transforms_.data(),
               num_nodes_ * sizeof(DualQuat), cudaMemcpyHostToDevice);
}

std::vector<DeformNode> WarpField::download_nodes() const
{
    return h_nodes_;
}

std::vector<DualQuat> WarpField::download_transforms() const
{
    return h_transforms_;
}

void WarpField::upload_nodes(const std::vector<DeformNode> &nodes)
{
    num_nodes_ = std::min((int)nodes.size(), max_nodes_);
    h_nodes_.assign(nodes.begin(), nodes.begin() + num_nodes_);
    CUDA_CHECK(cudaMemcpy(d_nodes_.data, h_nodes_.data(),
                          num_nodes_ * sizeof(DeformNode),
                          cudaMemcpyHostToDevice));
    if ((int)h_transforms_.size() != num_nodes_)
    {
        h_transforms_.assign(num_nodes_, DualQuat::identity());
        CUDA_CHECK(cudaMemcpy(d_transforms_.data, h_transforms_.data(),
                              num_nodes_ * sizeof(DualQuat),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_transforms_prev_.data, h_transforms_.data(),
                              num_nodes_ * sizeof(DualQuat),
                              cudaMemcpyHostToDevice));
    }
}

void WarpField::upload_transforms(const std::vector<DualQuat> &transforms)
{
    int n = std::min((int)transforms.size(), max_nodes_);
    if (num_nodes_ == 0)
        num_nodes_ = n;
    n = std::min(n, num_nodes_);
    h_transforms_.assign(transforms.begin(), transforms.begin() + n);
    CUDA_CHECK(cudaMemcpy(d_transforms_.data, h_transforms_.data(),
                          n * sizeof(DualQuat), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transforms_prev_.data, h_transforms_.data(),
                          n * sizeof(DualQuat), cudaMemcpyHostToDevice));
}
