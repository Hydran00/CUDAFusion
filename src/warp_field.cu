#include "warp_field.h"
#include "tsdf_volume.h"
#include "se3_math.cuh"
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <tuple>

// ─────────────────────────────────────────────
//  CPU spatial hash for O(1) proximity queries
// ─────────────────────────────────────────────

namespace {

using CellKey = std::tuple<int,int,int>;

struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        auto [x,y,z] = k;
        return (size_t)x * 73856093u ^ (size_t)y * 19349663u ^ (size_t)z * 83492791u;
    }
};

// Maps each cell → list of node indices whose position falls in that cell
using SpatialGrid = std::unordered_map<CellKey, std::vector<int>, CellKeyHash>;

inline CellKey cell_of(float3 p, float inv_cell) {
    return { (int)floorf(p.x * inv_cell),
             (int)floorf(p.y * inv_cell),
             (int)floorf(p.z * inv_cell) };
}

// True if any node in grid is within min_dist of v.
// With cell_size = min_dist, only 3³=27 neighbors need checking.
bool is_too_close(const SpatialGrid& grid, float3 v,
                  float inv_cell, float min_dist2,
                  const std::vector<DeformNode>& nodes)
{
    auto [cx,cy,cz] = cell_of(v, inv_cell);
    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++) {
        auto it = grid.find({cx+dx, cy+dy, cz+dz});
        if (it == grid.end()) continue;
        for (int idx : it->second) {
            float3 d = { v.x - nodes[idx].pos.x,
                         v.y - nodes[idx].pos.y,
                         v.z - nodes[idx].pos.z };
            if (d.x*d.x + d.y*d.y + d.z*d.z < min_dist2) return true;
        }
    }
    return false;
}

// Collect node indices within search_dist of v using grid.
// search_radius in cells = ceil(search_dist / cell_size).
void collect_neighbors(const SpatialGrid& grid, float3 v,
                       float inv_cell, float search_dist2,
                       const std::vector<DeformNode>& nodes,
                       std::vector<std::pair<float,int>>& out)
{
    int R = 2; // 2 cells radius covers up to ~2*min_dist
    auto [cx,cy,cz] = cell_of(v, inv_cell);
    for (int dx = -R; dx <= R; dx++)
    for (int dy = -R; dy <= R; dy++)
    for (int dz = -R; dz <= R; dz++) {
        auto it = grid.find({cx+dx, cy+dy, cz+dz});
        if (it == grid.end()) continue;
        for (int idx : it->second) {
            float3 d = { v.x - nodes[idx].pos.x,
                         v.y - nodes[idx].pos.y,
                         v.z - nodes[idx].pos.z };
            float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;
            if (dist2 < search_dist2)
                out.push_back({dist2, idx});
        }
    }
}

} // namespace

// ─────────────────────────────────────────────
//  Kernel: aggiorna trasformazioni con exp(Δx)
// ─────────────────────────────────────────────

__global__ void apply_twist_kernel(
    const float* delta_x,
    Mat4*        transforms,
    int          num_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    Mat4 dT = exp_se3(&delta_x[i * 6]);
    transforms[i] = dT * transforms[i];
}

// ─────────────────────────────────────────────
//  Kernel: calcola k-NN field nel volume
//  Per ogni voxel nella narrow band,
//  trova i K_NEIGHBORS nodi più vicini
// ─────────────────────────────────────────────

__global__ void compute_voxel_knn_kernel(
    const TSDFVoxel*  voxels,
    int3              dims,
    float3            origin,
    float             voxel_size,
    const DeformNode* nodes,
    int               num_nodes,
    int*              knn_ids,
    float*            knn_ws)
{
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;

    if (vx >= dims.x || vy >= dims.y || vz >= dims.z) return;

    int vidx = vz * dims.x * dims.y + vy * dims.x + vx;

    // Solo voxel non vuoti
    if (voxels[vidx].weight <= 0 || num_nodes == 0) {
        for (int k = 0; k < K_NEIGHBORS; k++) {
            knn_ids[vidx * K_NEIGHBORS + k] = -1;
            knn_ws [vidx * K_NEIGHBORS + k] = 0.0f;
        }
        return;
    }

    float3 p = make_float3(
        origin.x + vx * voxel_size,
        origin.y + vy * voxel_size,
        origin.z + vz * voxel_size);

    float best_dist[K_NEIGHBORS];
    int   best_id  [K_NEIGHBORS];

    for (int k = 0; k < K_NEIGHBORS; k++) {
        best_dist[k] = 1e30f;
        best_id[k]   = -1;
    }

    // KNN brute force
    for (int ni = 0; ni < num_nodes; ni++) {

        float3 d = make_float3(
            p.x - nodes[ni].pos.x,
            p.y - nodes[ni].pos.y,
            p.z - nodes[ni].pos.z);

        float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;

        if (dist2 < best_dist[K_NEIGHBORS - 1]) {

            best_dist[K_NEIGHBORS - 1] = dist2;
            best_id[K_NEIGHBORS - 1]   = ni;

            // bubble-up
            for (int k = K_NEIGHBORS - 2; k >= 0; k--) {
                if (best_dist[k+1] < best_dist[k]) {
                    float td = best_dist[k];
                    best_dist[k] = best_dist[k+1];
                    best_dist[k+1] = td;

                    int ti = best_id[k];
                    best_id[k] = best_id[k+1];
                    best_id[k+1] = ti;
                }
            }
        }
    }

    float w_sum = 0.0f;
    float weights[K_NEIGHBORS];

    // pesi robusti (evita NaN/instabilità)
    for (int k = 0; k < K_NEIGHBORS; k++) {

        if (best_id[k] < 0) {
            weights[k] = 0.0f;
            continue;
        }

        float r = fmaxf(nodes[best_id[k]].radius, 1e-6f);
        weights[k] = expf(-best_dist[k] / (2.0f * r * r));
        w_sum += weights[k];
    }

    float inv_sum = (w_sum > 1e-8f) ? 1.0f / w_sum : 0.0f;

    for (int k = 0; k < K_NEIGHBORS; k++) {

        knn_ids[vidx * K_NEIGHBORS + k] = best_id[k];
        knn_ws [vidx * K_NEIGHBORS + k] = weights[k] * inv_sum;
    }
}
// ─────────────────────────────────────────────
//  WarpField implementazione
// ─────────────────────────────────────────────

WarpField::WarpField(float node_radius, int max_nodes)
    : node_radius_(node_radius)
    , num_nodes_(0)
    , max_nodes_(max_nodes)
{
    d_nodes_.allocate(max_nodes);
    d_transforms_.allocate(max_nodes);
    d_transforms_prev_.allocate(max_nodes);
    h_nodes_.reserve(max_nodes);
    h_transforms_.reserve(max_nodes);
}

int WarpField::add_nodes_from_surface(
    const std::vector<float3>& surface_vertices,
    float min_dist)
{
    const float inv_cell   = 1.f / min_dist;
    const float min_dist2  = min_dist * min_dist;
    // For neighbor graph search: up to K_GRAPH*min_dist radius
    const float graph_dist2 = (K_GRAPH * min_dist) * (K_GRAPH * min_dist);

    // Build spatial grid from existing nodes
    SpatialGrid grid;
    grid.reserve(num_nodes_ * 2);
    for (int i = 0; i < num_nodes_; i++)
        grid[cell_of(h_nodes_[i].pos, inv_cell)].push_back(i);

    int added = 0;

    for (const float3& v : surface_vertices) {
        if (num_nodes_ >= max_nodes_) break;
        if (is_too_close(grid, v, inv_cell, min_dist2, h_nodes_)) continue;

        DeformNode node;
        node.pos    = v;
        node.radius = node_radius_;
        node.num_neighbors = 0;
        memset(node.neighbors, -1, sizeof(node.neighbors));
        memset(node.neighbor_w, 0,  sizeof(node.neighbor_w));

        // Find graph neighbors via spatial hash (avoids O(N_nodes) scan)
        std::vector<std::pair<float,int>> near;
        near.reserve(K_GRAPH * 4);
        collect_neighbors(grid, v, inv_cell, graph_dist2, h_nodes_, near);

        // If too few nearby, fall back to global brute-force (rare)
        if ((int)near.size() < std::min(K_GRAPH, num_nodes_)) {
            near.clear();
            for (int i = 0; i < num_nodes_; i++) {
                float3 d = { v.x - h_nodes_[i].pos.x,
                             v.y - h_nodes_[i].pos.y,
                             v.z - h_nodes_[i].pos.z };
                near.push_back({d.x*d.x+d.y*d.y+d.z*d.z, i});
            }
        }

        std::sort(near.begin(), near.end());
        int k = std::min((int)near.size(), K_GRAPH);
        for (int i = 0; i < k; i++) {
            node.neighbors[i]  = near[i].second;
            node.neighbor_w[i] = 1.f;
        }
        node.num_neighbors = k;

        // Init transform: distance-weighted avg of k nearest neighbors
        Mat4 T_init = Mat4::identity();
        if (k > 0) {
            float w_sum = 0;
            Mat4 T_acc;
            memset(T_acc.m, 0, sizeof(T_acc.m));
            for (int i = 0; i < k; i++) {
                float w = 1.f / (sqrtf(near[i].first) + 1e-6f);
                const Mat4& Tn = h_transforms_[near[i].second];
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        T_acc.m[r][c] += w * Tn.m[r][c];
                w_sum += w;
            }
            if (w_sum > 0) {
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        T_init.m[r][c] = T_acc.m[r][c] / w_sum;
                T_init.m[3][0] = T_init.m[3][1] = T_init.m[3][2] = 0;
                T_init.m[3][3] = 1;
            }
        }

        h_nodes_.push_back(node);
        h_transforms_.push_back(T_init);
        // Insert new node into grid so subsequent verts see it
        grid[cell_of(v, inv_cell)].push_back(num_nodes_);
        num_nodes_++;
        added++;
    }

    if (added > 0) {
        cudaMemcpy(d_nodes_.data, h_nodes_.data(),
                   num_nodes_ * sizeof(DeformNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transforms_.data, h_transforms_.data(),
                   num_nodes_ * sizeof(Mat4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transforms_prev_.data, h_transforms_.data(),
                   num_nodes_ * sizeof(Mat4), cudaMemcpyHostToDevice);
    }

    return added;
}

void WarpField::compute_voxel_knn(
    const TSDFVolume& volume,
    DeviceArray<int>&   d_voxel_knn,
    DeviceArray<float>& d_voxel_knn_w)
{
    if (num_nodes_ == 0) return;

    const auto& p = volume.params();
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

void WarpField::save_transforms() {
    cudaMemcpy(d_transforms_prev_.data, d_transforms_.data,
               num_nodes_ * sizeof(Mat4), cudaMemcpyDeviceToDevice);
}

void WarpField::apply_twist_increment(const DeviceArray<float>& delta_x) {
    if (num_nodes_ == 0) return;

    int block = 256;
    int grid  = (num_nodes_ + block - 1) / block;

    apply_twist_kernel<<<grid, block>>>(
        delta_x.data, d_transforms_.data, num_nodes_);

    cudaDeviceSynchronize();

    // Sincronizza su CPU
    h_transforms_.resize(num_nodes_);
    cudaMemcpy(h_transforms_.data(), d_transforms_.data,
               num_nodes_ * sizeof(Mat4), cudaMemcpyDeviceToHost);
}

void WarpField::reset_transforms() {
    h_transforms_.assign(num_nodes_, Mat4::identity());
    cudaMemcpy(d_transforms_.data, h_transforms_.data(),
               num_nodes_ * sizeof(Mat4), cudaMemcpyHostToDevice);
}

std::vector<DeformNode> WarpField::download_nodes() const {
    return h_nodes_;
}

std::vector<Mat4> WarpField::download_transforms() const {
    return h_transforms_;
}
