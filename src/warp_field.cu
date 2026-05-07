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
