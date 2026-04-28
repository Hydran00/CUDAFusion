#include "warp_field.h"
#include "tsdf_volume.h"
#include "se3_math.cuh"
#include <algorithm>
#include <cstring>

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
    int*              knn_ids,    // [vol_size * K_NEIGHBORS]
    float*            knn_ws)     // [vol_size * K_NEIGHBORS]
{
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;

    if (vx >= dims.x || vy >= dims.y || vz >= dims.z) return;

    int vidx = vz * dims.x * dims.y + vy * dims.x + vx;

    // Solo voxel nella narrow band
    if (voxels[vidx].weight <= 0 ||
        fabsf(voxels[vidx].tsdf) >= 0.99f) {
        for (int k = 0; k < K_NEIGHBORS; k++) {
            knn_ids[vidx * K_NEIGHBORS + k] = -1;
            knn_ws [vidx * K_NEIGHBORS + k] = 0;
        }
        return;
    }

    float3 p = make_float3(
        origin.x + vx * voxel_size,
        origin.y + vy * voxel_size,
        origin.z + vz * voxel_size);

    // Brute-force k-NN (efficiente per num_nodes < 4096)
    // Trova i K_NEIGHBORS nodi più vicini
    float  best_dist[K_NEIGHBORS];
    int    best_id  [K_NEIGHBORS];
    for (int k = 0; k < K_NEIGHBORS; k++) {
        best_dist[k] = 1e30f;
        best_id  [k] = -1;
    }

    for (int ni = 0; ni < num_nodes; ni++) {
        float3 d = make_float3(
            p.x - nodes[ni].pos.x,
            p.y - nodes[ni].pos.y,
            p.z - nodes[ni].pos.z);
        float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;

        // Insertion sort nei top-K
        if (dist2 < best_dist[K_NEIGHBORS-1]) {
            best_dist[K_NEIGHBORS-1] = dist2;
            best_id  [K_NEIGHBORS-1] = ni;
            // Bubble up
            for (int k = K_NEIGHBORS-2; k >= 0; k--) {
                if (best_dist[k+1] < best_dist[k]) {
                    float tmp_d = best_dist[k]; best_dist[k] = best_dist[k+1]; best_dist[k+1] = tmp_d;
                    int   tmp_i = best_id  [k]; best_id  [k] = best_id  [k+1]; best_id  [k+1] = tmp_i;
                }
            }
        }
    }

    // Calcola pesi e normalizza
    float w_sum = 0;
    float weights[K_NEIGHBORS];
    for (int k = 0; k < K_NEIGHBORS; k++) {
        if (best_id[k] < 0) { weights[k] = 0; continue; }
        float r = nodes[best_id[k]].radius;
        weights[k] = expf(-best_dist[k] / (2.f * r * r));
        w_sum += weights[k];
    }

    for (int k = 0; k < K_NEIGHBORS; k++) {
        knn_ids[vidx * K_NEIGHBORS + k] = best_id[k];
        knn_ws [vidx * K_NEIGHBORS + k] = (w_sum > 1e-8f) ?
                                            weights[k] / w_sum : 0;
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
    int added = 0;

    for (const float3& v : surface_vertices) {
        if (num_nodes_ >= max_nodes_) break;

        // Controlla distanza dai nodi esistenti
        bool too_close = false;
        for (int i = 0; i < num_nodes_; i++) {
            float3 d = make_float3(
                v.x - h_nodes_[i].pos.x,
                v.y - h_nodes_[i].pos.y,
                v.z - h_nodes_[i].pos.z);
            float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            if (dist < min_dist) { too_close = true; break; }
        }
        if (too_close) continue;

        // Aggiungi nuovo nodo
        DeformNode node;
        node.pos    = v;
        node.radius = node_radius_;
        node.num_neighbors = 0;
        memset(node.neighbors, -1, sizeof(node.neighbors));
        memset(node.neighbor_w, 0,  sizeof(node.neighbor_w));

        // Trova vicini nel grafo tra nodi esistenti
        // (K_GRAPH vicini più vicini)
        struct Neighbor { float dist; int id; };
        std::vector<Neighbor> candidates;
        candidates.reserve(num_nodes_);

        for (int i = 0; i < num_nodes_; i++) {
            float3 d = make_float3(
                v.x - h_nodes_[i].pos.x,
                v.y - h_nodes_[i].pos.y,
                v.z - h_nodes_[i].pos.z);
            float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            candidates.push_back({dist, i});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Neighbor& a, const Neighbor& b){
                      return a.dist < b.dist; });

        int k = std::min((int)candidates.size(), K_GRAPH);
        for (int i = 0; i < k; i++) {
            node.neighbors[i]  = candidates[i].id;
            node.neighbor_w[i] = 1.f;  // peso uniforme (semplificato)
        }
        node.num_neighbors = k;

        // Trasformazione iniziale: identità (o interpolata dai vicini)
        Mat4 T_init = Mat4::identity();
        if (k > 0) {
            // Media pesata delle trasformazioni dei vicini
            // (approssimazione — per rigore servirebbero le geodesiche su SE(3))
            float w_sum = 0;
            Mat4 T_acc;
            memset(T_acc.m, 0, sizeof(T_acc.m));
            for (int i = 0; i < k; i++) {
                float w = 1.f / (candidates[i].dist + 1e-6f);
                const Mat4& Tn = h_transforms_[candidates[i].id];
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        T_acc.m[r][c] += w * Tn.m[r][c];
                w_sum += w;
            }
            if (w_sum > 0) {
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        T_init.m[r][c] = T_acc.m[r][c] / w_sum;
                // Ri-ortogonalizza la parte rotazionale (approssimazione)
                // In produzione: SVD o Gram-Schmidt
                T_init.m[3][0] = T_init.m[3][1] = T_init.m[3][2] = 0;
                T_init.m[3][3] = 1;
            }
        }

        h_nodes_.push_back(node);
        h_transforms_.push_back(T_init);
        num_nodes_++;
        added++;
    }

    // Upload su GPU
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
