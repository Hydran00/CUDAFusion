#include "tsdf_volume.h"
#include "se3_math.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

__device__ __forceinline__
int voxel_index(int x, int y, int z, int3 dims)
{
    return z * dims.x * dims.y + y * dims.x + x;
}

__device__ __forceinline__
bool sample_tsdf_trilinear(
    const TSDFVoxel* voxels,
    int3             dims,
    float3           origin,
    float            voxel_size,
    float3           p_world,
    float&           tsdf,
    int&             base_vidx)
{
    float gx = (p_world.x - origin.x) / voxel_size;
    float gy = (p_world.y - origin.y) / voxel_size;
    float gz = (p_world.z - origin.z) / voxel_size;

    int x0 = __float2int_rd(gx);
    int y0 = __float2int_rd(gy);
    int z0 = __float2int_rd(gz);

    if (x0 < 0 || x0 >= dims.x - 1 ||
        y0 < 0 || y0 >= dims.y - 1 ||
        z0 < 0 || z0 >= dims.z - 1)
        return false;

    float fx = gx - x0;
    float fy = gy - y0;
    float fz = gz - z0;

    float accum = 0.0f;
    #pragma unroll
    for (int dz = 0; dz <= 1; dz++) {
        float wz = dz ? fz : (1.0f - fz);
        #pragma unroll
        for (int dy = 0; dy <= 1; dy++) {
            float wy = dy ? fy : (1.0f - fy);
            #pragma unroll
            for (int dx = 0; dx <= 1; dx++) {
                float wx = dx ? fx : (1.0f - fx);
                int idx = voxel_index(x0 + dx, y0 + dy, z0 + dz, dims);
                if (voxels[idx].weight <= 0.0f)
                    return false;
                accum += wx * wy * wz * voxels[idx].tsdf;
            }
        }
    }

    tsdf = accum;
    base_vidx = voxel_index(x0, y0, z0, dims);
    return true;
}

__device__ __forceinline__
bool sample_tsdf_value(
    const TSDFVoxel* voxels,
    int3             dims,
    float3           origin,
    float            voxel_size,
    float3           p_world,
    float&           tsdf)
{
    int unused = -1;
    return sample_tsdf_trilinear(voxels, dims, origin, voxel_size,
                                 p_world, tsdf, unused);
}

} // namespace

// ─────────────────────────────────────────────
//  Kernel: integrazione depth nel TSDF
// ─────────────────────────────────────────────

__global__ void tsdf_integrate_kernel(
    TSDFVoxel*           voxels,
    int3                 dims,
    float3               origin,
    float                voxel_size,
    float                truncation,
    const float*         depth,
    int                  img_w,
    int                  img_h,
    CameraIntrinsics     cam,
    Mat4                 T_cam_world,    // world → camera
    // Warp field (opzionale)
    const DeformNode*    nodes,
    const Mat4*          transforms,
    int                  num_nodes,
    const int*           voxel_knn,
    const float*         voxel_knn_w,
    bool                 use_warp)
{
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;

    if (vx >= dims.x || vy >= dims.y || vz >= dims.z) return;

    int voxel_idx = vz * dims.x * dims.y + vy * dims.x + vx;

    // Posizione mondo del voxel
    float3 p_world = make_float3(
        origin.x + (vx + 0.5f) * voxel_size,
        origin.y + (vy + 0.5f) * voxel_size,
        origin.z + (vz + 0.5f) * voxel_size);

    // Se c'è il warp field, deforma il punto prima di proiettare
    float3 p_live = p_world;
    if (use_warp && num_nodes > 0) {
        const int*   knn   = voxel_knn   + voxel_idx * K_NEIGHBORS;
        const float* knn_w = voxel_knn_w + voxel_idx * K_NEIGHBORS;
        p_live = warp_point(p_world, nodes, transforms, knn, knn_w);
    }

    // Proietta nel frame camera
    float3 p_cam = T_cam_world.transform_point(p_live);
    if (p_cam.z <= 0) return;

    float2 px = cam.project(p_cam);
    int u = __float2int_rn(px.x);
    int v = __float2int_rn(px.y);

    if (u < 0 || u >= img_w || v < 0 || v >= img_h) return;

    float d = depth[v * img_w + u];
    if (d <= 0) return;

    // Distanza SDF = depth - z_camera
    float sdf = d - p_cam.z;

    if (sdf < -truncation) return;  // dietro la superficie troppo in profondità

    float tsdf = fminf(1.f, sdf / truncation);

    // Aggiornamento running average (KinectFusion style)
    TSDFVoxel& voxel = voxels[voxel_idx];
    float w_new = 1.f;
    float w_old = voxel.weight;

    voxel.tsdf   = (w_old * voxel.tsdf + w_new * tsdf) / (w_old + w_new);
    voxel.weight = fminf(w_old + w_new, 100.f);  // cap weight
}

// ─────────────────────────────────────────────
//  Kernel: calcolo normali dal depth map
// ─────────────────────────────────────────────

__global__ void compute_depth_normals_kernel(
    const float*     depth,
    float3*          normals,
    int              w,
    int              h,
    CameraIntrinsics cam)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u <= 0 || u >= w-1 || v <= 0 || v >= h-1) {
        if (u < w && v < h) normals[v*w+u] = make_float3(0,0,0);
        return;
    }

    float d   = depth[v*w + u];
    float d_r = depth[v*w + u+1];
    float d_l = depth[v*w + u-1];
    float d_u = depth[(v-1)*w + u];
    float d_d = depth[(v+1)*w + u];

    if (d <= 0 || d_r <= 0 || d_l <= 0 || d_u <= 0 || d_d <= 0) {
        normals[v*w+u] = make_float3(0,0,0);
        return;
    }

    float3 p   = cam.unproject(u,   v,   d);
    float3 p_r = cam.unproject(u+1, v,   d_r);
    float3 p_u = cam.unproject(u,   v-1, d_u);

    float3 dx = make_float3(p_r.x-p.x, p_r.y-p.y, p_r.z-p.z);
    float3 dy = make_float3(p_u.x-p.x, p_u.y-p.y, p_u.z-p.z);

    // Orient depth normals toward the camera, matching the TSDF gradient
    // convention used by raycasting.
    float3 n = cross3(dx, dy);
    normals[v*w+u] = normalize3(n);
}

// ─────────────────────────────────────────────
//  Kernel: raycasting del TSDF → vertici e normali
// ─────────────────────────────────────────────

__global__ void raycast_kernel(
    const TSDFVoxel* voxels,
    int3             dims,
    float3           origin,
    float            voxel_size,
    float            truncation,
    float3*          out_vertices,
    float3*          out_normals,
    int*             out_canonical_vidx,  // canonical voxel index per pixel (may be null)
    int              img_w,
    int              img_h,
    CameraIntrinsics cam,
    Mat4             T_world_cam,
    const DeformNode* nodes,
    const Mat4*       transforms,
    int               num_nodes,
    const int*        voxel_knn,
    const float*      voxel_knn_w,
    bool              use_warp)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= img_w || v >= img_h) return;

    int px_idx = v * img_w + u;
    out_vertices[px_idx] = make_float3(0,0,0);
    out_normals [px_idx] = make_float3(0,0,0);
    if (out_canonical_vidx) out_canonical_vidx[px_idx] = -1;

    // Raggio in world space
    float3 ray_d_cam   = cam.unproject(u, v, 1.f);
    float3 ray_origin  = T_world_cam.transform_point(make_float3(0,0,0));
    float3 ray_dir     = normalize3(T_world_cam.transform_normal(ray_d_cam));

    // Marching lungo il raggio
    float t     = 0.05f; // start depth
    float t_max = 3.5f;
    float step  = voxel_size * 0.8f;

    float tsdf_prev = 0;
    bool  have_prev = false;
    float t_prev = t;

    for (; t < t_max; t += step) {
        float3 p_world = make_float3(
            ray_origin.x + t * ray_dir.x,
            ray_origin.y + t * ray_dir.y,
            ray_origin.z + t * ray_dir.z);

        float tsdf_cur = 1.0f;
        int vidx = -1;
        if (!sample_tsdf_trilinear(voxels, dims, origin, voxel_size,
                                   p_world, tsdf_cur, vidx)) {
            have_prev = false;
            continue;
        }

        // Zero crossing: tsdf cambia segno
        if (have_prev && tsdf_prev > 0 && tsdf_cur <= 0 && tsdf_prev < 1.0f) {
            // Interpolazione lineare per trovare la superficie esatta
            float t_surface = t - (t - t_prev) * tsdf_cur / (tsdf_cur - tsdf_prev);

            float3 p_surface = make_float3(
                ray_origin.x + t_surface * ray_dir.x,
                ray_origin.y + t_surface * ray_dir.y,
                ray_origin.z + t_surface * ray_dir.z);

            // Normale via gradiente trilineare del TSDF.
            float sxp, sxm, syp, sym, szp, szm;
            bool ok_grad =
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x + voxel_size, p_surface.y, p_surface.z), sxp) &&
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x - voxel_size, p_surface.y, p_surface.z), sxm) &&
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x, p_surface.y + voxel_size, p_surface.z), syp) &&
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x, p_surface.y - voxel_size, p_surface.z), sym) &&
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x, p_surface.y, p_surface.z + voxel_size), szp) &&
                sample_tsdf_value(voxels, dims, origin, voxel_size,
                                  make_float3(p_surface.x, p_surface.y, p_surface.z - voxel_size), szm);
            if (!ok_grad)
                break;

            float3 grad;
            grad.x = sxp - sxm;
            grad.y = syp - sym;
            grad.z = szp - szm;
            float3 normal = normalize3(grad);

            // Output canonical voxel index for pixel_knn lookup
            if (out_canonical_vidx) out_canonical_vidx[px_idx] = vidx;

            // Apply warp to get live-space vertex
            float3 p_live = p_surface;
            if (use_warp && num_nodes > 0) {
                const int*   knn   = voxel_knn   + vidx * K_NEIGHBORS;
                const float* knn_w = voxel_knn_w + vidx * K_NEIGHBORS;
                p_live = warp_point(p_surface, nodes, transforms, knn, knn_w);
                normal = warp_normal(normal, transforms, knn, knn_w);
            }

            out_vertices[px_idx] = p_live;
            out_normals [px_idx] = normal;
            break;
        }

        tsdf_prev = tsdf_cur;
        have_prev = true;
        t_prev = t;
        // Adaptive step: più piccolo vicino alla superficie
        if (fabsf(tsdf_cur) < 0.5f)
            step = voxel_size * 0.5f;
        else
            step = voxel_size * 0.8f;
    }
}

// ─────────────────────────────────────────────
//  Kernel: update trasformazioni con exp(Δx)
// ─────────────────────────────────────────────

__global__ void apply_twist_increment_kernel(
    const float* delta_x,   // [num_nodes * 6]
    Mat4*        transforms, // [num_nodes]
    int          num_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    Mat4 dT = exp_se3(&delta_x[i * 6]);

    // T_i ← dT · T_i  (composizione a sinistra)
    transforms[i] = dT * transforms[i];
}

// ─────────────────────────────────────────────
//  Kernel: trova corrispondenze ICP punto-piano
// ─────────────────────────────────────────────

__global__ void find_correspondences_kernel(
    const float3*      live_vertices,   // vertici raycasted [H*W]
    const float3*      live_normals,
    const float*       depth_map,       // depth frame corrente
    const float3*      depth_normals,   // normali depth frame
    Correspondence*    corrs,
    int*               num_valid,
    int                img_w,
    int                img_h,
    CameraIntrinsics   cam,
    Mat4               T_cam_world,
    // k-NN per ogni pixel live
    const int*         pixel_knn,       // [H*W*K] — nodi per pixel
    const float*       pixel_knn_w,
    float              dist_threshold,
    float              angle_threshold,
    float              view_threshold,
    int                search_radius_px,
    int*               debug_stats)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= img_w || v >= img_h) return;

    int px = v * img_w + u;

    Correspondence corr;
    corr.valid = false;
    corr.weight = 0.0f;

    float3 v_live = live_vertices[px];
    float3 n_live = live_normals[px];

    if (norm3(v_live) < 1e-6f || norm3(n_live) < 1e-6f) {
        if (debug_stats) atomicAdd(&debug_stats[0], 1);
        goto store;
    }

    {
        // Proietta vertice live nel depth frame
        float3 v_cam = T_cam_world.transform_point(v_live);
        if (v_cam.z <= 0) {
            if (debug_stats) atomicAdd(&debug_stats[1], 1);
            goto store;
        }

        float2 proj = cam.project(v_cam);
        int pu = __float2int_rn(proj.x);
        int pv = __float2int_rn(proj.y);

        if (pu < 0 || pu >= img_w || pv < 0 || pv >= img_h) {
            if (debug_stats) atomicAdd(&debug_stats[1], 1);
            goto store;
        }

        float3 n_cam   = T_cam_world.transform_normal(n_live);
        float3 best_dst = make_float3(0, 0, 0);
        float3 best_n = make_float3(0, 0, 0);
        float best_score = dist_threshold;
        float best_weight = 0.0f;
        float euclidean_cap = fmaxf(3.0f * dist_threshold, dist_threshold + 0.03f);
        float normal_eps = fmaxf(1.0f - angle_threshold, 1e-4f);
        float view_eps = fmaxf(1.0f - view_threshold, 1e-4f);
        float3 view_dir = normalize3(make_float3(-v_cam.x, -v_cam.y, -v_cam.z));
        float view_cos = fabsf(dot3(n_cam, view_dir));
        if (view_cos < view_threshold) {
            if (debug_stats) atomicAdd(&debug_stats[4], 1);
            goto store;
        }
        bool saw_depth = false;
        bool saw_close = false;
        bool saw_angle = false;

        // Projective ICP with a small local search. A single rounded pixel is
        // fragile when the warp is still catching up or the depth has holes.
        int radius = max(0, search_radius_px);
        for (int dy = -radius; dy <= radius; dy++) {
            int y = pv + dy;
            if (y < 0 || y >= img_h) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int x = pu + dx;
                if (x < 0 || x >= img_w) continue;

                float d_dst = depth_map[y * img_w + x];
                if (d_dst <= 0) continue;
                saw_depth = true;

                float3 v_dst = cam.unproject(x, y, d_dst);
                float3 diff = make_float3(v_cam.x - v_dst.x,
                               v_cam.y - v_dst.y,
                               v_cam.z - v_dst.z);
                float euclidean_dist = norm3(diff);
                if (euclidean_dist > euclidean_cap) continue;

                float3 n_dst = depth_normals[y * img_w + x];
                if (norm3(n_dst) < 1e-6f) continue;
                float cosine = fabsf(dot3(n_cam, n_dst));
                if (cosine < angle_threshold) continue;
                saw_angle = true;

                float plane_dist = fabsf(dot3(n_dst, diff));
                if (plane_dist > dist_threshold || plane_dist >= best_score) continue;
                saw_close = true;

                float phi_d = 1.0f - euclidean_dist / dist_threshold;
                float phi_n = 1.0f - (1.0f - cosine) / normal_eps;
                float phi_v = 1.0f - (1.0f - view_cos) / view_eps;
                float confidence = (phi_d + phi_n + phi_v) / 3.0f;
                confidence = confidence * confidence;

                best_score = plane_dist;
                best_weight = confidence;
                best_dst = v_dst;
                best_n = n_dst;
            }
        }

        if (!saw_depth) {
            if (debug_stats) atomicAdd(&debug_stats[2], 1);
            goto store;
        }
        if (!saw_close) {
            if (debug_stats) atomicAdd(&debug_stats[3], 1);
            goto store;
        }
        if (!saw_angle) {
            if (debug_stats) atomicAdd(&debug_stats[4], 1);
            goto store;
        }

        // Salva corrispondenza
        corr.src    = v_live;
        corr.dst    = best_dst;
        corr.normal = best_n;
        corr.weight = best_weight;
        corr.valid  = true;

        // Copia k-NN nodi per questo pixel
        #pragma unroll
        for (int k = 0; k < K_NEIGHBORS; k++) {
            corr.node_ids[k] = pixel_knn  [px * K_NEIGHBORS + k];
            corr.node_ws [k] = pixel_knn_w[px * K_NEIGHBORS + k];
        }

        atomicAdd(num_valid, 1);
        if (debug_stats) atomicAdd(&debug_stats[5], 1);
    }

store:
    corrs[px] = corr;
}

// ─────────────────────────────────────────────
//  Kernel: voxel_knn → pixel_knn
//  For each live pixel, look up the voxel that
//  contains the live vertex and copy its k-NN.
//  Approximation: canonical pos ≈ live pos
//  (valid for small deformations).
// ─────────────────────────────────────────────

__global__ void compute_pixel_knn_kernel(
    const int*    canonical_vidx,  // from raycast: canonical voxel hit per pixel
    int           img_w,
    int           img_h,
    const int*    voxel_knn,
    const float*  voxel_knn_w,
    int*          pixel_knn,
    float*        pixel_knn_w)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= img_w || v >= img_h) return;

    int px = v * img_w + u;

    for (int k = 0; k < K_NEIGHBORS; k++) {
        pixel_knn  [px * K_NEIGHBORS + k] = -1;
        pixel_knn_w[px * K_NEIGHBORS + k] = 0.f;
    }

    int vidx = canonical_vidx[px];
    if (vidx < 0) return;

    for (int k = 0; k < K_NEIGHBORS; k++) {
        pixel_knn  [px * K_NEIGHBORS + k] = voxel_knn  [vidx * K_NEIGHBORS + k];
        pixel_knn_w[px * K_NEIGHBORS + k] = voxel_knn_w[vidx * K_NEIGHBORS + k];
    }
}

// ─────────────────────────────────────────────
//  Bounding box filter
// ─────────────────────────────────────────────

__global__ void bbox_filter_kernel(float3* vertices, int n, float3 min_pt, float3 max_pt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 v = vertices[i];
    if (v.x < min_pt.x || v.x > max_pt.x ||
        v.y < min_pt.y || v.y > max_pt.y ||
        v.z < min_pt.z || v.z > max_pt.z)
        vertices[i] = make_float3(0.f, 0.f, 0.f);
}

__global__ void depth_bbox_filter_kernel(
    float*           depth,
    int              w,
    int              h,
    CameraIntrinsics cam,
    float3           min_pt,
    float3           max_pt)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= w || v >= h) return;

    float d = depth[v * w + u];
    if (d <= 0.f) return;

    float3 p = cam.unproject((float)u, (float)v, d);
    if (p.x < min_pt.x || p.x > max_pt.x ||
        p.y < min_pt.y || p.y > max_pt.y ||
        p.z < min_pt.z || p.z > max_pt.z)
        depth[v * w + u] = 0.f;
}
