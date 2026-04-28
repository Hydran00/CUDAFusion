#include "tsdf_volume.h"
#include "se3_math.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
        origin.x + vx * voxel_size,
        origin.y + vy * voxel_size,
        origin.z + vz * voxel_size);

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

    float3 n = cross3(dy, dx);
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
    int              img_w,
    int              img_h,
    CameraIntrinsics cam,
    Mat4             T_world_cam,    // camera → world
    // Warp field opzionale
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

    // Raggio in world space
    float3 ray_d_cam   = cam.unproject(u, v, 1.f);
    float3 ray_origin  = T_world_cam.transform_point(make_float3(0,0,0));
    float3 ray_dir     = normalize3(T_world_cam.transform_normal(ray_d_cam));

    // Marching lungo il raggio
    float t     = 0.3f;  // start depth
    float t_max = 3.5f;
    float step  = voxel_size * 0.8f;

    float tsdf_prev = 0;
    bool  found_zero_crossing = false;

    for (; t < t_max; t += step) {
        float3 p_world = make_float3(
            ray_origin.x + t * ray_dir.x,
            ray_origin.y + t * ray_dir.y,
            ray_origin.z + t * ray_dir.z);

        // Coordinate voxel
        int vx = __float2int_rn((p_world.x - origin.x) / voxel_size);
        int vy = __float2int_rn((p_world.y - origin.y) / voxel_size);
        int vz = __float2int_rn((p_world.z - origin.z) / voxel_size);

        if (vx < 0 || vx >= dims.x ||
            vy < 0 || vy >= dims.y ||
            vz < 0 || vz >= dims.z) continue;

        int vidx = vz * dims.x * dims.y + vy * dims.x + vx;
        const TSDFVoxel& voxel = voxels[vidx];

        if (voxel.weight <= 0) { tsdf_prev = 0; continue; }

        float tsdf_cur = voxel.tsdf;

        // Zero crossing: tsdf cambia segno
        if (tsdf_prev > 0 && tsdf_cur <= 0 && tsdf_prev < 1.0f) {
            // Interpolazione lineare per trovare la superficie esatta
            float t_surface = t - step * tsdf_cur / (tsdf_cur - tsdf_prev);

            float3 p_surface = make_float3(
                ray_origin.x + t_surface * ray_dir.x,
                ray_origin.y + t_surface * ray_dir.y,
                ray_origin.z + t_surface * ray_dir.z);

            // Normale via gradiente del TSDF
            // (differenze finite sui voxel vicini)
            float3 grad;
            auto sample = [&](int dx, int dy, int dz) -> float {
                int nx = vx+dx, ny = vy+dy, nz = vz+dz;
                if (nx<0||nx>=dims.x||ny<0||ny>=dims.y||nz<0||nz>=dims.z)
                    return 0;
                int ni = nz*dims.x*dims.y + ny*dims.x + nx;
                return voxels[ni].weight > 0 ? voxels[ni].tsdf : 0;
            };

            grad.x = sample(1,0,0) - sample(-1,0,0);
            grad.y = sample(0,1,0) - sample(0,-1,0);
            grad.z = sample(0,0,1) - sample(0,0,-1);
            float3 normal = normalize3(grad);

            out_vertices[px_idx] = p_surface;
            out_normals [px_idx] = normal;
            found_zero_crossing  = true;
            break;
        }

        tsdf_prev = tsdf_cur;
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
    float              angle_threshold)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= img_w || v >= img_h) return;

    int px = v * img_w + u;

    Correspondence corr;
    corr.valid = false;

    float3 v_live = live_vertices[px];
    float3 n_live = live_normals[px];

    if (norm3(v_live) < 1e-6f || norm3(n_live) < 1e-6f) goto store;

    {
        // Proietta vertice live nel depth frame
        float3 v_cam = T_cam_world.transform_point(v_live);
        if (v_cam.z <= 0) goto store;

        float2 proj = cam.project(v_cam);
        int pu = __float2int_rn(proj.x);
        int pv = __float2int_rn(proj.y);

        if (pu < 0 || pu >= img_w || pv < 0 || pv >= img_h) goto store;

        float d_dst = depth_map[pv * img_w + pu];
        if (d_dst <= 0) goto store;

        float3 v_dst   = cam.unproject(pu, pv, d_dst);
        float3 n_dst   = depth_normals[pv * img_w + pu];

        // Filtra per distanza e angolo
        float3 diff = make_float3(v_live.x - v_dst.x,
                                   v_live.y - v_dst.y,
                                   v_live.z - v_dst.z);
        float dist = norm3(diff);
        if (dist > dist_threshold) goto store;

        float cosine = fabsf(dot3(n_live, n_dst));
        if (cosine < angle_threshold) goto store;

        // Salva corrispondenza
        corr.src    = v_live;
        corr.dst    = v_dst;
        corr.normal = n_dst;
        corr.valid  = true;

        // Copia k-NN nodi per questo pixel
        #pragma unroll
        for (int k = 0; k < K_NEIGHBORS; k++) {
            corr.node_ids[k] = pixel_knn  [px * K_NEIGHBORS + k];
            corr.node_ws [k] = pixel_knn_w[px * K_NEIGHBORS + k];
        }

        atomicAdd(num_valid, 1);
    }

store:
    corrs[px] = corr;
}
