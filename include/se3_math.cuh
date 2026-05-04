#pragma once
#include "types.h"
#include <cuda_runtime.h>
#include <math.h>

// ─────────────────────────────────────────────
//  SE(3) / so(3) math — device functions
// ─────────────────────────────────────────────

__device__ __forceinline__
    float3
    cross3(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float norm3(float3 v) { return sqrtf(dot3(v, v)); }

__device__ __forceinline__
    float3
    normalize3(float3 v)
{
    float n = norm3(v);
    return (n > 1e-9f) ? make_float3(v.x / n, v.y / n, v.z / n) : v;
}

// ─────────────────────────────────────────────
//  Mappa esponenziale: twist ∈ ℝ⁶ → SE(3)
//  twist = [omega(3), v(3)]
// ─────────────────────────────────────────────

__device__ __forceinline__
    Mat4
    exp_se3(const float *twist)
{
    float ox = twist[0], oy = twist[1], oz = twist[2];
    float vx = twist[3], vy = twist[4], vz = twist[5];

    float theta2 = ox * ox + oy * oy + oz * oz;
    float theta = sqrtf(theta2);

    Mat4 T; // identity

    float A, B, C;
    if (theta < 1e-6f)
    {
        // First-order approx
        A = 1.f;
        B = 0.5f;
        C = 1.f / 6.f;
    }
    else
    {
        float inv_t = 1.f / theta;
        float inv_t2 = inv_t * inv_t;
        A = sinf(theta) * inv_t;
        B = (1.f - cosf(theta)) * inv_t2;
        C = (theta - sinf(theta)) * inv_t2 * inv_t;
    }

    // R = I + A [ω]× + B [ω]×²
    T.m[0][0] = 1.f - B * (oy * oy + oz * oz);
    T.m[0][1] = B * ox * oy - A * oz;
    T.m[0][2] = B * ox * oz + A * oy;

    T.m[1][0] = B * ox * oy + A * oz;
    T.m[1][1] = 1.f - B * (ox * ox + oz * oz);
    T.m[1][2] = B * oy * oz - A * ox;

    T.m[2][0] = B * ox * oz - A * oy;
    T.m[2][1] = B * oy * oz + A * ox;
    T.m[2][2] = 1.f - B * (ox * ox + oy * oy);

    // t = V * v,  V = I + B [ω]× + C [ω]×²
    float Vxx = 1.f - C * (oy * oy + oz * oz);
    float Vxy = C * ox * oy - B * oz;
    float Vxz = C * ox * oz + B * oy;
    float Vyx = C * ox * oy + B * oz;
    float Vyy = 1.f - C * (ox * ox + oz * oz);
    float Vyz = C * oy * oz - B * ox;
    float Vzx = C * ox * oz - B * oy;
    float Vzy = C * oy * oz + B * ox;
    float Vzz = 1.f - C * (ox * ox + oy * oy);

    T.m[0][3] = Vxx * vx + Vxy * vy + Vxz * vz;
    T.m[1][3] = Vyx * vx + Vyy * vy + Vyz * vz;
    T.m[2][3] = Vzx * vx + Vzy * vy + Vzz * vz;

    return T;
}

// ─────────────────────────────────────────────
//  Warp di un punto con il campo di deformazione
//  Interpolazione pesata sui K nodi più vicini
// ─────────────────────────────────────────────

__device__ __forceinline__
    float3
    warp_point(
        float3 p,
        const DeformNode *nodes,
        const Mat4 *transforms,
        const int *knn_ids,  // [K_NEIGHBORS]
        const float *knn_ws) // [K_NEIGHBORS]
{
    float3 result = make_float3(0, 0, 0);
    float w_sum = 0;

#pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        int nid = knn_ids[k];
        float w = knn_ws[k];
        if (nid < 0 || w < 1e-8f)
            continue;

        // Applica trasformazione centrata sulla posizione del nodo
        float3 warped = transforms[nid].transform_point_centered(p, nodes[nid].pos);

        result.x += w * warped.x;
        result.y += w * warped.y;
        result.z += w * warped.z;
        w_sum += w;
    }

    if (w_sum > 1e-8f)
    {
        result.x /= w_sum;
        result.y /= w_sum;
        result.z /= w_sum;
    }
    else
    {
        result = p; // nessun nodo vicino → no deformazione
    }
    return result;
}

__device__ __forceinline__
    float3
    warp_normal(
        float3 n,
        const Mat4 *transforms,
        const int *knn_ids,
        const float *knn_ws)
{
    float3 result = make_float3(0, 0, 0);
    float w_sum = 0;

#pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        int nid = knn_ids[k];
        float w = knn_ws[k];
        if (nid < 0 || w < 1e-8f)
            continue;

        float3 warped = transforms[nid].transform_normal(n);
        result.x += w * warped.x;
        result.y += w * warped.y;
        result.z += w * warped.z;
        w_sum += w;
    }

    if (w_sum > 1e-8f)
    {
        result.x /= w_sum;
        result.y /= w_sum;
        result.z /= w_sum;
        result = normalize3(result);
    }
    else
    {
        result = n;
    }
    return result;
}

// ─────────────────────────────────────────────
//  Peso di influenza nodo → punto
//  w(p, n_i) = exp(-||p - x_i||² / (2 * r_i²))
// ─────────────────────────────────────────────

__device__ __forceinline__ float node_weight(float3 p, float3 node_pos, float radius)
{
    float3 d = make_float3(p.x - node_pos.x,
                           p.y - node_pos.y,
                           p.z - node_pos.z);
    float dist2 = dot3(d, d);
    return expf(-dist2 / (2.f * radius * radius));
}

// ─────────────────────────────────────────────
//  Jacobiana del warp rispetto al twist del nodo k
//  ∂(warp(p)) / ∂(twist_k) — matrice 3×6
//  Riga i: [w_k * (R_k * p + t_k)]× e [w_k * I]
//  Per ICP punto-piano: proiettata su normale → vettore 1×6
// ─────────────────────────────────────────────

__device__ __forceinline__ void compute_jacobian_row(
    float3 warped_p, // punto warpato
    float3 node_pos, // centro del nodo k
    float3 normal,   // normale ICP
    float w_k,       // peso nodo k
    float *J_row)    // output: 6 floats
{
    // ICP punto-piano: r = n^T (p - dst).
    // With a left-applied small twist, p' ~= p + omega x (p-c) + v.
    // Thus dr/domega = n^T (omega x local_p)
    //                 = omega^T (local_p x n).
    // The opposite order, cross3(normal, local_p), flips the rotation update.

    float3 local_p = make_float3(warped_p.x - node_pos.x,
                                 warped_p.y - node_pos.y,
                                 warped_p.z - node_pos.z);
    float3 cp = cross3(local_p, normal);

    J_row[0] = w_k * cp.x;     // ∂/∂ω_x
    J_row[1] = w_k * cp.y;     // ∂/∂ω_y
    J_row[2] = w_k * cp.z;     // ∂/∂ω_z
    J_row[3] = w_k * normal.x; // ∂/∂v_x
    J_row[4] = w_k * normal.y; // ∂/∂v_y
    J_row[5] = w_k * normal.z; // ∂/∂v_z
}

// ─────────────────────────────────────────────
//  Inversa di matrice 6×6 (inline, Cholesky)
//  Per il precondizionatore block-diagonale
// ─────────────────────────────────────────────

__device__ __forceinline__ bool invert6x6_cholesky(const float *A, float *Ainv)
{
    // Cholesky L L^T = A, poi risolvi L L^T Ainv = I
    float L[6][6] = {};

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float s = A[i * 6 + j];
            for (int k = 0; k < j; k++)
                s -= L[i][k] * L[j][k];
            if (i == j)
            {
                if (s <= 0)
                    return false; // non PD
                L[i][j] = sqrtf(s);
            }
            else
            {
                L[i][j] = s / L[j][j];
            }
        }
    }

    // Forward/backward substitution per ogni colonna dell'identità
    for (int col = 0; col < 6; col++)
    {
        float y[6] = {};
        // Forward: L y = e_col
        for (int i = 0; i < 6; i++)
        {
            float s = (i == col) ? 1.f : 0.f;
            for (int k = 0; k < i; k++)
                s -= L[i][k] * y[k];
            y[i] = s / L[i][i];
        }
        // Backward: L^T x = y
        float x[6] = {};
        for (int i = 5; i >= 0; i--)
        {
            float s = y[i];
            for (int k = i + 1; k < 6; k++)
                s -= L[k][i] * x[k];
            x[i] = s / L[i][i];
        }
        for (int i = 0; i < 6; i++)
            Ainv[i * 6 + col] = x[i];
    }
    return true;
}
