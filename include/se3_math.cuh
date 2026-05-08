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

__host__ __device__ __forceinline__
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
    // The warp update is applied as a centered dual-quaternion increment. With
    // the quaternion convention used below, the observed descent direction is
    // obtained by projecting normal x local_p.

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

// Quaternion utils
__host__ __device__ __forceinline__
    float4
    quat_mul(float4 a, float4 b)
{
    return make_float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

__host__ __device__ __forceinline__
    float4
    quat_conj(float4 q)
{
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

__host__ __device__ __forceinline__
    float4
    quat_normalize(float4 q)
{
    float n = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-12f)
        return make_float4(0, 0, 0, 1);
    return make_float4(q.x / n, q.y / n, q.z / n, q.w / n);
}
__host__ __device__ __forceinline__ float quat_dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ __forceinline__
    DualQuat
    dq_identity()
{
    return DualQuat::identity();
}

__host__ __device__ __forceinline__
    DualQuat
    mat4_to_dq(const Mat4 &T)
{
    // rotation -> quaternion
    float trace = T.m[0][0] + T.m[1][1] + T.m[2][2];
    float4 qr;

    if (trace > 0)
    {
        float s = sqrtf(trace + 1.0f) * 2.f;
        qr.w = 0.25f * s;
        qr.x = (T.m[2][1] - T.m[1][2]) / s;
        qr.y = (T.m[0][2] - T.m[2][0]) / s;
        qr.z = (T.m[1][0] - T.m[0][1]) / s;
    }
    else if (T.m[0][0] > T.m[1][1] && T.m[0][0] > T.m[2][2])
    {
        float s = sqrtf(1.0f + T.m[0][0] - T.m[1][1] - T.m[2][2]) * 2.f;
        qr.w = (T.m[2][1] - T.m[1][2]) / s;
        qr.x = 0.25f * s;
        qr.y = (T.m[0][1] + T.m[1][0]) / s;
        qr.z = (T.m[0][2] + T.m[2][0]) / s;
    }
    else if (T.m[1][1] > T.m[2][2])
    {
        float s = sqrtf(1.0f + T.m[1][1] - T.m[0][0] - T.m[2][2]) * 2.f;
        qr.w = (T.m[0][2] - T.m[2][0]) / s;
        qr.x = (T.m[0][1] + T.m[1][0]) / s;
        qr.y = 0.25f * s;
        qr.z = (T.m[1][2] + T.m[2][1]) / s;
    }
    else
    {
        float s = sqrtf(1.0f + T.m[2][2] - T.m[0][0] - T.m[1][1]) * 2.f;
        qr.w = (T.m[1][0] - T.m[0][1]) / s;
        qr.x = (T.m[0][2] + T.m[2][0]) / s;
        qr.y = (T.m[1][2] + T.m[2][1]) / s;
        qr.z = 0.25f * s;
    }

    qr = quat_normalize(qr);

    float3 t = make_float3(T.m[0][3], T.m[1][3], T.m[2][3]);
    float4 qt = make_float4(t.x, t.y, t.z, 0.f);

    float4 qd = quat_mul(qt, qr);
    qd.x *= 0.5f;
    qd.y *= 0.5f;
    qd.z *= 0.5f;
    qd.w *= 0.5f;

    DualQuat dq;
    dq.real = qr;
    dq.dual = qd;
    return dq;
}

__host__ __device__ __forceinline__
    DualQuat
    dq_normalize(DualQuat dq)
{
    float norm = sqrtf(
        dq.real.x * dq.real.x +
        dq.real.y * dq.real.y +
        dq.real.z * dq.real.z +
        dq.real.w * dq.real.w);

    if (norm < 1e-12f)
        return dq_identity();

    float inv = 1.f / norm;

    dq.real.x *= inv;
    dq.real.y *= inv;
    dq.real.z *= inv;
    dq.real.w *= inv;

    dq.dual.x *= inv;
    dq.dual.y *= inv;
    dq.dual.z *= inv;
    dq.dual.w *= inv;

    return dq;
}

__host__ __device__ __forceinline__
    DualQuat
    dq_mul(DualQuat a, DualQuat b)
{
    DualQuat r;
    r.real = quat_mul(a.real, b.real);
    float4 ad = quat_mul(a.real, b.dual);
    float4 bd = quat_mul(a.dual, b.real);
    r.dual = make_float4(ad.x + bd.x, ad.y + bd.y, ad.z + bd.z, ad.w + bd.w);
    return dq_normalize(r);
}

__host__ __device__ __forceinline__
    DualQuat
    dq_from_twist(const float *twist)
{
    return mat4_to_dq(exp_se3(twist));
}

__host__ __device__ __forceinline__
    float3
    dq_rotate_vector(DualQuat dq, float3 p)
{
    float4 p_quat = make_float4(p.x, p.y, p.z, 0.f);

    float4 qr = dq.real;
    float4 qr_conj = quat_conj(qr);

    float4 r = quat_mul(quat_mul(qr, p_quat), qr_conj);
    return make_float3(r.x, r.y, r.z);
}

__host__ __device__ __forceinline__
    float3
    dq_translation(DualQuat dq)
{
    float4 t = quat_mul(dq.dual, quat_conj(dq.real));
    return make_float3(2.f * t.x, 2.f * t.y, 2.f * t.z);
}

__host__ __device__ __forceinline__
    float3
    dq_transform_point(DualQuat dq, float3 p)
{
    float3 r = dq_rotate_vector(dq, p);
    float3 t = dq_translation(dq);
    return make_float3(r.x + t.x, r.y + t.y, r.z + t.z);
}

__host__ __device__ __forceinline__
    float3
    dq_transform_point_centered(DualQuat dq, float3 p, float3 center)
{
    float3 rp = dq_transform_point(dq, p);
    float3 rc = dq_rotate_vector(dq, center);
    return make_float3(rp.x + center.x - rc.x,
                       rp.y + center.y - rc.y,
                       rp.z + center.z - rc.z);
}

__host__ __device__ __forceinline__
    DualQuat
    dq_from_rotation_translation(float4 qr, float3 t)
{
    DualQuat dq;
    dq.real = quat_normalize(qr);
    float4 qt = make_float4(t.x, t.y, t.z, 0.f);
    dq.dual = quat_mul(qt, dq.real);
    dq.dual.x *= 0.5f;
    dq.dual.y *= 0.5f;
    dq.dual.z *= 0.5f;
    dq.dual.w *= 0.5f;
    return dq_normalize(dq);
}

__host__ __device__ __forceinline__
    DualQuat
    dq_centered(DualQuat dq, float3 center)
{
    float3 t = dq_translation(dq);
    float3 rc = dq_rotate_vector(dq, center);
    float3 centered_t = make_float3(t.x + center.x - rc.x,
                                    t.y + center.y - rc.y,
                                    t.z + center.z - rc.z);
    return dq_from_rotation_translation(dq.real, centered_t);
}

__host__ __device__ __forceinline__
    Mat4
    dq_to_mat4(DualQuat dq)
{
    dq = dq_normalize(dq);
    float x = dq.real.x, y = dq.real.y, z = dq.real.z, w = dq.real.w;
    float3 t = dq_translation(dq);

    Mat4 T;
    T.m[0][0] = 1.f - 2.f * (y * y + z * z);
    T.m[0][1] = 2.f * (x * y - z * w);
    T.m[0][2] = 2.f * (x * z + y * w);
    T.m[0][3] = t.x;
    T.m[1][0] = 2.f * (x * y + z * w);
    T.m[1][1] = 1.f - 2.f * (x * x + z * z);
    T.m[1][2] = 2.f * (y * z - x * w);
    T.m[1][3] = t.y;
    T.m[2][0] = 2.f * (x * z - y * w);
    T.m[2][1] = 2.f * (y * z + x * w);
    T.m[2][2] = 1.f - 2.f * (x * x + y * y);
    T.m[2][3] = t.z;
    T.m[3][0] = 0.f;
    T.m[3][1] = 0.f;
    T.m[3][2] = 0.f;
    T.m[3][3] = 1.f;
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
        const DualQuat *transforms,
        const int *knn_ids,
        const float *knn_ws)
{
    DualQuat dq_blend;
    dq_blend.real = make_float4(0, 0, 0, 0);
    dq_blend.dual = make_float4(0, 0, 0, 0);

    float w_sum = 0;

#pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        int nid = knn_ids[k];
        float w = knn_ws[k];

        if (nid < 0 || w < 1e-8f)
            continue;

        DualQuat dq = dq_centered(transforms[nid], nodes[nid].pos);

        // IMPORTANT: antipodality fix
        if (quat_dot(dq.real, dq_blend.real) < 0)
        {
            dq.real.x *= -1;
            dq.real.y *= -1;
            dq.real.z *= -1;
            dq.real.w *= -1;

            dq.dual.x *= -1;
            dq.dual.y *= -1;
            dq.dual.z *= -1;
            dq.dual.w *= -1;
        }

        dq_blend.real.x += w * dq.real.x;
        dq_blend.real.y += w * dq.real.y;
        dq_blend.real.z += w * dq.real.z;
        dq_blend.real.w += w * dq.real.w;

        dq_blend.dual.x += w * dq.dual.x;
        dq_blend.dual.y += w * dq.dual.y;
        dq_blend.dual.z += w * dq.dual.z;
        dq_blend.dual.w += w * dq.dual.w;

        w_sum += w;
    }

    if (w_sum < 1e-8f)
        return p;

    // normalize
    dq_blend = dq_normalize(dq_blend);

    return dq_transform_point(dq_blend, p);
}

__device__ __forceinline__
    float3
    warp_normal(
        float3 n,
        const DualQuat *transforms,
        const int *knn_ids,
        const float *knn_ws)
{
    DualQuat dq_blend;
    dq_blend.real = make_float4(0, 0, 0, 0);
    dq_blend.dual = make_float4(0, 0, 0, 0);

#pragma unroll
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        int nid = knn_ids[k];
        float w = knn_ws[k];

        if (nid < 0 || w < 1e-8f)
            continue;

        DualQuat dq = transforms[nid];

        if (quat_dot(dq.real, dq_blend.real) < 0)
        {
            dq.real.x *= -1;
            dq.real.y *= -1;
            dq.real.z *= -1;
            dq.real.w *= -1;
        }

        dq_blend.real.x += w * dq.real.x;
        dq_blend.real.y += w * dq.real.y;
        dq_blend.real.z += w * dq.real.z;
        dq_blend.real.w += w * dq.real.w;
    }

    dq_blend = dq_normalize(dq_blend);

    float4 q = dq_blend.real;
    float4 q_conj = quat_conj(q);

    float4 n_quat = make_float4(n.x, n.y, n.z, 0.f);
    float4 r = quat_mul(quat_mul(q, n_quat), q_conj);

    return normalize3(make_float3(r.x, r.y, r.z));
}
