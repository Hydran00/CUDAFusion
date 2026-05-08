#include "solver.h"
#include "se3_math.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <vector>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

__device__ float compute_huber_weight(float residual, float delta)
{
    float abs_r = fabsf(residual);
    if (abs_r <= delta)
    {
        return 1.0f; // Zona quadratica (normale)
    }
    else
    {
        return delta / abs_r; // Zona lineare (smorza l'outlier)
    }
}
__device__ int find_block(
    int row, int col,
    const int *row_ptr,
    const int *col_idx)
{
    int lo = row_ptr[row];
    int hi = row_ptr[row + 1] - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        int v = col_idx[mid];
        if (v == col)
            return mid;
        if (v < col)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

// ─────────────────────────────────────────────
//  Kernel: assemblaggio sistema lineare
//  Contributo data term (ICP punto-piano)
//  Un thread per corrispondenza valida
// ─────────────────────────────────────────────

__global__ void assemble_data_term_kernel(
    const Correspondence *corrs,
    int num_corrs,
    const DeformNode *nodes,
    const DualQuat *transforms,
    float *A_values,
    float *b_values,
    const int *row_ptr,
    const int *col_idx,
    int num_nodes,
    float huber_delta) // <--- Nuovo parametro
{
    int cidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cidx >= num_corrs)
        return;

    const Correspondence &c = corrs[cidx];
    if (!c.valid)
        return;

    float3 src = c.src;
    float3 dst = c.dst;
    float3 normal = c.normal;

    // 1. Calcolo del residuo punto-piano
    float residual = dot3(normal, make_float3(src.x - dst.x, src.y - dst.y, src.z - dst.z));

    // 2. Calcolo del peso robusto (Huber)
    // Se il residuo è grande, huber_w sarà < 1.0
    float huber_w = compute_huber_weight(residual, huber_delta);
    float final_weight = fmaxf(c.weight, 0.0f) * huber_w;

    if (final_weight <= 1e-8f)
        return;

    // Cache delle Jacobiane per i vicini per evitare ricalcoli inutili
    float Ji_cache[K_NEIGHBORS][6];
    bool valid_neighbor[K_NEIGHBORS] = {false};

    // Pre-calcolo delle Jacobiane per questa riga
    for (int k = 0; k < K_NEIGHBORS; k++)
    {
        int ni = c.node_ids[k];
        float w_i = c.node_ws[k];
        if (ni >= 0 && ni < num_nodes && w_i > 1e-8f)
        {
            compute_jacobian_row(src, nodes[ni].pos, normal, w_i, Ji_cache[k]);
            valid_neighbor[k] = true;

            // Accumulo in b (RHS): -weight * J^T * r
            for (int a = 0; a < 6; a++)
            {
                atomicAdd(&b_values[ni * 6 + a], -final_weight * Ji_cache[k][a] * residual);
            }
        }
    }

    // 3. Accumulo in A (Hessiana approssimata): J^T * J
    for (int ki = 0; ki < K_NEIGHBORS; ki++)
    {
        if (!valid_neighbor[ki])
            continue;
        int ni = c.node_ids[ki];

        for (int kj = 0; kj < K_NEIGHBORS; kj++)
        {
            if (!valid_neighbor[kj])
                continue;
            int nj = c.node_ids[kj];

            // Trova il blocco (ni, nj) nella struttura BSR
            int block_pos = find_block(ni, nj, row_ptr, col_idx);
            if (block_pos < 0)
                continue;

            float *blk = A_values + block_pos * 36;
            for (int a = 0; a < 6; a++)
            {
                for (int b = 0; b < 6; b++)
                {
                    // Notare che usiamo final_weight (che include Huber)
                    atomicAdd(&blk[a * 6 + b], final_weight * Ji_cache[ki][a] * Ji_cache[kj][b]);
                }
            }
        }
    }
}

__device__ void compute_arap_jacobian(
    const float3 &p, // p = T * x
    float J[6][3])   // 6 parametri × 3 output (trasposta rispetto forma classica)
{
    // Rotazione (parte skew-symmetric)
    J[0][0] = 0.0f;
    J[0][1] = p.z;
    J[0][2] = -p.y;
    J[1][0] = -p.z;
    J[1][1] = 0.0f;
    J[1][2] = p.x;
    J[2][0] = p.y;
    J[2][1] = -p.x;
    J[2][2] = 0.0f;

    // Traslazione (identità)
    J[3][0] = 1.0f;
    J[3][1] = 0.0f;
    J[3][2] = 0.0f;
    J[4][0] = 0.0f;
    J[4][1] = 1.0f;
    J[4][2] = 0.0f;
    J[5][0] = 0.0f;
    J[5][1] = 0.0f;
    J[5][2] = 1.0f;
}

__global__ void add_diagonal_damping_kernel(
    float *A_values,
    const int *row_ptr,
    const int *col_idx,
    int num_nodes,
    float lambda)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes)
        return;

    int blk = find_block(ni, ni, row_ptr, col_idx);
    if (blk < 0)
        return;

    float *diag = A_values + blk * BLOCK_SIZE;
    for (int k = 0; k < BLOCK_DIM; k++)
    {
        float current_diag = diag[k * BLOCK_DIM + k];
        diag[k * BLOCK_DIM + k] = current_diag + lambda * (current_diag + 1.0f);
    }
    // diag[k * BLOCK_DIM + k] += lambda;
}

// ─────────────────────────────────────────────
//  Kernel: termine smoothness ARAP
//  Per ogni arco (i,j) del grafo:
//  E_smooth = ||T_i * x_j - T_j * x_j||²
// ─────────────────────────────────────────────

__global__ void assemble_smooth_term_kernel(
    const DeformNode *nodes,
    const DualQuat *transforms,
    int num_nodes,
    float lambda,
    float *A_values,
    float *b_values,
    const int *row_ptr,
    const int *col_idx)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes)
        return;

    const DeformNode &node_i = nodes[ni];
    // Delta per Huber sullo smoothness: es. 5cm.
    // Se un arco si allunga più di così, il suo contributo viene smorzato.
    const float huber_smooth_delta = 0.05f;
    // Adaptive strain weighting: if an edge stretches/compresses by roughly
    // alpha or more, its ARAP stiffness drops smoothly instead of forcing a
    // bad bridge to stay rigid.
    const float strain_alpha = 0.25f;

    for (int k = 0; k < node_i.num_neighbors; k++)
    {
        int nj = node_i.neighbors[k];
        if (nj < 0 || nj >= num_nodes)
            continue;

        // Processiamo ogni arco non orientato una sola volta per coppia
        if (nj <= ni)
            continue;

        float original_w = fmaxf(node_i.neighbor_w[k], 0.0f);
        if (original_w <= 1e-8f)
            continue;

        float3 xj = nodes[nj].pos;
        float3 xi = node_i.pos;

        float3 Ti_xj = dq_transform_point_centered(transforms[ni], xj, node_i.pos);
        float3 Ti_xi = dq_transform_point_centered(transforms[ni], xi, node_i.pos);
        float3 Tj_xj = dq_transform_point_centered(transforms[nj], xj, nodes[nj].pos);

        float3 rest_edge = make_float3(xi.x - xj.x, xi.y - xj.y, xi.z - xj.z);
        float3 curr_edge = make_float3(Ti_xi.x - Tj_xj.x,
                                       Ti_xi.y - Tj_xj.y,
                                       Ti_xi.z - Tj_xj.z);
        float dist_rest = sqrtf(rest_edge.x * rest_edge.x +
                                rest_edge.y * rest_edge.y +
                                rest_edge.z * rest_edge.z);
        float dist_curr = sqrtf(curr_edge.x * curr_edge.x +
                                curr_edge.y * curr_edge.y +
                                curr_edge.z * curr_edge.z);
        float strain = fabsf(dist_curr - dist_rest) / fmaxf(dist_rest, 1e-6f);
        float strain_ratio = strain / strain_alpha;
        float strain_weight = expf(-(strain_ratio * strain_ratio));

        float3 r = make_float3(
            Ti_xj.x - Tj_xj.x,
            Ti_xj.y - Tj_xj.y,
            Ti_xj.z - Tj_xj.z);

        // Calcolo della norma del residuo ARAP per questo arco
        float r_norm = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        // Calcolo del peso di Huber per lo smoothness
        float huber_w = compute_huber_weight(r_norm, huber_smooth_delta);
        float effective_lambda = lambda * original_w * huber_w * strain_weight;
        if (effective_lambda <= 1e-8f)
            continue;

        float Ji[6][3], Jj[6][3];

        float3 qi = make_float3(Ti_xj.x - node_i.pos.x,
                                Ti_xj.y - node_i.pos.y,
                                Ti_xj.z - node_i.pos.z);
        float3 qj = make_float3(Tj_xj.x - nodes[nj].pos.x,
                                Tj_xj.y - nodes[nj].pos.y,
                                Tj_xj.z - nodes[nj].pos.z);

        compute_arap_jacobian(qi, Ji);
        compute_arap_jacobian(qj, Jj);

        int ii = find_block(ni, ni, row_ptr, col_idx);
        int jj = find_block(nj, nj, row_ptr, col_idx);
        int ij = find_block(ni, nj, row_ptr, col_idx);
        int ji = find_block(nj, ni, row_ptr, col_idx);

        // Utilizziamo effective_lambda invece del lambda fisso per pesare i contributi

        // --- A(ii) ---
        if (ii >= 0)
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                {
                    float v = 0;
                    for (int c = 0; c < 3; c++)
                        v += Ji[a][c] * Ji[b][c];
                    atomicAdd(&A_values[ii * 36 + a * 6 + b], effective_lambda * v);
                }

        // --- A(jj) ---
        if (jj >= 0)
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                {
                    float v = 0;
                    for (int c = 0; c < 3; c++)
                        v += Jj[a][c] * Jj[b][c];
                    atomicAdd(&A_values[jj * 36 + a * 6 + b], effective_lambda * v);
                }

        // --- A(ij) ---
        if (ij >= 0)
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                {
                    float v = 0;
                    for (int c = 0; c < 3; c++)
                        v += Ji[a][c] * Jj[b][c];
                    atomicAdd(&A_values[ij * 36 + a * 6 + b], -effective_lambda * v);
                }

        // --- A(ji) ---
        if (ji >= 0)
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                {
                    float v = 0;
                    for (int c = 0; c < 3; c++)
                        v += Jj[a][c] * Ji[b][c];
                    atomicAdd(&A_values[ji * 36 + a * 6 + b], -effective_lambda * v);
                }

        // --- RHS (b) ---
        float r_arr[3] = {r.x, r.y, r.z};

        for (int a = 0; a < 6; a++)
        {
            float vi = 0, vj = 0;
            for (int c = 0; c < 3; c++)
            {
                vi += Ji[a][c] * r_arr[c];
                vj += Jj[a][c] * r_arr[c];
            }
            atomicAdd(&b_values[ni * 6 + a], -effective_lambda * vi);
            atomicAdd(&b_values[nj * 6 + a], effective_lambda * vj);
        }
    }
}
// ─────────────────────────────────────────────
//  Kernel: precondizionatore block-diagonale
//  Inverte ogni blocco diagonale 6x6
// ─────────────────────────────────────────────

__global__ void compute_preconditioner_kernel(
    const float *A_values,
    const int *row_ptr,
    const int *col_idx,
    float *M_inv,
    int num_nodes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes)
        return;

    // Trova il blocco diagonale (ni, ni)
    float diag[36] = {};
    for (int r = row_ptr[ni]; r < row_ptr[ni + 1]; r++)
    {
        if (col_idx[r] == ni)
        {
            for (int k = 0; k < 36; k++)
                diag[k] = A_values[r * BLOCK_SIZE + k];
            break;
        }
    }

    // Aggiungi damping per stabilità numerica (Levenberg)
    for (int k = 0; k < 6; k++)
        diag[k * 6 + k] += 1e-4f;

    // Inverti 6x6 con Cholesky
    float inv[36];
    if (!invert6x6_cholesky(diag, inv))
    {
        // Fallback: precondizionatore diagonale semplice
        const float eps = 1e-6f;
        for (int k = 0; k < 36; k++)
            inv[k] = 0.f;
        for (int k = 0; k < 6; k++)
            inv[k * 6 + k] = 1.f / (diag[k * 6 + k] + eps);
    }

    for (int k = 0; k < 36; k++)
        M_inv[ni * BLOCK_SIZE + k] = inv[k];
}

// ─────────────────────────────────────────────
//  Kernel: SpMV block-sparso  y = A * x
//  Un blocco per block-row (un nodo)
// ─────────────────────────────────────────────

__global__ void bsr_spmv_kernel(
    const float *A_values,
    const int *row_ptr,
    const int *col_idx,
    const float *x,
    float *y,
    int num_nodes)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = num_nodes * BLOCK_DIM;
    if (gid >= N)
        return;

    int ni = gid / BLOCK_DIM;
    int row = gid % BLOCK_DIM;

    float sum = 0.0f;
    for (int b = row_ptr[ni]; b < row_ptr[ni + 1]; b++)
    {
        int nj = col_idx[b];
        const float *blk = A_values + b * BLOCK_SIZE;
        const float *xj = x + nj * BLOCK_DIM;
        for (int c = 0; c < BLOCK_DIM; c++)
            sum += blk[row * BLOCK_DIM + c] * xj[c];
    }
    y[gid] = sum;
}

// ─────────────────────────────────────────────
//  Kernel: applica precondizionatore  z = M⁻¹ r
// ─────────────────────────────────────────────

__global__ void apply_precond_kernel(
    const float *M_inv,
    const float *r,
    float *z,
    int num_nodes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes)
        return;

    const float *Mi = M_inv + ni * BLOCK_SIZE;
    const float *ri = r + ni * BLOCK_DIM;
    float *zi = z + ni * BLOCK_DIM;

#pragma unroll
    for (int row = 0; row < BLOCK_DIM; row++)
    {
        float acc = 0;
#pragma unroll
        for (int col = 0; col < BLOCK_DIM; col++)
            acc += Mi[row * BLOCK_DIM + col] * ri[col];
        zi[row] = acc;
    }
}

// ─────────────────────────────────────────────
//  Kernel: AXPY  y = alpha*x + y
// ─────────────────────────────────────────────

__global__ void axpy_kernel(float alpha, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

// y = x + alpha*y
__global__ void xpay_kernel(float alpha, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + alpha * y[i];
}

// ─────────────────────────────────────────────
//  GaussNewtonSolver — implementazione
// ─────────────────────────────────────────────

GaussNewtonSolver::GaussNewtonSolver(const Params &p, int max_nodes)
    : params_(p), max_nodes_(max_nodes)
{
    int n6 = max_nodes * BLOCK_DIM;
    b_.allocate(n6);
    r_.allocate(n6);
    z_.allocate(n6);
    p_.allocate(n6);
    Ap_.allocate(n6);
    M_inv_.allocate(max_nodes * BLOCK_SIZE);
}

void GaussNewtonSolver::build_sparsity_pattern(
    const DeformNode *h_nodes, int num_nodes)
{
    // Each row contains the diagonal plus graph-neighbor blocks in both
    // directions. PCG expects a symmetric system; the node graph is stored
    // directionally because new nodes only point to previous neighbors.
    std::vector<std::vector<int>> row_cols(num_nodes);
    for (int i = 0; i < num_nodes; i++)
        row_cols[i].push_back(i);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int k = 0; k < h_nodes[i].num_neighbors; k++)
        {
            int nb = h_nodes[i].neighbors[k];
            if (nb < 0 || nb >= num_nodes)
                continue;
            row_cols[i].push_back(nb);
            row_cols[nb].push_back(i);
        }
    }

    h_row_ptr_.resize(num_nodes + 1);
    h_col_idx_.clear();

    for (int i = 0; i < num_nodes; i++)
    {
        h_row_ptr_[i] = (int)h_col_idx_.size();
        std::vector<int> &cols = row_cols[i];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        for (int c : cols)
            h_col_idx_.push_back(c);
    }
    h_row_ptr_[num_nodes] = (int)h_col_idx_.size();
    current_num_blocks_ = (int)h_col_idx_.size();

    // Upload to GPU
    A_.allocate(num_nodes, current_num_blocks_);
    cudaMemcpy(A_.row_ptr.data, h_row_ptr_.data(),
               (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_.col_idx.data, h_col_idx_.data(),
               current_num_blocks_ * sizeof(int), cudaMemcpyHostToDevice);
}

void GaussNewtonSolver::solve(
    const DeviceArray<Correspondence> &corrs,
    int num_corrs,
    const DeformNode *d_nodes,
    const DualQuat *d_transforms,
    int num_nodes,
    DeviceArray<float> &delta_x)
{
    delta_x.allocate(num_nodes * BLOCK_DIM);

    // Build sparsity pattern only when topology (num_nodes) changes
    if (num_nodes != last_num_nodes_)
    {
        std::vector<DeformNode> h_nodes(num_nodes);
        cudaMemcpy(h_nodes.data(), d_nodes,
                   num_nodes * sizeof(DeformNode), cudaMemcpyDeviceToHost);
        build_sparsity_pattern(h_nodes.data(), num_nodes);
        last_num_nodes_ = num_nodes;
    }

    // One linearization per frame. Repeating this loop without applying the
    // update and re-raycasting solves the same system again and can only waste
    // time; the caller applies the returned increment once.
    assemble_system(corrs, num_corrs, d_nodes, d_transforms, num_nodes);
    compute_preconditioner(num_nodes);
    delta_x.zero();
    pcg_solve(num_nodes, delta_x);
}

void GaussNewtonSolver::assemble_system(
    const DeviceArray<Correspondence> &corrs,
    int num_corrs,
    const DeformNode *d_nodes,
    const DualQuat *d_transforms,
    int num_nodes)
{
    A_.zero_values();
    b_.zero();

    int block = 256;
    int grid = (num_corrs + block - 1) / block;

    // Data term
    assemble_data_term_kernel<<<grid, block>>>(
        corrs.data, num_corrs, d_nodes, d_transforms,
        A_.values.data, b_.data,
        A_.row_ptr.data, A_.col_idx.data, num_nodes, 0.2f /*huber delta*/);

    // Smoothness term
    int grid_nodes = (num_nodes + block - 1) / block;
    assemble_smooth_term_kernel<<<grid_nodes, block>>>(
        d_nodes, d_transforms, num_nodes,
        params_.lambda_smooth,
        A_.values.data, b_.data,
        A_.row_ptr.data, A_.col_idx.data);

    add_diagonal_damping_kernel<<<grid_nodes, block>>>(
        A_.values.data, A_.row_ptr.data, A_.col_idx.data,
        num_nodes, params_.lambda_damping);

    cudaDeviceSynchronize();
}

void GaussNewtonSolver::compute_preconditioner(int num_nodes)
{
    int block = 128;
    int grid = (num_nodes + block - 1) / block;
    compute_preconditioner_kernel<<<grid, block>>>(
        A_.values.data, A_.row_ptr.data, A_.col_idx.data,
        M_inv_.data, num_nodes);
    cudaDeviceSynchronize();
}

float GaussNewtonSolver::pcg_solve(int num_nodes, DeviceArray<float> &x_out)
{
    int n = num_nodes * BLOCK_DIM;

    // r = b - A*x  (robust for non-zero initial x_out)
    // Ap_ used as temporary: Ap_ = A * x_out
    bsr_spmv(A_, x_out, Ap_, num_nodes);                                       // Ap_ = A*x_out
    cudaMemcpy(r_.data, b_.data, n * sizeof(float), cudaMemcpyDeviceToDevice); // r_ = b
    axpy(-1.f, Ap_, r_, n);                                                    // r = b - A*x_out

    // z = M⁻¹ r
    apply_preconditioner(r_, z_, num_nodes);

    // p = z
    cudaMemcpy(p_.data, z_.data, n * sizeof(float), cudaMemcpyDeviceToDevice);

    float rz_old = dot(r_, z_, n);
    // optional debug: print initial norms and PCG settings
    if (params_.debug)
    {
        float bnorm = sqrtf(dot(b_, b_, n));
        printf("[PCG] start nodes=%d n=%d ||b||=%e rz_old=%e tol=%e maxit=%d\n",
               num_nodes, n, bnorm, rz_old, params_.pcg_tolerance, params_.pcg_iterations);
    }
    int iter_used = 0;
    for (int iter = 0; iter < params_.pcg_iterations; iter++)
    {
        // Ap = A * p
        bsr_spmv(A_, p_, Ap_, num_nodes);

        float pAp = dot(p_, Ap_, n);
        if (fabsf(pAp) < 1e-12f)
        {
            if (params_.debug)
                printf("[PCG] break: pAp too small (%e) at iter %d\n", pAp, iter);
            iter_used = iter;
            break;
        }
        float alpha = rz_old / pAp;

        // x = x + alpha * p
        axpy(alpha, p_, x_out, n);

        // r = r - alpha * Ap
        axpy(-alpha, Ap_, r_, n);

        float r_norm = sqrtf(dot(r_, r_, n));
        if (params_.debug)
        {
            printf("[PCG] iter %d: r_norm=%e rz=%e pAp=%e alpha=%e\n",
                   iter, r_norm, rz_old, pAp, alpha);
        }
        iter_used = iter + 1;
        if (r_norm < params_.pcg_tolerance)
            break;

        // z = M⁻¹ r
        apply_preconditioner(r_, z_, num_nodes);

        float rz_new = dot(r_, z_, n);
        float beta = rz_new / rz_old;

        // p = z + beta * p
        xpay(beta, z_, p_, n);

        rz_old = rz_new;
    }
    float final_r = sqrtf(dot(r_, r_, n));
    if (params_.debug)
        printf("[PCG] end: iters=%d final_r=%e\n", iter_used, final_r);
    return final_r;
}

void GaussNewtonSolver::bsr_spmv(
    const BSRMatrix &A,
    const DeviceArray<float> &x,
    DeviceArray<float> &y,
    int num_nodes)
{
    int n = num_nodes * BLOCK_DIM;
    int block = 256;
    int grid = (n + block - 1) / block;
    bsr_spmv_kernel<<<grid, block>>>(
        A.values.data, A.row_ptr.data, A.col_idx.data,
        x.data, y.data, num_nodes);
    cudaDeviceSynchronize();
}

void GaussNewtonSolver::apply_preconditioner(
    const DeviceArray<float> &r,
    DeviceArray<float> &z,
    int num_nodes)
{
    int block = 128;
    int grid = (num_nodes + block - 1) / block;
    apply_precond_kernel<<<grid, block>>>(M_inv_.data, r.data, z.data, num_nodes);
    cudaDeviceSynchronize();
}

float GaussNewtonSolver::dot(
    const DeviceArray<float> &a,
    const DeviceArray<float> &b,
    int n)
{
    thrust::device_ptr<const float> pa(a.data);
    thrust::device_ptr<const float> pb(b.data);
    return thrust::inner_product(pa, pa + n, pb, 0.f);
}

void GaussNewtonSolver::axpy(float alpha, const DeviceArray<float> &x,
                             DeviceArray<float> &y, int n)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(alpha, x.data, y.data, n);
    cudaDeviceSynchronize();
}

void GaussNewtonSolver::xpay(float alpha, const DeviceArray<float> &x,
                             DeviceArray<float> &y, int n)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    xpay_kernel<<<grid, block>>>(alpha, x.data, y.data, n);
    cudaDeviceSynchronize();
}
