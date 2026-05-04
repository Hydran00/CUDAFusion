#include "solver.h"
#include "se3_math.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <vector>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>

// ─────────────────────────────────────────────
//  Kernel: assemblaggio sistema lineare
//  Contributo data term (ICP punto-piano)
//  Un thread per corrispondenza valida
// ─────────────────────────────────────────────

__global__ void assemble_data_term_kernel(
    const Correspondence* corrs,
    int                   num_corrs,
    const DeformNode*     nodes,
    const Mat4*           transforms,
    float*                A_values,    // BSR blocks [num_blocks * 36]
    float*                b_values,    // RHS [num_nodes * 6]
    const int*            row_ptr,     // BSR row pointers
    const int*            col_idx,     // BSR col indices
    int                   num_nodes)
{
    int cidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cidx >= num_corrs) return;

    const Correspondence& c = corrs[cidx];
    if (!c.valid) return;

    float3 src    = c.src;
    float3 dst    = c.dst;
    float3 normal = c.normal;
    float  weight = fmaxf(c.weight, 0.0f);
    if (weight <= 1e-8f) return;

    // Residuo ICP punto-piano
    float residual = dot3(normal, make_float3(src.x-dst.x, src.y-dst.y, src.z-dst.z));

    // Per ogni coppia di nodi (ki, kj) che influenzano questo punto
    for (int ki = 0; ki < K_NEIGHBORS; ki++) {
        int   ni  = c.node_ids[ki];
        float w_i = c.node_ws [ki];
        if (ni < 0 || w_i < 1e-8f) continue;

        // Jacobiana riga per nodo i
        float Ji[6];
        compute_jacobian_row(src, nodes[ni].pos, normal, w_i, Ji);

        // Contributo a b: J_i^T * r
        for (int a = 0; a < 6; a++)
            atomicAdd(&b_values[ni*6 + a], -weight * Ji[a] * residual);

        // Contributo a A: J_i^T * J_j per tutti j
        for (int kj = 0; kj < K_NEIGHBORS; kj++) {
            int   nj  = c.node_ids[kj];
            float w_j = c.node_ws [kj];
            if (nj < 0 || w_j < 1e-8f) continue;

            // Trova il blocco (ni, nj) nella BSR
            int block_pos = -1;
            for (int r = row_ptr[ni]; r < row_ptr[ni+1]; r++) {
                if (col_idx[r] == nj) { block_pos = r; break; }
            }
            if (block_pos < 0) continue;

            // Jacobiana per nodo j
            float Jj[6];
            compute_jacobian_row(src, nodes[nj].pos, normal, w_j, Jj);

            // Accumula J_i^T * J_j nel blocco (ni, nj)
            float* blk = A_values + block_pos * BLOCK_SIZE;
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                    atomicAdd(&blk[a*6 + b], weight * Ji[a] * Jj[b]);
        }
    }
}

__device__ void compute_arap_jacobian(
    const float3& p,   // p = T * x
    float J[6][3])     // 6 parametri × 3 output (trasposta rispetto forma classica)
{
    // Rotazione (parte skew-symmetric)
    J[0][0] = 0.0f;     J[0][1] =  p.z;    J[0][2] = -p.y;
    J[1][0] = -p.z;     J[1][1] = 0.0f;    J[1][2] =  p.x;
    J[2][0] =  p.y;     J[2][1] = -p.x;    J[2][2] = 0.0f;

    // Traslazione (identità)
    J[3][0] = 1.0f;     J[3][1] = 0.0f;    J[3][2] = 0.0f;
    J[4][0] = 0.0f;     J[4][1] = 1.0f;    J[4][2] = 0.0f;
    J[5][0] = 0.0f;     J[5][1] = 0.0f;    J[5][2] = 1.0f;
}

__device__ int find_block(
    int row, int col,
    const int* row_ptr,
    const int* col_idx)
{
    for (int r = row_ptr[row]; r < row_ptr[row+1]; r++)
        if (col_idx[r] == col) return r;
    return -1;
}

__global__ void add_diagonal_damping_kernel(
    float*      A_values,
    const int*  row_ptr,
    const int*  col_idx,
    int         num_nodes,
    float       lambda)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes) return;

    int blk = find_block(ni, ni, row_ptr, col_idx);
    if (blk < 0) return;

    float* diag = A_values + blk * BLOCK_SIZE;
    for (int k = 0; k < BLOCK_DIM; k++)
        diag[k * BLOCK_DIM + k] += lambda;
}

// ─────────────────────────────────────────────
//  Kernel: termine smoothness ARAP
//  Per ogni arco (i,j) del grafo:
//  E_smooth = ||T_i * x_j - T_j * x_j||²
// ─────────────────────────────────────────────

__global__ void assemble_smooth_term_kernel(
    const DeformNode* nodes,
    const Mat4*       transforms,
    int               num_nodes,
    float             lambda,
    float*            A_values,
    float*            b_values,
    const int*        row_ptr,
    const int*        col_idx)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes) return;

    const DeformNode& node_i = nodes[ni];

    for (int k = 0; k < node_i.num_neighbors; k++) {
        int nj = node_i.neighbors[k];
        if (nj < 0 || nj >= num_nodes) continue;

        float3 xj = nodes[nj].pos;

        float3 Ti_xj = transforms[ni].transform_point_centered(xj, node_i.pos);
        float3 Tj_xj = transforms[nj].transform_point_centered(xj, nodes[nj].pos);

        float3 r = make_float3(
            Ti_xj.x - Tj_xj.x,
            Ti_xj.y - Tj_xj.y,
            Ti_xj.z - Tj_xj.z);

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

        // --- A(ii) ---
        if (ii >= 0)
            for (int a=0;a<6;a++)
                for (int b=0;b<6;b++) {
                    float v = 0;
                    for (int c=0;c<3;c++)
                        v += Ji[a][c]*Ji[b][c];
                    atomicAdd(&A_values[ii*36 + a*6+b], lambda * v);
                }

        // --- A(jj) ---
        if (jj >= 0)
            for (int a=0;a<6;a++)
                for (int b=0;b<6;b++) {
                    float v = 0;
                    for (int c=0;c<3;c++)
                        v += Jj[a][c]*Jj[b][c];
                    atomicAdd(&A_values[jj*36 + a*6+b], lambda * v);
                }

        // --- A(ij) ---
        if (ij >= 0)
            for (int a=0;a<6;a++)
                for (int b=0;b<6;b++) {
                    float v = 0;
                    for (int c=0;c<3;c++)
                        v += Ji[a][c]*Jj[b][c];
                    atomicAdd(&A_values[ij*36 + a*6+b], -lambda * v);
                }

        // --- A(ji) ---
        if (ji >= 0)
            for (int a=0;a<6;a++)
                for (int b=0;b<6;b++) {
                    float v = 0;
                    for (int c=0;c<3;c++)
                        v += Jj[a][c]*Ji[b][c];
                    atomicAdd(&A_values[ji*36 + a*6+b], -lambda * v);
                }

        // --- RHS ---
        float r_arr[3] = {r.x, r.y, r.z};

        for (int a=0;a<6;a++) {
            float vi = 0, vj = 0;
            for (int c=0;c<3;c++) {
                vi += Ji[a][c]*r_arr[c];
                vj += Jj[a][c]*r_arr[c];
            }
            atomicAdd(&b_values[ni*6+a], -lambda * vi);
            atomicAdd(&b_values[nj*6+a],  lambda * vj);
        }
    }
}

// ─────────────────────────────────────────────
//  Kernel: precondizionatore block-diagonale
//  Inverte ogni blocco diagonale 6x6
// ─────────────────────────────────────────────

__global__ void compute_preconditioner_kernel(
    const float* A_values,
    const int*   row_ptr,
    const int*   col_idx,
    float*       M_inv,
    int          num_nodes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes) return;

    // Trova il blocco diagonale (ni, ni)
    float diag[36] = {};
    for (int r = row_ptr[ni]; r < row_ptr[ni+1]; r++) {
        if (col_idx[r] == ni) {
            for (int k = 0; k < 36; k++)
                diag[k] = A_values[r * BLOCK_SIZE + k];
            break;
        }
    }

    // Aggiungi damping per stabilità numerica (Levenberg)
    for (int k = 0; k < 6; k++)
        diag[k*6 + k] += 1e-4f;

    // Inverti 6x6 con Cholesky
    float inv[36];
    if (!invert6x6_cholesky(diag, inv)) {
        // Fallback: precondizionatore diagonale semplice
        for (int k = 0; k < 36; k++) inv[k] = 0;
        for (int k = 0; k < 6; k++)
            inv[k*6+k] = (diag[k*6+k] > 1e-8f) ? 1.f/diag[k*6+k] : 1.f;
    }

    for (int k = 0; k < 36; k++)
        M_inv[ni * BLOCK_SIZE + k] = inv[k];
}

// ─────────────────────────────────────────────
//  Kernel: SpMV block-sparso  y = A * x
//  Un blocco per block-row (un nodo)
// ─────────────────────────────────────────────

__global__ void bsr_spmv_kernel(
    const float* A_values,
    const int*   row_ptr,
    const int*   col_idx,
    const float* x,
    float*       y,
    int          num_nodes)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_nodes * 6) return;

    int ni  = gid / 6;
    int row = gid % 6;

    float sum = 0.0f;

    for (int b = row_ptr[ni]; b < row_ptr[ni+1]; b++) {
        int nj = col_idx[b];
        const float* blk = A_values + b * 36;
        const float* xj  = x + nj * 6;

        for (int c = 0; c < 6; c++)
            sum += blk[row*6 + c] * xj[c];
    }

    y[gid] = sum;
}

// ─────────────────────────────────────────────
//  Kernel: applica precondizionatore  z = M⁻¹ r
// ─────────────────────────────────────────────

__global__ void apply_precond_kernel(
    const float* M_inv,
    const float* r,
    float*       z,
    int          num_nodes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= num_nodes) return;

    const float* Mi = M_inv + ni * BLOCK_SIZE;
    const float* ri = r + ni * BLOCK_DIM;
    float*       zi = z + ni * BLOCK_DIM;

    #pragma unroll
    for (int row = 0; row < BLOCK_DIM; row++) {
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

__global__ void axpy_kernel(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// y = x + alpha*y
__global__ void xpay_kernel(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + alpha * y[i];
}

// ─────────────────────────────────────────────
//  GaussNewtonSolver — implementazione
// ─────────────────────────────────────────────

GaussNewtonSolver::GaussNewtonSolver(const Params& p, int max_nodes)
    : params_(p), max_nodes_(max_nodes)
{
    int n6 = max_nodes * BLOCK_DIM;
    b_.allocate(n6);
    x_.allocate(n6);
    r_.allocate(n6);
    z_.allocate(n6);
    p_.allocate(n6);
    Ap_.allocate(n6);
    M_inv_.allocate(max_nodes * BLOCK_SIZE);
}

void GaussNewtonSolver::build_sparsity_pattern(
    const DeformNode* h_nodes, int num_nodes)
{
    // Each row contains the diagonal plus graph-neighbor blocks in both
    // directions. PCG expects a symmetric system; the node graph is stored
    // directionally because new nodes only point to previous neighbors.
    std::vector<std::vector<int>> row_cols(num_nodes);
    for (int i = 0; i < num_nodes; i++)
        row_cols[i].push_back(i);

    for (int i = 0; i < num_nodes; i++) {
        for (int k = 0; k < h_nodes[i].num_neighbors; k++) {
            int nb = h_nodes[i].neighbors[k];
            if (nb < 0 || nb >= num_nodes) continue;
            row_cols[i].push_back(nb);
            row_cols[nb].push_back(i);
        }
    }

    h_row_ptr_.resize(num_nodes + 1);
    h_col_idx_.clear();

    for (int i = 0; i < num_nodes; i++) {
        h_row_ptr_[i] = (int)h_col_idx_.size();
        std::vector<int>& cols = row_cols[i];
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        for (int c : cols) h_col_idx_.push_back(c);
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
    const DeviceArray<Correspondence>& corrs,
    int num_corrs,
    const DeformNode* d_nodes,
    const Mat4*       d_transforms,
    int               num_nodes,
    DeviceArray<float>& delta_x)
{
    delta_x.allocate(num_nodes * BLOCK_DIM);

    // Download nodes to CPU to build sparsity pattern
    std::vector<DeformNode> h_nodes(num_nodes);
    cudaMemcpy(h_nodes.data(), d_nodes,
               num_nodes * sizeof(DeformNode), cudaMemcpyDeviceToHost);
    build_sparsity_pattern(h_nodes.data(), num_nodes);

    // One linearization per frame. Repeating this loop without applying the
    // update and re-raycasting solves the same system again and can only waste
    // time; the caller applies the returned increment once.
    assemble_system(corrs, num_corrs, d_nodes, d_transforms, num_nodes);
    compute_preconditioner(num_nodes);
    delta_x.zero();
    pcg_solve(num_nodes, delta_x);
}

void GaussNewtonSolver::assemble_system(
    const DeviceArray<Correspondence>& corrs,
    int num_corrs,
    const DeformNode* d_nodes,
    const Mat4*       d_transforms,
    int               num_nodes)
{
    A_.zero_values();
    b_.zero();

    int block = 256;
    int grid  = (num_corrs + block - 1) / block;

    // Data term
    assemble_data_term_kernel<<<grid, block>>>(
        corrs.data, num_corrs, d_nodes, d_transforms,
        A_.values.data, b_.data,
        A_.row_ptr.data, A_.col_idx.data, num_nodes);

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

void GaussNewtonSolver::compute_preconditioner(int num_nodes) {
    int block = 128;
    int grid  = (num_nodes + block - 1) / block;
    compute_preconditioner_kernel<<<grid, block>>>(
        A_.values.data, A_.row_ptr.data, A_.col_idx.data,
        M_inv_.data, num_nodes);
    cudaDeviceSynchronize();
}

float GaussNewtonSolver::pcg_solve(int num_nodes, DeviceArray<float>& x_out) {
    int n = num_nodes * BLOCK_DIM;

    // r = b - A*x  (x=0 → r=b)
    cudaMemcpy(r_.data, b_.data, n * sizeof(float), cudaMemcpyDeviceToDevice);

    // z = M⁻¹ r
    apply_preconditioner(r_, z_, num_nodes);

    // p = z
    cudaMemcpy(p_.data, z_.data, n * sizeof(float), cudaMemcpyDeviceToDevice);

    float rz_old = dot(r_, z_, n);
    float final_residual = sqrtf(rz_old);

    for (int iter = 0; iter < params_.pcg_iterations; iter++) {
        // Ap = A * p
        bsr_spmv(A_, p_, Ap_, num_nodes);

        float pAp   = dot(p_, Ap_, n);
        if (fabsf(pAp) < 1e-12f) break;
        float alpha = rz_old / pAp;

        // x = x + alpha * p
        axpy(alpha, p_, x_out, n);

        // r = r - alpha * Ap
        axpy(-alpha, Ap_, r_, n);

        float r_norm = sqrtf(dot(r_, r_, n));
        final_residual = r_norm;
        if (r_norm < params_.pcg_tolerance) break;

        // z = M⁻¹ r
        apply_preconditioner(r_, z_, num_nodes);

        float rz_new = dot(r_, z_, n);
        float beta   = rz_new / rz_old;

        // p = z + beta * p
        xpay(beta, z_, p_, n);

        rz_old = rz_new;
    }
    return final_residual;
}

void GaussNewtonSolver::bsr_spmv(
    const BSRMatrix& A,
    const DeviceArray<float>& x,
    DeviceArray<float>& y,
    int num_nodes)
{
    // Un blocco CUDA per node-row, BLOCK_DIM thread per blocco
    bsr_spmv_kernel<<<num_nodes, BLOCK_DIM>>>(
        A.values.data, A.row_ptr.data, A.col_idx.data,
        x.data, y.data, num_nodes);
    cudaDeviceSynchronize();
}

void GaussNewtonSolver::apply_preconditioner(
    const DeviceArray<float>& r,
    DeviceArray<float>& z,
    int num_nodes)
{
    int block = 128;
    int grid  = (num_nodes + block - 1) / block;
    apply_precond_kernel<<<grid, block>>>(M_inv_.data, r.data, z.data, num_nodes);
    cudaDeviceSynchronize();
}

float GaussNewtonSolver::dot(
    const DeviceArray<float>& a,
    const DeviceArray<float>& b,
    int n)
{
    thrust::device_ptr<const float> pa(a.data);
    thrust::device_ptr<const float> pb(b.data);
    return thrust::inner_product(pa, pa + n, pb, 0.0f);
}

void GaussNewtonSolver::axpy(float alpha, const DeviceArray<float>& x,
                              DeviceArray<float>& y, int n) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    axpy_kernel<<<grid, block>>>(alpha, x.data, y.data, n);
    cudaDeviceSynchronize();
}

void GaussNewtonSolver::xpay(float alpha, const DeviceArray<float>& x,
                              DeviceArray<float>& y, int n) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    xpay_kernel<<<grid, block>>>(alpha, x.data, y.data, n);
    cudaDeviceSynchronize();
}
