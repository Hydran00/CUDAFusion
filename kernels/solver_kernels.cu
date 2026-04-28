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

    // Residuo ICP punto-piano
    float residual = dot3(normal, make_float3(src.x-dst.x, src.y-dst.y, src.z-dst.z));

    // Per ogni coppia di nodi (ki, kj) che influenzano questo punto
    for (int ki = 0; ki < K_NEIGHBORS; ki++) {
        int   ni  = c.node_ids[ki];
        float w_i = c.node_ws [ki];
        if (ni < 0 || w_i < 1e-8f) continue;

        // Jacobiana riga per nodo i
        float Ji[6];
        compute_jacobian_row(src, normal, w_i, Ji);

        // Contributo a b: J_i^T * r
        for (int a = 0; a < 6; a++)
            atomicAdd(&b_values[ni*6 + a], -Ji[a] * residual);

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
            compute_jacobian_row(src, normal, w_j, Jj);

            // Accumula J_i^T * J_j nel blocco (ni, nj)
            float* blk = A_values + block_pos * BLOCK_SIZE;
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                    atomicAdd(&blk[a*6 + b], Ji[a] * Jj[b]);
        }
    }
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
    float             lambda_smooth,
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

        // Residuo smoothness: T_i * x_j - T_j * x_j
        float3 Ti_xj = transforms[ni].transform_point(xj);
        float3 Tj_xj = transforms[nj].transform_point(xj);

        float3 res = make_float3(Ti_xj.x - Tj_xj.x,
                                  Ti_xj.y - Tj_xj.y,
                                  Ti_xj.z - Tj_xj.z);

        // Jacobiana rispetto a twist di nodo i: ∂(T_i * xj) / ∂ξ_i
        // = [-(T_i * xj)×  |  I] per parametrizzazione a sinistra
        float3 Tixj = Ti_xj;
        float Ji[6][3];  // 6 parametri twist, 3 output
        Ji[0][0] = 0;       Ji[0][1] =  Tixj.z; Ji[0][2] = -Tixj.y;
        Ji[1][0] = -Tixj.z; Ji[1][1] = 0;       Ji[1][2] =  Tixj.x;
        Ji[2][0] =  Tixj.y; Ji[2][1] = -Tixj.x; Ji[2][2] = 0;
        Ji[3][0] = 1;       Ji[3][1] = 0;        Ji[3][2] = 0;
        Ji[4][0] = 0;       Ji[4][1] = 1;        Ji[4][2] = 0;
        Ji[5][0] = 0;       Ji[5][1] = 0;        Ji[5][2] = 1;

        // Contributo diagonale A(ni,ni) += λ * Ji^T Ji
        int diag_pos = -1;
        for (int r = row_ptr[ni]; r < row_ptr[ni+1]; r++)
            if (col_idx[r] == ni) { diag_pos = r; break; }

        if (diag_pos >= 0) {
            float* blk = A_values + diag_pos * BLOCK_SIZE;
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++) {
                    float JtJ = 0;
                    for (int c = 0; c < 3; c++)
                        JtJ += Ji[a][c] * Ji[b][c];
                    atomicAdd(&blk[a*6+b], lambda_smooth * JtJ);
                }
        }

        // Contributo RHS b(ni) -= λ * Ji^T * res
        for (int a = 0; a < 6; a++) {
            float Jt_r = 0;
            const float res_arr[3] = {res.x, res.y, res.z};
            for (int c = 0; c < 3; c++)
                Jt_r += Ji[a][c] * res_arr[c];
            atomicAdd(&b_values[ni*6+a], -lambda_smooth * Jt_r);
        }
        // Note: termini off-diagonale con nj omessi per semplicità
        // (approssimazione comune in implementazioni real-time)
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
    int ni = blockIdx.x;
    if (ni >= num_nodes) return;

    // Shared memory per accumulatore
    __shared__ float acc[BLOCK_DIM];

    int lane = threadIdx.x;  // 0..5 per le 6 righe del blocco
    if (lane >= BLOCK_DIM) return;

    float sum = 0;

    // Itera su tutti i blocchi della riga ni
    for (int b = row_ptr[ni]; b < row_ptr[ni+1]; b++) {
        int   nj  = col_idx[b];
        const float* blk = A_values + b * BLOCK_SIZE;
        const float* xj  = x + nj * BLOCK_DIM;

        // Prodotto blocco-vettore: riga 'lane' del blocco * xj
        #pragma unroll
        for (int c = 0; c < BLOCK_DIM; c++)
            sum += blk[lane * BLOCK_DIM + c] * xj[c];
    }

    y[ni * BLOCK_DIM + lane] = sum;
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
    // Each row i: diagonal block (i,i) + one block per graph neighbor
    h_row_ptr_.resize(num_nodes + 1);
    h_col_idx_.clear();

    for (int i = 0; i < num_nodes; i++) {
        h_row_ptr_[i] = (int)h_col_idx_.size();
        // Collect sorted unique col indices: diagonal + neighbors
        std::vector<int> cols;
        cols.push_back(i);
        for (int k = 0; k < h_nodes[i].num_neighbors; k++) {
            int nb = h_nodes[i].neighbors[k];
            if (nb >= 0 && nb < num_nodes) cols.push_back(nb);
        }
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

    for (int gn = 0; gn < params_.gn_iterations; gn++) {
        // 1. Assembla sistema
        assemble_system(corrs, num_corrs, d_nodes, d_transforms, num_nodes);

        // 2. Precondizionatore
        compute_preconditioner(num_nodes);

        // 3. PCG
        delta_x.zero();
        pcg_solve(num_nodes, delta_x);
    }
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
