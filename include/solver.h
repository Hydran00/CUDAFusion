#pragma once
#include "types.h"

// ─────────────────────────────────────────────
//  Gauss-Newton + PCG solver
//  Risolve: (J^T J) Δx = -J^T f
//  per il warp field non-rigido
// ─────────────────────────────────────────────

class GaussNewtonSolver {
 public:
  struct Params {
    int gn_iterations = 3;    // iterazioni Gauss-Newton esterne
    int pcg_iterations = 10;  // iterazioni PCG interne
    float pcg_tolerance = 1e-4f;
    float lambda_smooth = 1.0f;  // peso termine smoothness (ARAP)
    float lambda_rot = 1.0f;     // peso penalità rotazione
  };

  explicit GaussNewtonSolver(const Params& p, int max_nodes);
  ~GaussNewtonSolver() = default;

  // Costruisce il sistema e risolve per Δx
  // Output: delta_x [num_nodes * 6] — twist increment per ogni nodo
  void solve(
      // Corrispondenze ICP
      const DeviceArray<Correspondence>& corrs, int num_corrs,
      // Grafo di deformazione
      const DeformNode* d_nodes, const Mat4* d_transforms, int num_nodes,
      // Output
      DeviceArray<float>& delta_x);

  // Accesso al sistema assemblato (per debug)
  const BSRMatrix& system_matrix() const { return A_; }
  const DeviceArray<float>& rhs() const { return b_; }

 private:
  Params params_;
  int max_nodes_;

  // Sistema lineare
  BSRMatrix A_;
  DeviceArray<float> b_;      // [max_nodes * 6]
  DeviceArray<float> x_;      // soluzione corrente
  DeviceArray<float> M_inv_;  // precondizionatore block-diag [max_nodes * 36]

  // Vettori PCG
  DeviceArray<float> r_, z_, p_, Ap_;

  // BSR sparsity pattern (costruito dal grafo)
  std::vector<int> h_row_ptr_;
  std::vector<int> h_col_idx_;
  int current_num_blocks_ = 0;

  // ── Kernel wrappers ───────────────────────────

  void assemble_system(const DeviceArray<Correspondence>& corrs, int num_corrs,
                       const DeformNode* d_nodes, const Mat4* d_transforms,
                       int num_nodes);

  void build_sparsity_pattern(const DeformNode* h_nodes, int num_nodes);

  void compute_preconditioner(int num_nodes);

  // PCG interno
  float pcg_solve(int num_nodes, DeviceArray<float>& x_out);

  // SpMV block-sparso: y = A * x
  void bsr_spmv(const BSRMatrix& A, const DeviceArray<float>& x,
                DeviceArray<float>& y, int num_nodes);

  // Precondizionatore: z = M^{-1} r
  void apply_preconditioner(const DeviceArray<float>& r, DeviceArray<float>& z,
                            int num_nodes);

  // Operazioni vettoriali su GPU
  float dot(const DeviceArray<float>& a, const DeviceArray<float>& b, int n);

  void axpy(float alpha, const DeviceArray<float>& x, DeviceArray<float>& y,
            int n);

  void xpay(float alpha, const DeviceArray<float>& x, DeviceArray<float>& y,
            int n);  // y = x + alpha*y
};