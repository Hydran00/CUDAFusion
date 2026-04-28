#pragma once
#include <memory>
#include <string>

#include "solver.h"
#include "tsdf_volume.h"
#include "types.h"
#include "warp_field.h"

// ─────────────────────────────────────────────
//  DynamicFusion — pipeline principale
// ─────────────────────────────────────────────

class DynamicFusion {
 public:
  struct Params {
    TSDFVolume::Params tsdf;
    GaussNewtonSolver::Params solver;
    CameraIntrinsics camera;
    float node_radius = 0.05f;     // ~8x voxel_size
    float node_min_dist = 0.025f;  // distanza minima tra nodi
    int icp_pyramid_levels = 3;    // livelli piramide ICP
    float depth_scale = 0.001f;    // mm → m
    float depth_min = 0.3f;
    float depth_max = 3.0f;
    int max_correspondences = 200000;
  };

  explicit DynamicFusion(const Params& p);
  ~DynamicFusion() = default;

  // Processa un frame depth
  // depth: immagine depth raw (uint16 o float in mm)
  void process_frame(const cv::Mat& depth_raw);

  // Accessors per visualizzazione
  const TSDFVolume& volume() const { return *volume_; }
  const WarpField& warp_field() const { return *warp_field_; }

  // Vertici/normali del live frame (dopo raycasting)
  void get_live_surface(std::vector<float3>& verts,
                        std::vector<float3>& norms) const;

  // Mesh canonica corrente
  void get_canonical_mesh(std::vector<float3>& verts,
                          std::vector<float3>& norms,
                          std::vector<int3>& tris) const;

  int frame_count() const { return frame_count_; }

 private:
  Params params_;
  int frame_count_ = 0;
  bool is_initialized_ = false;

  // Componenti principali
  std::unique_ptr<TSDFVolume> volume_;
  std::unique_ptr<WarpField> warp_field_;
  std::unique_ptr<GaussNewtonSolver> solver_;

  // GPU buffers
  DeviceArray<float> d_depth_;           // depth float [H*W]
  DeviceArray<float3> d_normals_depth_;  // normali dal depth [H*W]
  DeviceArray<float3> d_vertices_live_;  // vertici raycasted [H*W]
  DeviceArray<float3> d_normals_live_;   // normali raycasted [H*W]
  DeviceArray<Correspondence> d_corrs_;  // corrispondenze ICP
  DeviceArray<int> d_num_valid_corrs_;

  // k-NN field nel volume
  DeviceArray<int> d_voxel_knn_;      // [vol_size * K]
  DeviceArray<float> d_voxel_knn_w_;  // [vol_size * K]

  // Trasformazione rigida camera (warm start tracking)
  Mat4 camera_pose_;

  // ── Passi della pipeline ──────────────────────

  // 1. Preprocessing depth
  void preprocess_depth(const cv::Mat& depth_raw);

  // 2. Primo frame: inizializza TSDF e grafo
  void initialize(const cv::Mat& depth_raw);

  // 3. Raycasting del modello canonico → live surface
  void raycast_live_surface();

  // 4. Trova corrispondenze ICP punto-piano
  int find_correspondences();

  // 5. Ottimizzazione Gauss-Newton → aggiorna warp field
  void optimize_warp_field(int num_corrs);

  // 6. Integrazione depth nel volume canonico
  void fuse_depth();

  // 7. Aggiorna grafo con nuovi nodi
  void update_node_graph();
};