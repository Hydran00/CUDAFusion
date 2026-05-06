#pragma once
#include <vector>

#include "types.h"
#include "tsdf_volume.h"

// ─────────────────────────────────────────────
//  WarpField — gestisce il grafo di deformazione
//  e le trasformazioni SE(3) per ogni nodo
// ─────────────────────────────────────────────

class WarpField {
 public:
  static constexpr int MAX_NODES = 8192;
  static constexpr float NODE_RADIUS_FACTOR =
      2.5f;  // radius = factor * voxel_size

  struct SpatialHash {
    // Hash 3D per trovare nodi vicini a un punto
    // Struttura flat su GPU
    DeviceArray<int> cell_start;  // inizio di ogni cella
    DeviceArray<int> cell_end;
    DeviceArray<int> sorted_ids;
    float cell_size;
    int3 grid_dims;
  };

  WarpField(float node_radius, int max_nodes = MAX_NODES);
  ~WarpField() = default;

  // ── Gestione nodi ──────────────────────────────

  // Aggiunge nodi dalla superficie appena estratta
  // Restituisce quanti nodi sono stati aggiunti
  int add_nodes_from_surface(const std::vector<float3>& surface_vertices,
                             float min_dist_between_nodes,
                             const std::vector<float3>* surface_normals = nullptr);

  // Inizializza trasformazione nuovo nodo
  // per interpolazione dai vicini esistenti
  void init_node_transform(int new_node_idx);

  int num_nodes() const { return num_nodes_; }

  // ── Dense k-NN field nel volume ────────────────

  // Costruisce/aggiorna il mapping voxel → k nodi più vicini
  // Output: d_voxel_knn [vol_size * K_NEIGHBORS]
  //         d_voxel_knn_w [vol_size * K_NEIGHBORS]
  void compute_voxel_knn(const TSDFVolume& volume,
                         DeviceArray<int>& d_voxel_knn,
                         DeviceArray<float>& d_voxel_knn_w);

  // ── Accesso GPU ───────────────────────────────

  DeformNode* device_nodes() { return d_nodes_.data; }
  DualQuat* device_transforms() { return d_transforms_.data; }
  DualQuat* device_transforms_prev() { return d_transforms_prev_.data; }

  const DeformNode* device_nodes() const { return d_nodes_.data; }
  const DualQuat* device_transforms() const { return d_transforms_.data; }

  // ── Aggiornamento trasformazioni ──────────────

  // Salva trasformazioni correnti come "prev" (warm start)
  void save_transforms();
  void restore_transforms();

  // Applica incremento twist: T_i ← exp(Δx_i) · T_i
  void apply_twist_increment(const DeviceArray<float>& delta_x,
                             float max_rot, float max_trans,
                             float update_scale = 1.0f);

  // Reset trasformazioni a identità
  void reset_transforms();

  // CPU access per debug/export
  std::vector<DeformNode> download_nodes() const;
  std::vector<DualQuat> download_transforms() const;

  // Upload nodi modificati su CPU (per aggiunta nodi)
  void upload_nodes(const std::vector<DeformNode>& nodes);
  void upload_transforms(const std::vector<DualQuat>& transforms);

 private:
  float node_radius_;
  int num_nodes_;
  int max_nodes_;

  // GPU arrays (pre-allocati a MAX_NODES)
  DeviceArray<DeformNode> d_nodes_;
  DeviceArray<DualQuat> d_transforms_;
  DeviceArray<DualQuat> d_transforms_prev_;

  // CPU mirror (per aggiornamenti incrementali)
  std::vector<DeformNode> h_nodes_;
  std::vector<DualQuat> h_transforms_;

  // Spatial hash per k-NN query
  SpatialHash spatial_hash_;

  void build_spatial_hash();
  float compute_node_weight(float3 point, float3 node_pos, float radius) const;
};
