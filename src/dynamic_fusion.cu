#include "dynamic_fusion.h"

#include <opencv2/opencv.hpp>

#include "tsdf_kernels.h"

DynamicFusion::DynamicFusion(const Params& p) : params_(p) {
  volume_ = std::make_unique<TSDFVolume>(p.tsdf);
  warp_field_ = std::make_unique<WarpField>(p.node_radius, WarpField::MAX_NODES);
  solver_ = std::make_unique<GaussNewtonSolver>(p.solver, WarpField::MAX_NODES);

  const int n = p.camera.width * p.camera.height;
  d_depth_.allocate(n);
  d_normals_depth_.allocate(n);
  d_vertices_live_.allocate(n);
  d_normals_live_.allocate(n);
  d_corrs_.allocate(p.max_correspondences);
  d_num_valid_corrs_.allocate(1);

  d_voxel_knn_.allocate(volume_->total_voxels() * K_NEIGHBORS);
  d_voxel_knn_w_.allocate(volume_->total_voxels() * K_NEIGHBORS);

  camera_pose_ = Mat4::identity();
}

void DynamicFusion::process_frame(const cv::Mat& depth_raw) {
  preprocess_depth(depth_raw);
  if (!is_initialized_) {
    initialize(depth_raw);
    is_initialized_ = true;
    frame_count_++;
    return;
  }

  raycast_live_surface();
  const int num_corrs = find_correspondences();
  if (num_corrs > 64) optimize_warp_field(num_corrs);
  fuse_depth();
  update_node_graph();
  frame_count_++;
}

void DynamicFusion::preprocess_depth(const cv::Mat& depth_raw) {
  cv::Mat depth_m;
  if (depth_raw.type() == CV_32F) {
    depth_m = depth_raw;
  } else {
    depth_raw.convertTo(depth_m, CV_32F, params_.depth_scale);
  }

  cv::threshold(depth_m, depth_m, params_.depth_min, 0, cv::THRESH_TOZERO);
  cv::threshold(depth_m, depth_m, params_.depth_max, 0, cv::THRESH_TOZERO_INV);

  std::vector<float> h_depth((float*)depth_m.datastart, (float*)depth_m.dataend);
  d_depth_.upload(h_depth);

  dim3 block(16, 16);
  dim3 grid((params_.camera.width + block.x - 1) / block.x,
            (params_.camera.height + block.y - 1) / block.y);
  compute_depth_normals_kernel<<<grid, block>>>(
      d_depth_.data, d_normals_depth_.data, params_.camera.width,
      params_.camera.height, params_.camera);
  cudaDeviceSynchronize();
}

void DynamicFusion::initialize(const cv::Mat&) {
  volume_->integrate(d_depth_, d_normals_depth_, params_.camera, camera_pose_);
  std::vector<float3> v, n;
  std::vector<int3> t;
  volume_->extract_surface(v, n, t);
  warp_field_->add_nodes_from_surface(v, params_.node_min_dist);
  warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
}

void DynamicFusion::raycast_live_surface() {
  volume_->raycast(d_vertices_live_, d_normals_live_, params_.camera, camera_pose_,
                   warp_field_->device_nodes(), warp_field_->device_transforms(),
                   warp_field_->num_nodes(), d_voxel_knn_.data,
                   d_voxel_knn_w_.data);
}

int DynamicFusion::find_correspondences() {
  d_num_valid_corrs_.zero();
  dim3 block(256);
  dim3 grid((params_.camera.width * params_.camera.height + block.x - 1) / block.x);
  find_correspondences_kernel<<<grid, block>>>(
      d_vertices_live_.data, d_normals_live_.data, d_depth_.data,
      d_normals_depth_.data, d_corrs_.data, d_num_valid_corrs_.data,
      params_.camera.width, params_.camera.height, params_.camera,
      camera_pose_, d_voxel_knn_.data, d_voxel_knn_w_.data,
      0.05f, 0.7f);
  cudaDeviceSynchronize();
  int n = 0;
  cudaMemcpy(&n, d_num_valid_corrs_.data, sizeof(int), cudaMemcpyDeviceToHost);
  return n;
}

void DynamicFusion::optimize_warp_field(int num_corrs) {
  DeviceArray<float> delta_x(warp_field_->num_nodes() * 6);
  solver_->solve(d_corrs_, num_corrs, warp_field_->device_nodes(),
                 warp_field_->device_transforms(), warp_field_->num_nodes(),
                 delta_x);
  warp_field_->apply_twist_increment(delta_x);
}

void DynamicFusion::fuse_depth() {
  volume_->integrate(d_depth_, d_normals_depth_, params_.camera, camera_pose_,
                     warp_field_->device_nodes(), warp_field_->device_transforms(),
                     warp_field_->num_nodes(), d_voxel_knn_.data,
                     d_voxel_knn_w_.data);
}

void DynamicFusion::update_node_graph() {
  std::vector<float3> v;
  std::vector<float3> n;
  std::vector<int3> t;
  volume_->extract_surface(v, n, t);
  warp_field_->add_nodes_from_surface(v, params_.node_min_dist);
  warp_field_->compute_voxel_knn(*volume_, d_voxel_knn_, d_voxel_knn_w_);
}

void DynamicFusion::get_live_surface(std::vector<float3>& verts,
                                     std::vector<float3>& norms) const {
  d_vertices_live_.download(verts);
  d_normals_live_.download(norms);
}

void DynamicFusion::get_canonical_mesh(std::vector<float3>& verts,
                                       std::vector<float3>& norms,
                                       std::vector<int3>& tris) const {
  volume_->extract_surface(verts, norms, tris);
}
