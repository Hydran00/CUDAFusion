# CUDA Non rigid RGB-D Fusion

**Warning: This project is still experimental. It does not work as supposed yet.**
CUDA implementation of a non-rigid RGB-D fusion pipeline. The system reconstructs dynamic scenes by combining a canonical TSDF volume, a deformation graph, projective ICP, sparse SIFT constraints, and GPU-accelerated mesh extraction/rasterization.

The design follows the main ideas of **DynamicFusion** and **VolumeDeform**, with several practical additions for robustness and speed: CUDA TSDF integration, VolumeDeform-style deformed mesh prediction, geodesic graph edges, node pruning, sparse CudaSift support, profiling, and Open3D visualization.

## Example Videos

Add your examples here. This table is intentionally visible near the top.

| Scene | Input | Output / Video | Notes |
| ----- | ----- | -------------- | ----- |
| TODO  | TODO  | TODO           | TODO  |
| TODO  | TODO  | TODO           | TODO  |
| TODO  | TODO  | TODO           | TODO  |

## What It Does

For each depth frame, the pipeline:

1. Loads depth and optional grayscale color.
2. Computes depth normals.
3. Predicts the current live surface from the canonical TSDF.
4. Tracks the surface with projective ICP and optional SIFT correspondences.
5. Optimizes a deformation graph with data and ARAP smoothness terms.
6. Integrates the new depth frame into the canonical TSDF.
7. Updates and prunes deformation graph nodes.
8. Displays or saves canonical/warped meshes.

The canonical model stays in a stable reference space. The warp field maps canonical geometry into the current live frame.

## Main Ideas

### DynamicFusion Basis

DynamicFusion introduced real-time non-rigid reconstruction by maintaining:

- a canonical volumetric model;
- a warp field represented by deformation nodes;
- an optimization loop that aligns the warped canonical surface to the current depth frame;
- smoothness regularization to keep neighboring nodes locally coherent.

This project keeps that structure, but implements the expensive parts in CUDA.

### VolumeDeform-style Prediction

Instead of relying only on TSDF raycasting, the pipeline can:

- extract the canonical mesh;
- skin/deform vertices with the current warp field;
- rasterize the deformed mesh into the current camera;
- use that rendered live surface for ICP correspondences.

This is closer to the VolumeDeform tracking style and makes the prediction stage more explicit and easier to debug.

### Deformation Graph

Nodes are sampled from the TSDF surface. Each node stores:

- canonical position;
- canonical normal;
- influence radius;
- graph neighbors;
- dual-quaternion transform.

Graph edges can be rebuilt using geodesic distances over the extracted mesh, which helps avoid connecting nearby but topologically unrelated surfaces, such as an arm close to the torso.

### Node Pruning

Nodes are pruned during graph updates when they no longer have TSDF surface support or when they form small disconnected islands. This is useful when nodes were created in regions that later become observed free space, or when floating fragments remain after fusion.

Important pruning parameters:

| Parameter                     | Meaning                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `prune_nodes`                 | Enable TSDF/graph pruning.                                              |
| `prune_disconnected`          | Remove small disconnected graph components.                             |
| `prune_min_observed_voxels`   | Minimum observed TSDF voxels needed before making a pruning decision.   |
| `prune_min_surface_voxels`    | Minimum near-surface voxels required to keep a node.                    |
| `prune_min_component_size`    | Graph islands smaller than this are removed.                            |
| `prune_support_radius_factor` | Radius multiplier used when checking TSDF support around a node.        |
| `prune_surface_tsdf_abs`      | A voxel counts as surface support when `abs(tsdf)` is below this value. |
| `prune_empty_tsdf`            | A voxel counts as free space when `tsdf` is above this value.           |

## Build

### Dependencies

Required:

- CUDA Toolkit
- CMake 3.18+
- OpenCV
- Eigen3
- Open3D
- yaml-cpp, fetched automatically by CMake
- CudaSift in `thirdparty/CudaSift`

On Ubuntu, the system packages are usually:

```bash
sudo apt-get install cmake build-essential libopencv-dev libeigen3-dev
```

Open3D is currently searched with:

```cmake
find_package(Open3D REQUIRED HINTS /home/hydran00/Open3D/lib/cmake/Open3D)
```

Adjust that path in `CMakeLists.txt` if your Open3D installation is elsewhere.

### CudaSift Branch

This repository expects CudaSift under:

```bash
thirdparty/CudaSift
```

Use the CudaSift branch that matches your GPU architecture setup. For example, for Ada Lovelace:

```bash
cd thirdparty/CudaSift
git checkout adalovelace
cd ../..
```

Use the corresponding branch for other architectures when available.

### Configure With CUDA Architecture

The CUDA architecture is configurable through:

```bash
-DDFUSION_CUDA_ARCHITECTURES=<arch>
```

Examples:

```bash
# Ada Lovelace, e.g. RTX 40xx
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDFUSION_CUDA_ARCHITECTURES=89

# Ampere, e.g. RTX 30xx
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDFUSION_CUDA_ARCHITECTURES=86

# Turing, e.g. RTX 20xx
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDFUSION_CUDA_ARCHITECTURES=75

# Multiple architectures
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDFUSION_CUDA_ARCHITECTURES="86;89"

# Native detection, if supported by your CMake/CUDA combination
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDFUSION_CUDA_ARCHITECTURES=native
```

Default is `89`, optimized for Ada Lovelace.

### Compile

```bash
cmake --build build -j$(nproc)
```

The Release build enables aggressive optimization flags:

- `-O3`
- CUDA `--use_fast_math`
- CUDA `--extra-device-vectorization`
- PTXAS `-O3`
- host `-march=native`
- interprocedural optimization

If tracking becomes numerically unstable, the first flags to relax are fast-math flags, not the CUDA architecture.

## Run

With the default config:

```bash
./build/run_fusion config/params.yaml
```

Useful CLI flags:

```bash
--vis              # enable visualization
--no-vis           # disable visualization
--debug-vis        # show debug panels
--debug-every N    # debug every N frames
--max-frames N     # process only N frames
tum | icl | raw    # override sequence format
```

Example:

```bash
./build/run_fusion config/params.yaml tum --vis --max-frames 300
```

Outputs:

- `mesh_final_canonical.ply`
- `mesh_final_warped.ply`
- optional intermediate checkpoints from the application loop

## Configuration

The default configuration is in `config/params.yaml`.

### Sequence

```yaml
sequence:
  path: "/path/to/sequence"
  format: "tum"       # tum | icl | raw
  depth_scale: 0.001
  start_frame: 0
  max_frames: -1
```

### Camera

```yaml
camera:
  fx: 570.342
  fy: 570.342
  cx: 320.0
  cy: 240.0
  width: 640
  height: 480
```

### TSDF

```yaml
tsdf:
  dims: [256, 256, 256]
  voxel_size: 0.005
  truncation: 0.03
  origin: [-0.7, -0.7, 0.5]
  decay_alpha: 1.0
  max_weight: 50.0
```

`decay_alpha < 1.0` makes the volume more reactive in dynamic scenes. `1.0` keeps accumulated observations.

### Solver

```yaml
solver:
  gn_iterations: 2
  pcg_iterations: 80
  lambda_smooth: 50.0
  lambda_damping: 1.0
  debug: false
```

Notes:

- More PCG iterations are not always better. Early stopping can act as a regularizer.
- Increase `lambda_smooth` if graph edges stretch.
- Increase `lambda_damping` if updates become unstable.

### Warp Field

```yaml
warp:
  node_radius: 0.02
  node_min_dist: 0.015
  max_nodes: 2048
  node_update_every_n: 2
  prune_nodes: true
  prune_disconnected: true
  prune_min_observed_voxels: 2
  prune_min_surface_voxels: 5
  prune_min_component_size: 16
  prune_support_radius_factor: 0.75
  prune_surface_tsdf_abs: 0.28
  prune_empty_tsdf: 0.55
  max_dx_mean: 0.2
  max_update_rot: 0.12
  max_update_trans: 0.04
  update_scale: 0.3
  integrate_warped: true
  integrate_min_optimized_count: 1
  integrate_unwarped_fallback: false
```

Useful tuning rules:

- Smaller `node_min_dist` creates denser graphs.
- Smaller `node_radius` gives more local deformation but can require more nodes.
- Lower `update_scale` is safer.
- Lower `max_update_trans` and `max_update_rot` prevent sudden graph collapse.
- `integrate_warped: true` is recommended for dynamic scenes.

### ICP

```yaml
icp:
  dist_threshold: 0.15
  angle_threshold: 0.1
  view_threshold: 0.1
  search_radius_px: 10
  min_valid_corrs: 5000
```

Large `search_radius_px` and permissive thresholds can recover from faster camera motion, but they also increase runtime and outlier risk. If geometry degrades, make these stricter.

### SIFT

```yaml
sift:
  enabled: true
  max_features: 8192
  max_history: 12
  max_matches_per_frame: 96
  octaves: 5
  threshold: 3.0
  max_match_error: 5.0
  max_ambiguity: 0.85
  max_3d_dist: 0.2
  weight: 0.4
```

Sparse CudaSift matches help when the camera moves too quickly for local projective ICP alone.

### Profiler and Debug

```yaml
profiler:
  enabled: true
  every_n: 1
  quiet: false
```

Debug visualization can be enabled with:

```bash
./build/run_fusion config/params.yaml --debug-vis --debug-every 5
```

The graph debug panel draws:

- nodes in yellow;
- normal graph edges in purple;
- suspicious long edges in red.

## Source Layout

```text
myfusion/
├── CMakeLists.txt
├── README.md
├── config/
│   └── params.yaml
├── include/
│   ├── fusion_app.h
│   ├── fusion_debug.h
│   ├── fusion_math.h
│   ├── tsdf_volume.h
│   ├── warp_field.h
│   ├── solver.h
│   └── se3_math.cuh
├── kernels/
│   ├── tsdf_kernels.cu
│   └── solver_kernels.cu
├── src/
│   ├── marching_cubes.cu
│   ├── mesh_raster_kernels.cu
│   ├── skinning_kernels.cu
│   ├── tsdf_volume.cu
│   └── warp_field.cu
├── tests/
│   └── main.cu
└── thirdparty/
    └── CudaSift/
```

## Performance Notes

The profiler prints high-level frame timing and tracking sub-timing:

```text
[profiler frame ...] upload=... normals=... tracking=... fusion=... graph_update=... total=...
[tracking profile ...] raster_total=... corr=... rigid=... sift_aug=... solve=... rmse=... line_search=... other=... raster_calls=...
```

Common bottlenecks:

- `corr`: projective correspondence search, especially with large `search_radius_px`;
- `raster_total`: repeated deformed mesh extraction/rasterization;
- `sift_aug`: sparse feature matching and correspondence augmentation;
- `rmse` / `line_search`: repeated trial updates and CPU/GPU transfers;
- `graph_update`: mesh extraction, node creation, geodesic graph rebuild, pruning.

If `raster_calls` is high, the line search is re-rendering the mesh many times per frame.

## Troubleshooting

### Many correspondences disappear

Try temporarily increasing:

```yaml
icp:
  dist_threshold: 0.015
  angle_threshold: 0.75
  search_radius_px: 3
  min_valid_corrs: 10000
```

Also consider increasing SIFT support:

```yaml
sift:
  max_history: 12
  max_matches_per_frame: 96
  weight: 0.4
```

### Graph edges stretch or collapse

Try:

```yaml
solver:
  pcg_iterations: 15
  lambda_smooth: 100.0
  lambda_damping: 3.0

warp:
  update_scale: 0.05
  max_update_rot: 0.08
  max_update_trans: 0.025
```

### Floating islands or flying nodes

Use more aggressive pruning:

```yaml
warp:
  prune_nodes: true
  prune_disconnected: true
  prune_min_surface_voxels: 5
  prune_min_component_size: 16
  prune_empty_tsdf: 0.55
```

### CUDA out of memory

Reduce:

- `tsdf.dims`
- `max_nodes`
- SIFT history/features

## References

- Newcombe et al., **KinectFusion: Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera**, UIST 2011.
- Newcombe et al., **DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time**, CVPR 2015.
- Innmann et al., **VolumeDeform: Real-time Volumetric Non-rigid Reconstruction**, ECCV 2016.

## License

See the repository license and third-party dependency licenses.
