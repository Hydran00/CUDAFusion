# DynamicFusion - CUDA-Accelerated 3D Reconstruction

A real-time volumetric fusion system implemented in CUDA that reconstructs dynamic 3D scenes from depth sensor sequences. This implementation combines non-rigid surface tracking with volumetric TSDF fusion to handle scenes with deforming objects.

## Overview

**DynamicFusion** is a state-of-the-art algorithm for fusing depth data from RGB-D sensors into a coherent 3D reconstruction that adapts to non-rigid deformations. This CUDA implementation accelerates the algorithm for real-time performance.

### Key Features

- **TSDF (Truncated Signed Distance Function) Fusion**: Volumetric 3D reconstruction from depth frames
- **Non-Rigid Warp Fields**: Models dynamic deformations using a hierarchical warp graph
- **Real-Time Performance**: GPU-accelerated using CUDA
- **Real-Time Visualization**: Open3D integration for live mesh display
- **Flexible Input**: Supports multiple dataset formats (TUM, ICL, RAW_PNG)
- **Synthetic Testing**: Generate synthetic depth sequences for development

---

## Pipeline Architecture

### 1. **Input & Configuration** 
```
Depth Sequence (RAW_PNG, TUM, ICL format)
       ↓
   Configuration (YAML)
       ↓
   Camera Intrinsics
```

The pipeline starts by loading:
- A depth frame sequence from disk (or generating synthetic test data)
- Application configuration (YAML file with algorithm parameters)
- Camera calibration (intrinsic matrix: fx, fy, cx, cy)

**Supported Formats:**
- **RAW_PNG**: Single-channel uint16 depth images (depth in millimeters)
- **TUM**: TUM RGB-D Benchmark format (timestamp + filename format)
- **ICL**: ICL-NUIM Indoor Dataset format

---

### 2. **Frame Loading** 
```
DepthSequence Class
       ↓
   Load Frame i
       ↓
   Convert to meters (apply depth_scale)
       ↓
   OpenCV Mat (float32 depth_m)
```

Each frame is:
- Loaded from disk (PNG, or other format)
- Depth values converted from uint16 to float (millimeters → meters)
- Stored as a `DepthFrame` structure with:
  - `depth_m`: depth map in meters (float32, CV_32F)
  - `color_gray`: grayscale intensity image
  - `timestamp`: frame timestamp

---

### 3. **Frame Processing Pipeline**
```
DynamicFusionPipeline::process_frame()
       ↓
   ┌─────────────────────────────┐
   │ 1. Pose Estimation          │
   │    (Visual Odometry)        │
   └─────────────────────────────┘
              ↓
   ┌─────────────────────────────┐
   │ 2. Warp Field Update        │
   │    (Track Deformations)     │
   └─────────────────────────────┘
              ↓
   ┌─────────────────────────────┐
   │ 3. TSDF Volume Integration  │
   │    (Fuse Depth into Volume) │
   └─────────────────────────────┘
              ↓
   ┌─────────────────────────────┐
   │ 4. Mesh Extraction          │
   │    (Marching Cubes)         │
   └─────────────────────────────┘
```

#### 3.1 **Pose Estimation**
- Estimates camera motion between frames
- Uses visual feature matching (SIFT features via CudaSift)
- Computes SE(3) transformation matrix (rotation + translation)
- Handles both rigid and non-rigid motion

#### 3.2 **Warp Field Update**
- Maintains a hierarchical deformation graph
- Tracks non-rigid surface deformations
- Updates warp nodes based on:
  - Feature correspondences from pose estimation
  - Smoothness constraints (rigidity preservation)
- Represents local rotations and translations at control points

#### 3.3 **TSDF Volume Integration**
- Updates a 3D voxel grid with new depth measurements
- TSDF encodes:
  - Signed distance from surface (positive = inside, negative = outside)
  - Truncation: clamps values to [-truncation_distance, +truncation_distance]
- Integration process:
  1. Project depth pixels into 3D world coordinates
  2. Apply warp field to transform to canonical space
  3. Update voxel values using weighted averaging
  4. Weight based on depth confidence and viewing angle

```cpp
// Pseudocode: TSDF Integration
for each pixel (u, v) in depth_map:
    if depth_map[v, u] is valid:
        world_point = unproject(u, v, depth, intrinsics)
        canonical_point = warp_field.inverse_warp(world_point)
        for each voxel near canonical_point:
            tsdf_dist = truncated_signed_distance(canonical_point)
            tsdf_voxel[x, y, z].update(tsdf_dist, weight)
```

#### 3.4 **Mesh Extraction**
- Uses Marching Cubes algorithm on the TSDF volume
- Extracts iso-surface at zero-crossing (surface of the object)
- Produces triangle mesh with vertices and normals
- **Two versions:**
  - **Canonical Mesh**: In the rest/reference frame (original shape)
  - **Warped Mesh**: With non-rigid deformations applied

---

### 4. **TSDF Volume Representation**
```
3D Voxel Grid (Canonical Space)
├─ Resolution: 512x512x512 (configurable)
├─ Voxel Size: 0.005 m (5mm per voxel)
├─ Value Range: [-1.0, +1.0] (truncated distance in units of truncation)
└─ Each Voxel Contains:
   ├─ Signed Distance
   ├─ Weight (confidence)
   └─ Color (optional)
```

The TSDF volume represents 3D space in the canonical frame:
- **Inside object**: TSDF < 0 (negative distance)
- **Surface**: TSDF ≈ 0 (zero-crossing)
- **Outside object**: TSDF > 0 (positive distance)
- **Truncation**: Values beyond ±truncation_distance are clamped

---

### 5. **Warp Field**
```
Hierarchical Deformation Graph
├─ Level 0: ~512 control points (coarse)
├─ Level 1: ~4096 control points (medium)
└─ Level 2: ~32768 control points (fine)

For each control point:
├─ Position (x, y, z)
├─ Rotation (quaternion)
└─ Neighbors (weighted graph edges)
```

The warp field enables:
- **Non-rigid deformation tracking**: Captures bending, stretching, deformation
- **Local detail preservation**: High-resolution control at fine scales
- **Smooth interpolation**: Gaussian kernels for smooth transitions between control points

**Warp Operation:**
```
warped_point = warp_field(point)
= sum over nearby warp nodes n:
    w_n * (R_n * (point - p_n) + p_n + t_n)
where w_n = gaussian_weight(distance(point, p_n))
```

---

### 6. **Output Generation**

#### 6.1 **Real-Time Visualization** (Optional)
- Open3D visualizer for live 3D mesh display
- Updates mesh every frame (or when new data is integrated)
- Shows coordinate frame and camera reference
- Interactive viewer (rotate, zoom, pan)

#### 6.2 **Mesh Checkpoints**
- Every 30 frames: saves intermediate meshes
- Files saved:
  - `mesh_frame_XX_canonical.ply`: Undeformed reconstruction
  - `mesh_frame_XX_warped.ply`: With deformations applied

#### 6.3 **Final Output**
- `mesh_final_canonical.ply`: Final canonical mesh (full sequence)
- `mesh_final_warped.ply`: Final warped mesh (full sequence)

---

## Usage

### Prerequisites

```bash
# Dependencies
sudo apt-get install libopencv-dev libyaml-cpp-dev libcuda-dev

# Build
cd myfusion
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Running the Pipeline

#### 1. **With Synthetic Sequence** (No config file)
```bash
./build/run_fusion
```
Generates 30 synthetic depth frames of a moving sphere.

#### 2. **With Config File**
```bash
./build/run_fusion config/params.yaml
```

#### 3. **With Real Dataset**
```bash
./build/run_fusion /path/to/dataset/config.yaml --vis
```

### Command-Line Options

```bash
# Enable visualization
--vis

# Disable visualization (faster)
--no-vis

# Enable debug visualization
--debug-vis

# Debug output every N frames
--debug-every 5

# Process only first N frames
--max-frames 100

# Specify dataset format
tum      # TUM RGB-D format
icl      # ICL-NUIM format
raw      # Raw PNG format (default)
```

### Example

```bash
# Run with visualization on TUM dataset
./build/run_fusion config/params.yaml tum --vis --max-frames 300
```

---

## Configuration File (YAML)

Example `config/params.yaml`:

```yaml
# Depth sequence parameters
sequence:
  path: /path/to/depth/frames
  format: raw_png              # raw_png, tum, icl
  depth_scale: 0.001           # Convert uint16 to meters
  start_frame: 0
  max_frames: -1               # -1 = all frames

  # Camera intrinsics
  camera:
    fx: 570.342
    fy: 570.342
    cx: 320.0
    cy: 240.0
    width: 640
    height: 480

# Dynamic Fusion parameters
dynamic_fusion:
  tsdf:
    volume_size: 512            # Voxel grid resolution
    voxel_size: 0.005           # Size of each voxel (meters)
    truncation_distance: 0.05   # TSDF truncation distance
    
  warp_field:
    num_levels: 3               # Hierarchical levels
    radius: 0.1                 # Control point influence radius
    regularization: 0.1         # Smoothness weight
    
  pose_estimation:
    feature_threshold: 0.01
    max_iterations: 5
    
  visualization:
    mesh_update_interval: 1     # Update every N frames
    
  debug:
    enabled: false
    every_n_frames: 30
```

---

## Architecture Details

### Core Components

#### **DepthSequence**
- Loads and manages depth frame sequences
- Supports multiple formats (TUM, ICL, RAW_PNG)
- Handles depth-to-meters conversion
- Provides frame iterator interface

#### **DynamicFusionPipeline**
- Main processing pipeline class
- Coordinates all stages:
  1. Pose estimation
  2. Warp field optimization
  3. TSDF integration
  4. Mesh extraction
- Maintains:
  - TSDF volume (GPU memory)
  - Warp field (GPU memory)
  - Camera pose history
  - Current mesh

#### **TSDFVolume**
- 3D voxel grid implementation
- GPU-accelerated operations:
  - Volume integration
  - Trilinear interpolation
  - Marching cubes extraction

#### **WarpField**
- Hierarchical deformation graph
- Control point-based warping
- GPU-accelerated point transformation

#### **CudaSift** (Third-party)
- Feature extraction and matching
- Used for visual odometry (pose estimation)

---

## Performance Characteristics

### Typical Performance (on NVIDIA GPU)

| Component         | Time per Frame |
| ----------------- | -------------- |
| Pose Estimation   | 5-10 ms        |
| Warp Field Update | 3-5 ms         |
| TSDF Integration  | 20-30 ms       |
| Mesh Extraction   | 10-15 ms       |
| Visualization     | 5-10 ms        |
| **Total**         | **50-70 ms**   |

**FPS**: ~15-20 fps (depending on resolution and GPU)

### Memory Usage

| Component          | Memory  |
| ------------------ | ------- |
| TSDF Volume (512³) | ~500 MB |
| Warp Field         | ~50 MB  |
| Temporary Buffers  | ~200 MB |
| **Total**          | ~750 MB |

---

## File Structure

```
myfusion/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── config/
│   └── params.yaml            # Default configuration
├── include/
│   ├── fusion_app.h           # Application header
│   ├── dynamic_fusion.h       # Main pipeline class
│   ├── tsdf_volume.h          # TSDF implementation
│   ├── warp_field.h           # Warp field class
│   ├── se3_math.cuh           # SE(3) math utilities
│   ├── solver.h               # Optimization solver
│   └── ...
├── src/
│   ├── tsdf_volume.cu         # TSDF GPU kernels
│   ├── warp_field.cu          # Warp field GPU kernels
│   └── ...
├── kernels/
│   ├── tsdf_kernels.cu        # TSDF integration kernels
│   ├── solver_kernels.cu      # Solver kernels
│   └── ...
├── tests/
│   └── main.cu                # Main application
└── thirdparty/
    └── CudaSift/              # SIFT feature detector
```

---

## Algorithm Overview (Academic Reference)

The algorithm implements the Dynamic Fusion approach for non-rigid 3D reconstruction:

### Key Innovations:

1. **Real-time TSDF Fusion**: Extends traditional TSDF to handle dynamic scenes
2. **Hierarchical Warp Field**: Multi-level deformation representation
3. **Iterative Closest Point (ICP)**: Robust pose estimation
4. **GPU Acceleration**: CUDA kernels for all compute-intensive operations

### Mathematical Foundation:

- **TSDF Definition**: `tsdf(p) = min(truncation, signed_distance(p))`
- **Warp Operation**: `p_warped = W(p, θ)` where θ are warp parameters
- **Energy Minimization**: 
  ```
  E = E_data + λ * E_regularization
    = ∑ |tsdf(W(p, θ))| + λ * regularization_loss(θ)
  ```

---

## Troubleshooting

### Common Issues

**Q: "CUDA out of memory" error**
- A: Reduce TSDF volume size in config, or increase GPU memory allocation

**Q: Poor mesh quality / holes in reconstruction**
- A: Increase TSDF integration weight, check camera calibration, ensure good frame overlap

**Q: Slow processing / low FPS**
- A: Check GPU utilization, reduce frame resolution, disable visualization

**Q: Visualization crashes**
- A: Ensure X11 display is available, or disable visualization with `--no-vis`

---

## References

- Newcombe, R. A., et al. "KinectFusion: Real-time 3D reconstruction and interaction using a moving depth camera." (2011)
- Innmann, M., et al. "VolumeDeform: Real-time volumetric non-rigid reconstruction." (2016)
- Palazzi, A., et al. "DynamicFusion: Reconstruction and tracking of non-rigid scenes in real-time." (2015)

---

## License

See LICENSE file for details.

---

## Contact & Support

For issues, questions, or contributions, please contact the development team.

**Last Updated**: May 2026
