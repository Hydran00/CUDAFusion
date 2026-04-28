# DynamicFusion — Implementazione CUDA/C++

Simulazione della pipeline DynamicFusion con:
- TSDF volume (integrazione e raycasting)
- Warp field non-rigido (grafo di deformazione)
- Gauss-Newton + PCG sparso su GPU

---

## Struttura del progetto

```
dynamicfusion/
├── include/
│   ├── types.h          — tipi base (Mat4, DeviceArray, BSRMatrix, ...)
│   ├── tsdf_volume.h    — volume TSDF
│   ├── warp_field.h     — grafo di deformazione
│   └── solver.h         — Gauss-Newton + PCG
├── kernels/
│   ├── se3_math.cuh     — math SE(3): exp_map, warp, Jacobiana, inv 6x6
│   ├── tsdf_kernels.cu  — integrate, raycast, normali, corrispondenze
│   └── solver_kernels.cu — assemblaggio sistema, PCG, SpMV BSR
├── src/
│   ├── tsdf_volume.cpp  — implementazione TSDFVolume
│   └── warp_field.cpp   — implementazione WarpField
└── tests/
    └── main.cpp         — pipeline completa + loader sequenze depth
```

---

## Build

### Requisiti
- CUDA >= 11.0
- CMake >= 3.18
- OpenCV >= 4.0
- Eigen3

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Architettura GPU
Modifica `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 86)
# 75 = Turing (RTX 20xx)
# 86 = Ampere (RTX 30xx)
# 89 = Ada (RTX 40xx)
# 70 = Volta (V100)
# 61 = Pascal (GTX 10xx)
```

---

## Uso

### Dataset sintetico (nessun argomento)
```bash
./run_fusion
```
Genera una sfera che si muove su 30 frame.

### TUM RGB-D dataset
```bash
./run_fusion /path/to/tum/fr1_desk tum
```
Scarica da: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

Struttura attesa:
```
fr1_desk/
├── depth/          (*.png, uint16, /5000 = metres)
├── rgb/
└── associations.txt
```

### ICL-NUIM
```bash
./run_fusion /path/to/icl/living_room_0 icl
```

### Cartella raw di PNG
```bash
./run_fusion /path/to/depth_images raw
```
Accetta PNG uint16 (scala automatica da mm a m) o float32 già in m.

---

## Dataset consigliati per test

| Dataset | Link | Note |
|---|---|---|
| TUM fr1/desk | [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) | Classico benchmark, camera fissa |
| TUM fr3/walking_xyz | stesso | Persona che cammina — non-rigido |
| ICL-NUIM lr kt0 | [ICL](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) | Sintetico, ground truth perfetto |
| Bonn RGB-D | [Bonn](http://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | Scene con oggetti in movimento |

---

## Parametri chiave

```cpp
// TSDF
tsdf.voxel_size = 0.006f;   // 6mm — aumenta per scene grandi
tsdf.truncation = 0.03f;    // 5x voxel_size è buona euristica

// Grafo di deformazione
node_radius   = 0.05f;      // raggio influenza nodo (m)
node_min_dist = 0.025f;     // distanza minima tra nodi

// Ottimizzatore
solver.gn_iterations  = 3;  // iterazioni Gauss-Newton
solver.pcg_iterations = 10; // iterazioni PCG (dal paper BodyFusion)
solver.lambda_smooth  = 5.0; // peso ARAP smoothness
```

---

## Note implementative

### Cosa è completo
- TSDF: integrazione depth + raycasting con zero-crossing
- SE(3): mappa esponenziale (twist → Mat4), Rodrigues completo
- Warp field: struttura dati, aggiunta nodi, k-NN field voxel
- PCG: SpMV BSR 6x6, precondizionatore block-diagonale, Cholesky 6x6 inline
- Loader: TUM, ICL, raw PNG/EXR, generatore sintetico

### Semplificazioni rispetto al paper originale
1. **k-NN brute force** per il voxel field (produzione: spatial hashing)
2. **atomicAdd** per assemblaggio sistema (produzione: color ordering)
3. **Termine off-diagonale smoothness** parzialmente omesso
4. **Marching cubes** semplificato (solo narrow band sampling)
5. **Tracking rigido camera** fisso (produzione: ICP rigido come KinectFusion)

### Per estendere a BodyFusion
Aggiungere in `solver_kernels.cu`:
- `assemble_skeleton_term_kernel` — Eskeleton con catena cinematica
- `assemble_binding_term_kernel`  — Ebinding tra nodi e ossa
In `warp_field`: gestione skin attachments frame-by-frame

---

## Output

- `mesh_frame_N.ply` — mesh al frame N (apribile con MeshLab)
- `mesh_final.ply`   — mesh finale
- Console: fps, numero nodi, corrispondenze ICP per frame
