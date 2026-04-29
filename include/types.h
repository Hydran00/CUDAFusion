#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>

// ─────────────────────────────────────────────
//  Math primitives
// ─────────────────────────────────────────────

struct Mat4 {
    float m[4][4];

    __host__ __device__ Mat4() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.f : 0.f;
    }

    __host__ __device__ static Mat4 identity() { return Mat4(); }

    __host__ __device__ float3 transform_point(float3 p) const {
        float x = m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3];
        float y = m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3];
        float z = m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3];
        return make_float3(x, y, z);
    }

    __host__ __device__ float3 transform_normal(float3 n) const {
        float x = m[0][0]*n.x + m[0][1]*n.y + m[0][2]*n.z;
        float y = m[1][0]*n.x + m[1][1]*n.y + m[1][2]*n.z;
        float z = m[2][0]*n.x + m[2][1]*n.y + m[2][2]*n.z;
        return make_float3(x, y, z);
    }

    __host__ __device__ Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                r.m[i][j] = 0;
                for (int k = 0; k < 4; k++)
                    r.m[i][j] += m[i][k] * o.m[k][j];
            }
        return r;
    }

    __host__ __device__ Mat4& operator+=(const Mat4& o) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] += o.m[i][j];
        return *this;
    }

    __host__ __device__ Mat4 operator*(float s) const {
        Mat4 r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r.m[i][j] = m[i][j] * s;
        return r;
    }
};

// ─────────────────────────────────────────────
//  GPU Device Array wrapper
// ─────────────────────────────────────────────

template<typename T>
struct DeviceArray {
    T*     data = nullptr;
    size_t count = 0;

    DeviceArray() = default;

    explicit DeviceArray(size_t n) { allocate(n); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& o) noexcept
        : data(o.data), count(o.count) { o.data = nullptr; o.count = 0; }

    ~DeviceArray() { free(); }

    void allocate(size_t n) {
        free();
        count = n;
        if (n > 0)
            cudaMalloc(&data, n * sizeof(T));
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; count = 0; }
    }

    void zero() { cudaMemset(data, 0, count * sizeof(T)); }

    void upload(const std::vector<T>& v) {
        if (count != v.size()) allocate(v.size());
        cudaMemcpy(data, v.data(), count * sizeof(T), cudaMemcpyHostToDevice);
    }

    void download(std::vector<T>& v) const {
        v.resize(count);
        cudaMemcpy(v.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    size_t size() const { return count; }
    size_t bytes() const { return count * sizeof(T); }
};

// ─────────────────────────────────────────────
//  Camera intrinsics
// ─────────────────────────────────────────────

struct CameraIntrinsics {
    float fx, fy, cx, cy;
    int   width, height;

    __host__ __device__ float2 project(float3 p) const {
        return make_float2(fx * p.x / p.z + cx,
                           fy * p.y / p.z + cy);
    }

    __host__ __device__ float3 unproject(float u, float v, float z) const {
        return make_float3(z * (u - cx) / fx,
                           z * (v - cy) / fy,
                           z);
    }
};

// ─────────────────────────────────────────────
//  TSDF Voxel
// ─────────────────────────────────────────────

struct TSDFVoxel {
    float tsdf;     // truncated signed distance
    float weight;   // running weight
    uchar3 color;   // RGB (optional)
    uint8_t pad;

    __host__ __device__ TSDFVoxel()
        : tsdf(1.f), weight(0.f), color{128,128,128}, pad(0) {}
};

// ─────────────────────────────────────────────
//  Deformation node
// ─────────────────────────────────────────────

static constexpr int K_NEIGHBORS = 4;   // nodi per vertice (skinning)
static constexpr int K_GRAPH     = 8;   // vicini nel grafo nodi

struct DeformNode {
    float3 pos;          // posizione nel canonical frame
    float  radius;       // sigma per i pesi (influenza)
    int    neighbors[K_GRAPH];    // indici nodi vicini nel grafo
    float  neighbor_w[K_GRAPH];   // pesi archi smoothness
    int    num_neighbors;
};

// ─────────────────────────────────────────────
//  Corrispondenza ICP
// ─────────────────────────────────────────────

struct Correspondence {
    float3 src;      // punto nel live frame (warped canonical)
    float3 dst;      // punto nel depth frame
    float3 normal;   // normale del punto dst
    int    node_ids[K_NEIGHBORS];
    float  node_ws[K_NEIGHBORS];
    bool   valid;
};

// ─────────────────────────────────────────────
//  BSR Matrix (Block Sparse Row, blocchi 6x6)
// ─────────────────────────────────────────────

static constexpr int BLOCK_DIM = 6;
static constexpr int BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM;  // 36

struct BSRMatrix {
    int num_block_rows;   // = num_nodes
    int num_blocks;       // totale blocchi non-zero

    DeviceArray<int>   row_ptr;    // [num_block_rows + 1]
    DeviceArray<int>   col_idx;    // [num_blocks]
    DeviceArray<float> values;     // [num_blocks * 36]

    void allocate(int n_rows, int n_blocks) {
        num_block_rows = n_rows;
        num_blocks     = n_blocks;
        row_ptr.allocate(n_rows + 1);
        col_idx.allocate(n_blocks);
        values.allocate(n_blocks * BLOCK_SIZE);
    }

    void zero_values() { values.zero(); }
};

// ─────────────────────────────────────────────
//  Bounding box filter
// ─────────────────────────────────────────────

struct BBoxFilter {
    float3 min_pt;
    float3 max_pt;
    bool   enabled = false;
};

// ─────────────────────────────────────────────
//  Helper: CUDA error check
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)  do { \
    cudaError_t err = call;    \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } } while(0)

inline dim3 grid1d(int n, int block = 256) {
    return dim3((n + block - 1) / block);
}
