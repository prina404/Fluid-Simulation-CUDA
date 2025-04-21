#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_types.h>
#include <curand_kernel.h>

#include <iostream>

extern __constant__ __device__ uint3 GRID_DIMS;
extern __constant__ __device__ uint NUM_PARTICLES;
extern __constant__ __device__ float3 VOLUME;
extern __constant__ __device__ float KERNEL_RADIUS;
// Generate the 27 neighbor coordinate triples for neighbor grid index calculation
extern __constant__ __device__ uint3 NBR_COORD_OFFSETS[27];

#define CHECK(ans)                            \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define GRID_SIZE (GRID_DIMS.x) * (GRID_DIMS.y) * (GRID_DIMS.z)

void init_nbr_coords();

// This function assumes that all the particles' coordinates lie in the interval [0, gridsize)
inline __device__ uint3 getGridCoordinate(const Particle& pos) {
    return make_uint3(static_cast<uint>((pos.position.x * (GRID_DIMS.x - 1)) / VOLUME.x),
                      static_cast<uint>((pos.position.y * (GRID_DIMS.y - 1)) / VOLUME.y),
                      static_cast<uint>((pos.position.z * (GRID_DIMS.z - 1)) / VOLUME.z));
}

/**
 * Given a linear index, return the voxel center coordinates relative to the VOLUME of the simulation
 */

inline __device__ uint linearGridIndex(const uint3& gridCoord) {
    return gridCoord.z * (GRID_DIMS.x * GRID_DIMS.y) + gridCoord.y * GRID_DIMS.x + gridCoord.x;
}

inline __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline __device__ __host__ uint innerProduct(const uint3 v){
    return v.x * v.y * v.z;
}

inline __device__ float L2Norm(const float3& v) {
    return sqrtf(dot(v, v));
}

inline __device__ __host__ float L1Norm(const float3& v) {
    return v.x + v.y + v.z;
}

inline __device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator-(const float& a, const float3& b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator+(const float3& a, const float& n) {
    return make_float3(a.x + n, a.y + n, a.z + n);
}

inline __device__ float3 operator*(const float3& a, const float& n) {
    return make_float3(a.x * n, a.y * n, a.z * n);
}

inline __device__ float3 operator*(const float& n, const float3& a) {
    return a * n;
}

inline __device__ __host__ float3 operator/(const float3& a, const float& n) {
    return make_float3(a.x / n, a.y / n, a.z / n);
}

inline __device__ float3 operator/(const uint3& a, const float& n) {
    return make_float3(a.x / n, a.y / n, a.z / n);
}

inline __device__ uint3 operator+(const uint3& a, const uint3& b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ bool operator==(const uint3& a, const uint3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __device__ float3& operator+=(float3& a, const float3& b) {
    a = a + b;
    return a;
}
inline __device__ float3& operator-=(float3& a, const float3& b) {
    a = a - b;
    return a;
}

inline __device__ float3 nextParticlePosition(Particle p, float deltaT){
    return p.position + p.velocity * (deltaT/2);
}

