#include <cuda_utils.cuh>

__constant__ __device__ uint3 GRID_DIMS;
__constant__ __device__ uint NUM_PARTICLES;
__constant__ __device__ float3 VOLUME;
__constant__ __device__ float KERNEL_RADIUS;
__constant__ __device__ uint3 NBR_COORD_OFFSETS[27];

void init_nbr_coords() {
    int idx = 0;
    uint3 array[27];
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                array[idx++] = make_uint3(x, y, z);

    CHECK(cudaMemcpyToSymbol(NBR_COORD_OFFSETS, array, 27 * sizeof(uint3)));
}
