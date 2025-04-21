#include <Grid.hpp>
#include <cub/cub.cuh>

__global__ void initGridKernel(const Particle*, uint*, uint*);
__global__ void updateGridKernel(const uint*, int*);
__global__ void sortParticleBuffer(uint* particle_ID, Particle* particles, Particle* temp);

Grid::Grid(float3 volume, float kernel_radius, uint num_parts) : num_particles(num_parts) {
    grid_size = {
        static_cast<uint>(ceil(volume.x / kernel_radius)),
        static_cast<uint>(ceil(volume.y / kernel_radius)),
        static_cast<uint>(ceil(volume.z / kernel_radius)),
    };
    CHECK(cudaMemcpyToSymbol(GRID_DIMS, &grid_size, sizeof(uint3)));

    // malloc grid main arrays
    CHECK(cudaMalloc(&grid_cell_ID, num_particles * sizeof(uint)));
    CHECK(cudaMalloc(&particle_ID, num_particles * sizeof(uint)));
    CHECK(cudaMalloc(&cellStartIdx, getLinearGridSize() * sizeof(int)));
    CHECK(cudaMalloc(&d_temp_particle, num_particles * sizeof(Particle)));

    // malloc support arrays needed for sorting the grid
    CHECK(cudaMalloc(&d_keys_out, num_particles * sizeof(uint)));
    CHECK(cudaMalloc(&d_values_out, num_particles * sizeof(uint)));

    CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, d_storage_size, grid_cell_ID, d_keys_out, particle_ID, d_values_out, num_particles));
    CHECK(cudaMalloc(&d_temp_storage, d_storage_size));
    printf("LinearGridSize: %d\n", getLinearGridSize());
}

Grid::~Grid() {
    CHECK(cudaFree(grid_cell_ID));
    CHECK(cudaFree(particle_ID));
    CHECK(cudaFree(cellStartIdx));
    CHECK(cudaFree(d_temp_storage));
    CHECK(cudaFree(d_keys_out));
    CHECK(cudaFree(d_values_out));
    CHECK(cudaFree(d_temp_particle));
}
void Grid::sortGrid() {
    CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, d_storage_size, grid_cell_ID, d_keys_out, particle_ID, d_values_out, num_particles));
    CHECK(cudaMemcpy(grid_cell_ID, d_keys_out, num_particles * sizeof(uint), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(particle_ID, d_values_out, num_particles * sizeof(uint), cudaMemcpyDeviceToDevice));
}

void Grid::updateGrid(Particle* particles) {
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    CHECK(cudaMemset(cellStartIdx, -1, getLinearGridSize() * sizeof(int)));

    initGridKernel<<<numBlocks, blockSize>>>(particles, grid_cell_ID, particle_ID);
    CHECK(cudaDeviceSynchronize());

    sortGrid();

    updateGridKernel<<<numBlocks, blockSize>>>(grid_cell_ID, cellStartIdx);
    CHECK(cudaDeviceSynchronize());

    sortParticleBuffer<<<numBlocks, blockSize>>>(particle_ID, particles, d_temp_particle);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(particles, d_temp_particle, num_particles * sizeof(Particle), cudaMemcpyDeviceToDevice));
}

__global__ void sortParticleBuffer(uint* particle_ID, Particle* particles, Particle* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        temp[idx] = particles[particle_ID[idx]];
        particle_ID[idx] = idx;
    }
}

__global__ void initGridKernel(const Particle* particles, uint* grid_cell, uint* part_ID) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        grid_cell[idx] = linearGridIndex(getGridCoordinate(particles[idx]));
        part_ID[idx] = idx;
    }
}

__global__ void updateGridKernel(const uint* grid_cell, int* cellStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        cellStart[0] = grid_cell[0];
    if (idx < NUM_PARTICLES - 1)
        if (grid_cell[idx] != grid_cell[idx + 1])
            cellStart[grid_cell[idx + 1]] = idx + 1;
}
