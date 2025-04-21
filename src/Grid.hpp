#pragma once
#include <cuda_runtime.h>
#include <helper_types.h>

#include <cuda_utils.cuh>

/**
 * @brief Manages the spatial grid used for the simulation.
 */
class Grid {
   public:
    /**
     * @brief Constructs a Grid object.
     *
     * Calculates the grid dimensions based on the provided volume and kernel radius.
     * Allocates device memory for grid cell IDs, particle IDs, cell start indices,
     * and auxiliary arrays used for sorting.
     *
     * @param Volume The overall simulation volume.
     * @param kernel_radius The smoothing kernel radius.
     * @param num_particles The total number of particles.
     */
    Grid(float3 Volume, float kernel_radius, uint num_particles);

    ~Grid();

    /**
     * @brief Returns the total number of grid cells.
     *
     * @return uint The linear size of the grid.
     */
    uint getLinearGridSize() { return grid_size.x * grid_size.y * grid_size.z; };

    /**
     * @brief Updates the grid based on the current particle positions.
     *
     * This function launches CUDA kernels to:
     * - Initialize grid cell indices for each particle.
     * - Sort the particle IDs using the grid cell indices.
     * - Compute the start index of each grid cell.
     *
     * @param particles Pointer to the array of Particle objects.
     */
    void updateGrid(Particle* particles);

    uint num_particles;
    uint* grid_cell_ID;
    uint* particle_ID;
    int* cellStartIdx;

   private:
    /**
     * @brief Sorts grid cell indices and associated particle IDs.
     *
     * Uses CUB to sort the grid_cell_ID and particle_ID arrays.
     */
    void sortGrid();

    uint3 grid_size;
    void* d_temp_storage = nullptr;
    size_t d_storage_size = 0;
    uint* d_keys_out;
    uint* d_values_out;
    Particle* d_temp_particle;
};