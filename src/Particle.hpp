#pragma once

#include <cuda_gl_interop.h>
#include <helper_types.h>

#include <Grid.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cuda_utils.cuh>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define PI 3.14159265358979323846f

/**
 * @brief Class for simulating particles using SPH (Smoothed Particle Hydrodynamics).
 */
class ParticleSim {
   public:
    /**
     * @brief Constructs a ParticleSim object.
     *
     * Initializes the simulation based on the given parameters.
     * Registers the provided OpenGL VBO for CUDA graphics interop,
     * allocates device buffers for particles,
     * and initializes the simulation grid.
     *
     * @param params Simulation parameters (volume, numParticles, initialVelocity, etc.).
     * @param VBO OpenGL Vertex Buffer Object used to store particle positions.
     * @param kernel_radius Smoothing kernel radius used in SPH estimations.
     */
    ParticleSim(const SimParams& params, GLuint& VBO, float kernel_radius);

    /**
     * @brief Destroys the ParticleSim object.
     */
    ~ParticleSim();

    /**
     * @brief Copies particle data from CUDA to the OpenGL buffer.
     */
    void CUDAToOpenGLParticles();

    /**
     * @brief Updates simulation parameters.
     *
     * @param newParams The new set of parameters to use in the simulation.
     */
    void updateParams(const SimParams& newParams);

    /**
     * @return The number of particles.
     */
    inline uint getNumParticles() { return this->params_.numParticles; }

    /**
     * @brief Starts the simulation thread.
     */
    void startSimulation() {
        std::cout << "Starting Simulation Thread!" << std::endl;
        this->running_.store(true);
        this->simulationThread_ = std::thread(&ParticleSim::simulate, this);
    }

    /**
     * @brief Pauses the simulation thread.
     */

    void pauseSimulation() {
        this->running_.store(false);
        if (this->simulationThread_.joinable())
            this->simulationThread_.join();
    }

    /**
     * @brief Sets the simulation speed.
     */
    void setSimulationTimestep(float deltaT) {
        if (deltaT <= 0.0)
            adaptiveStepTime.store(true);
        else {
            adaptiveStepTime.store(false);
            this->deltaT = deltaT;
        }
    }

    std::atomic<bool> running_ = false;

   private:
    /**
     * @brief Initializes the particles.
     *
     * Sets up the initial positions and states for all particles,
     * placing them randomly within the simulation volume.
     */
    void initParticles();

    void initRandomGenerator();

    /**
     * @brief Updates the state of particles and density field.
     *
     * Computes the new positions, velocities, and densities based on computed forces.
     * Updated the density field needed for visualization, and swaps the active buffers
     * after updating.
     */
    void updateParticles();

    /**
     * @brief Main simulation loop.
     */
    void simulate();

    SimParams params_;
    float kernel_radius;
    Particle* pBuffers[2];             // front & back device buffers
    cudaGraphicsResource_t sharedVBO;  // resource that points to the shared openGL buffer
    std::atomic<bool> adaptiveStepTime = false;
    float deltaT;

    std::atomic<uint> activeBuffer = 0;
    std::thread simulationThread_;

    std::mutex buffer_mutex;
    std::mutex param_mutex;

    curandState* d_states;  // random generator states

    Grid g;

    uint blockSize = 128;
};
