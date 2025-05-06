#include <Particle.hpp>
#include <cub/cub.cuh>
#include <random>

// Simulation parameters written in constant memory to be accessed by all threads
__constant__ __device__ float K;
__constant__ __device__ float RHO_ZERO;
__constant__ __device__ float MASS;
__constant__ __device__ float PRESSURE_MULT;
__constant__ __device__ float VISC_MULT;

__constant__ __device__ float TIMESTEP;
__device__ curandState* d_randStates;

// Function pointer type to do funny functional stuff in SPHEstimator
typedef float3 (*funcPointer_t)(Particle, Particle);

// compute particle densities using SPH estimation
__global__ void computeParticleDensity(Particle* particles, int* cellStartIdx, uint* cellKeys, uint* cellValues);
// main update particle states logic
__global__ void updateParticleKernel(Particle* particles, Particle* nextParticles, int* cellStart, uint* cellKeys, uint* cellValues, const float deltaT);
// applies an SPH kernel estimator over neighbors to compute desired quantity
__device__ float3 SPHEstimator(funcPointer_t SPH_func, Particle* particles, Particle current, int* cellStartIdx, uint* cellKeys, uint* cellValues);
// SPH functions for qty estimation
__device__ float3 density_(Particle current, Particle other);
__device__ float3 viscosity_(Particle current, Particle other);
__device__ float3 pressure_(Particle current, Particle other);

// smoothing kernels
__device__ float poly6Kernel(const float r);
__device__ float spikyKernel(const float r);
__device__ float spikyKernelGradient(const float r);
__device__ float viscosityKernel(const float r);
__device__ float viscosityKernelLaplacian(const float r);

// utils
__device__ void checkParticleBoundaries(Particle& pos);
__device__ float3 clampVelocity(float3 vel, float max_vel);
__global__ void initCurandStates(curandState* states, unsigned long seed, int numStates);
__device__ float3 randomDirection(int idx);

ParticleSim::ParticleSim(const SimParams& params, GLuint& VBO, float kernel_radius)
    : g(params.volume, kernel_radius, params.numParticles), kernel_radius(kernel_radius) {
    updateParams(params);

    CHECK(cudaMalloc(&pBuffers[0], params.numParticles * sizeof(Particle)));
    CHECK(cudaMalloc(&pBuffers[1], params.numParticles * sizeof(Particle)));
    CHECK(cudaMemcpyToSymbol(VOLUME, &(params.volume), sizeof(params.volume)));
    CHECK(cudaMemcpyToSymbol(KERNEL_RADIUS, &kernel_radius, sizeof(kernel_radius)));

    CHECK(cudaGraphicsGLRegisterBuffer(&sharedVBO, VBO, cudaGraphicsMapFlagsWriteDiscard));

    initParticles();
    init_nbr_coords();
    initRandomGenerator();

    g.updateGrid(pBuffers[activeBuffer]);
}

ParticleSim::~ParticleSim() {
    running_ = false;
    if (simulationThread_.joinable()) {
        simulationThread_.join();
    }
    CHECK(cudaFree(pBuffers[0]));
    CHECK(cudaFree(pBuffers[1]));
    CHECK(cudaFree(d_states));
}

void ParticleSim::simulate() {
    size_t loop = 0;
    while (running_) {
        auto start = std::chrono::high_resolution_clock::now();

        g.updateGrid(pBuffers[activeBuffer]);
        updateParticles();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;

        if (adaptiveStepTime.load())
            deltaT = 0.9f * deltaT + 0.1f * duration.count();

        if (++loop % 200 == 0)
            printf("Simulation frame time: %.5fs\n", duration.count());
    }
}

void ParticleSim::updateParams(const SimParams& newParams) {
    std::scoped_lock<std::mutex> lock(param_mutex);
    params_ = newParams;
    CHECK(cudaMemcpyToSymbol(K, &(newParams.gas_constant), sizeof(newParams.gas_constant)));
    CHECK(cudaMemcpyToSymbol(RHO_ZERO, &(newParams.pressure_target), sizeof(newParams.pressure_target)));
    CHECK(cudaMemcpyToSymbol(MASS, &(newParams.particle_mass), sizeof(newParams.particle_mass)));
    CHECK(cudaMemcpyToSymbol(PRESSURE_MULT, &(newParams.pressure_multiplier), sizeof(newParams.pressure_multiplier)));
    CHECK(cudaMemcpyToSymbol(VISC_MULT, &(newParams.viscosity_multiplier), sizeof(newParams.viscosity_multiplier)));
}

void ParticleSim::CUDAToOpenGLParticles() {
    std::scoped_lock<std::mutex> lock(buffer_mutex);  // Lock the mutex while I'm writing on the shared buffer

    void* devPtr;
    size_t size;
    CHECK(cudaGraphicsMapResources(1, &sharedVBO, 0));
    CHECK(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, sharedVBO));
    CHECK(cudaMemcpy(devPtr, pBuffers[activeBuffer], size, cudaMemcpyDeviceToDevice));
    CHECK(cudaGraphicsUnmapResources(1, &sharedVBO, 0));
}

void ParticleSim::initParticles() {
    uint numParticles = getNumParticles();
    float3 vol = params_.volume;
    std::random_device rd;
    std::default_random_engine e2(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    std::vector<Particle> particles(numParticles);
    for (uint i = 0; i < numParticles; ++i) // initialize particles at random positions
        particles[i].position = make_float3((vol.x / 5) + dist(e2) * (vol.x / 3), vol.y * dist(e2), (vol.z / 5) + dist(e2) * (vol.z / 2));

    CHECK(cudaMemcpyToSymbol(NUM_PARTICLES, &numParticles, sizeof(numParticles)));

    CHECK(cudaMemcpy(pBuffers[0], particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

void ParticleSim::updateParticles() {
    uint numBlocks = (getNumParticles() + blockSize - 1) / blockSize;
    uint current = activeBuffer.load();
    uint next = 1 - current;

    // run simulation only if I'm not updating parameters
    std::scoped_lock<std::mutex> p_lock(param_mutex);
    float t_half = deltaT / 2;
    CHECK(cudaMemcpyToSymbol(TIMESTEP, &t_half, sizeof(deltaT)));

    computeParticleDensity<<<numBlocks, blockSize>>>(pBuffers[current], g.cellStartIdx, g.grid_cell_ID, g.particle_ID);
    CHECK(cudaDeviceSynchronize());

    updateParticleKernel<<<numBlocks, blockSize>>>(pBuffers[current], pBuffers[next], g.cellStartIdx, g.grid_cell_ID, g.particle_ID, deltaT);
    CHECK(cudaDeviceSynchronize());

    // swap current and next buffer only if the mutex is unlocked, i.e. if openGL is not reading 'current'
    std::scoped_lock<std::mutex> b_lock(buffer_mutex);
    activeBuffer.store(next);
}

void ParticleSim::initRandomGenerator() {
    int numStates = getNumParticles();

    CHECK(cudaMalloc(&d_states, numStates * sizeof(curandState)));
    CHECK(cudaMemcpyToSymbol(d_randStates, &d_states, sizeof(curandState*)));

    int blocks = (numStates + blockSize - 1) / blockSize;
    initCurandStates<<<blocks, blockSize>>>(d_states, time(NULL), numStates);
    CHECK(cudaDeviceSynchronize());
}

__global__ void computeParticleDensity(Particle* particles, int* cellStartIdx, uint* cellKeys, uint* cellValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES)
        particles[idx].density = max(SPHEstimator(density_, particles, particles[idx], cellStartIdx, cellKeys, cellValues).x, 0.001);
}

__global__ void updateParticleKernel(Particle* particles, Particle* nextParticles, int* cellStart, uint* cellKeys, uint* cellValues, const float deltaT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        Particle current = particles[idx];
        uint gridIdx = linearGridIndex(getGridCoordinate(current));

        float3 pressure_force = -PRESSURE_MULT * SPHEstimator(pressure_, particles, current, cellStart, cellKeys, cellValues);
        float3 viscosity_force = VISC_MULT * SPHEstimator(viscosity_, particles, current, cellStart, cellKeys, cellValues);
        float3 gravity_force = make_float3(0.0, -9.81 * current.density, 0.0);

        float3 total_force = pressure_force + viscosity_force + gravity_force;
        float3 acceleration = total_force / current.density;
        float3 velocity = current.velocity + deltaT * acceleration;

        nextParticles[idx].velocity = clampVelocity(velocity, 15);
        nextParticles[idx].position = current.position + nextParticles[idx].velocity * deltaT;

        checkParticleBoundaries(nextParticles[idx]);
    }
}

__device__ float3 SPHEstimator(funcPointer_t SPH_func, Particle* particles, Particle current, int* cellStartIdx, uint* cellKeys, uint* cellValues) {
    float3 quantity = make_float3(0.0f, 0.0f, 0.0f);
    uint3 gridCoord = getGridCoordinate(current);

    // Iterate over the 27 neighboring cells
    for (int i = 0; i < 27; ++i) {
        uint3 nbrCoord = gridCoord + NBR_COORD_OFFSETS[i];
        // Ensure the neighbor cell is within grid bounds
        if (nbrCoord.x < GRID_DIMS.x && nbrCoord.y < GRID_DIMS.y && nbrCoord.z < GRID_DIMS.z) {
            int cellIdx = cellStartIdx[linearGridIndex(nbrCoord)];
            if (cellIdx < 0)  // if counter == -1 there are no particles in the cell
                continue;

            while (cellKeys[cellIdx] == cellKeys[++cellIdx] && cellIdx < NUM_PARTICLES)  // iterate over all particles in a cell
            {
                Particle other = particles[cellValues[cellIdx - 1]];
                quantity += (*SPH_func)(current, other);
            }
        }
    }
    return quantity;
}

__global__ void initCurandStates(curandState* states, unsigned long seed, int numStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numStates)
        curand_init(seed, idx, 0, &states[idx]);
}

__device__ float3 randomDirection(int idx) {
    // Use the global random states array
    curandState localState = d_randStates[idx];
    float theta = 2.0f * 3.14159265f * curand_uniform(&localState);
    float z = 2.0f * curand_uniform(&localState) - 1.0f;
    float r = sqrtf(1.0f - z * z);
    float3 dir = make_float3(r * cosf(theta), r * sinf(theta), z);
    // Save the state back
    d_randStates[idx] = localState;
    return dir;
}

__device__ float poly6Kernel(const float r) {
    const float h = KERNEL_RADIUS;
    return (r >= h || r == 0.) ? 0. : (315.f / (64.f * PI * powf(h, 9))) * powf(powf(h, 2) - powf(r, 2), 3);
}

__device__ float spikyKernel(const float r) {
    const float h = KERNEL_RADIUS;
    return (r >= h || r == 0.) ? 0. : (15.f / (2 * PI * powf(h, 6))) * powf(h - r, 3);
}

__device__ float spikyKernelGradient(const float r) {
    const float h = KERNEL_RADIUS;
    return (r >= h || r == 0.) ? 0. : (15.f / (PI * powf(h, 6))) * powf(h - r, 2);
}

__device__ float viscosityKernel(const float r) {
    const float h = KERNEL_RADIUS;
    return (r >= h || r == 0.) ? 0. : (15.f / (2 * PI * powf(h, 6))) * (-powf(r, 3) / (2 * powf(h, 3)) + powf(r, 2) / powf(h, 2) + h / (2 * r) - 1);
}

__device__ float viscosityKernelLaplacian(const float r) {
    const float h = KERNEL_RADIUS;
    return (r >= h || r == 0.) ? 0. : (45.f / (2 * PI * powf(h, 6)) * (h - r));
}

__device__ float3 density_(Particle current, Particle other) {
    float3 p_i = nextParticlePosition(current, TIMESTEP);
    float3 p_j = nextParticlePosition(other, TIMESTEP);
    return MASS * make_float3(spikyKernel(L2Norm(p_i - p_j)), 0.0, 0.0);
}

__device__ float3 viscosity_(Particle current, Particle other) {
    float3 p_i = nextParticlePosition(current, TIMESTEP);
    float3 p_j = nextParticlePosition(other, TIMESTEP);
    return (MASS / other.density) * (other.velocity - current.velocity) * viscosityKernelLaplacian(L2Norm(p_i - p_j));
}

__device__ float3 pressure_(Particle current, Particle other) {
    float3 p_i = nextParticlePosition(current, TIMESTEP);
    float3 p_j = nextParticlePosition(other, TIMESTEP);

    float pressure_i = K * (current.density - RHO_ZERO);
    float pressure_j = K * (other.density - RHO_ZERO);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float3 dir = (L2Norm(p_i - p_j) <= 0.001) ? randomDirection(idx) : ((p_j - p_i) / L2Norm(p_i - p_j));

    return (MASS / other.density) * ((pressure_i + pressure_j) / 2) * dir * spikyKernelGradient(L2Norm(p_i - p_j));
}

__device__ void checkParticleBoundaries(Particle& pos) {
    float damping_factor = 0.8;
    if (pos.position.x <= 0.0f) {
        pos.position.x = 0.0f;
        pos.velocity.x *= -1 * damping_factor;
    }
    if (pos.position.y <= 0.0f) {
        pos.position.y = 0.0f;
        pos.velocity.y *= -1 * damping_factor;
    }
    if (pos.position.z <= 0.0f) {
        pos.position.z = 0.0f;
        pos.velocity.z *= -1 * damping_factor;
    }
    if (pos.position.x >= VOLUME.x) {
        pos.position.x = VOLUME.x;
        pos.velocity.x *= -1 * damping_factor;
    }
    if (pos.position.y >= VOLUME.y) {
        pos.position.y = VOLUME.y;
        pos.velocity.y *= -1 * damping_factor;
    }
    if (pos.position.z >= VOLUME.z) {
        pos.position.z = VOLUME.z;
        pos.velocity.z *= -1 * damping_factor;
    }
}

__device__ float3 clampVelocity(float3 vel, float max_vel) {
    float speed = L2Norm(vel);
    if (speed > max_vel && speed > 0.f) {
        float scale = max_vel / speed;
        vel = vel * scale;
    }
    return vel;
}
