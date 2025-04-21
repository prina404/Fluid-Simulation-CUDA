#pragma once
#include <cuda_runtime.h>


struct SimParams {
    int numParticles;
    float3 volume;
    float gas_constant;
    float pressure_target;
    float pressure_multiplier;
    float viscosity_multiplier;
    float particle_mass;
};

struct Particle {
    float3 position;
    float3 velocity;
    float density;
};


