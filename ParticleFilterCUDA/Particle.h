#pragma once
#include <stdio.h>
#include <cmath>

typedef struct Particle {
    float* x;
    float* y;
    float* heading;
    float* weights;
};

typedef struct Vec2 {
    float x;
    float y;
};

void CreateParticleDim(Particle* p, float dim);
