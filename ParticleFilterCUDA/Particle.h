#pragma once
#include <stdio.h>
#include <cmath>

typedef struct Particles {
    float* x;
    float* y;
    float* heading;
    float* weights;
};

typedef struct floats2 {
    float* x;
    float* y;
};

void CreateParticleDim(Particles* p, float dim);
