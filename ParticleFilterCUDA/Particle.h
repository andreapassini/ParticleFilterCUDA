#pragma once
#include <stdio.h>
#include <cmath>

#include "Float2.h"

typedef struct Particles {
    float* x;
    float* y;
    float* heading;
    float* weights;
    unsigned int size;
};

void Create_Particles(Particles* p, float dim);
void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange);

void PrintParticle(const Particles* const p, int i);

float Lerp(float A, float B, float factor);

long BytesOfParticles(const Particles* const p);