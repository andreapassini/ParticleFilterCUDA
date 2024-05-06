#pragma once
#include <stdio.h>
#include <cmath>

typedef struct Particles {
    float* x;
    float* y;
    float* heading;
    float* weights;
    unsigned int size;
};

typedef struct Float2 {
    float x;
    float y;
};

typedef struct Floats2 {
    float* x;
    float* y;
};


void Create_Particles(Particles* p, float dim);
void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange);

void PrintParticle(const Particles* const p, int i);

float Lerp(float A, float B, float factor);