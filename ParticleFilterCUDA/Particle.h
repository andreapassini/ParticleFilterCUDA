#pragma once
#include <stdio.h>
#include <cmath>

#include "Float2.h"

#include "MyDefs.h"

typedef struct Particles {
    float x[N];
    float y[N];
    float heading[N];
    float weights[N];
};

void Create_Particles(Particles* p, float dim);
void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange);

void PrintParticle(const Particles* const p, int i);

float Lerp(float A, float B, float factor);