#include "Particle.h"

#include <time.h>
#include <stdlib.h>
#include <math.h>

#define PI2 2.0f * 3.141592f

float Lerp(float A, float B, float factor)
{
    //return A * (1 - factor) + B * (factor);
    return A + (B - A) * factor;
}

void Create_Particles(Particles* p, float dim) {
    p->size = dim;
    p->x = (float*)malloc(dim * sizeof(float));
    p->y = (float*)malloc(dim * sizeof(float));
    p->heading = (float*)malloc(dim * sizeof(float));
    p->weights = (float*)malloc(dim * sizeof(float));
}

void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange) {
    p->size = dim;
    p->x = (float*)malloc(dim * sizeof(float));
    p->y = (float*)malloc(dim * sizeof(float));
    p->heading = (float*)malloc(dim * sizeof(float));
    p->weights = (float*)malloc(dim * sizeof(float));

    srand((unsigned int)time(NULL));   // Initialization, should only be called once.
    float r = 0.0f;

    for (int i = 0; i < dim; i++) {
        r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
        p->x[i] = Lerp(xRange->x, xRange->y, r);

        r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
        p->y[i] = Lerp(yRange->x, yRange->y, r);

        r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
        p->heading[i] = Lerp(headingRange->x, headingRange->y, r);
        p->heading[i] = fmodf(p->heading[i], PI2);

        p->weights[i] = 1.0f / dim;

        //printf("particle: %d, \n\t weight: %f\n", i, p->weights[i]);
    }
}


