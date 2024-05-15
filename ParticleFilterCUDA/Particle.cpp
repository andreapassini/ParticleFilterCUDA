#include "Particle.h"

#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "Float2.h"

#include "MyDefs.h"

float Lerp(float A, float B, float factor)
{
    //return A * (1 - factor) + B * (factor);
    return A + (B - A) * factor;
}

void Create_Particles(Particles* p, float dim) {
    p = (Particles*)malloc(sizeof(Particles));
}

void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange) {

    p = (Particles*)malloc(sizeof(Particles));

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

void PrintParticle(const Particles* const p, int i)
{
    printf("particle: %d, \n\t x: %f \n\t y: %f \n\t heading: %f \n\t weight: %f\n", 
        i, 
        p->x[i],
        p->y[i],
        p->heading[i],
        p->weights[i]);
}
