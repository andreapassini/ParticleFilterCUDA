#include "Particle.h"

void CreateParticleDim(Particles* p, float dim) {
    p->x = (float*)malloc(dim * sizeof(float));
    p->y = (float*)malloc(dim * sizeof(float));
    p->weights = (float*)malloc(dim * sizeof(float));
    p->heading = (float*)malloc(dim * sizeof(float));
}