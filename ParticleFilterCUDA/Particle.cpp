#include "Particle.h"

void CreateParticleDim(Particle* p, float dim) {
    p->x = (float*)malloc(dim * sizeof(float));
    p->y = (float*)malloc(dim * sizeof(float));
    p->weight = (float*)malloc(dim * sizeof(float));
    p->heading = (float*)malloc(dim * sizeof(float));
}