#include "Particle.h"

void CreateParticleDim(Particles* p, float dim) {
    p->x = (float*)calloc(dim * sizeof(float));
    p->y = (float*)calloc(dim * sizeof(float));
    p->weights = (float*)calloc(dim * sizeof(float));
    p->heading = (float*)calloc(dim * sizeof(float));
}