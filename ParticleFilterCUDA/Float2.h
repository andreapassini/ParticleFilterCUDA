#pragma once

#include <cmath>

typedef struct Floats2 {
    float* x;
    float* y;
};

typedef struct Float2 {
    float x;
    float y;
};

static Float2 Minus(const Float2 a, const Float2 b) {
    Float2 res;
    res.x = 0.0f;
    res.y = 0.0f;

    res.x = a.x - b.x;
    res.y = a.y - b.y;

    return res;
}

static Float2 Minus(const Float2* const a, const Float2* const b) {
    Float2 res;
    res.x = 0.0f;
    res.y = 0.0f;

    res.x = a->x - b->x;
    res.y = a->y - b->y;

    return res;
}

static float Magnitude(const float const x, const float const y) {
    float mag = sqrt((x * x) + (y * y));
    return mag;
}

static float Magnitude(const Float2& const vec2) {
    return Magnitude(vec2.x, vec2.y);
}
