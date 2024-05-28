# Particle Filter

https://github.com/mattdean1/cuda.git

## Motivation

We have moving objects that we want to track, like fighter jets, missiles or a car, the problem will have these characteristics:

* **multimodal**: We want to track zero, one, or more than one object simultaneously.

* **occlusions**: One object can hide another, resulting in one measurement for multiple objects.

* **nonlinear behavior**: Aircraft are buffeted by winds, balls move in parabolas, and people collide into each other.

* **nonlinear measurements**: Radar gives us the distance to an object. Converting that to an (x,y,z) coordinate requires a square root, which is nonlinear.

* **non-Gaussian noise:** as objects move across a background the computer vision can mistake part of the background for the object.

* **continuous:** the object's position and velocity (i.e. the state space) can smoothly vary over time.

* **multivariate**: we want to track several attributes, such as position, velocity, turn rates, etc.

* **unknown process model**: we may not know the process model of the system.

## Generic Particle Filter Algorithm

1. **Randomly generate a bunch of particles**
    
  Particles can have position, heading, and/or whatever other state variable you need to estimate. Each has a weight (probability) indicating how likely it matches the actual state of the system. Initialize each with the same weight.
  
2. **Predict next state of the particles**

 Move the particles based on how you predict the real system is behaving.

3. **Update**

  Update the weighting of the particles based on the measurement. Particles that closely match the measurements are weighted higher than particles which don't match the measurements very well.
  
4. **Resample**

  Discard highly improbable particle and replace them with copies of the more probable particles.
  
5. **Compute Estimate**

  Optionally, compute weighted mean and covariance of the set of particles to get a state estimate.

## Problem to solve

Consider tracking a robot or a car in an urban environment. In this problem we tracked a robot that has a sensor which measures the range and bearing to known landmarks.

We start by creating several thousand **particles**.

**Particle**  
- **Position** (possible belief of where the robot is in the scene)
- **Heading**
- **Weight**

We would want to scatter the particles uniformly over the entire scene. If you think of all of the particles representing a probability distribution, locations where there are more particles represent a higher belief, and locations with fewer particles represents a lower belief. If there was a large clump of particles near a specific location that would imply that we were more certain that the robot is there.

**Weight** - ideally the probability that it represents the true position of the robot. This probability is rarely computable, so we only require it be *proportional*  to that probability, which is computable. At initialization we have no reason to favor one particle over another, so we assign a weight of $1/N$, for $N$ particles. We use $1/N$ so that the sum of all probabilities equals one.

The combination of particles and weights forms the *probability distribution* for our problem. In this problem the robot can move on a plane of some arbitrary dimension, with the lower right corner at (0,0).

To track our robot we need to maintain states for x, y, and heading. We will store `N` particles in a `(N, 4)` shaped array. The four columns contain x, y, heading and weight, in that order.

If you are passively tracking something (no control input), then you would need to include velocity in the state and use that estimate to make the prediction. More dimensions requires exponentially more particles to form a good estimate, so we always try to minimize the number of random variables in the state.


# CPU Implementation

## Particle Generation

```python
import numpy as np
from numpy.random import uniform

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles
```

### C/C++ implementation
```c++
void CreateAndRandomInitialize_Particles(Particles* p, float dim, Float2* xRange, Float2* yRange, Float2* headingRange) {
    p->size = dim;
    p->x = (float*)malloc(dim * sizeof(float));
    p->y = (float*)malloc(dim * sizeof(float));
    p->heading = (float*)malloc(dim * sizeof(float));
    p->weights = (float*)malloc(dim * sizeof(float));

    srand((unsigned int)time(NULL)); 
    float r = 0.0f;

    for (int i = 0; i < dim; i++) {
        r = ((float)rand() / (float)(RAND_MAX));      
        p->x[i] = Lerp(xRange->x, xRange->y, r);

        r = ((float)rand() / (float)(RAND_MAX));      
        p->y[i] = Lerp(yRange->x, yRange->y, r);

        r = ((float)rand() / (float)(RAND_MAX));
        p->heading[i] = Lerp(headingRange->x, headingRange->y, r);
        p->heading[i] = fmodf(p->heading[i], PI2);

        p->weights[i] = 1.0f / dim;
    }
}
```

## Predict Step

Each particle represents a possible position for the robot. Suppose we send a command to the robot to move 0.1 meters while turning by 0.007 radians. We could move each particle by this amount. If we did that we would soon run into a problem. The robot's controls are not perfect so it will not move exactly as commanded. Therefore we need to add noise to the particle's movements to have a reasonable chance of capturing the actual movement of the robot. If you do not model the uncertainty in the system the particle filter will not correctly model the probability distribution of our belief in the robot's position.

### Python implementation
```python
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
```

### C/C++ implementation
```c++
static void PredictCPU(Particles* const p, const Float2* const u, const Float2* const std, const float dt) {
    srand((unsigned int)time(NULL));   

    for (int i = 0; i < p->size; i++) {
        float r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.

        // update heading
        p->heading[i] += u->x + (r * std->x);
        p->heading[i] = fmodf(p->heading[i], PI2);

        float dist = (u->y * dt) + (r * std->y);

        // move in the(noisy) commanded direction
        p->x[i] += cos(p->heading[i]) * dist;
        p->y[i] += sin(p->heading[i]) * dist;
    }
}
```

## Update Step

Next we get a set of measurements - one for each landmark currently in view. How should these measurements be used to alter our probability distribution as modeled by the particles?

Like in a **Discrete Bayes Filter**. We assigned a probability to each position which we called the *prior*. When a new measurement came in we multiplied the current probability of that position (the *prior*) by the *likelihood* that the measurement matched that location:

which is an implementation of the equation

$$x = \| \mathcal L \bar x \|$$

which is a realization of Bayes theorem:

$$\begin{aligned}P(x \mid z) &= \frac{P(z \mid x)\, P(x)}{P(z)} \\
 &= \frac{\mathtt{likelihood}\times \mathtt{prior}}{\mathtt{normalization}}\end{aligned}$$

 We do the same with our particles. Each particle has a position and a weight which estimates how well it matches the measurement. Normalizing the weights so they sum to one turns them into a probability distribution. The particles those that are closest to the robot will generally have a higher weight than ones far from the robot.


### Python implementation

```python
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
```

### C/C++ implementation
```c++
static void UpdateCPU(Particles* const p, const float const* z, const float R, const Floats2 const* landmarks, const int numberOfLandmarks) {
    int size = p->size;

    for (int i = 0; i < numberOfLandmarks; i++) {
        Floats2 distance;
        distance.x = (float*)malloc(size * sizeof(float));
        memcpy(distance.x, p->x, size * sizeof(float));
        distance.y = (float*)malloc(size * sizeof(float));
        memcpy(distance.y, p->y, size * sizeof(float));

        for (int j = 0; j < size; j++) {
            distance.x[j] -= landmarks->x[i];
            distance.y[j] -= landmarks->y[i];
        }
        float* distanceMagnitudes = (float*)calloc(size, sizeof(float));
        for (int j = 0; j < size; j++) {
            distanceMagnitudes[j] = Magnitude(distance.x[j], distance.y[j]);
        }

        float* normPdfs = (float*)malloc(size * sizeof(float));
        for (int j = 0; j < size; j++) {
            normPdfs[j] = normpdf(z[i], distanceMagnitudes[j], R);;
        }
        for (int j = 0; j < size; j++) { 
            p->weights[j] *= normPdfs[j];
        }

        free(distance.x);
        free(distance.y);
        free(distanceMagnitudes);
        free(normPdfs);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        p->weights[i] += FLT_EPSILON; // avoid round - off to zero
        sum += p->weights[i];
    }

    for (int i = 0; i < size; i++) {
        p->weights[i] /= sum; // normalize
    }
}
```

## Estimate Step

In most applications you will want to know the estimated state after each update, but the filter consists of nothing but a collection of particles. Assuming that we are tracking one object (i.e. it is unimodal) we can compute the mean of the estimate as the sum of the weighted values of the particles.

$$\displaystyle \mu = \frac{1}{N}\sum_{i=1}^N w^ix^i$$

Here I adopt the notation $x^i$ to indicate the $\mathtt{i}^{th}$ particle. A superscript is used because we often need to use subscripts to denote time steps, yielding the unwieldy $x^i_{k+1}$ for the $\mathtt{k+1}^{th}$ time step for example.

This function computes both the mean and variance of the particles:

### Python implementation

```python
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var
```

### C/C++ implementation
```c++
static void EstimateCPU(const Particles* const p, Float2* const mean_out, Float2* const var_out) {

    Floats2 pos;
    pos.x = (float*)malloc(p->size * sizeof(float));
    memcpy(pos.x, p->x, p->size * sizeof(float));
    pos.y = (float*)malloc(p->size * sizeof(float));
    memcpy(pos.y, p->y, p->size * sizeof(float));

    (*mean_out) = WeightedAverage(&pos, p->weights, p->size);
    for (int i = 0; i < p->size; i++) {
        pos.x[i] = (pos.x[i] - mean_out->x) * (pos.x[i] - mean_out->x);
        pos.y[i] = (pos.y[i] - mean_out->y) * (pos.y[i] - mean_out->y);
    }

    (*var_out) = WeightedAverage(&pos, p->weights, p->size);

    free(pos.x);
    free(pos.y);
}
```

## Resampling Step

This algorithm suffers from the *degeneracy problem*. It starts with uniformly distributed particles with equal weights. There may only be a handful of particles near the robot. As the algorithm runs any particle that does not match the measurements will acquire an extremely low weight. Only the particles which are near the robot will have an appreciable weight. We could have 5,000 particles with only 3 contributing meaningfully to the state estimate! We say the filter has *degenerated*.This problem is usually solved by some form of *resampling* of the particles.

Particles with very small weights do not meaningfully describe the probability distribution of the robot. The resampling algorithm discards particles with very low probability and replaces them with new particles with higher probability. It does that by duplicating particles with relatively high probability. The duplicates are slightly dispersed by the noise added in the predict step. This results in a set of points in which a large majority of the particles accurately represent the probability distribution.

The **simple random resampling**, also called **multinomial resampling**. It samples from the current particle set $N$ times, making a new set of particles from the sample. The probability of selecting any given particle should be proportional to its weight.

We accomplish this with NumPy's `cumsum` function. `cumsum` computes the cumulative sum of an array. That is, element one is the sum of elements zero and one, element two is the sum of elements zero, one and two, etc. Then we generate random numbers in the range of 0.0 to 1.0 and do a binary search to find the weight that most closely matches that number:

### Python implementation

```python
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)
```

### C/C++ implementation

```c++
static float* CumSum(const float* const arr_in, const int dim) {
    float* cumSumArr = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j <= i; j++) {
            cumSumArr[i] += arr_in[j];
        }
    }

    return cumSumArr;
}

static void SimpleResample(Particles* const p) {
    int dim = p->size;

    float* cumSum_arr = CumSum(p->weights, dim);

    int* indexes = (int*)malloc(dim * sizeof(int));
    srand((unsigned int)time(NULL));   // Initialization, should only be called once.
    float r = 0.0f;
    for (int i = 0; i < dim; i++) {
        r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
        indexes[i] = SearchSorted(cumSum_arr, r, dim);
    }

    float equalWeight = 1.0f / dim;
    for (int i = 0; i < dim; i++) {
        p->x[i] = p->x[indexes[i]];
        p->y[i] = p->y[indexes[i]];
        p->heading[i] = p->heading[indexes[i]];
        p->weights[i] = equalWeight;
    }

    free(indexes);
    free(cumSum_arr);
}
```

### **Neff**

We don't resample at every epoch. For example, if you received no new measurements you have not received any information from which the resample can benefit. We can determine when to resample by using something called the *effective N*, which approximately measures the number of particles which meaningfully contribute to the probability distribution. The equation for this is

$$\hat{N}_\text{eff} = \frac{1}{\sum w^2}$$

If $\hat{N}_\text{eff}$ falls below some threshold it is time to resample. A useful starting point is $N/2$, but this varies by problem. It is also possible for $\hat{N}_\text{eff} = N$, which means the particle set has collapsed to one point (each has equal weight). It may not be theoretically pure, but if that happens I create a new distribution of particles in the hopes of generating particles with more diversity. If this happens to you often, you may need to increase the number of particles, or otherwise adjust your filter. We will talk more of this later.

#### **Python implementation**

```python
def neff(weights):
    return 1. / np.sum(np.square(weights))
```

#### **C/C++ implementation**

```c++
static float Neff(const float* const weights, const int dim) {
    float res = 0.0f;
    float sum = FLT_EPSILON;

    for (int i = 0; i < dim; i++) {
        float squaredWeight = weights[i] * weights[i];
        sum += squaredWeight;
    }

    res = 1.0f / sum;
    return res;
}
```


# GPU Implementation

## Data Representation

Structure of arrays

```c++
typedef struct Particles {
    float* x;
    float* y;
    float* heading;
    float* weights;
    unsigned int size;
};
```

## Particle Generation


**Kernel**
```c++
__global__ void GenerateParticles(Particles* D_in, Particles* C_out, curandState* states, float2 x_range, float2 y_range, float2 heading_range) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= D_in->size)
        return;

    curand_init(idx, 0, 0, &states[idx]);
    float pos_x = x_range.x + ((x_range.y - x_range.x) * curand_uniform(&states[idx]));
    float pos_y = y_range.x + ((y_range.y - y_range.x) * curand_uniform(&states[idx]));
    float heading = heading_range.x + ((heading_range.y - heading_range.x) * curand_uniform(&states[idx]));
    heading = std::fmod(heading, 2 * PI);

    C_out->x[idx] = pos_x;
    C_out->y[idx] = pos_y;
    C_out->heading[idx] = heading;
}
```

## Predict Step

**Kernel**
```c++
__global__ void PredictGPUKernel(Particles* D_in, Particles* C_out, curandState* states, float* u, float* std, float dt) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= D_in->size)
        return;

    curand_init(idx, 0, 0, &states[idx]);

    float heading = D_in->heading[idx];
    heading += u[0] + (curand_normal(&states[idx]) * std[1]);
    heading = std::fmod(heading, 2 * PI);

    float dist = (u[1] * dt) + (curand_uniform(&states[idx]) * std[1]);
    float pos_x = D_in->x[idx];
    float pos_y = D_in->y[idx];
    pos_x += std::cos(heading) * dist;
    pos_y += std::sin(heading) * dist;

    C_out->x[idx] = pos_x;
    C_out->y[idx] = pos_y;
    C_out->heading[idx] = heading;
}
```

#### Kernel Call

```c++
static void PredictGPU(Particles* const p, const Float2* const u, const Float2* const std, const float dt) {
    long particlesBytes = BytesOfParticles(p);

    Particles* d_particlesIn;
    CHECK(cudaMalloc((void**)&d_particlesIn, particlesBytes));
    CHECK(cudaMemcpy(d_particlesIn, p, particlesBytes, cudaMemcpyHostToDevice));

    Particles* d_particlesOut;
    CHECK(cudaMalloc((void**)&d_particlesOut, particlesBytes));
    CHECK(cudaMemcpy(d_particlesOut, p, particlesBytes, cudaMemcpyHostToDevice));

    uint numBlocks = (p->size + BLOCKSIZE - 1) / BLOCKSIZE;
    curandState* devStates;

    PredictGPUKernel << <numBlocks, BLOCKSIZE >> > (d_particlesIn, d_particlesOut, devStates, (*u), (*std), dt);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(p, d_particlesOut, particlesBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_particlesIn));
    CHECK(cudaFree(d_particlesOut));
}
```

## Update Step

## Estimate Step

## Resampling Step

### **Neff**

```c++
static float NeffGPU(const float* const weightsIn, const int dim) {
    float res = 0.0f;
    float sum = FLT_EPSILON;

    int blockSize = 1024;            // block dim 1D
    //int numBlock = 1024 * 1024;      // grid dim 1D
    int numBlock = (dim / blockSize);
    if (dim % blockSize != 0) {
        numBlock += 1;
    }

    long blocksBytes = numBlock * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* weightsOut, * d_weightsIn, * d_weightsOut;
    CHECK(cudaMalloc((void**)&d_weightsIn, arrayBytes));
    CHECK(cudaMemcpy(d_weightsIn, weightsIn, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_weightsOut, blocksBytes));
    CHECK(cudaMemset((void*)d_weightsOut, 0, blocksBytes));
    weightsOut = (float*)malloc(blocksBytes * sizeof(float));

    SumSquaredParReduce << <numBlock, blockSize >> > (d_weightsIn, d_weightsOut, dim);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // memcopy D2H
    CHECK(cudaMemcpy(weightsOut, d_weightsOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlock; i++) {
        sum += weightsOut[i];
    }

    res = 1.0f / sum;

    cudaFree(d_weightsOut);
    cudaFree(d_weightsIn);
    free(weightsOut);

    CHECK(cudaDeviceReset());

    return res;
}
```

#### Sum Squared on GPU

```c++
/*
 *  Block by block parallel implementation without divergence (interleaved schema)
 */
__global__ void SumSquaredParReduce(float* in, float* out, const ulong n) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    float* thisBlock = in + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            thisBlock[tid] += (thisBlock[tid + stride] * thisBlock[tid + stride]);

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = thisBlock[0];
}
```

```c++
static float NeffGPU(const float* const weightsIn, const int dim) {
    float res = 0.0f;
    float sum = FLT_EPSILON;

    int blockSize = 1024;            // block dim 1D
    //int numBlock = 1024 * 1024;      // grid dim 1D
    int numBlock = (dim / blockSize);
    if (dim % blockSize != 0) {
        numBlock += 1;
    }

    long blocksBytes = numBlock * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* weightsOut, * d_weightsIn, * d_weightsOut;
    CHECK(cudaMalloc((void**)&d_weightsIn, arrayBytes));
    CHECK(cudaMemcpy(d_weightsIn, weightsIn, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_weightsOut, blocksBytes));
    CHECK(cudaMemset((void*)d_weightsOut, 0, blocksBytes));
    weightsOut = (float*)malloc(blocksBytes * sizeof(float));

    SumSquaredParReduce << <numBlock, blockSize >> > (d_weightsIn, d_weightsOut, dim);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // memcopy D2H
    CHECK(cudaMemcpy(weightsOut, d_weightsOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlock; i++) {
        sum += weightsOut[i];
    }

    res = 1.0f / sum;

    cudaFree(d_weightsOut);
    cudaFree(d_weightsIn);
    free(weightsOut);

    CHECK(cudaDeviceReset());

    return res;
}
```

# Comparison

 - particleFilterC -
number of particles: 400000


 Total execution time:    5.2010 sec

device 0: NVIDIA GeForce GTX 1070
61 - particleFilter GPU -
number of particles: 400000


 Total execution EVENT time:    0.0612 sec



 Total execution time:    0.9390 sec