#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

// Not recommended by NVIDIA for unpredictable side effects but it works
//for __syncthreads() and CHECK
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include <stdio.h>
#include <cmath>
#include <string>
#include <time.h>

#include "common.h"
#include "Particle.h"
#include "Float2.h"

#define PI 3.141592f
#define PI2 2.0f * 3.141592f

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)

#define DIM 10'000

#define BLOCKSIZE 1024  // block dim 1D
#define NUMBLOCKS 1024  // grid dim 1D 
#define N (NUMBLOCKS * BLOCKSIZE)

#define MinX 0.0f
#define MaxX 1000.0f

#define MinY 0.0f
#define MaxY 1000.0f

#define MinHeading 0.0f
#define MaxHeading 3.0f

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*
 *  Block by block parallel implementation without divergence (interleaved schema)
 */
__global__ void SumParReduce(int* in, int* out, ulong n) {

    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    int* thisBlock = in + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            thisBlock[tid] += thisBlock[tid + stride];

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = thisBlock[0];
}

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

__global__ void GenerateParticles(Particles* D_in, Particles* C_out, curandState* states, float2 x_range, float2 y_range, float2 heading_range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(tid, 0, 0, &states[tid]);
    float pos_x = x_range.x + ((x_range.y - x_range.x) * curand_uniform(&states[tid])); // Lerp
    float pos_y = y_range.x + ((y_range.y - y_range.x) * curand_uniform(&states[tid]));
    float heading = heading_range.x + ((heading_range.y - heading_range.x) * curand_uniform(&states[tid]));
    heading = std::fmod(heading, 2 * PI);

    C_out->x[tid] = pos_x;
    C_out->y[tid] = pos_y;
    C_out->heading[tid] = heading;
}

//# def predict(particles, u, std, dt=1.):
//# """ move according to control input u (heading change, velocity)
//# with noise Q (std heading change, std velocity)`"""
__global__ void PredictGPUKernel(Particles* D_in, Particles* C_out, curandState* states, float* u, float* std, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dt = 1.0f;

    curand_init(tid, 0, 0, &states[tid]);

    //# update heading
    float heading = D_in->heading[tid];
    heading += u[0] + (curand_normal(&states[tid]) * std[1]);
    heading = std::fmod(heading, 2 * PI);

    //# move in the (noisy) commanded direction
    float dist = (u[1] * dt) + (curand_uniform(&states[tid]) * std[1]);
    float pos_x = D_in->x[tid];
    float pos_y = D_in->y[tid];
    pos_x += std::cos(heading) * dist;
    pos_y += std::sin(heading) * dist;

    C_out->x[tid] = pos_x;
    C_out->y[tid] = pos_y;
    C_out->heading[tid] = heading;
}

static float SumArrayGPU(const float* const arrIn, const int dim) {
    float sum = 0.0f;

    int blockSize = 1024;            // block dim 1D
    //int numBlock = 1024 * 1024;      // grid dim 1D
    int numBlock = (dim / blockSize);
    if (dim % blockSize != 0) {
        numBlock += 1;
    }

    long blocksBytes = numBlock * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* arrOut, * d_arrIn, * d_arrOut;
    CHECK(cudaMalloc((void**)&d_arrIn, arrayBytes));
    CHECK(cudaMemcpy(d_arrIn, arrIn, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_arrOut, blocksBytes));
    CHECK(cudaMemset((void*)d_arrOut, 0, blocksBytes));
    arrOut = (float*)malloc(blocksBytes * sizeof(float));

    SumParReduce << <numBlock, blockSize >> > (d_arrIn, d_arrOut, dim);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // memcopy D2H
    CHECK(cudaMemcpy(arrOut, d_arrOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlock; i++) {
        sum += arrOut[i];
    }

    cudaFree(d_arrOut);
    cudaFree(d_arrIn);
    free(arrOut);

    CHECK(cudaDeviceReset());

    return sum;
}

//  def normpdf(x, mu=0, sigma=1):
static float normpdf(const float x, const float mu = 0.0f, const float sigma = 1.0f) {
    //  u = float((x - mu) / abs(sigma))
    //  y = exp(-u * u / 2) / (sqrt(2 * pi) * abs(sigma))
    //  return y
    float u = (x - mu) / abs(sigma);
    float y = exp(-u * u / 2) / (sqrt(PI2) * abs(sigma));
    return y;
}

float sqrdMagnitude(const float* const X, const int dim) {
    float sqrdMag = FLT_EPSILON;
    for (int i = 0; i < dim; i++) {
        sqrdMag += X[i] * X[i];
    }
    return sqrdMag;
}

static float magnitudeXY(const Particles* const p) {
    float sqrdMagX = sqrdMagnitude(p->x, p->size);
    float sqrdMagY = sqrdMagnitude(p->y, p->size);
    float magnitude = sqrt(sqrdMagX + sqrdMagY);
    return magnitude;
}

static Float2 WeightedAverage(const Floats2* const pos, const float* const weights, const int dim) {
    Float2 avg;
    avg.x = 0.0f;
    avg.y = 0.0f;

    // numpy implementation: avg = sum(a * weights) / sum(weights)

    float sumWeights = FLT_EPSILON; // avoid div by 0
    Float2 sumPos;
    sumPos.x = 0.0f;
    sumPos.y = 0.0f;
    for (int i = 0; i < dim; i++) {
        sumWeights += weights[i];
        sumPos.x += pos->x[i] * weights[i];
        sumPos.y += pos->y[i] * weights[i];
    }

    avg.x = sumPos.x / sumWeights;
    avg.y = sumPos.y / sumWeights;

    return avg;
}

static float* CumSum(const float* const arr_in, const int dim) {
    float* cumSumArr = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j <= i; j++) {
            cumSumArr[i] += arr_in[j];
        }
    }

    return cumSumArr;
}

static float* CumSumGPU(const float* const arr_in, const int dim) {
    float* cumSumArr = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        cumSumArr[i] = SumArrayGPU(arr_in, i + 1);  // + 1 for the size of the subArr
    }

    return cumSumArr;
}

// find the index, in the sorted array (ascending order), to insert the element preserving the order
static int SearchSorted(const float* const sortedArry, const float element, const int dim) {
    int index = dim;

    for (int i = 0; i < dim - 1; i++) {
        if (element <= sortedArry[i] && element > sortedArry[i + 1]) {
            index = i;
        }
    }

    return index;
}

static void PredictCPU(Particles* const p, const Float2* const u, const Float2* const std, const float dt) {
    //""" move according to control input u (heading change, velocity)
    //    with noise Q(std heading change, std velocity)`"""
    srand((unsigned int)time(NULL));   // Initialization, should only be called once.

    for (int i = 0; i < p->size; i++) {
        float r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.

        // update heading
        p->heading[i] += u->x + (r * std->x);
        p->heading[i] = fmodf(p->heading[i], PI2);

        float dist = (u->y * dt) + (r * std->y);

        // move in the(noisy) commanded direction
        p->x[i] += cos(p->heading[i]) * dist;
        p->y[i] += sin(p->heading[i]) * dist;

        //PrintParticle(p, i);
    }
}

static void PredictGPU(Particles* const p, const Float2* const u, const Float2* const std, const float dt) {
    //""" move according to control input u (heading change, velocity)
    //    with noise Q(std heading change, std velocity)`"""
    srand((unsigned int)time(NULL));   // Initialization, should only be called once.

    for (int i = 0; i < p->size; i++) {
        float r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.

        // update heading
        p->heading[i] += u->x + (r * std->x);
        p->heading[i] = fmodf(p->heading[i], 2.0f * PI);

        float dist = (u->y * dt) + (r * std->y);

        // move in the(noisy) commanded direction
        p->x[i] += cos(p->heading[i]) * dist;
        p->y[i] += sin(p->heading[i]) * dist;

        //PrintParticle(p, i);
    }
}

//def update(particles, weights, z, R, landmarks) :
//  for i, landmark in enumerate(landmarks) :
//      distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
//      weights *= scipy.stats.norm(distance, R).pdf(z[i])
//
//  weights += 1.e-300      # avoid round - off to zero
//  weights /= sum(weights) # normalize
static void UpdateCPU(Particles* const p, const float const* z, const float R, const Floats2 const* landmarks, const int numberOfLandmarks) {
    int size = p->size;

    for (int i = 0; i < numberOfLandmarks; i++) {
        //  distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
        Floats2 distance;
        distance.x = (float*)malloc(size * sizeof(float));
        memcpy(distance.x, p->x, size * sizeof(float));
        distance.y = (float*)malloc(size * sizeof(float));
        memcpy(distance.y, p->y, size * sizeof(float));

        for (int j = 0; j < size; j++) {    // particles[:, 0 : 2] - landmark
            distance.x[j] -= landmarks->x[i];
            distance.y[j] -= landmarks->y[i];
        }
        float* distanceMagnitudes = (float*)calloc(size, sizeof(float));
        for (int j = 0; j < size; j++) {    // np.linalg.norm
            distanceMagnitudes[j] = Magnitude(distance.x[j], distance.y[j]);
        }

        //  weights *= scipy.stats.norm(distance, R).pdf(z[i])
        float* normPdfs = (float*)malloc(size * sizeof(float));
        for (int j = 0; j < size; j++) { // scipy.stats.norm(distance, R).pdf(z[i])
            normPdfs[j] *= normpdf(z[i], distanceMagnitudes[j], R);;
        }
        for (int j = 0; j < size; j++) { // weights *=  // element wise multiplication
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

static void UpdateGPU(Particles* const p, const float const* z, const float R, const Floats2 const* landmarks) {
    int size = p->size;

    for (int i = 0; i < size; i++) {

        //  distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
        Floats2 distance;
        distance.x = (float*)malloc(size * sizeof(float));
        memcpy(distance.x, p->x, size * sizeof(float));
        distance.y = (float*)malloc(size * sizeof(float));
        memcpy(distance.y, p->y, size * sizeof(float));

        for (int j = 0; j < size; j++) {    // particles[:, 0 : 2] - landmark
            distance.x[j] -= landmarks->x[i];
            distance.y[j] -= landmarks->y[i];
        }
        float* distanceMagnitudes = (float*)calloc(size, sizeof(float));
        for (int j = 0; j < size; j++) {    // np.linalg.norm
            distanceMagnitudes[j] = Magnitude(distance.x[j], distance.y[j]);
        }

        //  weights *= scipy.stats.norm(distance, R).pdf(z[i])
        float* normPdfs = (float*)malloc(size * sizeof(float));
        for (int j = 0; j < size; j++) { // scipy.stats.norm(distance, R).pdf(z[i])
            normPdfs[j] *= normpdf(z[i], distanceMagnitudes[j], R);;
        }
        for (int j = 0; j < size; j++) { // weights *=  // element wise multiplication
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

//def estimate(particles, weights) :
//    """returns mean and variance of the weighted particles"""
//
//    pos = particles[:, 0 : 2]
//    mean = np.average(pos, weights = weights, axis = 0)
//    var = np.average((pos - mean) * *2, weights = weights, axis = 0)
//    return mean, var
// returns mean and variance of the weighted particles
static void EstimateCPU(const Particles* const p, Float2* const mean_out, Float2* const var_out) {

    Floats2 pos;
    pos.x = (float*)malloc(p->size * sizeof(float));
    memcpy(pos.x, p->x, p->size * sizeof(float));
    pos.y = (float*)malloc(p->size * sizeof(float));
    memcpy(pos.y, p->y, p->size * sizeof(float));

    // mean = np.average(pos, weights = weights, axis = 0)
    (*mean_out) = WeightedAverage(&pos, p->weights, p->size);
    // var = np.average((pos - mean) **2, weights = weights, axis = 0)
    for (int i = 0; i < p->size; i++) {
        pos.x[i] = (pos.x[i] - mean_out->x) * (pos.x[i] - mean_out->x);
        pos.y[i] = (pos.y[i] - mean_out->y) * (pos.y[i] - mean_out->y);
    }

    (*var_out) = WeightedAverage(&pos, p->weights, p->size);

    free(pos.x);
    free(pos.y);
}

//def simple_resample(particles, weights) :
//    N = len(particles)
//    cumulative_sum = np.cumsum(weights)
//    cumulative_sum[-1] = 1. # avoid round - off error
//    indexes = np.searchsorted(cumulative_sum, random(N))  // Return random floats in the half-open interval [0.0, 1.0).
//
//    # resample according to indexes
//    particles[:] = particles[indexes]
//    weights.fill(1.0 / N)
static void SimpleResample(Particles* const p) {
    int dim = p->size;

    //    cumulative_sum = np.cumsum(weights)
    float* cumSum_arr = CumSum(p->weights, dim);

    // indexes = np.searchsorted(cumulative_sum, random(N))  // Return random floats in the half-open interval [0.0, 1.0).
    int* indexes = (int*)malloc(dim * sizeof(int));
    srand((unsigned int)time(NULL));   // Initialization, should only be called once.
    float r = 0.0f;
    for (int i = 0; i < dim; i++) {
        r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
        indexes[i] = SearchSorted(cumSum_arr, r, dim);
    }

    //  # resample according to indexes
    //  particles[:] = particles[indexes]
    //  weights.fill(1.0 / N)
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

// We don't resample at every epoch. 
// For example, if you received no new measurements you have not received any information from which the resample can benefit. 
// We can determine when to resample by using something called the *effective N*, 
// which approximately measures the number of particles which meaningfully contribute to the probability distribution.
//def neff(weights) :
//    return 1. / np.sum(np.square(weights))
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

/*
 *  Device function: block parallel reduction based on warp unrolling
 */
__device__ void blockWarpUnroll(float* thisBlock, int blockDim, uint tid) {
    // in-place reduction in global memory
    for (int stride = blockDim / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            thisBlock[tid] += thisBlock[tid + stride];

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile float* vmem = thisBlock;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
}

__global__ void Norm_BlockUnroll8(float* in, float* out, float add, ulong n) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    float* thisBlock = in + blockIdx.x * blockDim.x * 8;

    float a = 0.0f;
    float temp = 0.0f;

    // unrolling 8 blocks
    if (idx + 7 * blockDim.x < n) {
        for (int i = 0; i < 8; i++) {
            temp = in[idx + i * blockDim.x];
            temp += add;
            a += temp * temp;
        }
        in[idx] = a;
    }

    __syncthreads();

    // block parall. reduction based on warp unrolling 
    blockWarpUnroll(thisBlock, blockDim.x, tid);

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = thisBlock[0];
}

//# def update(particles, weights, z, R, landmarks):
__global__ void Update(float2 norm, Particles* particles, Particles* C_out, float* weights, float* z, float* landmarks, int numberOfLandmarks, float R) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (tid >= DIM)
        return;

    //  distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
    //  weights *= scipy.stats.norm(distance, R).pdf(z[i])

}


void particleFilterGPU(Particles* p) {

    Particles d_p;

    CHECK(cudaMalloc((void**)&d_p.x, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.y, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.y, p->y, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.weights, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.weights, p->weights, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.heading, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.heading, p->heading, DIM * sizeof(float), cudaMemcpyHostToDevice));

    // OUTPUT for Device
    Particles d_out;
    CHECK(cudaMalloc((void**)&d_out.x, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.y, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.y, p->y, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.weights, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.weights, p->weights, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.heading, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.heading, p->heading, DIM * sizeof(float), cudaMemcpyHostToDevice));


    cudaFree(d_p.x);
    cudaFree(d_p.y);
    cudaFree(d_p.heading);
    cudaFree(d_p.weights);
    cudaFree(d_out.x);
    cudaFree(d_out.y);
    cudaFree(d_out.heading);
    cudaFree(d_out.weights);
}

void particleFilterCPU(Particles* const p, const int iterations, const float sensorStdError) {
    clock_t start, stop;
    double timer;

    printf(" - particleFilterC - \n");

    unsigned int dim = p->size;

    printf("number of particles: %d \n", dim);

    start = clock();

    // Start

    Float2 u;
    Float2 std;
    float dt = 0.1f;

    Floats2 landmarks;
    int numberOfLandmarks = 4;
    landmarks.x = (float*)malloc(numberOfLandmarks * sizeof(float));
    landmarks.y = (float*)malloc(numberOfLandmarks * sizeof(float));

    landmarks.x[0] = -1.0f;
    landmarks.y[0] = 2.0f;

    landmarks.x[1] = 5.0f;
    landmarks.y[1] = 10.0f;

    landmarks.x[2] = 12.0f;
    landmarks.y[2] = 24.0f;

    landmarks.x[3] = 18.0f;
    landmarks.y[3] = 21.0f;

    Float2 robotPosition;
    robotPosition.x = 0.0f;
    robotPosition.y = 0.0f;

    Float2* xs;
    xs = (Float2*)malloc(iterations * sizeof(Float2));

    for (int i = 0; i < iterations; i++) {
        // Diagonal movement
        robotPosition.x += 1.0f;
        robotPosition.y += 1.0f;

        srand((unsigned int)time(NULL));   // Initialization, should only be called once.
        float r = 0.0f;
        float* zs = (float*)malloc(numberOfLandmarks * sizeof(float));
        for (int j = 0; j < numberOfLandmarks; j++) {
            Float2 landmark;
            landmark.x = landmarks.x[j];
            landmark.y = landmarks.y[j];
            Float2 distanceRobotLandmark = Minus(&landmark, &robotPosition);
            float magnitudeDistance = Magnitude(distanceRobotLandmark);
            r = ((float)rand() / (float)(RAND_MAX));      // rand Returns a pseudo-random integer between 0 and RAND_MAX.
            zs[j] = magnitudeDistance + (r * sensorStdError);
        }

        PredictCPU(p, &u, &std, dt);

        UpdateCPU(p, zs, sensorStdError, &landmarks, numberOfLandmarks);

        //# resample if too few effective particles
        //    if neff(weights) < N / 2:
        //indexes = systematic_resample(weights)
        //    resample_from_index(particles, weights, indexes)
        //    assert np.allclose(weights, 1 / N)
        //mu, var = estimate(particles, weights)
        //xs.append(mu)

        float neff = Neff(p->weights, p->size);
        if (neff < p->size / 2.0f) {
            // resample
            SimpleResample(p);
        }

        Float2 var;
        Float2 mean;

        EstimateCPU(p, &mean, &var);

        xs[i] = mean;

        free(zs);
    }

    // End

    stop = clock();
    timer = ((double)(stop - start)) / (double)CLOCKS_PER_SEC;
    printf("\n\n Total execution time: %9.4f sec \n\n", timer);

    free(landmarks.x);
    free(landmarks.y);
    free(xs);
}

void euclideanNorm(Particles* p, float2* norm, float2* landmark) {

    Particles d_p;

    CHECK(cudaMalloc((void**)&d_p.x, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.y, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.y, p->y, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.weights, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.weights, p->weights, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_p.heading, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.heading, p->heading, DIM * sizeof(float), cudaMemcpyHostToDevice));


    // OUTPUT for Device
    Particles d_out;
    CHECK(cudaMalloc((void**)&d_out.x, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.y, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.y, p->y, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.weights, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.weights, p->weights, DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_out.heading, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_out.heading, p->heading, DIM * sizeof(float), cudaMemcpyHostToDevice));


    Particles p_norm;

    Create_Particles(&p_norm, DIM);

    Norm_BlockUnroll8 << <NUMBLOCKS / 8, BLOCKSIZE >> > (d_p.x, d_out.x, -landmark->x, N);  // ERRROR at <<< can be simply ignored
    Norm_BlockUnroll8 << <NUMBLOCKS / 8, BLOCKSIZE >> > (d_p.y, d_out.y, -landmark->y, N);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(p_norm.x, d_out.x, DIM * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(p_norm.y, d_out.y, DIM * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum on CPU the last elements
    for (uint i = 0; i < NUMBLOCKS / 8; i++) {
        norm->x += p_norm.x[i];
        norm->y += p_norm.y[i];
    }

    cudaFree(d_p.x);
    cudaFree(d_p.y);
    cudaFree(d_p.heading);
    cudaFree(d_p.weights);
    cudaFree(d_out.x);
    cudaFree(d_out.y);
    cudaFree(d_out.heading);
    cudaFree(d_out.weights);
    free(p_norm.x);
    free(p_norm.y);
    free(p_norm.heading);
    free(p_norm.weights);
}

int main()
{
    Float2 xRange;
    xRange.x = MinX;
    xRange.y = MaxX;

    Float2 yRange;
    yRange.x = MinY;
    yRange.y = MaxY;

    Float2 headingRange;
    headingRange.x = MinHeading;
    headingRange.y = MaxHeading;

    Particles p;

    CreateAndRandomInitialize_Particles(&p, DIM, &xRange, &yRange, &headingRange);

    //particleFilterGPU(&p);

    particleFilterCPU(&p, 18, 0.1f);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK(cudaDeviceReset());

    return 0;
}


//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    //// Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}

