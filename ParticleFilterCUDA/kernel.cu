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
#include <curand_kernel.h>
#include <cuda.h>

#include <stdio.h>
#include <cmath>
#include <string>
#include <time.h>

#include "common.h"
#include "Particle.h"
#include "Float2.h"

#include "MyDefs.h"

int THREADS_PER_BLOCK = BLOCKSIZE;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scanLargeEvenDeviceArray(float* d_out, float* d_in, int length, bool bcao);
void scanLargeDeviceArray(float* d_out, float* d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(float* d_out, float* d_in, int length, bool bcao);

/*
 *  Block by block parallel implementation without divergence (interleaved schema)
 */
__global__ void SumParReduce(float* in, float* out, ulong n) {

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
        if (stride == blockDim.x / 2) { // first time, multiply since we touch every single elements
            thisBlock[tid] = (thisBlock[tid] * thisBlock[tid]) * (thisBlock[tid + stride] * thisBlock[tid + stride]);
        }
        else if (tid < stride) {
            thisBlock[tid] += thisBlock[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = thisBlock[0];
}

__global__ void GenerateParticles(Particles* D_in, Particles* C_out, curandState* states, float2 x_range, float2 y_range, float2 heading_range) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
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

static float SumArrayGPU(const float* const arrIn, const int dim, float* const sumOut) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    uint numBlocks = (dim + BLOCKSIZE - 1) / BLOCKSIZE;

    long blocksBytes = numBlocks * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* arrOut, * d_arrIn, * d_arrOut;
    CHECK(cudaMalloc((void**)&d_arrIn, arrayBytes));
    CHECK(cudaMemcpy(d_arrIn, arrIn, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_arrOut, blocksBytes));
    CHECK(cudaMemset((void*)d_arrOut, 0, blocksBytes));
    arrOut = (float*)malloc(blocksBytes * sizeof(float));

    cudaEventRecord(start);

    SumParReduce << <numBlocks, BLOCKSIZE >> > (d_arrIn, d_arrOut, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iterMilliseconds = 0;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // memcopy D2H
    CHECK(cudaMemcpy(arrOut, d_arrOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlocks; i++) {
        *sumOut += arrOut[i];
    }

    cudaFree(d_arrOut);
    cudaFree(d_arrIn);
    free(arrOut);

    //CHECK(cudaDeviceReset());

    return seconds;
}

__global__ void SumArrayWeightsGPUKernel(float* weights, float* posX, float* posY, float* outWeights, float* outPosY, float* outPosX, ulong n) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    float* thisBlockWeights = weights + blockIdx.x * blockDim.x;
    float* thisBlockPosX = posX + blockIdx.x * blockDim.x;
    float* thisBlockPosY = posY + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (stride == blockDim.x / 2) { // first time, multiply since we touch every single elements
            thisBlockPosX[tid] = (thisBlockPosX[tid] * thisBlockWeights[tid]) + (thisBlockPosX[tid + stride] * thisBlockWeights[tid + stride]);
            thisBlockPosY[tid] = (thisBlockPosY[tid] * thisBlockWeights[tid]) + (thisBlockPosY[tid + stride] * thisBlockWeights[tid + stride]);
            thisBlockWeights[tid] += thisBlockWeights[tid + stride];
        }
        else if (tid < stride) {
            thisBlockPosX[tid] += thisBlockPosX[tid + stride];
            thisBlockPosY[tid] += thisBlockPosY[tid + stride];
            thisBlockWeights[tid] += thisBlockWeights[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        outPosX[blockIdx.x] = thisBlockPosX[0];
        outPosY[blockIdx.x] = thisBlockPosY[0];
        outWeights[blockIdx.x] = thisBlockWeights[0];
    }

}
__global__ void SumArrayWeightsSqrdSubGPUKernel(float* weights, float* posX, float* posY, float* outWeights, float* outPosY,
    float* outPosX, ulong n, const float meanX, const float meanY) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    float* thisBlockWeights = weights + blockIdx.x * blockDim.x;
    float* thisBlockPosX = posX + blockIdx.x * blockDim.x;
    float* thisBlockPosY = posY + blockIdx.x * blockDim.x;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (stride == blockDim.x / 2) { // first time, multiply since we touch every single elements
            thisBlockPosX[tid] = (thisBlockPosX[tid] - meanX) * (thisBlockPosX[tid] - meanX);
            thisBlockPosX[tid] = (thisBlockPosX[tid] * thisBlockWeights[tid]) + (thisBlockPosX[tid + stride] * thisBlockWeights[tid + stride]);

            thisBlockPosY[tid] = (thisBlockPosY[tid] - meanY) * (thisBlockPosY[tid] - meanY);
            thisBlockPosY[tid] = (thisBlockPosY[tid] * thisBlockWeights[tid]) + (thisBlockPosY[tid + stride] * thisBlockWeights[tid + stride]);

            thisBlockWeights[tid] += thisBlockWeights[tid + stride];
        }
        else if (tid < stride) {
            thisBlockPosX[tid] += thisBlockPosX[tid + stride];
            thisBlockPosY[tid] += thisBlockPosY[tid + stride];
            thisBlockWeights[tid] += thisBlockWeights[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        outPosX[blockIdx.x] = thisBlockPosX[0];
        outPosY[blockIdx.x] = thisBlockPosY[0];
        outWeights[blockIdx.x] = thisBlockWeights[0];
    }

}
float SumArrayWeightsGPU(const float* const posX, const float* const posY, float* weights,
    const int dim, float3* const sumsOut) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    float sumX = FLT_EPSILON;
    float sumY = FLT_EPSILON;
    float sumWeights = FLT_EPSILON;

    uint numBlocks = (dim + BLOCKSIZE - 1) / BLOCKSIZE;

    long blocksBytes = numBlocks * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* d_weights, * d_weightsOut, * weightsOut;

    // To be removed
    float* a;
    CHECK(cudaMalloc((void**)&a, sizeof(float)));

    float* d_posX, * d_posXOut, * posXOut;
    CHECK(cudaMalloc((void**)&d_posX, arrayBytes));
    CHECK(cudaMemcpy(d_posX, posX, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_posXOut, blocksBytes));
    CHECK(cudaMemset((void*)d_posXOut, 0, blocksBytes));
    posXOut = (float*)malloc(blocksBytes);

    CHECK(cudaMalloc((void**)&d_weights, arrayBytes));
    CHECK(cudaMemcpy(d_weights, weights, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_weightsOut, blocksBytes));
    CHECK(cudaMemset((void*)d_weightsOut, 0, blocksBytes));
    weightsOut = (float*)malloc(blocksBytes);

    float* d_posY, * d_posYOut, * posYOut;
    CHECK(cudaMalloc((void**)&d_posY, arrayBytes));
    CHECK(cudaMemcpy(d_posY, posY, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_posYOut, blocksBytes));
    CHECK(cudaMemset((void*)d_posYOut, 0, blocksBytes));
    posYOut = (float*)malloc(blocksBytes);

    cudaEventRecord(start);

    SumArrayWeightsGPUKernel << <numBlocks, BLOCKSIZE >> > (d_weights, d_posX, d_posY, d_weightsOut, d_posXOut, d_posYOut, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    float iterMilliseconds = 0.0f;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    // memcopy D2H
    CHECK(cudaMemcpy(weightsOut, d_weightsOut, blocksBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(posXOut, d_posXOut, blocksBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(posYOut, d_posYOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlocks; i++) {
        sumX += posXOut[i];
        sumY += posYOut[i];
        sumWeights += weightsOut[i];
    }

    cudaFree(d_weights);
    cudaFree(d_weightsOut);
    free(weightsOut);

    cudaFree(d_posX);
    cudaFree(d_posXOut);
    free(posXOut);

    cudaFree(d_posY);
    cudaFree(d_posYOut);
    free(posYOut);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CHECK(cudaDeviceReset());

    sumsOut->x = sumX;
    sumsOut->y = sumY;
    sumsOut->z = sumWeights;

    return seconds;
}
static float SumArrayWeightsSqrdSubGPU(const float* const posX, const float* const posY,
    const float* const weights, const int dim, const float meanX, const float meanY, float3* const sumsOut) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    float sumX = FLT_EPSILON;
    float sumY = FLT_EPSILON;
    float sumWeights = FLT_EPSILON;

    uint numBlocks = (dim + BLOCKSIZE - 1) / BLOCKSIZE;

    long blocksBytes = numBlocks * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* d_weights, * d_weightsOut, * weightsOut;
    CHECK(cudaMalloc((void**)&d_weights, arrayBytes));
    CHECK(cudaMemcpy(d_weights, weights, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_weightsOut, blocksBytes));
    CHECK(cudaMemset((void*)d_weightsOut, 0, blocksBytes));
    weightsOut = (float*)malloc(blocksBytes * sizeof(float));

    float* d_posX, * d_posXOut, * posXOut;
    CHECK(cudaMalloc((void**)&d_posX, arrayBytes));
    CHECK(cudaMemcpy(d_posX, posX, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_posXOut, blocksBytes));
    CHECK(cudaMemset((void*)d_posXOut, 0, blocksBytes));
    posXOut = (float*)malloc(blocksBytes * sizeof(float));

    float* d_posY, * d_posYOut, * posYOut;
    CHECK(cudaMalloc((void**)&d_posY, arrayBytes));
    CHECK(cudaMemcpy(d_posY, posY, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_posYOut, blocksBytes));
    CHECK(cudaMemset((void*)d_posYOut, 0, blocksBytes));
    posYOut = (float*)malloc(blocksBytes * sizeof(float));

    cudaEventRecord(start);

    SumArrayWeightsSqrdSubGPUKernel << <numBlocks, BLOCKSIZE >> > (d_weights, d_posX, d_posY, d_weightsOut, d_posXOut, d_posYOut, dim, meanX, meanY);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iterMilliseconds = 0;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // memcopy D2H
    CHECK(cudaMemcpy(weightsOut, d_weightsOut, blocksBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(posXOut, d_posXOut, blocksBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(posYOut, d_posYOut, blocksBytes, cudaMemcpyDeviceToHost));

    // check result
    for (uint i = 0; i < numBlocks; i++) {
        sumX += posXOut[i];
        sumY += posYOut[i];
        sumWeights += weightsOut[i];
    }

    cudaFree(d_weights);
    cudaFree(d_weightsOut);
    free(weightsOut);

    cudaFree(d_posX);
    cudaFree(d_posXOut);
    free(posXOut);

    cudaFree(d_posY);
    cudaFree(d_posYOut);
    free(posYOut);

    //CHECK(cudaDeviceReset());

    sumsOut->x = sumX;
    sumsOut->y = sumY;
    sumsOut->z = sumWeights;

    return seconds;
}

// Norm Pdf
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
// https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
//  def normpdf(x, mu=0, sigma=1):
static float normpdf(const float x, const float mu = 0.0f, const float sigma = 1.0f) {
    //  u = float((x - mu) / abs(sigma))
    //  y = exp(-u * u / 2) / (sqrt(2 * pi) * abs(sigma))
    //  return y
    float u = (x - mu) / abs(sigma);
    float num = exp(-(u * u) / 2.0f);
    float y = num / (PI2SQRD * abs(sigma));
    return y;
}
__device__ float normPdfGPU(const float x, const float mu = 0.0f, const float sigma = 1.0f) {
    float u = (x - mu) / abs(sigma);
    float y = exp(-(u * u) / 2.0f) / (PI2SQRD * abs(sigma));
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
    float sqrdMagX = sqrdMagnitude(p->x, N);
    float sqrdMagY = sqrdMagnitude(p->y, N);
    float magnitude = sqrt(sqrdMagX + sqrdMagY);
    return magnitude;
}
__device__ float MagnitudeGPU(const float x, const float y) {
    float mag = sqrt((x * x) + (y * y));
    return mag;
}
__device__ float MagnitudeGPU(const Float2* const vec2) {
    return MagnitudeGPU(vec2->x, vec2->y);

}
__device__ float MagnitudeGPU(const Float2 vec2) {
    return MagnitudeGPU(vec2.x, vec2.y);
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
static float WeightedAverageGPU(const Floats2* const pos, float* weights, const int dim, Float2* const average) {
    float seconds = 0.0f;

    float3 sums;
    sums.x = 0.0f;
    sums.y = 0.0f;
    sums.z = 0.0f;

    seconds += SumArrayWeightsGPU(pos->x, pos->y, weights, dim, &sums);

    average->x = sums.x / sums.z;
    average->y = sums.y / sums.z;

    return seconds;
}
static float WeightedAverageSqrdSubGPU(const Floats2* const pos, const float* const weights,
    const int dim, float meanX, float meanY, Float2* const average) {

    ulong arrayBytes = dim * sizeof(float);

    float3 sums;
    sums.x = 0.0f;
    sums.y = 0.0f;
    sums.z = 0.0f;

    float seconds = SumArrayWeightsSqrdSubGPU(pos->x, pos->y, weights, dim, meanX, meanY, &sums);

    average->x = sums.x / sums.z;
    average->y = sums.y / sums.z;

    return seconds;
}

static float* CumSum(const float* const arr_in, const int dim) {
    float* cumSumArr = (float*)malloc(dim * sizeof(float));

    cumSumArr[0] = arr_in[0];

    for (int i = 1; i < dim; i++) {
        cumSumArr[i] += cumSumArr[i - 1] + arr_in[i];
    }

    return cumSumArr;
}

/*///////////////////////////////////*/
/*            kernels.cu             */
/*///////////////////////////////////*/
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5

// There were two BCAO optimisations in the paper - this one is fastest
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void prescan_arbitrary(float* output, float* input, int n, int powerOfTwo)
{
    extern __shared__ float temp[];// allocated on invocation
    int threadID = threadIdx.x;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


    if (threadID < n) {
        temp[ai + bankOffsetA] = input[ai];
        temp[bi + bankOffsetB] = input[bi];
    }
    else {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) {
        temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[ai] = temp[ai + bankOffsetA];
        output[bi] = temp[bi + bankOffsetB];
    }
}
__global__ void prescan_arbitrary_unoptimized(float* output, float* input, int n, int powerOfTwo) {
    extern __shared__ float temp[];// allocated on invocation
    int threadID = threadIdx.x;

    if (threadID < n) {
        temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
        temp[2 * threadID + 1] = input[2 * threadID + 1];
    }
    else {
        temp[2 * threadID] = 0;
        temp[2 * threadID + 1] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[2 * threadID] = temp[2 * threadID]; // write results to device memory
        output[2 * threadID + 1] = temp[2 * threadID + 1];
    }
}
__global__ void prescan_large(float* output, float* input, int n, float* sums) {
    extern __shared__ float temp[];

    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = input[blockOffset + ai];
    temp[bi + bankOffsetB] = input[blockOffset + bi];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[blockOffset + ai] = temp[ai + bankOffsetA];
    output[blockOffset + bi] = temp[bi + bankOffsetB];
}
__global__ void prescan_large_unoptimized(float* output, float* input, int n, float* sums) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;

    extern __shared__ float temp[];
    temp[2 * threadID] = input[blockOffset + (2 * threadID)];
    temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1];
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[blockOffset + (2 * threadID)] = temp[2 * threadID];
    output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}
__global__ void add(float* output, int length, float* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}
__global__ void add(float* output, int length, float* n1, float* n2) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}
bool isPowerOfTwo(int x) {
    return x && !(x & (x - 1));
}
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

void scanSmallDeviceArray(float* d_out, float* d_in, int length, bool bcao) {
    int powerOfTwo = nextPowerOfTwo(length);

    if (bcao) {
        prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(float) >> > (d_out, d_in, length, powerOfTwo);
    }
    else {
        prescan_arbitrary_unoptimized << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(float) >> > (d_out, d_in, length, powerOfTwo);
    }
}
void scanLargeDeviceArray(float* d_out, float* d_in, int length, bool bcao) {
    int remainder = length % (ELEMENTS_PER_BLOCK);
    if (remainder == 0) {
        scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
    }
    else {
        // perform a large scan on a compatible multiple of elements
        int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

        // scan the remaining elements and add the (inclusive) last element of the large scan to this
        float* startOfOutputArray = &(d_out[lengthMultiple]);
        scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

        add << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
    }
}
void scanLargeEvenDeviceArray(float* d_out, float* d_in, int length, bool bcao) {
    const int blocks = length / ELEMENTS_PER_BLOCK;
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(float);

    float* d_sums, * d_incr;
    cudaMalloc((void**)&d_sums, blocks * sizeof(float));
    cudaMalloc((void**)&d_incr, blocks * sizeof(float));

    if (bcao) {
        prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
    }
    else {
        prescan_large_unoptimized << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
    }

    const int sumsArrThreadsNeeded = (blocks + 1) / 2;
    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
        // perform a large scan on the sums arr
        scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
    }
    else {
        // only need one block to scan sums arr so can use small scan
        scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
    }

    add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

    cudaFree(d_sums);
    cudaFree(d_incr);
}
float scan(float* output, float* input, int length, bool bcao) {
    float* d_out, * d_in;
    const int arraySize = length * sizeof(float);

    cudaMalloc((void**)&d_out, arraySize);
    cudaMalloc((void**)&d_in, arraySize);
    cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

    // start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (length > ELEMENTS_PER_BLOCK) {
        scanLargeDeviceArray(d_out, d_in, length, bcao);
    }
    else {
        scanSmallDeviceArray(d_out, d_in, length, bcao);
    }

    // end timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}
//static float CumSumGPU(const float* const arr_in, const int dim, float* const cumSumArr) {
//    float seconds = 0.0f;
//
//    for (int i = 0; i < dim; i++) {
//        seconds += SumArrayGPU(arr_in, i + 1, &cumSumArr[i]);  // + 1 for the size of the subArr
//    }
//
//    return seconds;
//}
//static float PrefixSumGPU(float*  arr_in, const int dim, float* const cumSumArr) {
static float CumSumGPU(float* arr_in, const int dim, float* const cumSumArr) {
    float seconds = 0.0f;

    // full scan with BCAO
    seconds = scan(cumSumArr, arr_in, N, true);

    return seconds;
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
__global__ void SearchSortedGPU(const float* const sortedArry, const int dim, const float element, int* const indexOut, const ulong resPosition) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dim)
        return;

    if (element <= sortedArry[idx] && element > sortedArry[idx + 1]) {
        indexOut[resPosition] = sortedArry[idx];
    }
}

__global__ void SimpleResampleGPUKernel(
    const Particles* const particles, Particles* const particlesOut,
    const float* const sortedArry, const int dim, curandState* states, const float equalWeight) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dim)
        return;

    curand_init(idx, 0, 0, &states[idx]);
    float element = curand_uniform(&states[idx]);

    //dim3 grid(gridDim.x);
    //dim3 block(blockDim.x);

    // I would like to use dynamic parallelism
    // but the sync device side with child is deprecated
    // SearchSortedGPU
    // Starts with indexFound = dim -1
    // So search sorted will not branch that much
    //SearchSortedGPU << <grid, block >> > (sortedArry, dim - 1, element, indexOut, idx);

    // Wait for the child kernel to be finished
    //cudaDeviceSynchronize();  // deprecated, nice

    int index = 0;

    for (int i = 0; i < dim; i++) {
        if (element <= sortedArry[idx] && element > sortedArry[idx + 1]) {
            index = i;
        }
    }

    // When all the indexes are found
    __syncthreads();

    // resample according to indexes
    particlesOut->x[idx] = particles->x[index];
    particlesOut->y[idx] = particles->y[index];
    particlesOut->heading[idx] = particles->heading[index];
    particlesOut->weights[idx] = equalWeight;
}


__global__ void DivisionKernel(float* const dividend, const uint dividendDim, const float divisor) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dividendDim)
        return;

    dividend[idx] /= divisor;
}
static float ParallelDivisionGPU(float* const dividend, const uint dividendDim, const float divisor) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    uint numBlocks = (dividendDim + BLOCKSIZE - 1) / BLOCKSIZE;

    long arrayBytes = dividendDim * sizeof(float);

    float* d_dividend;
    CHECK(cudaMalloc((void**)&d_dividend, arrayBytes));
    CHECK(cudaMemcpy(d_dividend, dividend, arrayBytes, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    DivisionKernel << <numBlocks, BLOCKSIZE >> > (d_dividend, dividendDim, divisor);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    float iterMilliseconds = 0;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // memcopy D2H
    CHECK(cudaMemcpy(dividend, d_dividend, arrayBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_dividend);

    //CHECK(cudaDeviceReset());

    return seconds;
}


static void PredictCPU(Particles* const p, const Float2* const u, const Float2* const std, const float dt) {
    //""" move according to control input u (heading change, velocity)
    //    with noise Q(std heading change, std velocity)`"""
    srand((unsigned int)time(NULL));   // Initialization, should only be called once.

    for (int i = 0; i < N; i++) {
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
__global__ void PredictGPUKernel(Particles* const D_in, curandState* const states, const Float2 u, const Float2 std, const float dt) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    curand_init(idx, 0, 0, &states[idx]);

    //# update heading
    float heading = D_in->heading[idx];
    heading += u.x + (curand_normal(&states[idx]) * std.x);
    heading = std::fmod(heading, 2 * PI);

    //# move in the (noisy) commanded direction
    float dist = (u.y * dt) + (curand_uniform(&states[idx]) * std.y);
    float pos_x = D_in->x[idx];
    float pos_y = D_in->y[idx];
    pos_x += std::cos(heading) * dist;
    pos_y += std::sin(heading) * dist;

    D_in->x[idx] = pos_x;
    D_in->y[idx] = pos_y;
    D_in->heading[idx] = heading;
}
static float PredictGPU(Particles** particles, const Float2* const u, const Float2* const std, const float dt) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ulong particlesBytes = sizeof(Particles);

    Particles* d_particlesIn;
    CHECK(cudaMalloc((void**)&d_particlesIn, particlesBytes));
    CHECK(cudaMemcpy(d_particlesIn, *particles, particlesBytes, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_particlesIn, pp, particlesBytes, cudaMemcpyHostToDevice));
    CHECK(cudaGetLastError());

    uint numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    curandState* devStates;
    cudaMalloc((void**)&devStates, N * sizeof(curandState));
    CHECK(cudaGetLastError());

    cudaEventRecord(start);

    //__global__ void PredictGPUKernel(Particles * D_in, Particles * C_out, curandState * states, float* u, float* std, float dt) {
    PredictGPUKernel << <numBlocks, BLOCKSIZE >> > (d_particlesIn, devStates, (*u), (*std), dt);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CHECK(cudaMemcpy(*particles, d_particlesIn, particlesBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_particlesIn));
    CHECK(cudaFree(devStates));

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float sec = milliseconds / 1000.0f;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CHECK(cudaDeviceReset());

    return sec;
}

//def update(particles, weights, z, R, landmarks) :
//  for i, landmark in enumerate(landmarks) :
//      distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
//      weights *= scipy.stats.norm(distance, R).pdf(z[i])
//
//  weights += 1.e-300      # avoid round - off to zero
//  weights /= sum(weights) # normalize
static void UpdateCPU(Particles* const p, const float const* z, const float R, const Floats2 const* landmarks, const int numberOfLandmarks) {
    int size = N;

    Floats2 distance;
    distance.x = (float*)malloc(size * sizeof(float));
    distance.y = (float*)malloc(size * sizeof(float));
    float* normPdfs = (float*)malloc(size * sizeof(float));
    float* distanceMagnitudes = (float*)calloc(size, sizeof(float));

    for (int i = 0; i < numberOfLandmarks; i++) {
        //  distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
        memcpy(distance.x, p->x, size * sizeof(float));
        memcpy(distance.y, p->y, size * sizeof(float));

        for (int j = 0; j < size; j++) {    // particles[:, 0 : 2] - landmark
            distance.x[j] -= landmarks->x[i];
            distance.y[j] -= landmarks->y[i];
        }

        for (int j = 0; j < size; j++) {    // np.linalg.norm
            distanceMagnitudes[j] = Magnitude(distance.x[j], distance.y[j]);
        }

        //  weights *= scipy.stats.norm(distance, R).pdf(z[i])
        for (int j = 0; j < size; j++) { // scipy.stats.norm(distance, R).pdf(z[i])
            normPdfs[j] = normpdf(z[i], distanceMagnitudes[j], R);
        }
        for (int j = 0; j < size; j++) { // weights *=  // element wise multiplication
            p->weights[j] *= normPdfs[j];
        }
    }

    free(distanceMagnitudes);
    free(normPdfs);
    free(distance.x);
    free(distance.y);

    float sum = FLT_EPSILON;  // avoid round - off to zero
    for (int i = 0; i < size; i++) {
        sum += p->weights[i];
    }

    for (int i = 0; i < size; i++) {
        p->weights[i] /= sum; // normalize
    }
}
__global__ void UpdateGPUKernel(Particles* const p, float* distanceX, float* distanceY, const float const* z, const float R, const Float2 landmark) {
    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= N)
        return;

    distanceX[idx] -= landmark.x;
    distanceY[idx] -= landmark.y;

    float distanceMagnitude = MagnitudeGPU(distanceX[idx], distanceY[idx]);

    float normPdf = normPdfGPU(z[idx], distanceMagnitude, R);
    //printf("\nidx: %d, normPdf: %f", idx, normPdf);

    p->weights[idx] *= normPdf;
}
static float UpdateGPU(Particles** const p, const float const* z, const float R, const Floats2 const* landmarks, const int numberOfLandmarks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    ulong particlesBytes = sizeof(Particles);

    Particles* d_particles;
    CHECK(cudaMalloc((void**)&d_particles, particlesBytes));
    CHECK(cudaMemcpy(d_particles, *p, particlesBytes, cudaMemcpyHostToDevice));
    CHECK(cudaGetLastError());

    ulong arrayBytes = N * sizeof(float);
    ulong landmarkBytes = numberOfLandmarks * sizeof(float);

    float* d_distanceX, * d_distanceY;
    CHECK(cudaMalloc((void**)&d_distanceX, arrayBytes));
    CHECK(cudaMalloc((void**)&d_distanceY, arrayBytes));
    CHECK(cudaGetLastError());

    float* d_z;
    CHECK(cudaMalloc((void**)&d_z, landmarkBytes));
    CHECK(cudaMemcpy(d_z, z, landmarkBytes, cudaMemcpyHostToDevice));
    CHECK(cudaGetLastError());

    uint numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    for (int i = 0; i < numberOfLandmarks; i++) {
        Float2 landmark;
        landmark.x = landmarks->x[i];
        landmark.y = landmarks->y[i];

        CHECK(cudaMemcpy(d_distanceX, (*p)->x, arrayBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_distanceY, (*p)->y, arrayBytes, cudaMemcpyHostToDevice));

        cudaEventRecord(start);

        UpdateGPUKernel << <numBlocks, BLOCKSIZE >> > (d_particles, d_distanceX, d_distanceY, d_z, R, landmark);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        //CHECK(cudaDeviceSynchronize());

        // Update weights
        CHECK(cudaMemcpy(*p, d_particles, particlesBytes, cudaMemcpyDeviceToHost));
        CHECK(cudaGetLastError());

        float iterMilliseconds = 0;
        cudaEventElapsedTime(&iterMilliseconds, start, stop);
        float iterSeconds = iterMilliseconds / 1000.0;
        seconds += iterSeconds;
    }

    CHECK(cudaFree(d_particles));
    CHECK(cudaFree(d_distanceX));
    CHECK(cudaFree(d_distanceY));
    CHECK(cudaFree(d_z));

    // Normalization
    float sum = FLT_EPSILON;  // avoid round - off to zero

    //cudaEventRecord(start);

    seconds += SumArrayGPU((*p)->weights, N, &sum);

    seconds += ParallelDivisionGPU((*p)->weights, N, sum);

    //CHECK(cudaDeviceReset());

    return seconds;
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
    pos.x = (float*)malloc(N * sizeof(float));
    memcpy(pos.x, p->x, N * sizeof(float));
    pos.y = (float*)malloc(N * sizeof(float));
    memcpy(pos.y, p->y, N * sizeof(float));

    // mean = np.average(pos, weights = weights, axis = 0)
    (*mean_out) = WeightedAverage(&pos, p->weights, N);
    // var = np.average((pos - mean) **2, weights = weights, axis = 0)
    for (int i = 0; i < N; i++) {
        pos.x[i] = (pos.x[i] - mean_out->x) * (pos.x[i] - mean_out->x);
        pos.y[i] = (pos.y[i] - mean_out->y) * (pos.y[i] - mean_out->y);
    }

    (*var_out) = WeightedAverage(&pos, p->weights, N);

    free(pos.x);
    free(pos.y);
}
static float EstimateGPU(Particles** p, Float2* const mean_out, Float2* const var_out) {
    float seconds = 0.0f;

    Floats2 pos;
    pos.x = (float*)malloc(N * sizeof(float));
    memcpy(pos.x, (*p)->x, N * sizeof(float));
    pos.y = (float*)malloc(N * sizeof(float));
    memcpy(pos.y, (*p)->y, N * sizeof(float));

    seconds += WeightedAverageGPU(&pos, (*p)->weights, N, mean_out);

    seconds += WeightedAverageSqrdSubGPU(&pos, (*p)->weights, N, mean_out->x, mean_out->y, var_out);

    free(pos.x);
    free(pos.y);

    return seconds;
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
static float NeffGPU(const float* const weightsIn, const int dim, float* const neffOut) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    float res = 0.0f;
    float sum = FLT_EPSILON;

    uint numBlock = (dim + BLOCKSIZE - 1) / BLOCKSIZE;


    long blocksBytes = numBlock * sizeof(float);
    long arrayBytes = dim * sizeof(float);

    float* weightsOut, * d_weightsIn, * d_weightsOut;
    CHECK(cudaMalloc((void**)&d_weightsIn, arrayBytes));
    CHECK(cudaMemcpy(d_weightsIn, weightsIn, arrayBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_weightsOut, blocksBytes));
    CHECK(cudaMemset((void*)d_weightsOut, 0, blocksBytes));
    weightsOut = (float*)malloc(blocksBytes * sizeof(float));

    cudaEventRecord(start);

    SumSquaredParReduce << <numBlock, BLOCKSIZE >> > (d_weightsIn, d_weightsOut, dim);

    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iterMilliseconds = 0;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //CHECK(cudaDeviceSynchronize());
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

    //CHECK(cudaDeviceReset());
    (*neffOut) = res;

    return seconds;
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
    int dim = N;

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
static float SimpleResampleGPU(Particles** const p) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float seconds = 0.0f;

    int dim = N;

    float* cumSum_arr = (float*)malloc(N * sizeof(float));

    seconds += CumSumGPU((*p)->weights, dim, cumSum_arr);
    printf("\n\t\t CumSumGPU: %9.4f sec", seconds);

    uint numBlocks = (dim + BLOCKSIZE - 1) / BLOCKSIZE;

    curandState* devStates;
    cudaMalloc((void**)&devStates, N * sizeof(curandState));

    float equalWeight = 1.0f / dim;

    float* d_cumSum_arr;
    CHECK(cudaMalloc((void**)&d_cumSum_arr, N * sizeof(float)));
    CHECK(cudaMemcpy(d_cumSum_arr, cumSum_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    Particles* d_particles, * d_particlesOut;
    ulong particlesBytes = sizeof(Particles);
    CHECK(cudaMalloc((void**)&d_particles, particlesBytes));
    CHECK(cudaMemcpy(d_particles, *p, particlesBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_particlesOut, particlesBytes));
    // it does not need to be initialized since we will overwrite the value of every element

    cudaEventRecord(start);

    SimpleResampleGPUKernel << <numBlocks, BLOCKSIZE >> > (
        d_particles, d_particlesOut,
        d_cumSum_arr, dim,
        devStates, equalWeight);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    float iterMilliseconds = 0;
    cudaEventElapsedTime(&iterMilliseconds, start, stop);
    float iterSeconds = iterMilliseconds / 1000.0f;
    seconds += iterSeconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK(cudaMemcpy(*p, d_particlesOut, sizeof(Particles), cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    cudaFree(d_cumSum_arr);
    cudaFree(d_particles);
    cudaFree(d_particlesOut);

    //CHECK(cudaDeviceReset());

    return seconds;
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

__global__ void Norm_BlockUnroll8(float* in, float* out, const float add, const ulong n) {
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


void particleFilterGPU(Particles** p, const int iterations, const float sensorStdError) {
    clock_t startClock, stopClock;
    double timer;
    startClock = clock();

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%d", prop.major * 10 + prop.minor);

    printf(" - particleFilter GPU - \n");

    unsigned int dim = N;

    printf("number of particles: %d \n", dim);

    // Start
    float duration = 0.0f;

    Float2 u;
    Float2 std;
    float dt = 0.1f;

    Floats2 landmarks;
    int numberOfLandmarks = 4;
    landmarks.x = (float*)malloc(numberOfLandmarks * sizeof(float));
    landmarks.y = (float*)malloc(numberOfLandmarks * sizeof(float));

    landmarks.x[0] = 0.0f;
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
        printf("\n\n Iteration: %d", i);

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

        float tempTime = PredictGPU(p, &u, &std, dt);
        duration += tempTime;
        printf("\n\t PredictGPU time: %9.4f sec", tempTime);

        tempTime = UpdateGPU(p, zs, sensorStdError, &landmarks, numberOfLandmarks);
        duration += tempTime;
        printf("\n\t UpdateGPU: %9.4f sec", tempTime);

        float neff;
        tempTime = NeffGPU((*p)->weights, N, &neff);
        duration += tempTime;
        printf("\n\t\t NeffGPU: %9.4f sec", tempTime);

        //if (neff < N / 2.0f) {
        if (1) { // Only to test Re-sample
            // re-sample
            tempTime = SimpleResampleGPU(p);
            duration += tempTime;
            printf("\n\t SimpleResampleGPU: %9.4f sec", tempTime);
        }

        Float2 var;
        Float2 mean;

        tempTime = EstimateGPU(p, &mean, &var);
        duration += tempTime;
        printf("\n\t EstimateGPU: %9.4f sec", tempTime);

        xs[i] = mean;

        free(zs);
    }

    // End
    printf("\n\n Total execution EVENT time: %9.4f sec \n\n", duration);

    stopClock = clock();
    timer = ((double)(stopClock - startClock)) / (double)CLOCKS_PER_SEC;
    printf("\n\n Total execution time: %9.4f sec \n\n", timer);

    // write to file the results
    FILE* fptr;

    // Open a file in append mode
    fptr = fopen("C:\\Users\\andre\\ANDREA\\UNI\\GPU-COMPUTING\\ParticleFilterCUDA\\particleFilterGPU.txt", "a");

    // Append some text to the file
    fprintf(fptr, "Total execution EVENT time: %9.4f sec", duration);

    // Close the file
    fclose(fptr);

    free(landmarks.x);
    free(landmarks.y);
    free(xs);

    CHECK(cudaDeviceReset());
}

void particleFilterCPU(Particles* const p, const int iterations, const float sensorStdError) {
    clock_t start, stop;
    double timer;

    printf(" - particleFilter CPU - \n");

    unsigned int dim = N;

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
        printf("Iteration: %d\n", i);

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
        printf("\tPredictCPU\n");

        UpdateCPU(p, zs, sensorStdError, &landmarks, numberOfLandmarks);
        printf("\tUpdateCPU\n");

        float neff = Neff((p)->weights, N);
        //if (neff < N / 2.0f) {
        if (1) { // Only to test Resample

            // resample
            SimpleResample(p);
            printf("\tSimpleResample\n");
        }

        Float2 var;
        Float2 mean;

        EstimateCPU(p, &mean, &var);
        printf("\tEstimateCPU\n");

        xs[i] = mean;

        free(zs);
    }

    // End

    stop = clock();
    timer = ((double)(stop - start)) / (double)CLOCKS_PER_SEC;
    printf("\n\n Total CPU execution time: %9.4f sec \n\n", timer);

    // write to file the results
    FILE* fptr;

    // Open a file in append mode
    fptr = fopen("C:\\Users\\andre\\ANDREA\\UNI\\GPU-COMPUTING\\ParticleFilterCUDA\\particleFilterCPU.txt", "a");

    // Append some text to the file
    fprintf(fptr, "Total CPU execution time: %9.4f sec", timer);

    // Close the file
    fclose(fptr);

    free(landmarks.x);
    free(landmarks.y);
    free(xs);
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

    Particles* p = nullptr;

    CreateAndRandomInitialize_Particles(&p, N, &xRange, &yRange, &headingRange);

    //particleFilterCPU(p, ITERAIONS, 0.1f);

    particleFilterGPU(&p, ITERAIONS, 0.1f);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK(cudaDeviceReset());

    return 0;
}
