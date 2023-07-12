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

#include "common.h"
#include "Particle.h"

#include <stdio.h>
#include <cmath>
#include <string>

#define PI 3.141592f

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)

#define DIM 100

#define BLOCKSIZE 1024  // block dim 1D
#define NUMBLOCKS 1024  // grid dim 1D 
#define N (NUMBLOCKS * BLOCKSIZE)


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void GenerateParticles(Particles* D_in, Particles* C_out, curandState* states, float2 x_range, float2 y_range, float2 heading_range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(tid, 0, 0, &states[tid]);
    float pos_x = x_range.x + ((x_range.y - x_range.x) * curand_uniform(&states[tid]));
    float pos_y = y_range.x + ((y_range.y - y_range.x) * curand_uniform(&states[tid]));
    float heading = curand_uniform(&states[tid]);
    heading = std::fmod(heading, 2 * PI);

    C_out->x[tid] = pos_x;
    C_out->y[tid] = pos_y;
    C_out->heading[tid] = heading;
}

//# def predict(particles, u, std, dt=1.):
//# """ move according to control input u (heading change, velocity)
//# with noise Q (std heading change, std velocity)`"""
__global__ void Predict(Particles* D_in, Particles* C_out, curandState* states, float* u, float* std, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dt = 1.0f;

    curand_init(tid, 0, 0, &states[tid]);

    //# update heading
    float heading = D_in->heading[tid];
    heading += u[0] + curand_normal(&states[tid]);
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

    //    weights *= scipy.stats.norm(distance, R).pdf(z[i]) // norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)


    //weights += 1.e-300      # avoid round - off to zero
    //weights /= sum(weights) # normalize

}

void particleFilter(Particles* p);

int main()
{
    Particles p;

    CreateParticleDim(&p, DIM);

    particleFilter(&p);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK(cudaDeviceReset());

    return 0;
}

void particleFilter(Particles* p) {

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

    CreateParticleDim(&p_norm, DIM);

    Norm_BlockUnroll8<<<NUMBLOCKS / 8, BLOCKSIZE >>>(d_p.x, d_out.x, -landmark->x, N);  // ERRROR at <<< can be simply ignored
    Norm_BlockUnroll8<<<NUMBLOCKS / 8, BLOCKSIZE >>>(d_p.y, d_out.y, -landmark->y, N);

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

