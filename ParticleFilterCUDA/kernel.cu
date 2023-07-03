
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

#include "common.h"
#include "Particle.h"

#include <stdio.h>
#include <cmath>

#define PI 3.141592f

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)
#define DIM 100


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void GenerateParticles(Particle* D_in, Particle* C_out, curandState* states, Vec2 x_range, Vec2 y_range, Vec2 heading_range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(tid, 0, 0, &states[tid]);

    D_in->x = x_range.x + ((x_range.y - x_range.x) * curand_uniform(&states[tid]));

    p.position.x = x_range.x + ((x_range.y - x_range.x) * curand_uniform(&states[tid]));
    p.position.y = y_range.x + ((y_range.y - y_range.x) * curand_uniform(&states[tid]));
    p.heading = curand_uniform(&states[tid]);
    p.heading = std::fmod(p.heading, 2 * PI);

    C_out[tid] = p;
}

int main()
{
    Particle p;
    Particle d_p;

    CreateParticleDim(&p, DIM);

    CHECK(cudaMalloc((void**)&d_p.x, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.x, p.x, DIM*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**)&d_p.y, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.y, p.y, DIM * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**)&d_p.weight, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.weight, p.weight, DIM * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc((void**)&d_p.heading, DIM * sizeof(float)));
    CHECK(cudaMemcpy(d_p.heading, p.heading, DIM * sizeof(float), cudaMemcpyHostToDevice));


    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    Particle p;

    p.x = (float*)malloc(DIM * sizeof(Particle));

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    //// Choose which GPU to run on, change this on a multi-GPU system.
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    //// Allocate GPU buffers for three vectors (two input, one output)    .
    //cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //}

    //// Copy input vectors from host memory to GPU buffers.
    //cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    //cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

    //// Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}
    //
    //// cudaDeviceSynchronize waits for the kernel to finish, and returns
    //// any errors encountered during the launch.
    //cudaStatus = cudaDeviceSynchronize();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //    goto Error;
    //}

    //// Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
