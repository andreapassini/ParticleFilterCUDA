
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
__global__ void Predict(Particle* D_in, Particle* C_out, curandState* states, float* u, float* std, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dt = 1.0f;

    curand_init(tid, 0, 0, &states[tid]);

    //# update heading
    float heading = D_in->heading[tid];
    heading += u[0] + curand_normal(&states[tid]);
    heading = std::fmod(D_in->heading[tid], 2 * PI);

    //# move in the (noisy) commanded direction
    float dist = (u[1] * dt) + (curand_uniform(&states[tid]) * std[1]);
    float pos_x = D_in->x[tid];
    float pos_y = D_in->y[tid];
    pos_x += std::cos(D_in->heading[tid]) * dist;
    pos_y += std::sin(D_in->heading[tid]) * dist;

    C_out->x[tid] = pos_x;
    C_out->y[tid] = pos_y;
    C_out->heading[tid] = heading;
}

//# def update(particles, weights, z, R, landmarks):
__global__ void Update(Particle* particles, Particle* C_out, float* weights, float* z, float* landmarks, int numberOfLandmarks, float R) {
    //# weights init as ones / N

    for (int i = 0; i < numberOfLandmarks; i++) {
        float distance =
    }

    for i, landmark in enumerate(landmarks) :
        distance = np.linalg.norm(particles[:, 0 : 2] - landmark, axis = 1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.e-300      # avoid round - off to zero
        weights /= sum(weights) # normalize
}

int main()
{
    Particle p;

    CreateParticleDim(&p, DIM);

    cudaError_t cudaStatus = particleFilter(&p);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Particle filter failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t particleFilter(Particle* p) {
    cudaError_t cudaStatus;
    
    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    Particle d_p;

    //CHECK(cudaMalloc((void**)&d_p.x, DIM * sizeof(float)));
    //CHECK(cudaMemcpy(d_p.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMalloc((void**)&d_p.y, DIM * sizeof(float)));
    //CHECK(cudaMemcpy(d_p.y, p.y, DIM * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMalloc((void**)&d_p.weight, DIM * sizeof(float)));
    //CHECK(cudaMemcpy(d_p.weight, p.weight, DIM * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMalloc((void**)&d_p.heading, DIM * sizeof(float)));
    //CHECK(cudaMemcpy(d_p.heading, p.heading, DIM * sizeof(float), cudaMemcpyHostToDevice));

    cudaStatus = cudaMalloc((void**)&d_p.x, DIM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_p.x, p->x, DIM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_p.y, DIM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_p.y, p->y, DIM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_p.weight, DIM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_p.weight, p->weight, DIM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_p.heading, DIM * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_p.heading, p->heading, DIM * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

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
    cudaFree(d_p.x);
    cudaFree(d_p.y);
    cudaFree(d_p.weight);
    cudaFree(d_p.heading);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

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
