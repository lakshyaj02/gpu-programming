#include "reduction.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

// Initial:
// s0 s1 s2 s3 s4 s5 s6 s7

// Stride 4:
// (s0+s4) (s1+s5) (s2+s6) (s3+s7)

// Stride 2:
// (s0+s4+s2+s6) (s1+s5+s3+s7)

// Stride 1:
// sum

__global__ void shared_reduction_kernel(float *result, const float *data, std::size_t size) {
    extern __shared__ float sdata[];

    const std::size_t tid = threadIdx.x;
    const std::size_t idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx<size) ? data[idx] : 0.0f;
    __syncthreads();

    for(std::size_t stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid+stride];
        }
        // Do not place __syncthreads() inside the if(tid < stride) block, as it will cause deadlock when the number of threads is odd.
        __syncthreads();
        // The barrier inside the loop is necessary because the output from one reduction stage becomes the input to the next stage.
    }

    if(tid == 0){
        atomicAdd(result, sdata[0]);
    }

}

__global__ void shared_reduction_grid_stride_kernel(float *result, const float *data, std::size_t size) {
    extern __shared__ float sdata[];

    const std::size_t tid = static_cast<std::size_t>(threadIdx.x);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t grid_stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    float thread_sum = 0.0f;
    for(std::size_t i = idx; i < size; i += grid_stride){
        thread_sum += data[i];
    }

    sdata[tid] = thread_sum;

    __syncthreads();

    for(std::size_t stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        atomicAdd(result, sdata[0]);
    }
}

void launch_shared_reduction_kernel(float *result,
                                    const float *data,
                                    std::size_t size,
                                    std::size_t blockSize,
                                    cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024) {
        std::fprintf(stderr, "Unsupported blockSize=%zu (must be in [1,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }
    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    const std::size_t sharedBytes = blockSize * sizeof(float);
    shared_reduction_kernel<<<blocks, static_cast<int>(blockSize), sharedBytes, stream>>>(result, data, size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_shared_reduction_grid_stride_kernel(float *result,
                                                 const float *data,
                                                 std::size_t size,
                                                 std::size_t blockSize,
                                                 cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024) {
        std::fprintf(stderr, "Unsupported blockSize=%zu (must be in [1,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }
    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    const std::size_t sharedBytes = blockSize * sizeof(float);
    shared_reduction_grid_stride_kernel<<<blocks, static_cast<int>(blockSize), sharedBytes, stream>>>(result, data, size);
    CUDA_CHECK(cudaGetLastError());
}