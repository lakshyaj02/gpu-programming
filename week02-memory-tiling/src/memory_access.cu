#include "memory_tiling.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

__global__ void contiguous_copy_kernel(float *output, const float *data, std::size_t size){
    const std::size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        output[idx] = data[idx];
    }
}

__global__ void strided_copy_kernel(float *output, const float *data, std::size_t size, std::size_t stride){
    const std::size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        const std::size_t data_idx = (idx*stride)%size;
        output[idx] = data[data_idx];
    }
}

void launch_contiguous_copy_kernel(float *output,
                                   const float *data,
                                   std::size_t size,
                                   std::size_t blockSize,
                                   cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024) {
        std::fprintf(stderr, "Unsupported copy blockSize=%zu (must be in [1,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }

    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    contiguous_copy_kernel<<<blocks, static_cast<int>(blockSize), 0, stream>>>(output, data, size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_strided_copy_kernel(float *output,
                                const float *data,
                                std::size_t size,
                                std::size_t stride,
                                std::size_t blockSize,
                                cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024) {
        std::fprintf(stderr, "Unsupported copy blockSize=%zu (must be in [1,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }

    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    strided_copy_kernel<<<blocks, static_cast<int>(blockSize), 0, stream>>>(output, data, size, stride);
    CUDA_CHECK(cudaGetLastError());
}