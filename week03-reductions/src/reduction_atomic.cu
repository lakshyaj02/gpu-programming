#include "reduction.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

__global__ void atomic_reduction_kernel(float *result, const float *data, std::size_t size) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(result, data[idx]);
    }
}

void launch_atomic_reduction_kernel(float *result,
                                    const float *data,
                                    std::size_t size,
                                    std::size_t blockSize,
                                    cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024) {
        std::fprintf(stderr, "Unsupported blockSize=%zu (must be in [1,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }
    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    atomic_reduction_kernel<<<blocks, static_cast<int>(blockSize), 0, stream>>>(result, data, size);
    CUDA_CHECK(cudaGetLastError());
}
