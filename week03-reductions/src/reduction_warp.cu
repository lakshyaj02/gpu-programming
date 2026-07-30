#include "reduction.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float value) {
    __shared__ float shared[32];

    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;

    float sum = warp_reduce_sum(value);

    if(lane == 0){
        shared[warpId] = sum;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + 31) / 32;

    value = threadIdx.x < warp_count ? shared[lane] : 0.0f;

    if (warpId == 0) {
        value = warp_reduce_sum(value);
    }

    return value;

}

__global__ void warp_reduction_kernel(float *result, const float *data, std::size_t size) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t grid_stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;
    for(std::size_t i = idx; i<size; i+=grid_stride){
        thread_sum += data[i];
    }
    const float block_sum = block_reduce_sum(thread_sum);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(result, block_sum);
    }
}

void launch_warp_reduction_kernel(float *result,
                                  const float *data,
                                  std::size_t size,
                                  std::size_t blockSize,
                                  cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024 || (blockSize & 31) != 0) {
        std::fprintf(stderr, "Unsupported blockSize=%zu (must be a multiple of 32 in [32,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }
    const int blocks = static_cast<int>((size + blockSize - 1) / blockSize);
    const std::size_t warpsPerBlock = blockSize / 32;
    const std::size_t sharedBytes = warpsPerBlock * sizeof(float);
    warp_reduction_kernel<<<blocks, static_cast<int>(blockSize), sharedBytes, stream>>>(result, data, size);
    CUDA_CHECK(cudaGetLastError());
}
