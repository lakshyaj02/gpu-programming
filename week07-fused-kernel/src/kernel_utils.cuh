#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace week07 {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

inline void checkLaunch(const char* kernel) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline int blocksFor(int count) {
    return (count + kBlockSize - 1) / kBlockSize;
}

__device__ inline float warpSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ inline float warpMax(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ inline float blockSum(float value) {
    __shared__ float partial[kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warpSum(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / kWarpSize ? partial[lane] : 0.0f;
    if (warp == 0) value = warpSum(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__device__ inline float blockMax(float value) {
    __shared__ float partial[kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warpMax(value);
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / kWarpSize ? partial[lane] : -CUDART_INF_F;
    if (warp == 0) value = warpMax(value);
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

}  // namespace week07