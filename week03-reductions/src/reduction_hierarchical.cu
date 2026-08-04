#include "reduction.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float warp_reduce_sum_h(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Phase 1: reduce each block into a single partial sum stored in partials[blockIdx.x]
__global__ void hierarchical_phase1_kernel(float *partials, const float *data, std::size_t size) {
    extern __shared__ float sdata[];

    const std::size_t tid = threadIdx.x;
    const std::size_t idx = blockIdx.x * blockDim.x + tid;
    const int lane = tid & 31;
    const int warpId = tid >> 5;

    float val = (idx < size) ? data[idx] : 0.0f;
    val = warp_reduce_sum_h(val);

    if (lane == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();

    const int warpsPerBlock = static_cast<int>(blockDim.x) >> 5;
    if (warpId == 0) {
        val = (lane < warpsPerBlock) ? sdata[lane] : 0.0f;
        val = warp_reduce_sum_h(val);
        if (lane == 0) {
            partials[blockIdx.x] = val;
        }
    }
}

// Phase 2: reduce the partials array (single block)
__global__ void hierarchical_phase2_kernel(float *result, const float *partials, std::size_t count) {
    extern __shared__ float sdata[];

    const std::size_t tid = threadIdx.x;
    float threadSum = 0.0f;
    for (std::size_t i = tid; i < count; i += blockDim.x) {
        threadSum += partials[i];
    }
    sdata[tid] = threadSum;
    __syncthreads();

    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}

void launch_hierarchical_reduction_kernel(float *result,
                                          const float *data,
                                          std::size_t size,
                                          std::size_t blockSize,
                                          cudaStream_t stream) {
    if (blockSize == 0 || blockSize > 1024 || (blockSize & 31) != 0) {
        std::fprintf(stderr, "Unsupported blockSize=%zu (must be a multiple of 32 in [32,1024])\n", blockSize);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t blocks = (size + blockSize - 1) / blockSize;
    const std::size_t warpsPerBlock = blockSize / 32;
    const std::size_t sharedBytes1 = warpsPerBlock * sizeof(float);

    float *d_partials = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_partials, blocks * sizeof(float), stream));

    hierarchical_phase1_kernel<<<static_cast<int>(blocks), static_cast<int>(blockSize), sharedBytes1, stream>>>(
        d_partials, data, size);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2 uses a single block; round up to next power-of-two for the shared-memory tree
    std::size_t p2blocks = 1;
    while (p2blocks < blocks) p2blocks <<= 1;
    if (p2blocks > 1024) p2blocks = 1024;
    const std::size_t sharedBytes2 = p2blocks * sizeof(float);

    hierarchical_phase2_kernel<<<1, static_cast<int>(p2blocks), sharedBytes2, stream>>>(
        result, d_partials, blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_partials, stream));
}
