#include "attention_kernels.cuh"
#include "cuda_check.h"

__global__ void scale_mask_kernel(float* scores, int m, int n, float scale,
                                  bool causal) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        scores[row * n + col] *= scale;

        if (causal && col > row) {
            scores[row * n + col] = -1e9f;
        }
    }
}

void launch_scale_mask(float* scores, int m, int n, float scale, bool causal,
                       cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }

    const int TILE_DIM = 32;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    scale_mask_kernel<<<gridDim, blockDim, 0, stream>>>(scores, m, n, scale, causal);
    CUDA_CHECK(cudaGetLastError());
}