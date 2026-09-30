#include "attention_kernels.cuh"
#include "cuda_check.h"

const int TILE_DIM = 32;
__global__ void qk_matmul_kernel(const float* q, const float* k, float* scores,
                                 int m, int n, int d) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (row < m && col < n) {
        float value = 0.0f;
        for (int k_idx = 0; k_idx < d; ++k_idx) {
            value += q[row * d + k_idx] * k[col * d + k_idx];
        }
        scores[row * n + col] = value;
    }
}

void launch_qk_matmul(const float* q, const float* k, float* scores,
                      int m, int n, int d, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    qk_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(q, k, scores, m, n, d);
    CUDA_CHECK(cudaGetLastError());
}