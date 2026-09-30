#include "attention_kernels.cuh"
#include "cuda_check.h"

__global__ void pv_matmul_kernel(const float* probabilities, const float* v,
                                 float* output, int m, int n, int dv) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < dv) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += probabilities[row * n + k] * v[k * dv + col];
        }
        output[row * dv + col] = value;
    }
}

void launch_pv_matmul(const float* probabilities, const float* v, float* output,
                      int m, int n, int dv, cudaStream_t stream) {
    if (m <= 0 || dv <= 0) {
        return;
    }

    const int TILE_DIM = 32;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((dv + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    pv_matmul_kernel<<<gridDim, blockDim, 0, stream>>>(probabilities, v, output, m, n, dv);
    CUDA_CHECK(cudaGetLastError());
}