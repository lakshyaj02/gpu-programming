#include "attention_kernels.cuh"

__global__ void softmax_kernel(const float* scores, float* probabilities,
                               int m, int n) {
    // TODO: Compute the maximum value for each row.
    // TODO: Exponentiate the shifted row values.
    // TODO: Compute the sum for each row.
    // TODO: Normalize each row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
}

void launch_softmax(const float* scores, float* probabilities, int m, int n,
                    cudaStream_t stream) {
    const int TILE_DIM = 32;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    softmax_kernel<<<gridDim, blockDim, 0, stream>>>(scores, probabilities, m, n);
}