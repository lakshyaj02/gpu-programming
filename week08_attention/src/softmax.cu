#include "attention_kernels.cuh"
#include "cuda_check.h"

#include <cfloat>

__global__ void softmax_kernel(const float* scores, float* probabilities,
                               int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        // Compute the maximum value for the row
        float max_val = -FLT_MAX;
        for (int j = 0; j < n; ++j) {
            max_val = fmaxf(max_val, scores[row * n + j]);
        }
        // Exponentiate the shifted row values and compute the sum
        float sum = 0.0f;
        for(int j = 0; j < n; j++) {
            float shifted_val = expf(scores[row * n + j] - max_val);
            sum += shifted_val;
            probabilities[row * n + j] = shifted_val;
        }
        // Normalize each row
        for(int j = 0; j < n; j++) {
            probabilities[row * n + j] /= sum;
        }
    }
}

void launch_softmax(const float* scores, float* probabilities, int m, int n,
                    cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = (m + threads - 1) / threads;
    softmax_kernel<<<blocks, threads, 0, stream>>>(scores, probabilities, m, n);
    CUDA_CHECK(cudaGetLastError());
}