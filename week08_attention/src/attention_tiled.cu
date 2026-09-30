#include "attention_kernels.cuh"
#include "cuda_check.h"

void launch_attention_tiled(const float* q, const float* k, const float* v,
                            float* output, int m, int n, int d, int dv,
                            float scale, bool causal, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || d <= 0 || dv <= 0) {
        return;
    }

    float* scores = nullptr;
    float* probabilities = nullptr;
    const size_t intermediate_bytes = static_cast<size_t>(m) * n * sizeof(float);

    CUDA_CHECK(cudaMallocAsync(&scores, intermediate_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&probabilities, intermediate_bytes, stream));

    launch_attention_naive(q, k, v, scores, probabilities, output,
                           m, n, d, dv, scale, causal, stream);

    CUDA_CHECK(cudaFreeAsync(probabilities, stream));
    CUDA_CHECK(cudaFreeAsync(scores, stream));
}