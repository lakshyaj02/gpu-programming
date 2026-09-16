#include "attention_kernels.cuh"

void launch_attention_naive(const float* q, const float* k, const float* v,
                            float* scores, float* probabilities, float* output,
                            int m, int n, int d, int dv, float scale,
                            bool causal, cudaStream_t stream) {
    // TODO: Launch QK^T.
    // TODO: Launch scale/mask.
    // TODO: Launch row-wise softmax.
    // TODO: Launch P x V.
}