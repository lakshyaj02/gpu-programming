#include "attention_kernels.cuh"

void launch_attention_naive(const float* q, const float* k, const float* v,
                            float* scores, float* probabilities, float* output,
                            int m, int n, int d, int dv, float scale,
                            bool causal, cudaStream_t stream) {
    launch_qk_matmul(q, k, scores, m, n, d, stream);
    launch_scale_mask(scores, m, n, scale, causal, stream);
    launch_softmax(scores, probabilities, m, n, stream);
    launch_pv_matmul(probabilities, v, output, m, n, dv, stream);
}