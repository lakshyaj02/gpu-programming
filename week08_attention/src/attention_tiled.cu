#include "attention_kernels.cuh"

void launch_attention_tiled(const float* q, const float* k, const float* v,
                            float* output, int m, int n, int d, int dv,
                            float scale, bool causal, cudaStream_t stream) {
    // TODO: Tile Q, K, and V.
    // TODO: Reuse data through shared memory.
    // TODO: Reduce global-memory traffic.
    // TODO: Avoid materializing intermediate scores.
}