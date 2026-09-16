#include "attention_kernels.cuh"

__global__ void pv_matmul_kernel(const float* probabilities, const float* v,
                                 float* output, int m, int n, int dv) {
    // TODO: Define the thread mapping and matrix indexing for P x V.
    // TODO: Explore tiling after establishing correctness.
}

void launch_pv_matmul(const float* probabilities, const float* v, float* output,
                      int m, int n, int dv, cudaStream_t stream) {
    // TODO: Configure and launch pv_matmul_kernel.
}