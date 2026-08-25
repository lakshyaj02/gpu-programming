#include "norm_kernels.cuh"

__global__ void layernorm_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < N; ++i) {
        mean += input[i];
    }
    mean /= N;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= N;
    float inv_std = rsqrtf(var + 1e-5f);

    // Normalize
    output[idx] = (input[idx] - mean) * inv_std;
}

struct WelfordData {
    float mean;
    float m2;
    int count;
};

__global__ void welford_layernorm_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    WelfordData wd = {0.0f, 0.0f, 0};
    for (int i = 0; i < N; ++i) {
        wd.count++;
        float delta = input[i] - wd.mean;
        wd.mean += delta / wd.count;
        float delta2 = input[i] - wd.mean;
        wd.m2 += delta * delta2;
    }
    float var = wd.m2 / N;
    float inv_std = rsqrtf(var + 1e-5f);

    output[idx] = (input[idx] - wd.mean) * inv_std;
}

void launch_layernorm(const float* input, float* output, int N, int threads_per_block) {
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    layernorm_kernel<<<blocks, threads_per_block>>>(input, output, N);
}

void launch_welford_layernorm(const float* input, float* output, int N, int threads_per_block) {
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    welford_layernorm_kernel<<<blocks, threads_per_block>>>(input, output, N);
}