#include "fused_kernels.cuh"
#include "kernel_utils.cuh"

#include <cmath>

namespace {

__device__ float gelu(float value) {
    constexpr float coefficient = 0.7978845608f;
    return 0.5f * value * (1.0f + tanhf(coefficient * (value + 0.044715f * value * value * value)));
}

__global__ void addBiasKernel(const float* input, const float* bias, float* output,
                              int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] + bias[index % columns];
}

__global__ void geluKernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = gelu(input[index]);
}

__global__ void biasGeluKernel(const float* input, const float* bias, float* output,
                               int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = gelu(input[index] + bias[index % columns]);
}

}  // namespace

void launch_bias_gelu_unfused(const float* input, const float* bias, float* scratch,
                              float* output, int rows, int columns, cudaStream_t stream) {
    const int count = rows * columns;
    addBiasKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, bias, scratch, count, columns);
    geluKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(scratch, output, count);
    week07::checkLaunch("bias + GELU unfused");
}

void launch_bias_gelu_fused(const float* input, const float* bias, float* output,
                            int rows, int columns, cudaStream_t stream) {
    const int count = rows * columns;
    biasGeluKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, bias, output, count, columns);
    week07::checkLaunch("bias + GELU fused");
}