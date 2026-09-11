#include "fused_kernels.cuh"
#include "kernel_utils.cuh"

namespace {

__global__ void addBiasKernel(const float* input, const float* bias, float* output,
                              int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] + bias[index % columns];
}

__global__ void reluKernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = fmaxf(input[index], 0.0f);
}

__global__ void biasReluKernel(const float* input, const float* bias, float* output,
                               int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = fmaxf(input[index] + bias[index % columns], 0.0f);
}

}  // namespace

void launch_bias_relu_unfused(const float* input, const float* bias, float* scratch,
                              float* output, int rows, int columns, cudaStream_t stream) {
    const int count = rows * columns;
    addBiasKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, bias, scratch, count, columns);
    reluKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(scratch, output, count);
    week07::checkLaunch("bias + ReLU unfused");
}

void launch_bias_relu_fused(const float* input, const float* bias, float* output,
                            int rows, int columns, cudaStream_t stream) {
    const int count = rows * columns;
    biasReluKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, bias, output, count, columns);
    week07::checkLaunch("bias + ReLU fused");
}