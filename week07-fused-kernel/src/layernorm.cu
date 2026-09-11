#include "fused_kernels.cuh"
#include "kernel_utils.cuh"

namespace {

constexpr float kEpsilon = 1.0e-5f;

__global__ void statsKernel(const float* input, float* stats, int columns) {
    const int row_offset = blockIdx.x * columns;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[row_offset + column];
        sum += value;
        square_sum += value * value;
    }
    sum = week07::blockSum(sum);
    square_sum = week07::blockSum(square_sum);
    if (threadIdx.x == 0) {
        const float mean = sum / columns;
        stats[2 * blockIdx.x] = mean;
        stats[2 * blockIdx.x + 1] = rsqrtf(fmaxf(square_sum / columns - mean * mean, 0.0f) + kEpsilon);
    }
}

__global__ void normalizeKernel(const float* input, const float* gamma, const float* beta,
                                const float* stats, float* output, int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const int row = index / columns;
        output[index] = (input[index] - stats[2 * row]) * stats[2 * row + 1] *
                        gamma[index % columns] + beta[index % columns];
    }
}

__global__ void layernormKernel(const float* input, const float* gamma, const float* beta,
                                float* output, int columns) {
    const int row_offset = blockIdx.x * columns;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[row_offset + column];
        sum += value;
        square_sum += value * value;
    }
    const float mean = week07::blockSum(sum) / columns;
    const float variance = fmaxf(week07::blockSum(square_sum) / columns - mean * mean, 0.0f);
    const float inverse_std = rsqrtf(variance + kEpsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        output[row_offset + column] = (input[row_offset + column] - mean) * inverse_std *
                                      gamma[column] + beta[column];
    }
}

}  // namespace

void launch_layernorm_unfused(const float* input, const float* gamma, const float* beta,
                              float*, float* row_stats, float* output, int rows, int columns,
                              cudaStream_t stream) {
    const int count = rows * columns;
    statsKernel<<<rows, week07::kBlockSize, 0, stream>>>(input, row_stats, columns);
    normalizeKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, gamma, beta, row_stats, output, count, columns);
    week07::checkLaunch("LayerNorm unfused");
}

void launch_layernorm_fused(const float* input, const float* gamma, const float* beta,
                            float* output, int rows, int columns, cudaStream_t stream) {
    layernormKernel<<<rows, week07::kBlockSize, 0, stream>>>(input, gamma, beta, output, columns);
    week07::checkLaunch("LayerNorm fused");
}