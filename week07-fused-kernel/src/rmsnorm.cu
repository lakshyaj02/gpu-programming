#include "fused_kernels.cuh"
#include "kernel_utils.cuh"

namespace {

constexpr float kEpsilon = 1.0e-6f;

__global__ void squareKernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * input[index];
}

__global__ void rowMeanKernel(const float* input, float* means, int columns) {
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += input[row * columns + column];
    }
    sum = week07::blockSum(sum);
    if (threadIdx.x == 0) means[row] = sum / columns;
}

__global__ void normalizeKernel(const float* input, const float* gamma,
                                const float* means, float* output, int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * rsqrtf(means[index / columns] + kEpsilon) *
                        gamma[index % columns];
    }
}

__global__ void rmsnormKernel(const float* input, const float* gamma, float* output,
                              int columns) {
    const int row_offset = blockIdx.x * columns;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[row_offset + column];
        square_sum += value * value;
    }
    const float inverse_rms = rsqrtf(week07::blockSum(square_sum) / columns + kEpsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        output[row_offset + column] = input[row_offset + column] * inverse_rms * gamma[column];
    }
}

}  // namespace

void launch_rmsnorm_unfused(const float* input, const float* gamma, float* scratch,
                            float* row_stats, float* output, int rows, int columns,
                            cudaStream_t stream) {
    const int count = rows * columns;
    squareKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, scratch, count);
    rowMeanKernel<<<rows, week07::kBlockSize, 0, stream>>>(scratch, row_stats, columns);
    normalizeKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, gamma, row_stats, output, count, columns);
    week07::checkLaunch("RMSNorm unfused");
}

void launch_rmsnorm_fused(const float* input, const float* gamma, float* output,
                          int rows, int columns, cudaStream_t stream) {
    rmsnormKernel<<<rows, week07::kBlockSize, 0, stream>>>(input, gamma, output, columns);
    week07::checkLaunch("RMSNorm fused");
}