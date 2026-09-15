#include "fused_kernels.cuh"
#include "kernel_utils.cuh"

#include <cmath>

namespace {

__global__ void rowMaxKernel(const float* input, float* maxima, int columns) {
    const int row_offset = blockIdx.x * columns;
    float maximum = -FLT_MAX;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        maximum = fmaxf(maximum, input[row_offset + column]);
    }
    maximum = week07::blockMax(maximum);
    if (threadIdx.x == 0) maxima[blockIdx.x] = maximum;
}

__global__ void shiftedExpKernel(const float* input, const float* maxima, float* output,
                                 int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = expf(input[index] - maxima[index / columns]);
}

__global__ void rowSumKernel(const float* input, float* sums, int columns) {
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += input[row * columns + column];
    }
    sum = week07::blockSum(sum);
    if (threadIdx.x == 0) sums[row] = sum;
}

__global__ void divideByRowKernel(const float* input, const float* sums, float* output,
                                  int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] / sums[index / columns];
}

__global__ void softmaxKernel(const float* input, float* output, int columns) {
    const int row_offset = blockIdx.x * columns;
    float maximum = -FLT_MAX;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        maximum = fmaxf(maximum, input[row_offset + column]);
    }
    maximum = week07::blockMax(maximum);
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += expf(input[row_offset + column] - maximum);
    }
    sum = week07::blockSum(sum);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        output[row_offset + column] = expf(input[row_offset + column] - maximum) / sum;
    }
}

}  // namespace

void launch_softmax_unfused(const float* input, float* scratch, float* row_stats,
                            float* output, int rows, int columns, cudaStream_t stream) {
    const int count = rows * columns;
    rowMaxKernel<<<rows, week07::kBlockSize, 0, stream>>>(input, row_stats, columns);
    shiftedExpKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(input, row_stats, scratch, count, columns);
    rowSumKernel<<<rows, week07::kBlockSize, 0, stream>>>(scratch, row_stats + rows, columns);
    divideByRowKernel<<<week07::blocksFor(count), week07::kBlockSize, 0, stream>>>(scratch, row_stats + rows, output, count, columns);
    week07::checkLaunch("softmax unfused");
}

void launch_softmax_fused(const float* input, float* output, int rows, int columns,
                          cudaStream_t stream) {
    softmaxKernel<<<rows, week07::kBlockSize, 0, stream>>>(input, output, columns);
    week07::checkLaunch("softmax fused");
}