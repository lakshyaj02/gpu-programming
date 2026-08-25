#include "norm_kernels.cuh"

namespace {

constexpr int warpSizeValue = 32;

__device__ float warpReduceSum(float value) {
    for (int offset = warpSizeValue / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void warp_rmsnorm_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta, int hidden_size){
    __shared__ float warp_sums[warpSizeValue];
    const int lane = threadIdx.x % warpSizeValue;
    const int warp = threadIdx.x / warpSizeValue;
    const int warp_count = blockDim.x / warpSizeValue;
    float sum = 0.0f;

    for(int i=threadIdx.x; i<hidden_size; i+=blockDim.x){
        sum += input[i] * input[i];
    }
    sum = warpReduceSum(sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            warp_sums[0] = sum;
        }
    }
    __syncthreads();

    const float rms_inv = rsqrtf(warp_sums[0] / hidden_size + 1e-6f);
    for(int i=threadIdx.x; i<hidden_size; i+=blockDim.x){
        output[i] = input[i] * rms_inv * gamma[i] + beta[i];
    }
}

}  // namespace

void launch_warp_rmsnorm(const float* input, float* output, const float* gamma, const float* beta, int hidden_size, int block_size){
    if (block_size < warpSizeValue || block_size > 1024 || block_size % warpSizeValue != 0) {
        return;
    }
    warp_rmsnorm_kernel<<<1, block_size>>>(input, output, gamma, beta, hidden_size);
    cudaDeviceSynchronize();
}