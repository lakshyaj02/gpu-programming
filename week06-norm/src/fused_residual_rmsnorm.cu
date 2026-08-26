#include "norm_kernels.cuh"

namespace {

constexpr int kWarpSize = 32;

__device__ float warpReduceSum(float value){
    for(int offset = kWarpSize/2; offset > 0; offset /= 2){
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float blockReduceSum(float value){
    __shared__ float shared[kWarpSize];

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warp_count = blockDim.x / kWarpSize;

    value = warpReduceSum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();

    if (warp == 0){
        value = lane < warp_count ? shared[lane] : 0.0f;
        value = warpReduceSum(value);
        if(lane == 0){
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__global__ void fused_residual_rmsnorm_kernel(const float* __restrict__ input, const float* __restrict__ residual, const float* __restrict__ gamma, float* __restrict__ output, int N) {
    float square_sum = 0.0f;

    for(int i = threadIdx.x; i < N; i += blockDim.x){
        float val = input[i] + residual[i];
        square_sum += val * val;
    }
    square_sum = blockReduceSum(square_sum);

    const float inverse_rms = rsqrtf(square_sum / N + 1e-6f);

    for(int i = threadIdx.x; i < N; i += blockDim.x){
        float val = input[i] + residual[i];
        output[i] = val * inverse_rms * gamma[i];
    }
}

}  // namespace

void launch_fused_residual_rmsnorm(const float* input, const float* residual, const float* gamma, float* output, int N, int threads_per_block) {
    if (N <= 0 || threads_per_block < kWarpSize || threads_per_block > 1024 ||
        threads_per_block % kWarpSize != 0) {
        return;
    }
    fused_residual_rmsnorm_kernel<<<1, threads_per_block>>>(input, residual, gamma, output, N);
}
