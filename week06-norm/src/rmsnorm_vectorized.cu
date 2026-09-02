#include "norm_kernels.cuh"
#include "common.cuh"

namespace {

constexpr int kWarpSize = 32;

__device__ float warpReduceSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float blockReduceSum(float value){
    __shared__ float shared[kWarpSize];

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warpCount = (blockDim.x + kWarpSize -1) / kWarpSize;

    value = warpReduceSum(value);

    if(lane == 0){
        shared[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < warpCount) ? shared[lane] : 0.0f;

    if(warp == 0){
        value = warpReduceSum(value);
        if(lane == 0){
            shared[0] = value;
        }
    }
    __syncthreads();

    return shared[0];
}

__global__ void vectorized_rmsnorm_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta, int hidden_size){
    const int vectorCount = hidden_size / 4;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);

    float square_sum = 0.0f;

    for(int i = threadIdx.x; i < vectorCount; i += blockDim.x){
        const float4 values = input4[i];

        square_sum += values.x * values.x + values.y * values.y + values.z * values.z + values.w * values.w;
    }

    for(int i = vectorCount * 4 + threadIdx.x; i < hidden_size; i += blockDim.x){
        const float value = input[i];
        square_sum += value * value;
    }

    square_sum = blockReduceSum(square_sum);
    const float inverse_rms = rsqrtf(square_sum / hidden_size + 1e-6f);

    for(int i = threadIdx.x; i < vectorCount; i += blockDim.x){
        const float4 values = input4[i];
        const float4 gammas = gamma4[i];
        const float4 betas = beta4[i];

        output4[i] = make_float4(
            values.x * inverse_rms * gammas.x + betas.x,
            values.y * inverse_rms * gammas.y + betas.y,
            values.z * inverse_rms * gammas.z + betas.z,
            values.w * inverse_rms * gammas.w + betas.w
        );
    }

    for(int i = vectorCount * 4 + threadIdx.x; i < hidden_size; i += blockDim.x){
        output[i] = input[i] * inverse_rms * gamma[i] + beta[i];
    }

}

__global__ void vectorized_rmsnorm_half2_kernel(const half* __restrict__ input, half* __restrict__ output, const half* __restrict__ gamma, const half* __restrict__ beta, int hidden_size){
    const int pairCount = hidden_size / 2;

    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
    const half2* beta2 = reinterpret_cast<const half2*>(beta);

    float square_sum = 0.0f;

    for(int i = threadIdx.x; i < pairCount; i += blockDim.x){
        const float2 values = __half22float2(input2[i]);
        square_sum += values.x * values.x + values.y * values.y;
    }

    if((hidden_size & 1) != 0 && threadIdx.x == 0){
        const float value = __half2float(input[hidden_size - 1]);
        square_sum += value * value;
    }

    square_sum = blockReduceSum(square_sum);
    const float inverse_rms = rsqrtf(square_sum / hidden_size + 1e-6f);

    for(int i = threadIdx.x; i < pairCount; i += blockDim.x){
        const float2 values = __half22float2(input2[i]);
        const float2 gammas = __half22float2(gamma2[i]);
        const float2 betas = __half22float2(beta2[i]);

        output2[i] = __floats2half2_rn(
            values.x * inverse_rms * gammas.x + betas.x,
            values.y * inverse_rms * gammas.y + betas.y);
    }

    if((hidden_size & 1) != 0 && threadIdx.x == 0){
        const int index = hidden_size - 1;
        output[index] = __float2half_rn(
            __half2float(input[index]) * inverse_rms * __half2float(gamma[index])
            + __half2float(beta[index]));
    }
}
} // namespace

void launch_vectorized_rmsnorm(const float* input, float* output, const float* gamma, const float* beta, int hidden_size, int block_size){
    if (hidden_size <= 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        block_size % 32 != 0) {
        return;
    }
    vectorized_rmsnorm_kernel<<<1, block_size>>>(input, output, gamma, beta, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_vectorized_rmsnorm_half2(const half* input, half* output, const half* gamma, const half* beta, int hidden_size, int block_size){
    if (hidden_size <= 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        block_size % 32 != 0) {
        return;
    }
    vectorized_rmsnorm_half2_kernel<<<1, block_size>>>(input, output, gamma, beta, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}