#include "norm_kernels.cuh"

__global__ void naive_rmsnorm_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta, int hidden_size){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= hidden_size) return;

    float sum = 0.0f;
    for(int i=0; i< hidden_size; i++){
        sum += input[i] * input[i];
    }
    float rms = sqrtf(sum/hidden_size + 1e-6f);

    output[idx] = input[idx] / rms * gamma[idx] + beta[idx];
}

void launch_naive_rmsnorm(const float* input, float* output, const float* gamma, const float* beta, int hidden_size, int block_size){
    int grid_size = (hidden_size + block_size-1)/block_size;
    dim3 grid(grid_size);
    dim3 block(block_size);
    naive_rmsnorm_kernel<<<grid, block>>>(input, output, gamma, beta, hidden_size);
    cudaDeviceSynchronize();
}