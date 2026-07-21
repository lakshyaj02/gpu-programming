#include "vector_ops.cuh"
#include "cuda_check.h"

__global__ void vector_kernel_add(const float *a, const float *b, float *c, const float alpha, std::size_t n){
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = alpha * a[idx] + b[idx];
    }
}

void launch_saxpy(const float *a, const float *b, float *c, const float alpha, std::size_t n, int block_size){
    const int grid_size = (n + block_size - 1) / block_size;
    vector_kernel_add<<<grid_size, block_size>>>(a, b, c, alpha, n);
    CUDA_CHECK(cudaGetLastError());
}