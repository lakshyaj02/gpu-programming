#include "memory_tiling.cuh"

__global__ void contiguous_copy_kernel(float *output, const float *data, std::size_t size){
    const std::size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        output[idx] = data[idx];
    }
}

__global__ void strided_copy_kernel(float *output, const float *data, std::size_t size, std::size_t stride){
    const std::size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        const std::size_t data_idx = (idx*stride)%size;
        output[idx] = data[data_idx];
    }
}

void launch_contiguous_copy_kernel(float *output, const float *data, std::size_t size, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((size + static_cast<std::size_t>(kBlockSize) - 1) /
                                        static_cast<std::size_t>(kBlockSize));
    contiguous_copy_kernel<<<blocks, kBlockSize, 0, stream>>>(output, data, size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_strided_copy_kernel(float *output, const float *data, std::size_t size, std::size_t stride, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((size + static_cast<std::size_t>(kBlockSize) - 1) /
                                        static_cast<std::size_t>(kBlockSize));
    strided_copy_kernel<<<blocks, kBlockSize, 0, stream>>>(output, data, size, stride);
    CUDA_CHECK(cudaGetLastError());
}