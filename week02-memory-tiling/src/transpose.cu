#include "memory_tiling.cuh"

namespace {
constexpr int kTransposeTileDim = 32;
constexpr int kTransposeBlockRows = 8;
}

__global__ void transpose_kernel(float *output, const float *data, std::size_t rows, std::size_t cols){
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x<cols && y<rows){
        output[x * rows + y] = data[y * cols + x];
    }
}

__global__ void transpose_tiled_kernel(float *output, const float *data, std::size_t rows, std::size_t cols){
    __shared__ float tile[kTransposeTileDim][kTransposeTileDim + 1]; // +1 to avoid bank conflicts
    // Shared memory used to avoid reordering of memory accesses and to improve coalescing.

    // all threads load a normal tile,
    // synchronize,
    // then all threads read the tile “across” (swapped indices) and write transposed output.

    const std::size_t x = blockIdx.x * kTransposeTileDim + threadIdx.x;
    const std::size_t y = blockIdx.y * kTransposeTileDim + threadIdx.y;

    if(x<cols && y<rows){
        for(std::size_t j = 0; j < kTransposeTileDim; j += kTransposeBlockRows){
            if(x< cols && y+j<rows){
                tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*cols + x];
            }
        }
    }
    __syncthreads();

    const std::size_t transposed_x = blockIdx.y * kTransposeTileDim + threadIdx.x;
    const std::size_t transposed_y = blockIdx.x * kTransposeTileDim + threadIdx.y;

    for(std::size_t j = 0; j < kTransposeTileDim; j += kTransposeBlockRows){
        if(transposed_x<rows && transposed_y+j<cols){
            output[(transposed_y+j)*rows + transposed_x] = tile[threadIdx.x][threadIdx.y+j];
        }
    }
}

void launch_transpose_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockSize, cudaStream_t stream) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((cols + blockSize - 1) / blockSize, (rows + blockSize - 1) / blockSize);
    transpose_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(output, data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_tiled_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockRows, std::size_t tileSize, cudaStream_t stream) {
    (void)blockRows;
    (void)tileSize;
    dim3 threadsPerBlock(kTransposeTileDim, kTransposeBlockRows);
    dim3 numBlocks((cols + kTransposeTileDim - 1) / kTransposeTileDim,
                   (rows + kTransposeTileDim - 1) / kTransposeTileDim);
    transpose_tiled_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(output, data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}