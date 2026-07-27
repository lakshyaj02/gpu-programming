#include "memory_tiling.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

namespace {
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_tiled_kernel_impl(float *output, const float *data, std::size_t rows, std::size_t cols){
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    const std::size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
    const std::size_t y = blockIdx.y * TILE_DIM + threadIdx.y;

    if(x<cols && y<rows){
        for(std::size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS){
            if(x< cols && y+j<rows){
                tile[threadIdx.y+j][threadIdx.x] = data[(y+j)*cols + x];
            }
        }
    }
    __syncthreads();

    const std::size_t transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    const std::size_t transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for(std::size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        if(transposed_x<rows && transposed_y+j<cols){
            output[(transposed_y+j)*rows + transposed_x] = tile[threadIdx.x][threadIdx.y+j];
        }
    }
}

template <int TILE_DIM>
void launch_tiled_with_block_rows(float *output,
                                  const float *data,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t blockRows,
                                  cudaStream_t stream) {
    dim3 numBlocks((cols + TILE_DIM - 1) / TILE_DIM,
                   (rows + TILE_DIM - 1) / TILE_DIM);
    switch (blockRows) {
    case 4:
        transpose_tiled_kernel_impl<TILE_DIM, 4><<<numBlocks, dim3(TILE_DIM, 4), 0, stream>>>(output, data, rows, cols);
        break;
    case 8:
        transpose_tiled_kernel_impl<TILE_DIM, 8><<<numBlocks, dim3(TILE_DIM, 8), 0, stream>>>(output, data, rows, cols);
        break;
    case 16:
        transpose_tiled_kernel_impl<TILE_DIM, 16><<<numBlocks, dim3(TILE_DIM, 16), 0, stream>>>(output, data, rows, cols);
        break;
    default:
        std::fprintf(stderr, "Unsupported transpose blockRows=%zu (supported: 4,8,16)\n", blockRows);
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

__global__ void transpose_kernel(float *output, const float *data, std::size_t rows, std::size_t cols){
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x<cols && y<rows){
        output[x * rows + y] = data[y * cols + x];
    }
}

void launch_transpose_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockSize, cudaStream_t stream) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((cols + blockSize - 1) / blockSize, (rows + blockSize - 1) / blockSize);
    transpose_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(output, data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_transpose_tiled_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockRows, std::size_t tileSize, cudaStream_t stream) {
    switch (tileSize) {
    case 8:
        launch_tiled_with_block_rows<8>(output, data, rows, cols, blockRows, stream);
        break;
    case 16:
        launch_tiled_with_block_rows<16>(output, data, rows, cols, blockRows, stream);
        break;
    case 32:
        launch_tiled_with_block_rows<32>(output, data, rows, cols, blockRows, stream);
        break;
    default:
        std::fprintf(stderr, "Unsupported transpose tileSize=%zu (supported: 8,16,32)\n", tileSize);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
}