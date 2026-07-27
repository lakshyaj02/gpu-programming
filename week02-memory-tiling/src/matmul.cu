#include "memory_tiling.cuh"
#include "cuda_check.h"

#include <cstdio>
#include <cstdlib>

template <int TILE>
__global__ void tiled_matrix_multiply_kernel_impl(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    std::size_t x = blockIdx.x * TILE + threadIdx.x;
    std::size_t y = blockIdx.y * TILE + threadIdx.y;

    float sum = 0.0f;

    const int num_tiles = static_cast<int>((K + TILE - 1) / TILE);

    for(int t=0; t < num_tiles; ++t){
        if(x<M && (t*TILE + threadIdx.y)<K){
            As[threadIdx.x][threadIdx.y] = A[x*K + t*TILE + threadIdx.y];
        } else{
            As[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if(y<N && (t*TILE + threadIdx.x)<K){
            Bs[threadIdx.x][threadIdx.y] = B[(t*TILE + threadIdx.x)*N + y];
        } else{
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int k=0; k<TILE; ++k){
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
        }

        __syncthreads();
    }
    if(x<M && y<N){
        C[x*N+y] = sum;
    }
}

__global__ void naive_matrix_multiply_kernel(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K){
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x<M && y<N){
        float sum = 0.0f;
        for(std::size_t k=0; k<K; ++k){
            sum += A[x*K + k] * B[k*N + y];
        }
        C[x*N+y] = sum;
    }
}

void launch_naive_matrix_multiply_kernel(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K, std::size_t blockSize, cudaStream_t stream) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((M + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    naive_matrix_multiply_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled_matrix_multiply_kernel(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K, std::size_t blockSize, cudaStream_t stream) {
    switch (blockSize) {
    case 8: {
        dim3 threadsPerBlock(8, 8);
        dim3 numBlocks((M + 8 - 1) / 8, (N + 8 - 1) / 8);
        tiled_matrix_multiply_kernel_impl<8><<<numBlocks, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
        break;
    }
    case 16: {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((M + 16 - 1) / 16, (N + 16 - 1) / 16);
        tiled_matrix_multiply_kernel_impl<16><<<numBlocks, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
        break;
    }
    case 32: {
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((M + 32 - 1) / 32, (N + 32 - 1) / 32);
        tiled_matrix_multiply_kernel_impl<32><<<numBlocks, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
        break;
    }
    default:
        std::fprintf(stderr, "Unsupported matmul blockSize=%zu (supported: 8,16,32)\n", blockSize);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
}