#include "memory_tiling.cuh"
#include "cuda_check.h"

namespace {
constexpr int kMatmulTileSize = 16;
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

__global__ void tiled_matrix_multiply_kernel(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K){
    __shared__ float As[kMatmulTileSize][kMatmulTileSize];
    __shared__ float Bs[kMatmulTileSize][kMatmulTileSize];

    std::size_t x = blockIdx.x * kMatmulTileSize + threadIdx.x;
    std::size_t y = blockIdx.y * kMatmulTileSize + threadIdx.y;

    float sum = 0.0f;

    const int num_tiles = static_cast<int>((K + kMatmulTileSize - 1) / kMatmulTileSize);

    for(int t=0; t < num_tiles; ++t){
        if(x<M && (t*kMatmulTileSize + threadIdx.y)<K){
            As[threadIdx.x][threadIdx.y] = A[x*K + t*kMatmulTileSize + threadIdx.y];
        } else{
            As[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if(y<N && (t*kMatmulTileSize + threadIdx.x)<K){
            Bs[threadIdx.x][threadIdx.y] = B[(t*kMatmulTileSize + threadIdx.x)*N + y];
        } else{
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int k=0; k<kMatmulTileSize; ++k){
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
        }

        __syncthreads();
    }
    if(x<M && y<N){
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
    (void)blockSize;
    dim3 threadsPerBlock(kMatmulTileSize, kMatmulTileSize);
    dim3 numBlocks((M + kMatmulTileSize - 1) / kMatmulTileSize,
                   (N + kMatmulTileSize - 1) / kMatmulTileSize);
    tiled_matrix_multiply_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(C, A, B, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}