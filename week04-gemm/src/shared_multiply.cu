#include "cuda_check.h"
#include "gemm.cuh"

#include <stdexcept>

template <int TILE_SIZE>
__global__ void sharedGemmKernel(const float *A, const float *B, float *C, int M, int N, int K){
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++){
        const int a_col = tile_idx * TILE_SIZE + threadIdx.x;
        const int b_row = tile_idx * TILE_SIZE + threadIdx.y;
        if(row < M && a_col < K){
            sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        }
        else{
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col < N && b_row < K){
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        }
        else{
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Ensures complete loading of shared memory before proceeding to computation
        __syncthreads();

        #pragma unroll
        for(int k=0; k<TILE_SIZE; k++){
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        // Ensures all threads have completed computation before loading the next tile
        // Ensures no overwriting of shared memory after all threads have completed their computation
        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

void launchSharedGemmKernel(const float *A, const float *B, float *C, int M, int N, int K, dim3 blockDim, cudaStream_t stream){
    if (blockDim.x != blockDim.y) {
        throw std::invalid_argument("shared GEMM requires a square thread block");
    }

    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    switch (blockDim.x) {
        case 8:
            sharedGemmKernel<8><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
            break;
        case 16:
            sharedGemmKernel<16><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
            break;
        case 32:
            sharedGemmKernel<32><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
            break;
        default:
            throw std::invalid_argument("supported shared GEMM tile sizes are 8, 16, and 32");
    }
    CUDA_CHECK(cudaGetLastError());
}