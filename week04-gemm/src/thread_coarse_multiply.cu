#include "gemm.cuh"
#include "cuda_check.h"

#include <stdexcept>

template <int TILE_SIZE, int COARSE_ROWS>
__global__ void threadCoarseGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    const int localRow = threadIdx.y * COARSE_ROWS;
    const int localCol = threadIdx.x;

    const int globalRow = blockIdx.y * TILE_SIZE + localRow;
    const int globalCol = blockIdx.x * TILE_SIZE + localCol;

    float sum[COARSE_ROWS] = {0.0f};

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        const int aCol = tileIdx * TILE_SIZE + localCol;

        #pragma unroll
        for (int rowOffset = 0; rowOffset < COARSE_ROWS; ++rowOffset) {
            const int sharedRow = localRow + rowOffset;
            const int matrixRow = globalRow + rowOffset;
            const int bRow = tileIdx * TILE_SIZE + sharedRow;
            sA[sharedRow][localCol] =
                matrixRow < M && aCol < K ? A[matrixRow * K + aCol] : 0.0f;
            sB[sharedRow][localCol] =
                bRow < K && globalCol < N ? B[bRow * N + globalCol] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int rowOffset = 0; rowOffset < COARSE_ROWS; ++rowOffset) {
                sum[rowOffset] += sA[localRow + rowOffset][k] * sB[k][localCol];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rowOffset = 0; rowOffset < COARSE_ROWS; ++rowOffset) {
        if (globalRow + rowOffset < M && globalCol < N) {
            C[(globalRow + rowOffset) * N + globalCol] = sum[rowOffset];
        }
    }
}

template <int TILE_SIZE>
void launchThreadCoarse(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int coarseRows = 4;
    const dim3 blockDim(TILE_SIZE, TILE_SIZE / coarseRows);
    const dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    threadCoarseGemmKernel<TILE_SIZE, coarseRows><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}

void launchThreadCoarseGemmKernel(const float* A, const float* B, float* C, int M, int N, int K, int tileSize, cudaStream_t stream) {
    switch (tileSize) {
        case 8:
            launchThreadCoarse<8>(A, B, C, M, N, K, stream);
            break;
        case 16:
            launchThreadCoarse<16>(A, B, C, M, N, K, stream);
            break;
        case 32:
            launchThreadCoarse<32>(A, B, C, M, N, K, stream);
            break;
        default:
            throw std::invalid_argument("supported thread-coarsened GEMM tile sizes are 8, 16, and 32");
    }
    CUDA_CHECK(cudaGetLastError());
}