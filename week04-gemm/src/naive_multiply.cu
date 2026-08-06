#include "cpu_timer.h"
#include "cuda_check.h"
#include "gemm.cuh"

__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row< M && col < N){
       float sum = 0.0f;
       for(int k=0; k<K; ++k){
           sum += A[row*K + k] * B[k*N + col];
       }
       C[row*N + col] = sum;
   }
}

void launchNaiveGemmKernel(const float* A, const float* B, float* C, int M, int N, int K, dim3 blockDim, cudaStream_t stream) {
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    naiveGemmKernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}