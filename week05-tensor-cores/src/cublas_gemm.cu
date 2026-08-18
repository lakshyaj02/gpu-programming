#include "common.cuh"

void launchCublasGemm(cublasHandle_t handle, const half* matrixA, const half* matrixB,
					  float* matrixC, int m, int n, int k) {
	constexpr float alpha = 1.0f;
	constexpr float beta = 0.0f;

	// Reversing A and B maps row-major C=A*B onto cuBLAS's column-major API.
	CUBLAS_CHECK(cublasGemmEx(handle,
							  CUBLAS_OP_N, CUBLAS_OP_N,
							  n, m, k,
							  &alpha,
							  matrixB, CUDA_R_16F, n,
							  matrixA, CUDA_R_16F, k,
							  &beta,
							  matrixC, CUDA_R_32F, n,
							  CUBLAS_COMPUTE_32F,
							  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}