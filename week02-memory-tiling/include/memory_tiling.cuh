#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void launch_contiguous_copy_kernel(float *output, const float *data, std::size_t size, cudaStream_t stream);
void launch_strided_copy_kernel(float *output, const float *data, std::size_t size, std::size_t stride, cudaStream_t stream);
void launch_transpose_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockSize, cudaStream_t stream);
void launch_transpose_tiled_kernel(float *output, const float *data, std::size_t rows, std::size_t cols, std::size_t blockRows, std::size_t tileSize, cudaStream_t stream);

void launch_naive_matrix_multiply_kernel(float *C,
										 const float *A,
										 const float *B,
										 std::size_t M,
										 std::size_t N,
										 std::size_t K,
										 std::size_t blockSize,
										 cudaStream_t stream);
void launch_tiled_matrix_multiply_kernel(float *C,
										 const float *A,
										 const float *B,
										 std::size_t M,
										 std::size_t N,
										 std::size_t K,
										 std::size_t blockSize,
										 cudaStream_t stream);