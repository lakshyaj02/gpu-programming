#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_naive_rmsnorm(const float* input, float* output, const float* gamma,
						  const float* beta, int hidden_size, int block_size = 256);

void launch_warp_rmsnorm(const float* input, float* output, const float* gamma,
						 const float* beta, int hidden_size, int block_size = 256);

void launch_warp_rmsnorm_half(const half* input, half* output, const half* gamma,
							  const half* beta, int hidden_size, int block_size = 256);

void launch_vectorized_rmsnorm(const float* input, float* output, const float* gamma,
                                const float* beta, int hidden_size, int block_size = 256);

void launch_vectorized_rmsnorm_half2(const half* input, half* output, const half* gamma,
									 const half* beta, int hidden_size, int block_size = 256);

void launch_layernorm(const float* input, float* output, int hidden_size,
					  int block_size = 256);

void launch_welford_layernorm(const float* input, float* output, int hidden_size,
						  int block_size = 256);
