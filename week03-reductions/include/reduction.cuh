#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void launch_atomic_reduction_kernel(float *result,
                                    const float *data,
                                    std::size_t size,
                                    std::size_t blockSize,
                                    cudaStream_t stream);

void launch_shared_reduction_kernel(float *result,
                                    const float *data,
                                    std::size_t size,
                                    std::size_t blockSize,
                                    cudaStream_t stream);

void launch_shared_reduction_grid_stride_kernel(float *result,
                                                const float *data,
                                                std::size_t size,
                                                std::size_t blockSize,
                                                cudaStream_t stream);

void launch_warp_reduction_kernel(float *result,
                                  const float *data,
                                  std::size_t size,
                                  std::size_t blockSize,
                                  cudaStream_t stream);

void launch_hierarchical_reduction_kernel(float *result,
                                          const float *data,
                                          std::size_t size,
                                          std::size_t blockSize,
                                          cudaStream_t stream);
