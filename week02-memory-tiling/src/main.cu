#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_check.h"
#include "cuda_timer.h"
#include "memory_tiling.cuh"

static void cpuContiguousCopy(float *output, const float *data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = data[i];
    }
}

static void cpuStridedCopy(float *output, const float *data, std::size_t size, std::size_t stride) {
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t data_idx = (i * stride) % size;
        output[i] = data[data_idx];
    }
}

static void cpuTranspose(float *output, const float *data, std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            output[c * rows + r] = data[r * cols + c];
        }
    }
}

static void cpuNaiveMatrixMultiply(float *C, const float *A, const float *B, std::size_t M, std::size_t N, std::size_t K) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static bool almostEqual(const float *a, const float *b, std::size_t size, float eps) {
    for (std::size_t i = 0; i < size; ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

static void initializePattern(float *data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>((i % 1024) * 0.25f);
    }
}

int main(int argc, char **argv) {
    int iterations = -1;
    int warmupIterations = 10;

    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) warmupIterations = std::atoi(argv[2]);

    if ((iterations == 0) || warmupIterations < 0) {
        std::fprintf(stderr, "Usage: %s [iterations] [warmup_iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<int> iterationCounts = {10, 50, 100};
    if (iterations > 0) {
        iterationCounts = {iterations};
    }

    const std::vector<std::size_t> sizes = {
        1u << 20,   // ~4 MiB
        1u << 22,   // ~16 MiB
        1u << 24,   // ~64 MiB
        1u << 26    // ~256 MiB
    };
    const std::vector<std::size_t> strides = {1, 2, 4, 8, 16, 32, 64};
    constexpr std::size_t copyBlockSize = 256;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::puts("copy_pattern,size,bytes,stride,iterations,avg_kernel_ms,min_kernel_ms,effective_gbps,correct");

    for (const std::size_t size : sizes) {
        const std::size_t bytes = size * sizeof(float);

        float *h_in = static_cast<float *>(std::malloc(bytes));
        float *h_out = static_cast<float *>(std::malloc(bytes));
        float *h_ref = static_cast<float *>(std::malloc(bytes));
        if (!h_in || !h_out || !h_ref) {
            std::fprintf(stderr, "Host allocation failed for size=%zu\n", size);
            std::free(h_in);
            std::free(h_out);
            std::free(h_ref);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }

        initializePattern(h_in, size);

        float *d_in = nullptr;
        float *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (const std::size_t stride : strides) {
            const bool isContiguous = (stride == 1);
            if (isContiguous) {
                cpuContiguousCopy(h_ref, h_in, size);
            } else {
                cpuStridedCopy(h_ref, h_in, size, stride);
            }

            for (int w = 0; w < warmupIterations; ++w) {
                if (isContiguous) {
                    launch_contiguous_copy_kernel(d_out, d_in, size, copyBlockSize, stream);
                } else {
                    launch_strided_copy_kernel(d_out, d_in, size, stride, copyBlockSize, stream);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CudaTimer timer;
            for (int currentIterations : iterationCounts) {
                std::vector<float> kernelMs;
                kernelMs.reserve(static_cast<std::size_t>(currentIterations));

                for (int it = 0; it < currentIterations; ++it) {
                    timer.start(stream);
                    if (isContiguous) {
                        launch_contiguous_copy_kernel(d_out, d_in, size, copyBlockSize, stream);
                    } else {
                        launch_strided_copy_kernel(d_out, d_in, size, stride, copyBlockSize, stream);
                    }
                    kernelMs.push_back(timer.stop(stream));
                }

                CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                bool correct = true;
                for (std::size_t i = 0; i < size; ++i) {
                    if (std::fabs(h_out[i] - h_ref[i]) > 1e-6f) {
                        correct = false;
                        break;
                    }
                }

                float sumMs = 0.0f;
                float minMs = kernelMs.front();
                for (float ms : kernelMs) {
                    sumMs += ms;
                    minMs = std::min(minMs, ms);
                }
                const float avgMs = sumMs / static_cast<float>(kernelMs.size());
                const double bytesMoved = 2.0 * static_cast<double>(bytes); // 1 read + 1 write
                const double effectiveGbps = (bytesMoved / (static_cast<double>(avgMs) * 1e-3)) / 1e9;

                std::printf("%s,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%s\n",
                            isContiguous ? "contiguous" : "strided",
                            size,
                            bytes,
                            stride,
                            currentIterations,
                            avgMs,
                            minMs,
                            effectiveGbps,
                            correct ? "PASS" : "FAIL");
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        std::free(h_in);
        std::free(h_out);
        std::free(h_ref);
    }

    std::puts("transpose_variant,rows,cols,bytes,iterations,avg_kernel_ms,min_kernel_ms,effective_gbps,correct");
    const std::vector<std::pair<std::size_t, std::size_t>> transposeSizes = {
        {1024, 1024},
        {2048, 1024},
        {4096, 2048},
        {1536, 1000},
        {3072, 777},
    };
    constexpr std::size_t transposeBlockSize = 16;
    const std::vector<std::pair<std::size_t, std::size_t>> transposeTileConfigs = {
        {8, 8},
        {16, 8},
        {32, 8},
    };

    for (const auto &dims : transposeSizes) {
        const std::size_t rows = dims.first;
        const std::size_t cols = dims.second;
        const std::size_t elements = rows * cols;
        const std::size_t bytes = elements * sizeof(float);

        float *h_in = static_cast<float *>(std::malloc(bytes));
        float *h_out = static_cast<float *>(std::malloc(bytes));
        float *h_ref = static_cast<float *>(std::malloc(bytes));
        if (!h_in || !h_out || !h_ref) {
            std::fprintf(stderr, "Host allocation failed for transpose rows=%zu cols=%zu\n", rows, cols);
            std::free(h_in);
            std::free(h_out);
            std::free(h_ref);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }
        initializePattern(h_in, elements);
        cpuTranspose(h_ref, h_in, rows, cols);

        float *d_in = nullptr;
        float *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int w = 0; w < warmupIterations; ++w) {
            launch_transpose_kernel(d_out, d_in, rows, cols, transposeBlockSize, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int currentIterations : iterationCounts) {
            CudaTimer timer;
            std::vector<float> kernelMs;
            kernelMs.reserve(static_cast<std::size_t>(currentIterations));

            for (int it = 0; it < currentIterations; ++it) {
                timer.start(stream);
                launch_transpose_kernel(d_out, d_in, rows, cols, transposeBlockSize, stream);
                kernelMs.push_back(timer.stop(stream));
            }

            CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const bool correct = almostEqual(h_out, h_ref, elements, 1e-6f);

            float sumMs = 0.0f;
            float minMs = kernelMs.front();
            for (float ms : kernelMs) {
                sumMs += ms;
                minMs = std::min(minMs, ms);
            }
            const float avgMs = sumMs / static_cast<float>(kernelMs.size());
            const double bytesMoved = 2.0 * static_cast<double>(bytes);
            const double effectiveGbps = (bytesMoved / (static_cast<double>(avgMs) * 1e-3)) / 1e9;

            std::printf("naive_b%zu,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%s\n",
                        transposeBlockSize,
                        rows,
                        cols,
                        bytes,
                        currentIterations,
                        avgMs,
                        minMs,
                        effectiveGbps,
                        correct ? "PASS" : "FAIL");
        }

        for (const auto &tileConfig : transposeTileConfigs) {
            const std::size_t tileSize = tileConfig.first;
            const std::size_t blockRows = tileConfig.second;

            for (int w = 0; w < warmupIterations; ++w) {
                launch_transpose_tiled_kernel(d_out, d_in, rows, cols, blockRows, tileSize, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int currentIterations : iterationCounts) {
                CudaTimer timer;
                std::vector<float> kernelMs;
                kernelMs.reserve(static_cast<std::size_t>(currentIterations));

                for (int it = 0; it < currentIterations; ++it) {
                    timer.start(stream);
                    launch_transpose_tiled_kernel(d_out, d_in, rows, cols, blockRows, tileSize, stream);
                    kernelMs.push_back(timer.stop(stream));
                }

                CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const bool correct = almostEqual(h_out, h_ref, elements, 1e-6f);

                float sumMs = 0.0f;
                float minMs = kernelMs.front();
                for (float ms : kernelMs) {
                    sumMs += ms;
                    minMs = std::min(minMs, ms);
                }
                const float avgMs = sumMs / static_cast<float>(kernelMs.size());
                const double bytesMoved = 2.0 * static_cast<double>(bytes);
                const double effectiveGbps = (bytesMoved / (static_cast<double>(avgMs) * 1e-3)) / 1e9;

                std::printf("tiled_t%zu_br%zu,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%s\n",
                            tileSize,
                            blockRows,
                            rows,
                            cols,
                            bytes,
                            currentIterations,
                            avgMs,
                            minMs,
                            effectiveGbps,
                            correct ? "PASS" : "FAIL");
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        std::free(h_in);
        std::free(h_out);
        std::free(h_ref);
    }

    std::puts("matmul_variant,M,N,K,iterations,avg_kernel_ms,min_kernel_ms,gflops,correct");
    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> matmulSizes = {
        {256, 256, 256},
        {512, 512, 512},
        {768, 768, 768},
        {384, 1024, 160},
        {511, 257, 769},
    };
    const std::vector<std::size_t> matmulTileSizes = {8, 16, 32};

    for (const auto &dims : matmulSizes) {
        const std::size_t M = std::get<0>(dims);
        const std::size_t N = std::get<1>(dims);
        const std::size_t K = std::get<2>(dims);
        const std::size_t aElems = M * K;
        const std::size_t bElems = K * N;
        const std::size_t cElems = M * N;
        const std::size_t aBytes = aElems * sizeof(float);
        const std::size_t bBytes = bElems * sizeof(float);
        const std::size_t cBytes = cElems * sizeof(float);

        float *h_a = static_cast<float *>(std::malloc(aBytes));
        float *h_b = static_cast<float *>(std::malloc(bBytes));
        float *h_c = static_cast<float *>(std::malloc(cBytes));
        float *h_ref = static_cast<float *>(std::malloc(cBytes));
        if (!h_a || !h_b || !h_c || !h_ref) {
            std::fprintf(stderr, "Host allocation failed for matmul M=%zu N=%zu K=%zu\n", M, N, K);
            std::free(h_a);
            std::free(h_b);
            std::free(h_c);
            std::free(h_ref);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }
        initializePattern(h_a, aElems);
        initializePattern(h_b, bElems);
        std::memset(h_ref, 0, cBytes);
        cpuNaiveMatrixMultiply(h_ref, h_a, h_b, M, N, K);

        float *d_a = nullptr;
        float *d_b = nullptr;
        float *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, aBytes));
        CUDA_CHECK(cudaMalloc(&d_b, bBytes));
        CUDA_CHECK(cudaMalloc(&d_c, cBytes));
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, aBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int w = 0; w < warmupIterations; ++w) {
            CUDA_CHECK(cudaMemsetAsync(d_c, 0, cBytes, stream));
            launch_naive_matrix_multiply_kernel(d_c, d_a, d_b, M, N, K, 16, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int currentIterations : iterationCounts) {
            CudaTimer timer;
            std::vector<float> kernelMs;
            kernelMs.reserve(static_cast<std::size_t>(currentIterations));

            for (int it = 0; it < currentIterations; ++it) {
                CUDA_CHECK(cudaMemsetAsync(d_c, 0, cBytes, stream));
                timer.start(stream);
                launch_naive_matrix_multiply_kernel(d_c, d_a, d_b, M, N, K, 16, stream);
                kernelMs.push_back(timer.stop(stream));
            }

            CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, cBytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const bool correct = almostEqual(h_c, h_ref, cElems, 1e-2f);

            float sumMs = 0.0f;
            float minMs = kernelMs.front();
            for (float ms : kernelMs) {
                sumMs += ms;
                minMs = std::min(minMs, ms);
            }
            const float avgMs = sumMs / static_cast<float>(kernelMs.size());
            const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
            const double gflops = flops / (static_cast<double>(avgMs) * 1e6);

            std::printf("naive_b16,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%s\n",
                        M,
                        N,
                        K,
                        currentIterations,
                        avgMs,
                        minMs,
                        gflops,
                        correct ? "PASS" : "FAIL");
        }

        for (const std::size_t tileSize : matmulTileSizes) {
            for (int w = 0; w < warmupIterations; ++w) {
                CUDA_CHECK(cudaMemsetAsync(d_c, 0, cBytes, stream));
                launch_tiled_matrix_multiply_kernel(d_c, d_a, d_b, M, N, K, tileSize, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int currentIterations : iterationCounts) {
                CudaTimer timer;
                std::vector<float> kernelMs;
                kernelMs.reserve(static_cast<std::size_t>(currentIterations));

                for (int it = 0; it < currentIterations; ++it) {
                    CUDA_CHECK(cudaMemsetAsync(d_c, 0, cBytes, stream));
                    timer.start(stream);
                    launch_tiled_matrix_multiply_kernel(d_c, d_a, d_b, M, N, K, tileSize, stream);
                    kernelMs.push_back(timer.stop(stream));
                }

                CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, cBytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const bool correct = almostEqual(h_c, h_ref, cElems, 1e-2f);

                float sumMs = 0.0f;
                float minMs = kernelMs.front();
                for (float ms : kernelMs) {
                    sumMs += ms;
                    minMs = std::min(minMs, ms);
                }
                const float avgMs = sumMs / static_cast<float>(kernelMs.size());
                const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
                const double gflops = flops / (static_cast<double>(avgMs) * 1e6);

                std::printf("tiled_t%zu,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%s\n",
                            tileSize,
                            M,
                            N,
                            K,
                            currentIterations,
                            avgMs,
                            minMs,
                            gflops,
                            correct ? "PASS" : "FAIL");
            }
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        std::free(h_a);
        std::free(h_b);
        std::free(h_c);
        std::free(h_ref);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
