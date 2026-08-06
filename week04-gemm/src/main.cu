#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.h"
#include "cuda_timer.h"
#include "gemm.cuh"

static void cpuMultiply(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void initializeMatrix(std::vector<float>& matrix, int modulus) {
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = static_cast<float>(static_cast<int>(index % modulus) - modulus / 2) / modulus;
    }
}

static void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %s failed with status %d\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int iterations = 20;
    int warmupIterations = 5;

    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) warmupIterations = std::atoi(argv[2]);
    if (iterations <= 0 || warmupIterations < 0) {
        std::fprintf(stderr, "Usage: %s [iterations] [warmup_iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    struct Problem {
        int M;
        int N;
        int K;
    };
    const std::vector<Problem> problems = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {1000, 1500, 768},
        {4093, 2049, 1027},
    };
    const std::vector<int> tileSizes = {8, 16, 32};

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t cublasHandle = nullptr;
    checkCublas(cublasCreate(&cublasHandle), "cublasCreate");
    checkCublas(cublasSetStream(cublasHandle, stream), "cublasSetStream");

    constexpr float absoluteTolerance = 1.0e-3f;
    constexpr float relativeTolerance = 1.0e-3f;
    int failedRuns = 0;

    std::puts("kernel,M,N,K,tile_size,iterations,avg_kernel_ms,min_kernel_ms,gflops,max_abs_error,max_rel_error,error_count,status");

    for (const Problem& problem : problems) {
        const std::size_t elementsA = static_cast<std::size_t>(problem.M) * problem.K;
        const std::size_t elementsB = static_cast<std::size_t>(problem.K) * problem.N;
        const std::size_t elementsC = static_cast<std::size_t>(problem.M) * problem.N;
        std::vector<float> hostA(elementsA);
        std::vector<float> hostB(elementsB);
        std::vector<float> reference(elementsC);
        std::vector<float> result(elementsC);
        initializeMatrix(hostA, 17);
        initializeMatrix(hostB, 13);
        cpuMultiply(hostA.data(), hostB.data(), reference.data(), problem.M, problem.N, problem.K);

        float* deviceA = nullptr;
        float* deviceB = nullptr;
        float* deviceC = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceA, elementsA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&deviceB, elementsB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&deviceC, elementsC * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(deviceA, hostA.data(), elementsA * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceB, hostB.data(), elementsB * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto runBenchmark = [&](const char* kernelName, int tileSize, auto launchKernel) {
            for (int warmup = 0; warmup < warmupIterations; ++warmup) {
                launchKernel();
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CudaTimer timer;
            std::vector<float> timings;
            timings.reserve(static_cast<std::size_t>(iterations));
            for (int iteration = 0; iteration < iterations; ++iteration) {
                timer.start(stream);
                launchKernel();
                timings.push_back(timer.stop(stream));
            }

            CUDA_CHECK(cudaMemcpyAsync(result.data(), deviceC, elementsC * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float totalMs = 0.0f;
            float minMs = timings.front();
            for (float timing : timings) {
                totalMs += timing;
                minMs = std::min(minMs, timing);
            }
            const float avgMs = totalMs / static_cast<float>(timings.size());
            float maxAbsError = 0.0f;
            float maxRelError = 0.0f;
            std::size_t errorCount = 0;
            for (std::size_t index = 0; index < elementsC; ++index) {
                const float absError = std::fabs(result[index] - reference[index]);
                const float relError = absError / (std::fabs(reference[index]) + 1.0e-6f);
                maxAbsError = std::max(maxAbsError, absError);
                maxRelError = std::max(maxRelError, relError);
                if (absError > absoluteTolerance + relativeTolerance * std::fabs(reference[index])) {
                    ++errorCount;
                }
            }
            const bool passed = errorCount == 0;
            const double operations = 2.0 * problem.M * problem.N * problem.K;
            const double gflops = operations / (static_cast<double>(avgMs) * 1.0e6);

            std::printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.6e,%.6e,%zu,%s\n",
                        kernelName, problem.M, problem.N, problem.K, tileSize, iterations,
                        avgMs, minMs, gflops, maxAbsError, maxRelError, errorCount,
                        passed ? "PASS" : "FAIL");

            if (!passed) {
                ++failedRuns;
                std::fprintf(stderr,
                             "ERROR: %s M=%d N=%d K=%d tile=%d failed at %zu/%zu elements "
                             "(max_abs=%.6e, max_rel=%.6e)\n",
                             kernelName, problem.M, problem.N, problem.K, tileSize,
                             errorCount, elementsC, maxAbsError, maxRelError);
            }
        };

        for (int tileSize : tileSizes) {
            for (const char* kernelName : {"naive", "shared", "thread_coarse"}) {
                const dim3 blockDim(tileSize, tileSize);
                auto launchKernel = [&]() {
                    if (kernelName[0] == 'n') {
                        launchNaiveGemmKernel(deviceA, deviceB, deviceC,
                                              problem.M, problem.N, problem.K, blockDim, stream);
                    } else if (kernelName[0] == 's') {
                        launchSharedGemmKernel(deviceA, deviceB, deviceC,
                                               problem.M, problem.N, problem.K, blockDim, stream);
                    } else {
                        launchThreadCoarseGemmKernel(deviceA, deviceB, deviceC,
                                                    problem.M, problem.N, problem.K, tileSize, stream);
                    }
                };
                runBenchmark(kernelName, tileSize, launchKernel);
            }
        }

        runBenchmark("cublas", 0, [&]() {
            launchCublasGemm(cublasHandle, deviceA, deviceB, deviceC,
                             problem.M, problem.N, problem.K);
        });

        CUDA_CHECK(cudaFree(deviceA));
        CUDA_CHECK(cudaFree(deviceB));
        CUDA_CHECK(cudaFree(deviceC));
    }

    checkCublas(cublasDestroy(cublasHandle), "cublasDestroy");
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (failedRuns > 0) {
        std::fprintf(stderr, "ERROR REPORT: %d benchmark run(s) exceeded atol=%.1e, rtol=%.1e\n",
                     failedRuns, absoluteTolerance, relativeTolerance);
        return EXIT_FAILURE;
    }

    std::fprintf(stderr, "ERROR REPORT: all benchmark runs passed atol=%.1e, rtol=%.1e\n",
                 absoluteTolerance, relativeTolerance);
    return EXIT_SUCCESS;
}



