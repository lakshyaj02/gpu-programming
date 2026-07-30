#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_check.h"
#include "cuda_timer.h"
#include "reduction.cuh"

static float cpuSum(const float *data, std::size_t size) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

static void initializeData(float *data, std::size_t size) {
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
        1u << 26,   // ~256 MiB
    };

    struct Variant {
        const char *name;
        std::size_t blockSize;
    };

    const std::vector<Variant> variants = {
        {"atomic_b128",        128},
        {"atomic_b256",        256},
        {"atomic_b512",        512},
        {"shared_b128",              128},
        {"shared_b256",              256},
        {"shared_b512",              512},
        {"shared_b1024",            1024},
        {"shared_gs_b128",           128},
        {"shared_gs_b256",           256},
        {"shared_gs_b512",           512},
        {"shared_gs_b1024",         1024},
        {"warp_b128",          128},
        {"warp_b256",          256},
        {"warp_b512",          512},
        {"warp_b1024",        1024},
        {"hierarchical_b128",  128},
        {"hierarchical_b256",  256},
        {"hierarchical_b512",  512},
        {"hierarchical_b1024",1024},
    };

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::puts("kernel,size,bytes,block_size,iterations,avg_kernel_ms,min_kernel_ms,effective_gbps,abs_error,rel_error");

    for (const std::size_t size : sizes) {
        const std::size_t bytes = size * sizeof(float);

        float *h_in = static_cast<float *>(std::malloc(bytes));
        if (!h_in) {
            std::fprintf(stderr, "Host allocation failed for size=%zu\n", size);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }

        initializeData(h_in, size);
        const float refSum = cpuSum(h_in, size);

        float *d_in = nullptr;
        float *d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (const auto &v : variants) {
            // Skip atomic for large sizes (too slow to be useful)
            if (std::strncmp(v.name, "atomic", 6) == 0 && size > (1u << 22)) {
                continue;
            }

            auto launchKernel = [&]() {
                CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float), stream));
                if (std::strncmp(v.name, "atomic", 6) == 0) {
                    launch_atomic_reduction_kernel(d_result, d_in, size, v.blockSize, stream);
                } else if (std::strncmp(v.name, "shared_gs", 9) == 0) {
                    launch_shared_reduction_grid_stride_kernel(d_result, d_in, size, v.blockSize, stream);
                } else if (std::strncmp(v.name, "shared", 6) == 0) {
                    launch_shared_reduction_kernel(d_result, d_in, size, v.blockSize, stream);
                } else if (std::strncmp(v.name, "warp", 4) == 0) {
                    launch_warp_reduction_kernel(d_result, d_in, size, v.blockSize, stream);
                } else {
                    launch_hierarchical_reduction_kernel(d_result, d_in, size, v.blockSize, stream);
                }
            };

            for (int w = 0; w < warmupIterations; ++w) {
                launchKernel();
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CudaTimer timer;
            for (int currentIterations : iterationCounts) {
                std::vector<float> kernelMs;
                kernelMs.reserve(static_cast<std::size_t>(currentIterations));

                for (int it = 0; it < currentIterations; ++it) {
                    timer.start(stream);
                    launchKernel();
                    kernelMs.push_back(timer.stop(stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                float h_result = 0.0f;
                CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(float), stream));
                launchKernel();
                CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const float absErr = std::fabs(h_result - refSum);
                const float relErr = absErr / (std::fabs(refSum) + 1e-6f);

                float sumMs = 0.0f;
                float minMs = kernelMs.front();
                for (float ms : kernelMs) {
                    sumMs += ms;
                    minMs = std::min(minMs, ms);
                }
                const float avgMs = sumMs / static_cast<float>(kernelMs.size());
                const double effectiveGbps =
                    (static_cast<double>(bytes) / (static_cast<double>(avgMs) * 1e-3)) / 1e9;

                std::printf("%s,%zu,%zu,%zu,%d,%.6f,%.6f,%.3f,%.6f,%.6e\n",
                            v.name,
                            size,
                            bytes,
                            v.blockSize,
                            currentIterations,
                            avgMs,
                            minMs,
                            effectiveGbps,
                            absErr,
                            relErr);
            }
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_result));
        std::free(h_in);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
