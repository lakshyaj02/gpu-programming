#include "common.cuh"
#include "timer.cuh"
#include "verify.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct Problem {
    int m;
    int n;
    int k;
};

void initializeMatrix(std::vector<float>& matrix, int modulus) {
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = static_cast<float>(static_cast<int>(index % modulus) - modulus / 2) /
                        static_cast<float>(modulus);
    }
}

float quantizeTf32(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x00001000u + ((bits >> 13u) & 1u);
    bits &= 0xffffe000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float quantizeFp8E4m3(float value) {
    const float sign = std::copysign(1.0f, value);
    const float magnitude = std::min(std::fabs(value), 448.0f);
    if (magnitude < 0.015625f) {
        return sign * std::round(magnitude * 512.0f) / 512.0f;
    }
    const int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
    const float quantum = std::ldexp(1.0f, exponent - 3);
    return sign * std::min(std::round(magnitude / quantum) * quantum, 448.0f);
}

float quantizeFp4E2m1(float value) {
    const float sign = std::copysign(1.0f, value);
    const float magnitude = std::min(std::fabs(value), 6.0f);
    if (magnitude < 1.0f) {
        return sign * std::round(magnitude * 2.0f) / 2.0f;
    }
    const int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
    const float quantum = std::ldexp(1.0f, exponent - 1);
    return sign * std::min(std::round(magnitude / quantum) * quantum, 6.0f);
}

template <typename StorageType, typename Convert>
std::vector<StorageType> quantize(std::vector<float>& values, Convert convert) {
    std::vector<StorageType> storage(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = convert(values[index]);
        storage[index] = static_cast<StorageType>(values[index]);
    }
    return storage;
}

template <typename StorageType, typename Launcher>
bool benchmarkMode(const char* precision, const char* execution,
                   const Problem& problem, const std::vector<float>& hostA,
                   const std::vector<float>& hostB, const std::vector<StorageType>& storageA,
                   const std::vector<StorageType>& storageB, int iterations,
                   int warmupIterations, cudaStream_t stream, Launcher launch) {
    const std::size_t elementsC = static_cast<std::size_t>(problem.m) * problem.n;
    const std::vector<float> reference = referenceGemm(
        hostA, hostB, problem.m, problem.n, problem.k);
    std::vector<float> result(elementsC);

    StorageType* deviceA = nullptr;
    StorageType* deviceB = nullptr;
    float* deviceC = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, storageA.size() * sizeof(StorageType)));
    CUDA_CHECK(cudaMalloc(&deviceB, storageB.size() * sizeof(StorageType)));
    CUDA_CHECK(cudaMalloc(&deviceC, elementsC * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(deviceA, storageA.data(), storageA.size() * sizeof(StorageType),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceB, storageB.data(), storageB.size() * sizeof(StorageType),
                               cudaMemcpyHostToDevice, stream));

    for (int warmup = 0; warmup < warmupIterations; ++warmup) {
        launch(deviceA, deviceB, deviceC);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuTimer timer;
    float totalMilliseconds = 0.0f;
    float minimumMilliseconds = 0.0f;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        timer.start(stream);
        launch(deviceA, deviceB, deviceC);
        const float elapsed = timer.stop(stream);
        totalMilliseconds += elapsed;
        minimumMilliseconds = iteration == 0
            ? elapsed : std::min(minimumMilliseconds, elapsed);
    }

    CUDA_CHECK(cudaMemcpyAsync(result.data(), deviceC, elementsC * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const VerificationResult verification = verifyResult(reference, result, 5.0e-2f, 5.0e-3f);
    const float averageMilliseconds = totalMilliseconds / iterations;
    const double operations = 2.0 * problem.m * problem.n * problem.k;
    const double tflops = operations / (averageMilliseconds * 1.0e9);
    const bool passed = verification.errorCount == 0;
    std::printf("wmma,%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6e,%.6e,%zu,%s\n",
                precision, execution, problem.m, problem.n, problem.k, iterations,
                averageMilliseconds, minimumMilliseconds, tflops,
                verification.maxAbsoluteError, verification.maxRelativeError,
                verification.errorCount, passed ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 20;
    int warmupIterations = 5;
    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) warmupIterations = std::atoi(argv[2]);
    if (argc > 3 || iterations <= 0 || warmupIterations < 0) {
        std::fprintf(stderr, "Usage: %s [iterations] [warmup_iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8) {
        std::fprintf(stderr, "TF32 and BF16 WMMA require compute capability 8.0 or newer; found %d.%d\n",
                     properties.major, properties.minor);
        return EXIT_FAILURE;
    }

    runPrecisionDemo();

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const std::vector<Problem> problems = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {768, 1024, 512},
    };

    int failedRuns = 0;
    std::puts("kernel,precision,execution,M,N,K,iterations,avg_ms,min_ms,tflops,max_abs_error,max_rel_error,error_count,status");

    for (const Problem& problem : problems) {
        std::vector<float> sourceA(static_cast<std::size_t>(problem.m) * problem.k);
        std::vector<float> sourceB(static_cast<std::size_t>(problem.k) * problem.n);
        initializeMatrix(sourceA, 17);
        initializeMatrix(sourceB, 13);

        auto tf32AValues = sourceA;
        auto tf32BValues = sourceB;
        const auto tf32A = quantize<float>(tf32AValues, quantizeTf32);
        const auto tf32B = quantize<float>(tf32BValues, quantizeTf32);
        failedRuns += !benchmarkMode("fp32_tf32", "native", problem, tf32AValues, tf32BValues,
            tf32A, tf32B, iterations, warmupIterations, stream,
            [&](const float* matrixA, const float* matrixB, float* matrixC) {
                launchWmmaTf32Gemm(matrixA, matrixB, matrixC,
                                   problem.m, problem.n, problem.k, stream);
            });

        auto bf16AValues = sourceA;
        auto bf16BValues = sourceB;
        const auto bf16A = quantize<__nv_bfloat16>(bf16AValues, [](float value) {
            return __bfloat162float(__float2bfloat16(value));
        });
        const auto bf16B = quantize<__nv_bfloat16>(bf16BValues, [](float value) {
            return __bfloat162float(__float2bfloat16(value));
        });
        failedRuns += !benchmarkMode("bf16", "native", problem, bf16AValues, bf16BValues,
            bf16A, bf16B, iterations, warmupIterations, stream,
            [&](const __nv_bfloat16* matrixA, const __nv_bfloat16* matrixB, float* matrixC) {
                launchWmmaBf16Gemm(matrixA, matrixB, matrixC,
                                   problem.m, problem.n, problem.k, stream);
            });

        for (const auto& mode : {std::pair{"fp8_e4m3", quantizeFp8E4m3},
                                 std::pair{"fp4_e2m1", quantizeFp4E2m1}}) {
            auto lowAValues = sourceA;
            auto lowBValues = sourceB;
            const auto lowA = quantize<half>(lowAValues, mode.second);
            const auto lowB = quantize<half>(lowBValues, mode.second);
            failedRuns += !benchmarkMode(mode.first, "emulated_fp16_wmma", problem,
                lowAValues, lowBValues, lowA, lowB, iterations, warmupIterations, stream,
                [&](const half* matrixA, const half* matrixB, float* matrixC) {
                    launchWmmaGemm(matrixA, matrixB, matrixC,
                                   problem.m, problem.n, problem.k, stream);
                });
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (failedRuns != 0) {
        std::fprintf(stderr, "%d benchmark result(s) failed verification\n", failedRuns);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}