#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "cpu_timer.h"
#include "cuda_check.h"
#include "cuda_timer.h"
#include "vector_ops.cuh"


static void cpuVectorAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

static void cpuSaxpy(const float* a, const float* b, float* c, float alpha, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = alpha * a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    // Parse command-line arguments.
    int n = 1 << 20;      // number of elements
    int blockSize = 256;  // threads per block
    int iterations = 100; // number of timing iterations
    int warmupIterations = 10; // number of warm-up iterations
    const char* operation = "vector_add";
    float alpha = 2.0f;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) blockSize = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    if (argc > 4) operation = argv[4];
    if (argc > 5) alpha = static_cast<float>(atof(argv[5]));
    if (n <= 0 || blockSize <= 0 || iterations <= 0) {
        fprintf(
            stderr,
            "Usage: %s [num_elements] [block_size] [iterations] [vector_add|saxpy] [alpha_for_saxpy]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    const bool useSaxpy = strcmp(operation, "saxpy") == 0;
    if (!useSaxpy && strcmp(operation, "vector_add") != 0) {
        fprintf(stderr, "Unsupported operation: %s (expected vector_add or saxpy)\n", operation);
        return EXIT_FAILURE;
    }

    size_t bytes = (size_t)n * sizeof(float);

    // Allocate host and device buffers.
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_a || !h_b || !h_c || !h_ref) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Initialize input data.
    srand(42);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Run CPU reference.
    if (useSaxpy) {
        cpuSaxpy(h_a, h_b, h_ref, alpha, n);
    } else {
        cpuVectorAdd(h_a, h_b, h_ref, n);
    }

    // Run CUDA warm-up.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < warmupIterations; ++i) {
        if (useSaxpy) {
            launch_saxpy(d_a, d_b, d_c, alpha, n, blockSize);
        } else {
            launch_vector_add(d_a, d_b, d_c, n, blockSize);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer cudaTimer;
    CpuTimer cpuTimer;

    // Measure each kernel launch independently.
    std::vector<float> kernelTimingsMs;
    kernelTimingsMs.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        cudaTimer.start();
        if (useSaxpy) {
            launch_saxpy(d_a, d_b, d_c, alpha, n, blockSize);
        } else {
            launch_vector_add(d_a, d_b, d_c, n, blockSize);
        }
        kernelTimingsMs.push_back(cudaTimer.stop());
    }

    // Measure host-to-device plus kernel plus device-to-host time.
    std::vector<float> totalTimingsMs;
    totalTimingsMs.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        cpuTimer.start();
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        if (useSaxpy) {
            launch_saxpy(d_a, d_b, d_c, alpha, n, blockSize);
        } else {
            launch_vector_add(d_a, d_b, d_c, n, blockSize);
        }
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        totalTimingsMs.push_back(cpuTimer.stop());
    }

    // Copy results back (already copied in the timed section above; ensure final copy).
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Check correctness.
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_c[i] - h_ref[i]) > 1e-5f) {
            correct = false;
            break;
        }
    }

    // Print one CSV-formatted row per measured kernel launch.
    // Columns: n,block_size,iteration,kernel_ms,total_ms,correct
    for (int i = 0; i < iterations; ++i) {
        printf("%d,%d,%d,%.6f,%.6f,%s\n", n, blockSize, i,
               kernelTimingsMs[i], totalTimingsMs[i], correct ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
