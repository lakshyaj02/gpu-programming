#include "fused_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename Launch>
float timeKernel(Launch launch, int warmups, int iterations) {
    for (int iteration = 0; iteration < warmups; ++iteration) launch();
    checkCuda(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start event");
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronize stop event");
    float elapsed_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "measure elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms / iterations;
}

void report(const char* assignment, const char* operation, int rows, int columns,
            float unfused_ms, float fused_ms, double unfused_bytes, double fused_bytes) {
    const int elements = rows * columns;
    const double unfused_bandwidth = unfused_bytes / (unfused_ms * 1.0e6);
    const double fused_bandwidth = fused_bytes / (fused_ms * 1.0e6);
    std::printf("%s,%s,unfused,%d,%d,%d,%.6f,%.3f,1.000\n",
                assignment, operation, rows, columns, elements, unfused_ms, unfused_bandwidth);
    std::printf("%s,%s,fused,%d,%d,%d,%.6f,%.3f,%.3f\n",
                assignment, operation, rows, columns, elements, fused_ms, fused_bandwidth,
                unfused_ms / fused_ms);
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
    const int warmups = argc > 2 ? std::atoi(argv[2]) : 10;
    if (iterations <= 0 || warmups < 0) {
        std::fprintf(stderr, "Usage: %s [iterations] [warmup_iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::puts("assignment,operation,variant,rows,columns,elements,avg_ms,effective_bandwidth_gbps,speedup_vs_unfused");
    for (int rows : {1, 32, 256}) {
        for (int columns : {256, 1024, 4096}) {
            const int count = rows * columns;
            const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
            const std::size_t parameter_bytes = static_cast<std::size_t>(columns) * sizeof(float);
            std::mt19937 generator(rows * 10000 + columns);
            std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
            std::vector<float> input(count), residual(count), gamma(columns), beta(columns);
            for (float& value : input) value = distribution(generator);
            for (float& value : residual) value = distribution(generator);
            for (float& value : gamma) value = 1.0f + 0.1f * distribution(generator);
            for (float& value : beta) value = 0.1f * distribution(generator);

            float *device_input, *device_residual, *device_gamma, *device_beta;
            float *device_scratch, *device_stats, *device_output;
            checkCuda(cudaMalloc(&device_input, bytes), "allocate input");
            checkCuda(cudaMalloc(&device_residual, bytes), "allocate residual");
            checkCuda(cudaMalloc(&device_gamma, parameter_bytes), "allocate gamma");
            checkCuda(cudaMalloc(&device_beta, parameter_bytes), "allocate beta");
            checkCuda(cudaMalloc(&device_scratch, bytes), "allocate scratch");
            checkCuda(cudaMalloc(&device_stats, 2 * rows * sizeof(float)), "allocate stats");
            checkCuda(cudaMalloc(&device_output, bytes), "allocate output");
            checkCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
            checkCuda(cudaMemcpy(device_residual, residual.data(), bytes, cudaMemcpyHostToDevice), "copy residual");
            checkCuda(cudaMemcpy(device_gamma, gamma.data(), parameter_bytes, cudaMemcpyHostToDevice), "copy gamma");
            checkCuda(cudaMemcpy(device_beta, beta.data(), parameter_bytes, cudaMemcpyHostToDevice), "copy beta");

            auto measurePair = [&](const char* assignment, const char* operation,
                                   double unfused_bytes, double fused_bytes,
                                   auto unfused, auto fused) {
                const float unfused_ms = timeKernel(unfused, warmups, iterations);
                const float fused_ms = timeKernel(fused, warmups, iterations);
                report(assignment, operation, rows, columns, unfused_ms, fused_ms,
                       unfused_bytes, fused_bytes);
            };
            const double tensor_bytes = static_cast<double>(bytes);
            const double parameters = static_cast<double>(parameter_bytes);

            measurePair("7.1", "bias_relu", 4.0 * tensor_bytes + parameters,
                        2.0 * tensor_bytes + parameters,
                        [&] { launch_bias_relu_unfused(device_input, device_beta, device_scratch, device_output, rows, columns); },
                        [&] { launch_bias_relu_fused(device_input, device_beta, device_output, rows, columns); });
            measurePair("7.2", "bias_gelu", 4.0 * tensor_bytes + parameters,
                        2.0 * tensor_bytes + parameters,
                        [&] { launch_bias_gelu_unfused(device_input, device_beta, device_scratch, device_output, rows, columns); },
                        [&] { launch_bias_gelu_fused(device_input, device_beta, device_output, rows, columns); });
            measurePair("7.3", "rmsnorm", 6.0 * tensor_bytes + parameters,
                        3.0 * tensor_bytes + parameters,
                        [&] { launch_rmsnorm_unfused(device_input, device_gamma, device_scratch, device_stats, device_output, rows, columns); },
                        [&] { launch_rmsnorm_fused(device_input, device_gamma, device_output, rows, columns); });
            measurePair("7.4", "layernorm", 3.0 * tensor_bytes + 2.0 * parameters,
                        3.0 * tensor_bytes + 2.0 * parameters,
                        [&] { launch_layernorm_unfused(device_input, device_gamma, device_beta, device_scratch, device_stats, device_output, rows, columns); },
                        [&] { launch_layernorm_fused(device_input, device_gamma, device_beta, device_output, rows, columns); });
            measurePair("7.5", "residual_rmsnorm", 8.0 * tensor_bytes + parameters,
                        4.0 * tensor_bytes + parameters,
                        [&] { launch_residual_rmsnorm_unfused(device_input, device_residual, device_gamma, device_scratch, device_stats, device_output, rows, columns); },
                        [&] { launch_residual_rmsnorm_fused(device_input, device_residual, device_gamma, device_output, rows, columns); });
            measurePair("7.6", "softmax", 6.0 * tensor_bytes, 4.0 * tensor_bytes,
                        [&] { launch_softmax_unfused(device_input, device_scratch, device_stats, device_output, rows, columns); },
                        [&] { launch_softmax_fused(device_input, device_output, rows, columns); });

            cudaFree(device_input); cudaFree(device_residual); cudaFree(device_gamma);
            cudaFree(device_beta); cudaFree(device_scratch); cudaFree(device_stats);
            cudaFree(device_output);
        }
    }
    return EXIT_SUCCESS;
}