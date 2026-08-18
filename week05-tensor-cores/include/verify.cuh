#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct VerificationResult {
    float maxAbsoluteError = 0.0f;
    float maxRelativeError = 0.0f;
    std::size_t errorCount = 0;
};

std::vector<float> referenceGemm(const std::vector<float>& matrixA,
                                 const std::vector<float>& matrixB,
                                 int m, int n, int k);

VerificationResult verifyResult(const std::vector<float>& expected,
                                const std::vector<float>& actual,
                                float absoluteTolerance,
                                float relativeTolerance);

inline std::vector<float> referenceGemm(const std::vector<float>& matrixA,
                                        const std::vector<float>& matrixB,
                                        int m, int n, int k) {
    std::vector<float> result(static_cast<std::size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int inner = 0; inner < k; ++inner) {
            const float valueA = matrixA[static_cast<std::size_t>(row) * k + inner];
            for (int column = 0; column < n; ++column) {
                result[static_cast<std::size_t>(row) * n + column] +=
                    valueA * matrixB[static_cast<std::size_t>(inner) * n + column];
            }
        }
    }
    return result;
}

inline VerificationResult verifyResult(const std::vector<float>& expected,
                                       const std::vector<float>& actual,
                                       float absoluteTolerance,
                                       float relativeTolerance) {
    VerificationResult result;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float absoluteError = std::fabs(actual[index] - expected[index]);
        const float relativeError = absoluteError / (std::fabs(expected[index]) + 1.0e-6f);
        result.maxAbsoluteError = std::max(result.maxAbsoluteError, absoluteError);
        result.maxRelativeError = std::max(result.maxRelativeError, relativeError);
        if (absoluteError > absoluteTolerance + relativeTolerance * std::fabs(expected[index])) {
            ++result.errorCount;
        }
    }
    return result;
}