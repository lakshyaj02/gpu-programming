#!/usr/bin/env python3

"""Benchmark PyTorch normalization operations using the Week 6 CUDA CSV schema."""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as functional


HIDDEN_SIZES = (768, 1024, 2048, 4096, 8192)
EPSILON = 1e-6


def measure_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float, float, int]:
    difference = (actual.double().cpu() - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-6)
    tolerance = absolute_tolerance + relative_tolerance * expected.abs()
    return (
        difference.max().item(),
        relative.max().item(),
        math.sqrt(difference.square().mean().item()),
        int((difference > tolerance).sum().item()),
    )


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    warmup_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if iterations <= 0 or warmup_iterations < 0 or not torch.cuda.is_available():
        raise SystemExit(f"Usage: {sys.argv[0]} [positive_iterations] [nonnegative_warmup_iterations] (CUDA required)")

    print("kernel,precision,hidden_size,block_size,iterations,avg_kernel_ms,min_kernel_ms,bandwidth_gbps,max_abs_error,max_rel_error,rmse,error_count,status")
    for hidden_size in HIDDEN_SIZES:
        generator = torch.Generator(device="cpu").manual_seed(42 + hidden_size)
        input_float = torch.empty(hidden_size).uniform_(-2.0, 2.0, generator=generator)
        residual_float = torch.empty(hidden_size).uniform_(-2.0, 2.0, generator=generator)
        gamma_float = torch.empty(hidden_size).uniform_(0.5, 1.5, generator=generator)
        beta_float = torch.empty(hidden_size).uniform_(-0.25, 0.25, generator=generator)
        input_double = input_float.double()
        rmsnorm_reference = input_double * torch.rsqrt(input_double.square().mean() + EPSILON) * gamma_float.double() + beta_float.double()
        layernorm_reference = functional.layer_norm(input_double, (hidden_size,), eps=1e-5)
        residual_double = residual_float.double()
        fused_input = input_double + residual_double
        fused_reference = fused_input * torch.rsqrt(fused_input.square().mean() + EPSILON) * gamma_float.double()

        for dtype, precision in ((torch.float32, "float32"), (torch.float16, "float16")):
            input_tensor = input_float.to(device="cuda", dtype=dtype)
            residual = residual_float.to(device="cuda", dtype=dtype)
            gamma = gamma_float.to(device="cuda", dtype=dtype)
            beta = beta_float.to(device="cuda", dtype=dtype)
            operations = (
                ("pytorch_rmsnorm", rmsnorm_reference, 4, lambda: functional.rms_norm(input_tensor, (hidden_size,), weight=gamma, eps=EPSILON) + beta),
                ("pytorch_layernorm", layernorm_reference, 2, lambda: functional.layer_norm(input_tensor, (hidden_size,), eps=1e-5)),
                ("pytorch_fused", fused_reference, 4, lambda: functional.rms_norm(input_tensor + residual, (hidden_size,), weight=gamma, eps=EPSILON)),
            )

            for kernel, reference, transferred_arrays, launch in operations:
                with torch.no_grad():
                    for _ in range(warmup_iterations):
                        output = launch()
                    torch.cuda.synchronize()
                    timings = []
                    for _ in range(iterations):
                        start = torch.cuda.Event(enable_timing=True)
                        stop = torch.cuda.Event(enable_timing=True)
                        start.record()
                        output = launch()
                        stop.record()
                        stop.synchronize()
                        timings.append(start.elapsed_time(stop))

                tolerance = 1e-4 if dtype == torch.float32 else 5e-3
                max_abs, max_rel, rmse, error_count = measure_error(
                    output, reference, tolerance, tolerance
                )
                average_ms = sum(timings) / len(timings)
                element_bytes = torch.tensor([], dtype=dtype).element_size()
                bandwidth = transferred_arrays * hidden_size * element_bytes / (average_ms * 1e6)
                status = "PASS" if error_count == 0 else "FAIL"
                print(
                    f"{kernel},{precision},{hidden_size},0,{iterations},{average_ms:.6f},{min(timings):.6f},"
                    f"{bandwidth:.3f},{max_abs:.6e},{max_rel:.6e},{rmse:.6e},{error_count},{status}"
                )


if __name__ == "__main__":
    main()