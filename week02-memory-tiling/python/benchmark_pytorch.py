#!/usr/bin/env python3

"""Run Week 2 PyTorch baselines and export sectioned CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week02-memory-tiling" / "results" / "raw_out"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def time_op(op, iterations: int, warmup: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(warmup):
            op()
        torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(iterations):
            start.record()
            op()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
    return timings


def summarize(timings: list[float]) -> tuple[float, float]:
    total = 0.0
    min_v = timings[0]
    for v in timings:
        total += v
        min_v = min(min_v, v)
    return total / float(len(timings)), min_v


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 2 PyTorch baseline benchmarks")
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if args.iterations == 0 or args.warmup < 0:
        raise SystemExit("--iterations must be positive or -1, and --warmup must be >= 0")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is not available")

    iteration_counts = [10, 50, 100] if args.iterations < 0 else [args.iterations]
    device = torch.device("cuda")

    copy_rows: list[dict[str, str]] = []
    transpose_rows: list[dict[str, str]] = []
    matmul_rows: list[dict[str, str]] = []

    copy_sizes = [1 << 20, 1 << 22, 1 << 24, 1 << 26]
    strides = [1, 2, 4, 8, 16, 32, 64]
    for size in copy_sizes:
        data = torch.arange(size, device=device, dtype=torch.float32) * 0.25
        out = torch.empty_like(data)
        bytes_moved = 2.0 * float(size * 4)
        for stride in strides:
            idx = (torch.arange(size, device=device, dtype=torch.int64) * stride) % size

            def copy_op() -> None:
                if stride == 1:
                    out.copy_(data)
                else:
                    out.copy_(torch.index_select(data, 0, idx))

            for current_iterations in iteration_counts:
                timings = time_op(copy_op, current_iterations, args.warmup)
                avg_ms, min_ms = summarize(timings)
                reference = data if stride == 1 else torch.index_select(data, 0, idx)
                correct = torch.allclose(out, reference, atol=1e-6)
                gbps = (bytes_moved / (avg_ms * 1e-3)) / 1e9
                copy_rows.append(
                    {
                        "copy_pattern": "contiguous" if stride == 1 else "strided",
                        "size": str(size),
                        "bytes": str(size * 4),
                        "stride": str(stride),
                        "iterations": str(current_iterations),
                        "avg_kernel_ms": f"{avg_ms:.6f}",
                        "min_kernel_ms": f"{min_ms:.6f}",
                        "effective_gbps": f"{gbps:.3f}",
                        "correct": "PASS" if bool(correct) else "FAIL",
                    }
                )

    transpose_sizes = [(1024, 1024), (2048, 1024), (4096, 2048)]
    for rows, cols in transpose_sizes:
        matrix = torch.arange(rows * cols, device=device, dtype=torch.float32).reshape(rows, cols)
        bytes_moved = 2.0 * float(rows * cols * 4)

        def transpose_op() -> None:
            nonlocal_transpose[0] = matrix.t().contiguous()

        nonlocal_transpose = [torch.empty((cols, rows), device=device, dtype=torch.float32)]
        for current_iterations in iteration_counts:
            timings = time_op(transpose_op, current_iterations, args.warmup)
            avg_ms, min_ms = summarize(timings)
            ref = matrix.t().contiguous()
            correct = torch.allclose(nonlocal_transpose[0], ref, atol=1e-6)
            gbps = (bytes_moved / (avg_ms * 1e-3)) / 1e9
            transpose_rows.append(
                {
                    "transpose_variant": "torch",
                    "rows": str(rows),
                    "cols": str(cols),
                    "bytes": str(rows * cols * 4),
                    "iterations": str(current_iterations),
                    "avg_kernel_ms": f"{avg_ms:.6f}",
                    "min_kernel_ms": f"{min_ms:.6f}",
                    "effective_gbps": f"{gbps:.3f}",
                    "correct": "PASS" if bool(correct) else "FAIL",
                }
            )

    matmul_sizes = [256, 512, 768]
    for n in matmul_sizes:
        a = torch.arange(n * n, device=device, dtype=torch.float32).reshape(n, n) * 1e-3
        b = torch.arange(n * n, device=device, dtype=torch.float32).reshape(n, n) * 2e-3

        nonlocal_mm = [torch.empty((n, n), device=device, dtype=torch.float32)]

        def matmul_op() -> None:
            nonlocal_mm[0] = torch.matmul(a, b)

        for current_iterations in iteration_counts:
            timings = time_op(matmul_op, current_iterations, args.warmup)
            avg_ms, min_ms = summarize(timings)
            ref = torch.matmul(a, b)
            correct = torch.allclose(nonlocal_mm[0], ref, atol=1e-2)
            flops = 2.0 * float(n) * float(n) * float(n)
            gflops = flops / (avg_ms * 1e6)
            matmul_rows.append(
                {
                    "matmul_variant": "torch",
                    "M": str(n),
                    "N": str(n),
                    "K": str(n),
                    "iterations": str(current_iterations),
                    "avg_kernel_ms": f"{avg_ms:.6f}",
                    "min_kernel_ms": f"{min_ms:.6f}",
                    "gflops": f"{gflops:.3f}",
                    "correct": "PASS" if bool(correct) else "FAIL",
                }
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_path = output_dir / f"pytorch_timings_copy_{args.timestamp}.csv"
    transpose_path = output_dir / f"pytorch_timings_transpose_{args.timestamp}.csv"
    matmul_path = output_dir / f"pytorch_timings_matmul_{args.timestamp}.csv"
    metadata_path = output_dir / f"pytorch_run_metadata_{args.timestamp}.json"

    write_csv(
        copy_path,
        [
            "copy_pattern",
            "size",
            "bytes",
            "stride",
            "iterations",
            "avg_kernel_ms",
            "min_kernel_ms",
            "effective_gbps",
            "correct",
        ],
        copy_rows,
    )
    write_csv(
        transpose_path,
        [
            "transpose_variant",
            "rows",
            "cols",
            "bytes",
            "iterations",
            "avg_kernel_ms",
            "min_kernel_ms",
            "effective_gbps",
            "correct",
        ],
        transpose_rows,
    )
    write_csv(
        matmul_path,
        [
            "matmul_variant",
            "M",
            "N",
            "K",
            "iterations",
            "avg_kernel_ms",
            "min_kernel_ms",
            "gflops",
            "correct",
        ],
        matmul_rows,
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "rows": {
            "copy": len(copy_rows),
            "transpose": len(transpose_rows),
            "matmul": len(matmul_rows),
        },
        "outputs": {
            "copy_csv": str(copy_path),
            "transpose_csv": str(transpose_path),
            "matmul_csv": str(matmul_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {copy_path}")
    print(f"Wrote {transpose_path}")
    print(f"Wrote {matmul_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
