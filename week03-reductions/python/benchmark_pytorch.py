#!/usr/bin/env python3

"""Run Week 3 PyTorch reduction baselines and export CSV outputs."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week03-reductions" / "results" / "raw"


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
    parser = argparse.ArgumentParser(description="Run Week 3 PyTorch reduction benchmarks")
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    if args.iterations == 0 or args.warmup < 0:
        raise SystemExit("--iterations must be positive or -1, and --warmup must be >= 0")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is not available")

    iteration_counts = [10, 50, 100] if args.iterations < 0 else [args.iterations]
    device = torch.device("cuda")

    reduction_rows: list[dict[str, str]] = []

    sizes = [
        1,
        17,
        31,
        32,
        33,
        1_000,
        1_000_003,
        1 << 20,
        1 << 22,
        1 << 24,
        16_777_219,
        1 << 26,
    ]

    for size in sizes:
        data = torch.arange(size, device=device, dtype=torch.float32) * 0.25
        ref_sum = float(data.sum())
        bytes_moved = float(size * 4)

        result_holder: list[torch.Tensor] = [torch.empty(1, device=device, dtype=torch.float32)]

        def reduce_op() -> None:
            result_holder[0] = data.sum()

        for current_iterations in iteration_counts:
            timings = time_op(reduce_op, current_iterations, args.warmup)
            avg_ms, min_ms = summarize(timings)
            result_val = float(result_holder[0])
            abs_err = abs(result_val - ref_sum)
            rel_err = abs_err / (abs(ref_sum) + 1e-6)
            gbps = (bytes_moved / (avg_ms * 1e-3)) / 1e9
            reduction_rows.append(
                {
                    "kernel": "torch_sum",
                    "size": str(size),
                    "bytes": str(size * 4),
                    "block_size": "N/A",
                    "iterations": str(current_iterations),
                    "avg_kernel_ms": f"{avg_ms:.6f}",
                    "min_kernel_ms": f"{min_ms:.6f}",
                    "effective_gbps": f"{gbps:.3f}",
                    "abs_error": f"{abs_err:.6f}",
                    "rel_error": f"{rel_err:.6e}",
                }
            )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reduction_path = output_dir / f"pytorch_timings_reduction_{args.timestamp}.csv"
    metadata_path = output_dir / f"pytorch_run_metadata_{args.timestamp}.json"

    write_csv(
        reduction_path,
        ["kernel", "size", "bytes", "block_size", "iterations", "avg_kernel_ms", "min_kernel_ms", "effective_gbps", "abs_error", "rel_error"],
        reduction_rows,
    )

    props = torch.cuda.get_device_properties(0)
    metadata = {
        "timestamp": args.timestamp,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 1024**3, 2),
        "gpu_sm_count": props.multi_processor_count,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {reduction_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
