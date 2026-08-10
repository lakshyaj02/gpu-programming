#!/usr/bin/env python3

"""Create Week 4 GEMM visualizations from a benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "week04-gemm" / "results" / "raw"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "week04-gemm" / "results" / "plots"
TIMESTAMP_RE = re.compile(r"^cuda_timings_gemm_(\d{8}_\d{6})\.csv$")
CUSTOM_KERNELS = ["naive", "shared", "thread_coarse"]
KERNEL_LABELS = {
    "naive": "Naive",
    "shared": "Shared",
    "thread_coarse": "Thread coarsened",
    "cublas": "cuBLAS",
}
COLORS = {
    "naive": "#d1495b",
    "shared": "#00798c",
    "thread_coarse": "#5f8f5f",
    "cublas": "#333333",
}
TILE_COLORS = {8: "#d1495b", 16: "#edae49", 32: "#00798c"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "kernel", "M", "N", "K", "tile_size", "avg_kernel_ms",
        "gflops", "max_abs_error", "max_rel_error", "error_count", "status",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def pick_input(raw_dir: Path, timestamp: str | None) -> tuple[Path, str]:
    available: list[tuple[str, Path]] = []
    for path in raw_dir.glob("cuda_timings_gemm_*.csv"):
        match = TIMESTAMP_RE.match(path.name)
        if match:
            available.append((match.group(1), path))
    if timestamp is not None:
        for candidate_timestamp, path in available:
            if candidate_timestamp == timestamp:
                return path, timestamp
        raise SystemExit(f"No GEMM benchmark CSV found for timestamp {timestamp}")
    if not available:
        raise SystemExit(f"No GEMM benchmark CSV files found under {raw_dir}")
    return max(available)[1], max(available)[0]


def problem_key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["M"]), int(row["N"]), int(row["K"])


def best_rows(rows: list[dict[str, str]]) -> dict[tuple[str, tuple[int, int, int]], dict[str, str]]:
    grouped: dict[tuple[str, tuple[int, int, int]], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["kernel"], problem_key(row))].append(row)
    return {
        key: max(group, key=lambda row: float(row["gflops"]))
        for key, group in grouped.items()
    }


def plot_square_scaling(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    best = best_rows(rows)
    sizes = sorted({int(row["M"]) for row in rows if row["M"] == row["N"] == row["K"]})
    fig, (throughput_ax, latency_ax) = plt.subplots(1, 2, figsize=(13, 5.2))
    for kernel in [*CUSTOM_KERNELS, "cublas"]:
        selected = [best[(kernel, (size, size, size))] for size in sizes]
        label = KERNEL_LABELS[kernel]
        throughput_ax.plot(sizes, [float(row["gflops"]) for row in selected],
                           marker="o", linewidth=2, color=COLORS[kernel], label=label)
        latency_ax.plot(sizes, [float(row["avg_kernel_ms"]) for row in selected],
                        marker="o", linewidth=2, color=COLORS[kernel], label=label)
    for axis in (throughput_ax, latency_ax):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sizes, [str(size) for size in sizes])
        axis.set_xlabel("Square matrix dimension")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    throughput_ax.set_title("Compute throughput")
    throughput_ax.set_ylabel("GFLOP/s")
    latency_ax.set_title("Kernel latency")
    latency_ax.set_ylabel("Average kernel time (ms)")
    latency_ax.legend(frameon=False)
    fig.suptitle("GEMM scaling: best tile per custom kernel")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tile_sensitivity(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    square_rows = [row for row in rows if row["M"] == row["N"] == row["K"]]
    sizes = sorted({int(row["M"]) for row in square_rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, kernel in zip(axes, CUSTOM_KERNELS):
        for tile_size in (8, 16, 32):
            selected = sorted(
                (row for row in square_rows
                 if row["kernel"] == kernel and int(row["tile_size"]) == tile_size),
                key=lambda row: int(row["M"]),
            )
            axis.plot(sizes, [float(row["gflops"]) for row in selected], marker="o",
                      linewidth=2, color=TILE_COLORS[tile_size], label=f"Tile {tile_size}")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sizes, [str(size) for size in sizes], rotation=35)
        axis.set_xlabel("Square matrix dimension")
        axis.set_title(KERNEL_LABELS[kernel])
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("GFLOP/s")
    axes[-1].legend(frameon=False)
    fig.suptitle("Tile-size sensitivity")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cublas_gap(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    best = best_rows(rows)
    square_sizes = sorted({int(row["M"]) for row in rows if row["M"] == row["N"] == row["K"]})
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for kernel in CUSTOM_KERNELS:
        ratios = []
        for size in square_sizes:
            key = (size, size, size)
            custom = float(best[(kernel, key)]["gflops"])
            cublas = float(best[("cublas", key)]["gflops"])
            ratios.append(100.0 * custom / cublas)
        axis.plot(square_sizes, ratios, marker="o", linewidth=2,
                  color=COLORS[kernel], label=KERNEL_LABELS[kernel])
    axis.axhline(100.0, color=COLORS["cublas"], linestyle="--", linewidth=1.5,
                label="cuBLAS parity")
    axis.set_xscale("log", base=2)
    axis.set_xticks(square_sizes, [str(size) for size in square_sizes])
    axis.set_xlabel("Square matrix dimension")
    axis.set_ylabel("Percent of cuBLAS throughput")
    axis.set_title("Custom-kernel performance relative to cuBLAS")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_irregular_and_correctness(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    best = best_rows(rows)
    problems = sorted({problem_key(row) for row in rows if not (row["M"] == row["N"] == row["K"])})
    labels = [f"{M}x{N}x{K}" for M, N, K in problems]
    kernels = [*CUSTOM_KERNELS, "cublas"]
    positions = np.arange(len(problems))
    width = 0.19
    fig, (performance_ax, correctness_ax) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    for index, kernel in enumerate(kernels):
        values = [float(best[(kernel, problem)]["gflops"]) for problem in problems]
        performance_ax.bar(positions + (index - 1.5) * width, values, width,
                           color=COLORS[kernel], label=KERNEL_LABELS[kernel])
    performance_ax.set_xticks(positions, labels)
    performance_ax.set_ylabel("GFLOP/s")
    performance_ax.set_title("Irregular GEMM throughput (best tile)")
    performance_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    performance_ax.legend(frameon=False, fontsize=9)

    total = len(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    failed = total - passed
    correctness_ax.bar(["Passed", "Failed"], [passed, failed],
                       color=["#5f8f5f", "#d1495b"], width=0.55)
    correctness_ax.set_ylabel("Benchmark rows")
    correctness_ax.set_title("Correctness summary")
    correctness_ax.text(0, passed, str(passed), ha="center", va="bottom", fontweight="bold")
    correctness_ax.text(1, max(failed, 0), str(failed), ha="center", va="bottom", fontweight="bold")
    max_abs = max(float(row["max_abs_error"]) for row in rows)
    max_rel = max(float(row["max_rel_error"]) for row in rows)
    correctness_ax.text(0.5, 0.88, f"Max abs error: {max_abs:.2e}\nMax rel error: {max_rel:.2e}",
                        transform=correctness_ax.transAxes, ha="center", va="top")
    correctness_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Week 4 GEMM benchmark results")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--timestamp", help="Run timestamp in YYYYMMDD_HHMMSS format")
    args = parser.parse_args()

    input_path, timestamp = pick_input(args.raw_dir, args.timestamp)
    rows = read_rows(input_path)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.plots_dir / f"gemm_scaling_{timestamp}.png",
        args.plots_dir / f"tile_sensitivity_{timestamp}.png",
        args.plots_dir / f"cublas_gap_{timestamp}.png",
        args.plots_dir / f"irregular_correctness_{timestamp}.png",
    ]
    plot_square_scaling(rows, outputs[0])
    plot_tile_sensitivity(rows, outputs[1])
    plot_cublas_gap(rows, outputs[2])
    plot_irregular_and_correctness(rows, outputs[3])
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()