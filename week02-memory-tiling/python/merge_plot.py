#!/usr/bin/env python3

"""Merge Week 2 CUDA/PyTorch benchmark CSVs and create comparison plots."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "week02-memory-tiling" / "results" / "raw_out"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "week02-memory-tiling" / "results" / "plots"

FAMILY_NAMES = ("copy", "transpose", "matmul")
TIMESTAMP_RE = re.compile(r"^(cuda|pytorch)_timings_(copy|transpose|matmul)_(\d{8}_\d{6})\.csv$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def available_timestamps(raw_dir: Path) -> set[str]:
    by_framework: dict[str, dict[str, set[str]]] = {
        "cuda": {family: set() for family in FAMILY_NAMES},
        "pytorch": {family: set() for family in FAMILY_NAMES},
    }
    for path in raw_dir.glob("*_timings_*_*.csv"):
        match = TIMESTAMP_RE.match(path.name)
        if not match:
            continue
        framework, family, timestamp = match.groups()
        by_framework[framework][family].add(timestamp)

    cuda_common = set.intersection(*(by_framework["cuda"][family] for family in FAMILY_NAMES))
    torch_common = set.intersection(*(by_framework["pytorch"][family] for family in FAMILY_NAMES))
    return cuda_common & torch_common


def pick_timestamp(raw_dir: Path, requested: str | None) -> str:
    timestamps = available_timestamps(raw_dir)
    if requested is not None:
        if requested not in timestamps:
            raise SystemExit(f"No full CUDA+PyTorch data set found for timestamp {requested}")
        return requested
    if not timestamps:
        raise SystemExit("No matching CUDA+PyTorch benchmark CSV sets found in raw_out")
    return sorted(timestamps)[-1]


def parse_float(value: str) -> float:
    return float(value)


def parse_int(value: str) -> int:
    return int(value)


def max_iterations(rows: Iterable[dict[str, str]]) -> int:
    values = [parse_int(row["iterations"]) for row in rows]
    if not values:
        raise SystemExit("Encountered an empty CSV while building plots")
    return max(values)


def filter_rows(rows: list[dict[str, str]], iterations: int) -> list[dict[str, str]]:
    return [row for row in rows if parse_int(row["iterations"]) == iterations and row.get("correct", "PASS") == "PASS"]


def merge_family(cuda_rows: list[dict[str, str]], pytorch_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for row in cuda_rows:
        merged.append({"framework": "cuda", **row})
    for row in pytorch_rows:
        merged.append({"framework": "pytorch", **row})
    return merged


def make_copy_plot(cuda_rows: list[dict[str, str]], torch_rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    cuda_iter = max_iterations(cuda_rows)
    torch_iter = max_iterations(torch_rows)
    cuda = filter_rows(cuda_rows, cuda_iter)
    torch = filter_rows(torch_rows, torch_iter)

    cuda_size = max(parse_int(row["size"]) for row in cuda)
    torch_size = max(parse_int(row["size"]) for row in torch)

    cuda = [row for row in cuda if parse_int(row["size"]) == cuda_size]
    torch = [row for row in torch if parse_int(row["size"]) == torch_size]

    cuda.sort(key=lambda row: parse_int(row["stride"]))
    torch.sort(key=lambda row: parse_int(row["stride"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot([parse_int(r["stride"]) for r in cuda], [parse_float(r["avg_kernel_ms"]) for r in cuda], marker="o", label=f"CUDA (size={cuda_size})")
    ax1.plot([parse_int(r["stride"]) for r in torch], [parse_float(r["avg_kernel_ms"]) for r in torch], marker="s", label=f"PyTorch (size={torch_size})")
    ax1.set_xlabel("Stride")
    ax1.set_ylabel("Average kernel time (ms)")
    ax1.set_title("Copy: Latency vs Stride")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.plot([parse_int(r["stride"]) for r in cuda], [parse_float(r["effective_gbps"]) for r in cuda], marker="o", label="CUDA")
    ax2.plot([parse_int(r["stride"]) for r in torch], [parse_float(r["effective_gbps"]) for r in torch], marker="s", label="PyTorch")
    ax2.set_xlabel("Stride")
    ax2.set_ylabel("Effective bandwidth (GB/s)")
    ax2.set_title("Copy: Bandwidth vs Stride")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_labeled_plot(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    family: str,
    x_key: str,
    label_key: str,
    latency_key: str,
    throughput_key: str,
    throughput_label: str,
) -> None:
    import matplotlib.pyplot as plt

    family_iter = max_iterations(rows)
    rows = filter_rows(rows, family_iter)

    x_labels = sorted({row[x_key] for row in rows}, key=lambda text: tuple(int(v) for v in text.split("x")))
    x_pos = list(range(len(x_labels)))

    series_names = sorted({row[label_key] for row in rows}, key=lambda s: (0 if s == "torch" else 1, s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for series in series_names:
        by_label = {row[x_key]: row for row in rows if row[label_key] == series}
        present_labels = [label for label in x_labels if label in by_label]
        if not present_labels:
            continue
        x_idx = [x_labels.index(label) for label in present_labels]
        y_latency = [parse_float(by_label[label][latency_key]) for label in present_labels]
        y_perf = [parse_float(by_label[label][throughput_key]) for label in present_labels]
        marker = "s" if series == "torch" else "o"
        display = "PyTorch" if series == "torch" else f"CUDA ({series})"
        ax1.plot(x_idx, y_latency, marker=marker, label=display)
        ax2.plot(x_idx, y_perf, marker=marker, label=display)

    ax1.set_xticks(x_pos, x_labels, rotation=20)
    ax1.set_xlabel("Matrix shape")
    ax1.set_ylabel("Average kernel time (ms)")
    ax1.set_title(f"{family}: Latency")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.set_xticks(x_pos, x_labels, rotation=20)
    ax2.set_xlabel("Matrix shape")
    ax2.set_ylabel(throughput_label)
    ax2.set_title(f"{family}: Throughput")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Week 2 CUDA/PyTorch CSVs and generate comparison plots")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--timestamp", type=str, default=None, help="Specific timestamp (YYYYMMDD_HHMMSS). Defaults to latest complete set.")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    plots_dir = args.plots_dir

    timestamp = pick_timestamp(raw_dir, args.timestamp)

    cuda_copy = read_rows(raw_dir / f"cuda_timings_copy_{timestamp}.csv")
    cuda_transpose = read_rows(raw_dir / f"cuda_timings_transpose_{timestamp}.csv")
    cuda_matmul = read_rows(raw_dir / f"cuda_timings_matmul_{timestamp}.csv")

    torch_copy = read_rows(raw_dir / f"pytorch_timings_copy_{timestamp}.csv")
    torch_transpose = read_rows(raw_dir / f"pytorch_timings_transpose_{timestamp}.csv")
    torch_matmul = read_rows(raw_dir / f"pytorch_timings_matmul_{timestamp}.csv")

    merged_copy = merge_family(cuda_copy, torch_copy)
    merged_transpose = merge_family(cuda_transpose, torch_transpose)
    merged_matmul = merge_family(cuda_matmul, torch_matmul)

    write_rows(
        raw_dir / f"comparison_copy_{timestamp}.csv",
        ["framework", "copy_pattern", "size", "bytes", "stride", "iterations", "avg_kernel_ms", "min_kernel_ms", "effective_gbps", "correct"],
        merged_copy,
    )
    write_rows(
        raw_dir / f"comparison_transpose_{timestamp}.csv",
        ["framework", "transpose_variant", "rows", "cols", "bytes", "iterations", "avg_kernel_ms", "min_kernel_ms", "effective_gbps", "correct"],
        merged_transpose,
    )
    write_rows(
        raw_dir / f"comparison_matmul_{timestamp}.csv",
        ["framework", "matmul_variant", "M", "N", "K", "iterations", "avg_kernel_ms", "min_kernel_ms", "gflops", "correct"],
        merged_matmul,
    )

    make_copy_plot(cuda_copy, torch_copy, plots_dir / f"comparison_copy_{timestamp}.png")

    transpose_rows = merged_transpose
    for row in transpose_rows:
        row["shape"] = f"{row['rows']}x{row['cols']}"
        row["variant"] = row["transpose_variant"] if row["framework"] == "cuda" else "torch"
    make_labeled_plot(
        transpose_rows,
        plots_dir / f"comparison_transpose_{timestamp}.png",
        family="Transpose",
        x_key="shape",
        label_key="variant",
        latency_key="avg_kernel_ms",
        throughput_key="effective_gbps",
        throughput_label="Effective bandwidth (GB/s)",
    )

    matmul_rows = merged_matmul
    for row in matmul_rows:
        row["shape"] = f"{row['M']}x{row['N']}"
        row["variant"] = row["matmul_variant"] if row["framework"] == "cuda" else "torch"
    make_labeled_plot(
        matmul_rows,
        plots_dir / f"comparison_matmul_{timestamp}.png",
        family="Matmul",
        x_key="shape",
        label_key="variant",
        latency_key="avg_kernel_ms",
        throughput_key="gflops",
        throughput_label="Throughput (GFLOP/s)",
    )

    print(f"Using timestamp: {timestamp}")
    print(f"Wrote {raw_dir / f'comparison_copy_{timestamp}.csv'}")
    print(f"Wrote {raw_dir / f'comparison_transpose_{timestamp}.csv'}")
    print(f"Wrote {raw_dir / f'comparison_matmul_{timestamp}.csv'}")
    print(f"Wrote {plots_dir / f'comparison_copy_{timestamp}.png'}")
    print(f"Wrote {plots_dir / f'comparison_transpose_{timestamp}.png'}")
    print(f"Wrote {plots_dir / f'comparison_matmul_{timestamp}.png'}")


if __name__ == "__main__":
    main()
