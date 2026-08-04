#!/usr/bin/env python3

"""Create Week 3 reduction plots from matching CUDA and PyTorch CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "week03-reductions" / "results" / "raw"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "week03-reductions" / "results" / "plots"
TIMESTAMP_RE = re.compile(r"^cuda_timings_reduction_(\d{8}_\d{6})\.csv$")
FAMILY_ORDER = ["atomic", "shared", "shared_gs", "warp", "hierarchical"]
FAMILY_LABELS = {
    "atomic": "Atomic",
    "shared": "Shared",
    "shared_gs": "Shared grid-stride",
    "warp": "Warp",
    "hierarchical": "Hierarchical",
}
COLORS = {
    "atomic": "#d1495b",
    "shared": "#00798c",
    "shared_gs": "#edae49",
    "warp": "#30638e",
    "hierarchical": "#5f8f5f",
    "torch": "#4a4a4a",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family(kernel: str) -> str:
    if kernel.startswith("shared_gs"):
        return "shared_gs"
    return kernel.split("_b", maxsplit=1)[0]


def pick_timestamp(raw_dir: Path, requested: str | None) -> str:
    available = []
    for path in raw_dir.glob("cuda_timings_reduction_*.csv"):
        match = TIMESTAMP_RE.match(path.name)
        if match and (raw_dir / f"pytorch_timings_reduction_{match.group(1)}.csv").is_file():
            available.append(match.group(1))
    if requested is not None:
        if requested not in available:
            raise SystemExit(f"No matching CUDA/PyTorch reduction CSVs for {requested}")
        return requested
    if not available:
        raise SystemExit("No matching CUDA/PyTorch reduction CSVs found")
    return sorted(available)[-1]


def highest_iteration_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    highest = max(int(row["iterations"]) for row in rows)
    return [row for row in rows if int(row["iterations"]) == highest]


def best_by_family_and_size(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(family(row["kernel"]), int(row["size"]))].append(row)
    return {
        key: min(group, key=lambda row: float(row["avg_kernel_ms"]))
        for key, group in grouped.items()
    }


def make_scaling_plot(
    cuda_rows: list[dict[str, str]],
    torch_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    best = best_by_family_and_size(cuda_rows)
    torch_by_size = {int(row["size"]): row for row in torch_rows}
    sizes = sorted(torch_by_size)
    fig, (latency_ax, bandwidth_ax) = plt.subplots(1, 2, figsize=(13, 5))

    for family_name in FAMILY_ORDER:
        present = [size for size in sizes if (family_name, size) in best]
        latency_ax.plot(
            present,
            [float(best[(family_name, size)]["avg_kernel_ms"]) for size in present],
            marker="o",
            markersize=4,
            color=COLORS[family_name],
            label=FAMILY_LABELS[family_name],
        )
        bandwidth_ax.plot(
            present,
            [float(best[(family_name, size)]["effective_gbps"]) for size in present],
            marker="o",
            markersize=4,
            color=COLORS[family_name],
            label=FAMILY_LABELS[family_name],
        )

    latency_ax.plot(
        sizes,
        [float(torch_by_size[size]["avg_kernel_ms"]) for size in sizes],
        marker="s",
        markersize=4,
        color=COLORS["torch"],
        linewidth=2,
        label="PyTorch",
    )
    bandwidth_ax.plot(
        sizes,
        [float(torch_by_size[size]["effective_gbps"]) for size in sizes],
        marker="s",
        markersize=4,
        color=COLORS["torch"],
        linewidth=2,
        label="PyTorch",
    )

    for axis in (latency_ax, bandwidth_ax):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Input length (elements)")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    latency_ax.set_ylabel("Average kernel time (ms)")
    latency_ax.set_title("Reduction latency")
    bandwidth_ax.set_ylabel("Effective input bandwidth (GB/s)")
    bandwidth_ax.set_title("Reduction bandwidth")
    bandwidth_ax.legend(fontsize=8)
    fig.suptitle("Best block size per reduction family (100 iterations)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_block_size_plot(cuda_rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    available_sizes = {int(row["size"]) for row in cuda_rows}
    selected_sizes = [size for size in (1_000_003, 16_777_219) if size in available_sizes]
    fig, axes = plt.subplots(1, len(selected_sizes), figsize=(6.5 * len(selected_sizes), 4.8), squeeze=False)

    for axis, size in zip(axes[0], selected_sizes):
        for family_name in FAMILY_ORDER:
            rows = [
                row
                for row in cuda_rows
                if int(row["size"]) == size and family(row["kernel"]) == family_name
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["block_size"]))
            axis.plot(
                [int(row["block_size"]) for row in rows],
                [float(row["effective_gbps"]) for row in rows],
                marker="o",
                color=COLORS[family_name],
                label=FAMILY_LABELS[family_name],
            )
        axis.set_xticks([128, 256, 512, 1024])
        axis.set_xlabel("Block size")
        axis.set_ylabel("Effective input bandwidth (GB/s)")
        axis.set_title(f"N = {size:,}")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Block-size sensitivity (100 iterations)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_error_plot(cuda_rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    best = best_by_family_and_size(cuda_rows)
    sizes = sorted({int(row["size"]) for row in cuda_rows})
    floor = 1e-12
    fig, (absolute_ax, relative_ax) = plt.subplots(1, 2, figsize=(13, 5))

    for family_name in FAMILY_ORDER:
        present = [size for size in sizes if (family_name, size) in best]
        absolute_ax.plot(
            present,
            [max(float(best[(family_name, size)]["abs_error"]), floor) for size in present],
            marker="o",
            markersize=4,
            color=COLORS[family_name],
            label=FAMILY_LABELS[family_name],
        )
        relative_ax.plot(
            present,
            [max(float(best[(family_name, size)]["rel_error"]), floor) for size in present],
            marker="o",
            markersize=4,
            color=COLORS[family_name],
            label=FAMILY_LABELS[family_name],
        )

    for axis in (absolute_ax, relative_ax):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Input length (elements)")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    absolute_ax.set_ylabel("Absolute error")
    absolute_ax.set_title("Absolute error")
    relative_ax.set_ylabel("Relative error")
    relative_ax.set_title("Relative error")
    relative_ax.legend(fontsize=8)
    fig.suptitle("Recorded error for the fastest block size in each family")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Week 3 reduction benchmark results")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--timestamp")
    args = parser.parse_args()

    timestamp = pick_timestamp(args.raw_dir, args.timestamp)
    cuda_rows = highest_iteration_rows(
        read_rows(args.raw_dir / f"cuda_timings_reduction_{timestamp}.csv")
    )
    torch_rows = highest_iteration_rows(
        read_rows(args.raw_dir / f"pytorch_timings_reduction_{timestamp}.csv")
    )
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        args.plots_dir / f"reduction_scaling_{timestamp}.png",
        args.plots_dir / f"block_size_sensitivity_{timestamp}.png",
        args.plots_dir / f"reduction_error_{timestamp}.png",
    ]
    make_scaling_plot(cuda_rows, torch_rows, outputs[0])
    make_block_size_plot(cuda_rows, outputs[1])
    make_error_plot(cuda_rows, outputs[2])
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()