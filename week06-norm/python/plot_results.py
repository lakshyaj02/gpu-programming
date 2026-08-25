#!/usr/bin/env python3

"""Create Week 6 RMSNorm plots from matching CUDA and PyTorch CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "week06-norm" / "results" / "raw"
PLOTS_DIR = PROJECT_ROOT / "week06-norm" / "results" / "plots"
COLORS = {"naive": "#d1495b", "warp": "#00798c", "warp_half": "#edae49", "pytorch": "#333333"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"kernel", "precision", "hidden_size", "avg_kernel_ms", "bandwidth_gbps", "max_abs_error", "rmse", "status"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def label(row: dict[str, str]) -> str:
    name = "PyTorch" if row["kernel"] == "pytorch" else row["kernel"].replace("_", " ").title()
    return f"{name} {row['precision'].replace('float', 'FP')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    rows = read_rows(RAW_DIR / f"cuda_timings_norm_{args.timestamp}.csv")
    rows += read_rows(RAW_DIR / f"pytorch_timings_norm_{args.timestamp}.csv")

    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(label(row), []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for series, selected in groups.items():
        selected.sort(key=lambda row: int(row["hidden_size"]))
        sizes = [int(row["hidden_size"]) for row in selected]
        color = COLORS[selected[0]["kernel"]]
        axes[0].plot(sizes, [float(row["avg_kernel_ms"]) for row in selected], marker="o", label=series, color=color)
        axes[1].plot(sizes, [float(row["bandwidth_gbps"]) for row in selected], marker="o", label=series, color=color)
        axes[2].plot(sizes, [float(row["rmse"]) for row in selected], marker="o", label=series, color=color)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([768, 1024, 2048, 4096, 8192], ["768", "1024", "2048", "4096", "8192"], rotation=30)
        axis.set_xlabel("Hidden size")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Average kernel time (ms)"); axes[0].set_title("Latency")
    axes[1].set_ylabel("Effective bandwidth (GB/s)"); axes[1].set_title("Memory throughput")
    axes[2].set_yscale("log"); axes[2].set_ylabel("RMSE vs FP64 reference"); axes[2].set_title("Numerical precision")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("RMSNorm: custom CUDA kernels vs PyTorch")
    fig.tight_layout()
    output = PLOTS_DIR / f"norm_comparison_{args.timestamp}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()