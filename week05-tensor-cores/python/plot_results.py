#!/usr/bin/env python3

"""Plot Week 05 Tensor Core GEMM benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


WEEK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = WEEK_DIR / "results" / "gemm_results.csv"
DEFAULT_OUTPUT_DIR = WEEK_DIR / "results" / "plots"
PRECISION_ORDER = ["fp32_tf32", "bf16", "fp8_e4m3", "fp4_e2m1"]
LABELS = {
    "fp32_tf32": "FP32 / TF32",
    "bf16": "BF16",
    "fp8_e4m3": "FP8 E4M3\n(emulated)",
    "fp4_e2m1": "FP4 E2M1\n(emulated)",
}
COLORS = {
    "fp32_tf32": "#35618f",
    "bf16": "#2f7d58",
    "fp8_e4m3": "#d08b28",
    "fp4_e2m1": "#b54d4d",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "precision", "execution", "M", "N", "K", "avg_ms", "tflops", "status",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(f"{path} contains failed benchmark rows")
    return rows


def problem_label(row: dict[str, str]) -> str:
    return f'{row["M"]}x{row["N"]}x{row["K"]}'


def rows_by_precision(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        precision: [row for row in rows if row["precision"] == precision]
        for precision in PRECISION_ORDER
    }


def plot_performance(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = rows_by_precision(rows)
    problems = [problem_label(row) for row in grouped[PRECISION_ORDER[0]]]
    positions = np.arange(len(problems))
    width = 0.2
    fig, (throughput_axis, latency_axis) = plt.subplots(1, 2, figsize=(14, 5.4))

    for index, precision in enumerate(PRECISION_ORDER):
        selected = grouped[precision]
        offsets = positions + (index - 1.5) * width
        hatch = "//" if selected[0]["execution"].startswith("emulated") else None
        throughput_axis.bar(
            offsets,
            [float(row["tflops"]) for row in selected],
            width,
            color=COLORS[precision],
            hatch=hatch,
            label=LABELS[precision],
        )
        latency_axis.bar(
            offsets,
            [float(row["avg_ms"]) for row in selected],
            width,
            color=COLORS[precision],
            hatch=hatch,
            label=LABELS[precision],
        )

    for axis in (throughput_axis, latency_axis):
        axis.set_xticks(positions, problems, rotation=22, ha="right")
        axis.set_xlabel("GEMM shape (M x N x K)")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    throughput_axis.set_title("Tensor Core throughput")
    throughput_axis.set_ylabel("TFLOP/s")
    latency_axis.set_title("Average kernel latency")
    latency_axis.set_ylabel("Milliseconds")
    latency_axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Week 05 WMMA precision comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_relative_throughput(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = rows_by_precision(rows)
    problems = [problem_label(row) for row in grouped[PRECISION_ORDER[0]]]
    tf32 = {
        problem_label(row): float(row["tflops"])
        for row in grouped["fp32_tf32"]
    }

    fig, axis = plt.subplots(figsize=(10, 5.4))
    for precision in PRECISION_ORDER[1:]:
        selected = grouped[precision]
        axis.plot(
            problems,
            [float(row["tflops"]) / tf32[problem_label(row)] for row in selected],
            marker="o",
            linewidth=2,
            color=COLORS[precision],
            label=LABELS[precision].replace("\n", " "),
        )

    axis.axhline(1.0, color=COLORS["fp32_tf32"], linestyle="--", label="TF32 baseline")
    axis.set_xlabel("GEMM shape (M x N x K)")
    axis.set_ylabel("Throughput relative to TF32")
    axis.set_title("Relative WMMA throughput")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Week 05 GEMM results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "precision_performance.png",
        args.output_dir / "relative_throughput.png",
    ]
    plot_performance(rows, outputs[0])
    plot_relative_throughput(rows, outputs[1])
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()