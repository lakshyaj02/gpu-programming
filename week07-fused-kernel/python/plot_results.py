#!/usr/bin/env python3

"""Plot fused-kernel speedups from the Week 07 benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == "fused"]
    if not rows:
        raise SystemExit(f"No fused benchmark rows found in {args.csv}")

    import matplotlib.pyplot as plt

    operations = list(dict.fromkeys(row["operation"] for row in rows))
    colors = ["#00798c", "#d1495b", "#2a9d8f", "#edae49", "#6f4e7c", "#59a14f"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, operation, color in zip(axes.flat, operations, colors):
        selected = [row for row in rows if row["operation"] == operation]
        for token_count in sorted({int(row["rows"]) for row in selected}):
            series = sorted(
                (row for row in selected if int(row["rows"]) == token_count),
                key=lambda row: int(row["columns"]),
            )
            axis.plot(
                [int(row["columns"]) for row in series],
                [float(row["speedup_vs_unfused"]) for row in series],
                marker="o",
                label=f"{token_count} rows",
                color=color,
                alpha=0.45 + 0.25 * sorted({int(row["rows"]) for row in selected}).index(token_count),
            )
        axis.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_title(operation.replace("_", " ").title())
        axis.set_ylabel("Speedup over unfused")
        axis.grid(True, linestyle="--", alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Columns")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Week 07: when kernel fusion helps")
    fig.tight_layout()
    output = args.output or args.csv.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()