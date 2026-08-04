#!/usr/bin/env python3

"""Run Week 2 native CUDA benchmarks and write structured CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "week02-memory-tiling" / "build" / "week02_benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week02-memory-tiling" / "results" / "raw_out"

COPY_HEADER = "copy_pattern,size,bytes,stride,iterations,avg_kernel_ms,min_kernel_ms,effective_gbps,correct"
TRANSPOSE_HEADER = "transpose_variant,rows,cols,bytes,iterations,avg_kernel_ms,min_kernel_ms,effective_gbps,correct"
MATMUL_HEADER = "matmul_variant,M,N,K,iterations,avg_kernel_ms,min_kernel_ms,gflops,correct"


@dataclass
class ParsedRows:
    copy_rows: list[dict[str, str]]
    transpose_rows: list[dict[str, str]]
    matmul_rows: list[dict[str, str]]


def parse_output(stdout: str) -> ParsedRows:
    section = ""
    copy_rows: list[dict[str, str]] = []
    transpose_rows: list[dict[str, str]] = []
    matmul_rows: list[dict[str, str]] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == COPY_HEADER:
            section = "copy"
            continue
        if line == TRANSPOSE_HEADER:
            section = "transpose"
            continue
        if line == MATMUL_HEADER:
            section = "matmul"
            continue

        values = line.split(",")
        if section == "copy":
            if len(values) != 9:
                raise ValueError(f"Unexpected copy row: {line!r}")
            copy_rows.append(
                {
                    "copy_pattern": values[0],
                    "size": values[1],
                    "bytes": values[2],
                    "stride": values[3],
                    "iterations": values[4],
                    "avg_kernel_ms": values[5],
                    "min_kernel_ms": values[6],
                    "effective_gbps": values[7],
                    "correct": values[8],
                }
            )
        elif section == "transpose":
            if len(values) != 9:
                raise ValueError(f"Unexpected transpose row: {line!r}")
            transpose_rows.append(
                {
                    "transpose_variant": values[0],
                    "rows": values[1],
                    "cols": values[2],
                    "bytes": values[3],
                    "iterations": values[4],
                    "avg_kernel_ms": values[5],
                    "min_kernel_ms": values[6],
                    "effective_gbps": values[7],
                    "correct": values[8],
                }
            )
        elif section == "matmul":
            if len(values) != 9:
                raise ValueError(f"Unexpected matmul row: {line!r}")
            matmul_rows.append(
                {
                    "matmul_variant": values[0],
                    "M": values[1],
                    "N": values[2],
                    "K": values[3],
                    "iterations": values[4],
                    "avg_kernel_ms": values[5],
                    "min_kernel_ms": values[6],
                    "gflops": values[7],
                    "correct": values[8],
                }
            )
        else:
            raise ValueError(f"Unexpected output before section header: {line!r}")

    return ParsedRows(copy_rows=copy_rows, transpose_rows=transpose_rows, matmul_rows=matmul_rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_combined_rows(parsed: ParsedRows) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in parsed.copy_rows:
        rows.append({"section": "copy", **row})
    for row in parsed.transpose_rows:
        rows.append({"section": "transpose", **row})
    for row in parsed.matmul_rows:
        rows.append({"section": "matmul", **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 2 native CUDA benchmarks")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if args.iterations == 0 or args.warmup < 0:
        raise SystemExit("--iterations must be positive or -1, and --warmup must be >= 0")

    command = [str(args.executable), str(args.iterations), str(args.warmup)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    parsed = parse_output(result.stdout)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_path = output_dir / f"cuda_timings_copy_{args.timestamp}.csv"
    transpose_path = output_dir / f"cuda_timings_transpose_{args.timestamp}.csv"
    matmul_path = output_dir / f"cuda_timings_matmul_{args.timestamp}.csv"
    combined_path = output_dir / f"cuda_timings_all_{args.timestamp}.csv"
    metadata_path = output_dir / f"cuda_run_metadata_{args.timestamp}.json"

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
        parsed.copy_rows,
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
        parsed.transpose_rows,
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
        parsed.matmul_rows,
    )

    combined_rows = build_combined_rows(parsed)
    combined_fieldnames = [
        "section",
        "copy_pattern",
        "size",
        "bytes",
        "stride",
        "transpose_variant",
        "rows",
        "cols",
        "matmul_variant",
        "M",
        "N",
        "K",
        "iterations",
        "avg_kernel_ms",
        "min_kernel_ms",
        "effective_gbps",
        "gflops",
        "correct",
    ]

    normalized_rows: list[dict[str, str]] = []
    for row in combined_rows:
        normalized = {key: "" for key in combined_fieldnames}
        normalized.update(row)
        normalized_rows.append(normalized)
    write_csv(combined_path, combined_fieldnames, normalized_rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "executable": str(args.executable),
        "rows": {
            "copy": len(parsed.copy_rows),
            "transpose": len(parsed.transpose_rows),
            "matmul": len(parsed.matmul_rows),
        },
        "outputs": {
            "copy_csv": str(copy_path),
            "transpose_csv": str(transpose_path),
            "matmul_csv": str(matmul_path),
            "combined_csv": str(combined_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {copy_path}")
    print(f"Wrote {transpose_path}")
    print(f"Wrote {matmul_path}")
    print(f"Wrote {combined_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
