import argparse
import csv
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "build" / "week01_benchmark"
DEFAULT_CSV = PROJECT_ROOT / "week01-vector-ops" / "results" / "raw" / "cuda_timings.csv"
DEFAULT_PYTORCH_CSV = (
    PROJECT_ROOT / "week01-vector-ops" / "results" / "raw" / "pytorch_timings.csv"
)
DEFAULT_PLOT_DIR = PROJECT_ROOT / "week01-vector-ops" / "results" / "plots"
FIELDNAMES = [
    "n",
    "block_size",
    "iteration",
    "kernel_ms",
    "total_ms",
    "correct",
]


def run_benchmark(executable, n, block_size, iterations):
    command = [str(executable), str(n), str(block_size), str(iterations)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split(",")
        if len(values) != len(FIELDNAMES):
            raise ValueError(f"Unexpected benchmark output: {line!r}")
        rows.append(
            {
                "n": int(values[0]),
                "block_size": int(values[1]),
                "iteration": int(values[2]),
                "kernel_ms": float(values[3]),
                "total_ms": float(values[4]),
                "correct": values[5],
            }
        )

    if len(rows) != iterations:
        raise ValueError(
            f"Expected {iterations} samples for n={n}, block_size={block_size}; "
            f"received {len(rows)}"
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["block_size"])].append(row["kernel_ms"])

    summaries = []
    for (n, block_size), timings in sorted(grouped.items()):
        mean_ms = statistics.fmean(timings)
        summaries.append(
            {
                "n": n,
                "block_size": block_size,
                "mean_ms": mean_ms,
                "stdev_ms": statistics.pstdev(timings),
                "bandwidth_gbps": 3 * n * 4 / (mean_ms * 1e-3) / 1e9,
            }
        )
    return summaries


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_pytorch_benchmarks(sizes, iterations, warmup):
    try:
        import benchmark_pytorch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                "PyTorch is not installed; install a CUDA-enabled PyTorch build or "
                "use --skip-pytorch"
            ) from error
        raise

    if not benchmark_pytorch.torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA device; use --skip-pytorch to continue")

    rows = []
    device = benchmark_pytorch.torch.device("cuda")
    for n in sizes:
        timings_ms, correct = benchmark_pytorch.benchmark(
            n, iterations, warmup, device
        )
        rows.extend(
            {
                "n": n,
                "iteration": iteration,
                "kernel_ms": kernel_ms,
                "correct": "PASS" if correct else "FAIL",
            }
            for iteration, kernel_ms in enumerate(timings_ms)
        )
        print(
            f"PyTorch: n={n}, mean={statistics.fmean(timings_ms):.6f} ms, "
            f"correct={rows[-1]['correct']}"
        )
    return rows


def summarize_pytorch(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["n"]].append(row["kernel_ms"])

    summaries = []
    for n, timings in sorted(grouped.items()):
        mean_ms = statistics.fmean(timings)
        summaries.append(
            {
                "n": n,
                "mean_ms": mean_ms,
                "stdev_ms": statistics.pstdev(timings),
                "bandwidth_gbps": 3 * n * 4 / (mean_ms * 1e-3) / 1e9,
            }
        )
    return summaries


def write_pytorch_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["n", "iteration", "kernel_ms", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)


def create_plots(summaries, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV was written but plots were skipped")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    block_sizes = sorted({row["block_size"] for row in summaries})

    fig, ax = plt.subplots(figsize=(8, 5))
    for block_size in block_sizes:
        series = [row for row in summaries if row["block_size"] == block_size]
        ns = [row["n"] for row in series]
        means = [row["mean_ms"] for row in series]
        lower = [max(0.0, row["mean_ms"] - row["stdev_ms"]) for row in series]
        upper = [row["mean_ms"] + row["stdev_ms"] for row in series]
        ax.plot(ns, means, marker="o", label=f"{block_size} threads")
        ax.fill_between(ns, lower, upper, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Kernel time (ms)")
    ax.set_title("CUDA Vector Add Latency")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "cuda_latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for block_size in block_sizes:
        series = [row for row in summaries if row["block_size"] == block_size]
        ax.plot(
            [row["n"] for row in series],
            [row["bandwidth_gbps"] for row in series],
            marker="o",
            label=f"{block_size} threads",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("CUDA Vector Add Effective Bandwidth")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "cuda_bandwidth.png", dpi=150)
    plt.close(fig)


def create_comparison_plots(cuda_summaries, pytorch_summaries, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    block_sizes = sorted({row["block_size"] for row in cuda_summaries})

    for metric, ylabel, filename, log_y in [
        ("mean_ms", "Kernel time (ms)", "comparison_latency.png", True),
        (
            "bandwidth_gbps",
            "Effective bandwidth (GB/s)",
            "comparison_bandwidth.png",
            False,
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for block_size in block_sizes:
            series = [
                row for row in cuda_summaries if row["block_size"] == block_size
            ]
            ax.plot(
                [row["n"] for row in series],
                [row[metric] for row in series],
                marker="o",
                label=f"CUDA ({block_size} threads)",
            )
        ax.plot(
            [row["n"] for row in pytorch_summaries],
            [row[metric] for row in pytorch_summaries],
            marker="s",
            linewidth=2.5,
            label="PyTorch",
        )
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Number of elements")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Native CUDA vs PyTorch {ylabel}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Publish CUDA vector-add timings")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1 << exponent for exponent in range(10, 25)]
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pytorch-csv", type=Path, default=DEFAULT_PYTORCH_CSV)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--skip-pytorch", action="store_true")
    args = parser.parse_args()

    if args.iterations <= 0 or args.warmup < 0 or any(size <= 0 for size in args.sizes):
        parser.error("iterations and sizes must be positive; warmup cannot be negative")
    if any(block_size <= 0 for block_size in args.block_sizes):
        parser.error("block sizes must be positive")
    if not args.executable.is_file():
        parser.error(f"benchmark executable not found: {args.executable}")

    rows = []
    for n in args.sizes:
        for block_size in args.block_sizes:
            samples = run_benchmark(args.executable, n, block_size, args.iterations)
            rows.extend(samples)
            mean_ms = statistics.fmean(row["kernel_ms"] for row in samples)
            print(
                f"n={n}, block_size={block_size}, mean={mean_ms:.6f} ms, "
                f"correct={samples[0]['correct']}"
            )

    failed = [row for row in rows if row["correct"] != "PASS"]
    write_csv(rows, args.csv)
    create_plots(summarize(rows), args.plot_dir)
    print(f"Wrote {args.csv}")
    print(f"Wrote plots to {args.plot_dir}")

    if not args.skip_pytorch:
        try:
            pytorch_rows = run_pytorch_benchmarks(
                args.sizes, args.iterations, args.warmup
            )
        except RuntimeError as error:
            parser.error(str(error))
        write_pytorch_csv(pytorch_rows, args.pytorch_csv)
        create_comparison_plots(
            summarize(rows), summarize_pytorch(pytorch_rows), args.plot_dir
        )
        print(f"Wrote {args.pytorch_csv}")
        print(f"Wrote comparison plots to {args.plot_dir}")
        failed.extend(row for row in pytorch_rows if row["correct"] != "PASS")

    if failed:
        raise SystemExit(f"Correctness failed for {len(failed)} timing samples")


if __name__ == "__main__":
    main()
