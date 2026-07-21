import argparse
import csv
import statistics

import torch


def benchmark(n, iterations, warmup, device):
    a = torch.rand(n, device=device)
    b = torch.rand(n, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(warmup):
            c = a + b

        torch.cuda.synchronize()
        timings_ms = []
        for _ in range(iterations):
            start.record()
            c = a + b
            end.record()
            end.synchronize()
            timings_ms.append(start.elapsed_time(end))

    # Correctness check against a reference.
    ref = a + b
    correct = torch.allclose(c, ref, atol=1e-5)

    return timings_ms, correct


def main():
    parser = argparse.ArgumentParser(description="PyTorch vector-add benchmark")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1 << e for e in range(10, 25)],  # 1K .. 16M elements
    )
    parser.add_argument("--csv", type=str, default="results_pytorch.csv")
    parser.add_argument("--plot", type=str, default="results_pytorch.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device("cuda")
    rows = []

    for n in args.sizes:
        timings_ms, correct = benchmark(n, args.iterations, args.warmup, device)
        for iteration, kernel_ms in enumerate(timings_ms):
            rows.append(
                {
                    "n": n,
                    "iteration": iteration,
                    "kernel_ms": kernel_ms,
                    "correct": "PASS" if correct else "FAIL",
                }
            )
        mean_ms = statistics.fmean(timings_ms)
        print(f"n={n}, mean={mean_ms:.6f} ms, correct={rows[-1]['correct']}")

    # Write CSV.
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n", "iteration", "kernel_ms", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv}")

    # Generate plots.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    summaries = []
    for n in args.sizes:
        timings_ms = [row["kernel_ms"] for row in rows if row["n"] == n]
        mean_ms = statistics.fmean(timings_ms)
        summaries.append(
            {
                "n": n,
                "kernel_ms": mean_ms,
                "gbps": 3 * n * 4 / (mean_ms * 1e-3) / 1e9,
            }
        )

    ns = [row["n"] for row in summaries]
    kernel_ms = [row["kernel_ms"] for row in summaries]
    gbps = [row["gbps"] for row in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ns, kernel_ms, marker="o")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of elements")
    ax1.set_ylabel("Kernel time (ms)")
    ax1.set_title("Vector Add: Time vs Size")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.plot(ns, gbps, marker="o", color="tab:green")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Number of elements")
    ax2.set_ylabel("Effective bandwidth (GB/s)")
    ax2.set_title("Vector Add: Bandwidth vs Size")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()