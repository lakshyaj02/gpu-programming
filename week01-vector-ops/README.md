## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/week01_benchmark
```

The executable prints one CSV-formatted row per measured launch:

```text
n,block_size,iteration,kernel_ms,total_ms,correct
```

## Publish benchmark results

Install PyTorch with CUDA support and Matplotlib, then build the executable,
run native CUDA and PyTorch over the same input sizes, write raw per-launch
timings, and generate latency and bandwidth comparison plots with:

```bash
./scripts/run_week01.sh
```

The generated artifacts are:

- `results/raw/cuda_timings.csv`
- `results/raw/pytorch_timings.csv`
- `results/plots/cuda_latency.png`
- `results/plots/cuda_bandwidth.png`
- `results/plots/comparison_latency.png`
- `results/plots/comparison_bandwidth.png`

To run a smaller benchmark or choose specific launch configurations:

```bash
./scripts/run_week01.sh --sizes 1048576 4194304 --block-sizes 128 256 --iterations 50
```

Use `--skip-pytorch` to publish only the native CUDA results when PyTorch is
not available.