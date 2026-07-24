# Week 2: Memory Hierarchy and Tiling

This week focuses on how memory access patterns and shared-memory tiling affect
effective bandwidth and kernel runtime on GPU workloads.

## Environment

- GPU: GB200 NVL72
- CUDA version: 12.8
- Compiler: nvcc 12.4

## Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Week 2 layout

- `include/`: shared declarations for Week 2 kernels/utilities
- `src/`: CUDA kernels and benchmark drivers
- `python/`: optional plotting/comparison scripts
- `results/raw_out/`: raw benchmark CSV outputs
- `results/plots_normal/`: clean plots for reports
- `results/plots_ragged/`: diagnostic/debug plots
- `results/observations.md`: final write-up

## Experiments

### 1) Memory access

Goal: measure cost of coalesced vs strided global memory reads/writes.

Implement:

- contiguous access kernel
- strided access kernels (for several strides)

Record:

- contiguous bandwidth (GB/s)
- strided bandwidth (GB/s)
- trend vs stride size

Suggested metrics table:

| stride | kernel_ms | bytes_moved | effective_gbps |
|---|---:|---:|---:|
| 1 |  |  |  |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |
| ... |  |  |  |

### 2) Matrix transpose

Goal: compare naive transpose with a coalesced/shared-memory transpose.

Implement:

- naive global-memory transpose
- tiled transpose using shared memory (with bank-conflict-safe padding)

Record:

- naive time (ms)
- coalesced/tiled time (ms)
- effective bandwidth (GB/s)
- speedup factor

Suggested summary:

| version | kernel_ms | effective_gbps | speedup |
|---|---:|---:|---:|
| naive |  |  | 1.00 |
| tiled/coalesced |  |  |  |

### 3) Matrix multiplication

Goal: compare arithmetic intensity and memory behavior of naive vs tiled GEMM.

Implement:

- CPU reference matmul (correctness baseline)
- naive CUDA matmul
- tiled CUDA matmul using shared memory

Record:

- CPU reference time (ms)
- naive CUDA time (ms)
- tiled CUDA time (ms)
- speedup vs naive and vs CPU
- max absolute error (or relative error) vs CPU reference

Suggested summary:

| version | time_ms | speedup_vs_cpu | speedup_vs_naive | max_abs_error |
|---|---:|---:|---:|---:|
| cpu_reference |  | 1.00 | - | - |
| cuda_naive |  |  | 1.00 |  |
| cuda_tiled |  |  |  |  |

## Reproducibility checklist

- Fix problem sizes and iteration count before comparisons
- Use warm-up launches before timing
- Report block size and tile size
- Synchronize before/after timed regions
- Store raw output in `results/raw_out/`
- Keep plots and write-up in `results/`

## Main observations template

Use this in `results/observations.md`:

1. Which access pattern most impacted bandwidth, and why?
2. How much did transpose coalescing improve throughput?
3. When did tiled GEMM outperform naive most strongly?
