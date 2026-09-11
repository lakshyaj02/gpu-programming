# Week 07: Fused Kernels

This week compares deliberately staged CUDA baselines with fused kernels. Tensors are
row-major `[rows, columns]`; normalization and softmax operate independently per row.

## Main assignments

| Assignment | Kernel | Main concept |
| --- | --- | --- |
| 7.1 | bias + ReLU | Basic elementwise fusion |
| 7.2 | bias + GELU | Realistic activation fusion |
| 7.3 | RMSNorm | Reduction + normalization fusion |
| 7.4 | LayerNorm | Multiple reductions + elementwise work |
| 7.5 | residual + RMSNorm | Transformer-style fusion |
| 7.6 | fused softmax | Reduction + exponentiation + normalization |
| 7.7 | benchmark suite | Quantify when fusion helps |

The public launch API is in `include/fused_kernels.cuh`. Each assignment has its own
source file under `src`, with staged and fused paths next to each other. Shared launch
checks and block reductions live in `src/kernel_utils.cuh`. The fused reduction kernels
assign one block to each row and use warp shuffles plus shared memory for block-wide
reductions.

## Build and verify

```bash
cmake -S week07-fused-kernel -B build/week07-fused-kernel -DCMAKE_BUILD_TYPE=Release
cmake --build build/week07-fused-kernel -j
./build/week07-fused-kernel/week07_fused_test
```

The test uses 513 columns so the kernels are checked on a width that is not divisible
by the block size.

## Benchmark

```bash
./scripts/run_week07.sh 100 10
```

The suite tests 1, 32, and 256 rows at widths 256, 1024, and 4096. It writes CSV data
under `results/raw` and a speedup plot under `results/plots`. Effective bandwidth is
based on the algorithmic global-memory traffic of each implementation. A speedup below
1 is useful data: fusion can lose when recomputation or limited occupancy costs more
than the eliminated launches and intermediate traffic.

## Experiments

1. Profile staged and fused paths with Nsight Compute and compare DRAM bytes.
2. Replace repeated softmax exponentiation with shared-memory caching and find the width
   where the extra shared memory starts reducing occupancy.
3. Add `half2` inputs with FP32 accumulation.
4. Sweep row widths beyond 4096 and compare one-block-per-row against multi-block rows.