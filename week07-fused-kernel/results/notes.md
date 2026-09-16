# Week 07 observations

Results analyzed from `raw/fusion_20260917_025510.csv` across 1, 32, and 256 rows
with widths of 256, 1024, and 4096. The correctness executable completed before the
benchmark in `run_week07.sh`, so these timings follow the CPU-reference checks.

## Summary

Fusion improved every measured case. The geometric-mean speedup across all 54 fused
measurements was **2.38x**. The benefit was largest when fusion removed several kernel
launches and intermediate tensors, and smallest when the fused kernel still had to make
the same number of passes over each row.

| Operation | Mean speedup | Measured range | Main explanation |
| --- | ---: | ---: | --- |
| bias + ReLU | 2.08x | 1.51-3.24x | Two elementwise launches become one; scratch traffic is removed. |
| bias + GELU | 2.10x | 1.74-3.16x | The GELU arithmetic does not erase the launch and traffic savings. |
| RMSNorm | 2.54x | 1.75-2.99x | Three staged launches become one and the squared-value buffer disappears. |
| LayerNorm | 1.63x | 1.11-2.00x | Fusion removes one launch, but both versions still read the input for statistics and normalization. |
| residual + RMSNorm | 3.23x | 2.24-4.00x | Four launches become one and two full-size intermediate passes are removed. |
| softmax | 3.51x | 2.50-4.00x | Four launches become one and scratch traffic is removed despite rereading input and recomputing exponentials. |

Softmax had the highest mean speedup, followed by residual + RMSNorm. LayerNorm was
consistently the least profitable fusion because this implementation reduces launch
overhead but does not reduce the number of global input passes.

## Scale effects

At widths 256 and 1024, the average speedup over all operations was 2.72x and 2.77x.
At width 4096 it fell to 2.05x. Wider rows make reduction and transcendental work a
larger fraction of runtime, so eliminating launches matters less.

This is clearest in LayerNorm: speedup falls to 1.11-1.25x for 4096 columns. RMSNorm
falls to 1.75-2.18x, while residual + RMSNorm remains at 2.24-2.60x because it removes
more intermediate work. Softmax remains strong at 2.50-3.09x for 4096 columns, so its
recomputation cost did not outweigh fusion in the measured range.

The smallest measurements are only a few microseconds and are dominated by launch
latency. Near-integer 2x, 3x, and 4x results should therefore be interpreted mainly as
launch-count effects, not proportional gains in arithmetic throughput. There is no
measured crossover where fusion loses; testing substantially wider rows is needed to
find one.

## Bandwidth interpretation

The CSV reports **effective** bandwidth from estimated algorithmic traffic, not DRAM
bytes measured by a profiler. Values such as 1.64 TB/s for fused residual + RMSNorm and
1.94 TB/s for fused softmax at `[256, 4096]` are useful for comparing these variants,
but they should not be presented as measured memory-controller throughput. Cache reuse,
repeated input reads, and the traffic model all affect this metric.

## Conclusions

1. Fusion is most valuable for chains that otherwise materialize full-size temporary
	 tensors and require several launches.
2. Elementwise fusion gives a dependable result near 2x once enough rows are available,
	 matching the reduction from two launches to one and removal of one temporary.
3. Reduction fusion can exceed the elementwise gain when it also removes preprocessing
	 and intermediate buffers, as shown by RMSNorm and residual + RMSNorm.
4. Fusion alone does not eliminate algorithmic passes. LayerNorm remains a two-pass
	 operation within one kernel, limiting its gain at large row widths.
5. Fused softmax is profitable throughout this sweep even though it recomputes
	 exponentials during output. A shared-memory or online-softmax variant is the next
	 useful comparison for wider rows.

## Follow-up measurements

- Record GPU model, driver, CUDA version, clock state, and benchmark iteration count
	alongside each CSV so results are reproducible.
- Use Nsight Compute to compare actual DRAM bytes, achieved occupancy, and memory
	throughput instead of relying only on the effective-bandwidth model.
- Add repeated benchmark trials with median and percentile reporting for microsecond
	kernels.
- Extend widths beyond 4096 to locate the softmax and normalization crossover points.
- Compare against optimized library or framework implementations before drawing claims
	about production performance.