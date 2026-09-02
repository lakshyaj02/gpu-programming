# Week 06 normalization observations

Results analyzed: `20260903_040849`. This run was collected after removing
`cudaDeviceSynchronize()` from the kernel launch wrappers. All CUDA and PyTorch
rows passed their precision-specific correctness checks.

## RMSNorm performance

- The warp reduction is increasingly better than the naive kernel as the hidden
	size grows. Its speedup rises from 2.23x at 768 to 6.79x at 8192. The naive
	implementation scales poorly because every output thread independently scans
	the entire input vector.
- `float4` provides little benefit at 1024 (1.03x over scalar warp RMSNorm), but
	the gain grows with the amount of data per launch: 1.31x at 2048, 1.49x at
	4096, and 1.94x at 8192.
- At hidden size 4096, `float4` takes 0.006960 ms versus 0.010378 ms for the
	scalar warp kernel. At 8192, it takes 0.008378 ms versus 0.016288 ms.
- `float4` is the fastest custom kernel at the two largest sizes. Its reported
	effective bandwidth reaches 9.416 GB/s at 4096 and 15.646 GB/s at 8192.
- PyTorch FP32 RMSNorm takes 0.047163 ms at 4096 and 0.047541 ms at 8192.
	The custom `float4` kernel is therefore 6.78x and 5.67x faster, respectively,
	for this single-vector workload.

## Half precision and `half2`

- `half2` has almost no effect at 768 and is 8.5% slower at 1024, where launch
	overhead and run-to-run noise dominate this very small workload.
- The packed path becomes useful at larger sizes. It is 1.29x faster at 2048,
	1.30x at 4096, and 1.53x at 8192 compared with scalar FP16.
- At 8192, `half2` reaches 6.014 GB/s versus 3.929 GB/s for scalar FP16 and runs
	in 0.010898 ms. It is 4.00x faster than PyTorch FP16 RMSNorm at that size.
- `half2` and scalar FP16 have essentially identical numerical error because
	both accumulate the reduction and perform normalization arithmetic in FP32
	before storing FP16 results. Their worst maximum absolute error is 2.26e-3,
	and RMSE stays near 3.7e-4.
- Large maximum relative errors for FP16, up to 0.464 in the custom kernels,
	occur near reference values close to zero. Absolute error and RMSE are more
	informative here, and every row passes the combined absolute/relative test.

## LayerNorm

- The current LayerNorm kernels are correctness baselines, not optimized
	reductions. Every output thread scans the complete row, producing quadratic
	work in the hidden size.
- The two-pass LayerNorm grows from 0.020298 ms at 768 to 0.165280 ms at 8192.
	Welford grows from 0.051486 ms to 0.512064 ms and is 2.54x to 3.10x slower
	than the two-pass implementation.
- PyTorch LayerNorm remains approximately flat around 0.026 ms over these
	sizes. At 8192, the custom two-pass implementation is 6.42x slower than
	PyTorch. A useful next implementation is one block per row with cooperative
	sum/variance or cooperative Welford reduction.
- Welford is somewhat more accurate than the two-pass kernel at larger sizes,
	but the improvement is small compared with its current performance cost.

## Fused residual RMSNorm

- Fused residual RMSNorm takes 0.010766 ms at 4096 and 0.016453 ms at 8192. It
	is approximately equal to scalar warp RMSNorm even though it performs an
	additional residual load and addition.
- The fused kernel is slower than `float4` RMSNorm by 1.55x at 4096 and 1.96x
	at 8192. This is not a direct fusion penalty because the two kernels perform
	different operations and the fused implementation is not vectorized.
- Compared with matching PyTorch fused residual RMSNorm, the custom fused
	kernel is 4.71x faster at 4096 and 2.90x faster at 8192.
- The experiment does not yet establish the speedup from fusion. That requires
	an unfused baseline consisting of a residual-add kernel followed by RMSNorm
	on the intermediate buffer.

## Precision

- All implementations pass: FP32 uses `atol=rtol=1e-4`, while FP16 uses
	`atol=rtol=5e-3`.
- Warp and `float4` RMSNorm match each other and have maximum absolute error no
	larger than 2.38e-7. Fused RMSNorm remains similarly accurate, with RMSE below
	9.34e-8.
- The largest FP32 absolute error is 2.50e-6 from the two-pass LayerNorm at
	hidden size 4096, still comfortably inside tolerance.

## Interpretation limits

- Only block size 256 was tested, so these results cannot show whether a larger
	or smaller block improves performance.
- Each benchmark processes one vector with one cooperative block. This is a
	latency experiment and does not provide enough parallel work to saturate a
	modern GPU. Batched rows are needed for a meaningful peak-throughput test.
- `bandwidth_gbps` is calculated from an estimated number of logical array
	transfers divided by kernel time. It is not a hardware DRAM counter and does
	not include cache effects or redundant traffic accurately.
- The maximum reported custom bandwidth is 15.646 GB/s, but the fraction of
	peak GPU bandwidth cannot be calculated without the GPU model or measured
	peak bandwidth.
- No Nsight Compute report is included in the downloaded results. Therefore
	coalescing efficiency, achieved occupancy, active warps per SM, DRAM
	utilization, shared-memory effects, and the compute-bound versus
	bandwidth-bound classification at 4096 cannot be confirmed from hardware
	counters.
