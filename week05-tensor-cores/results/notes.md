# Results Notes

Run `scripts/run_benchmarks.sh` on a Tensor Core capable NVIDIA GPU, then record observations here.

The `execution` CSV column distinguishes native TF32/BF16 WMMA from FP8/FP4
value emulation through FP16 WMMA. Do not interpret emulated rows as native
low-precision Tensor Core throughput.