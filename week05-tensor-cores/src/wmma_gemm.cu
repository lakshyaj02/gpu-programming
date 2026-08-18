#include "common.cuh"

#include <mma.h>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int tileSize = 16;
constexpr int warpsPerBlock = 4;

template <typename StorageType, typename FragmentType, int tileK>
__global__ void wmmaGemmKernel(const StorageType* matrixA, const StorageType* matrixB,
							   float* matrixC, int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
	const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	const int tileColumns = n / tileSize;
	const int warpRow = warpId / tileColumns;
	const int warpCol = warpId % tileColumns;
	if (warpRow >= m / tileSize) {
		return;
	}

	wmma::fragment<wmma::matrix_a, tileSize, tileSize, tileK,
				   FragmentType, wmma::row_major> aFrag;
	wmma::fragment<wmma::matrix_b, tileSize, tileSize, tileK,
				   FragmentType, wmma::row_major> bFrag;
	wmma::fragment<wmma::accumulator, tileSize, tileSize, tileK, float> cFrag;
	wmma::fill_fragment(cFrag, 0.0f);

	for (int tileIdx = 0; tileIdx < k / tileK; ++tileIdx) {
		const int aRow = warpRow * tileSize;
		const int aCol = tileIdx * tileK;
		const int bRow = tileIdx * tileK;
		const int bCol = warpCol * tileSize;

		wmma::load_matrix_sync(aFrag, matrixA + aRow * k + aCol, k);
		wmma::load_matrix_sync(bFrag, matrixB + bRow * n + bCol, n);
		wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
	}

	const int cRow = warpRow * tileSize;
	const int cCol = warpCol * tileSize;
	wmma::store_matrix_sync(matrixC + cRow * n + cCol, cFrag, n, wmma::mem_row_major);
#endif
}

template <typename StorageType, typename FragmentType, int tileK>
void launch(const StorageType* matrixA, const StorageType* matrixB, float* matrixC,
			int m, int n, int k, cudaStream_t stream) {
	if (m % tileSize != 0 || n % tileSize != 0 || k % tileK != 0) {
		std::fprintf(stderr, "WMMA dimensions must be multiples of M=%d, N=%d, K=%d\n",
					 tileSize, tileSize, tileK);
		std::exit(EXIT_FAILURE);
	}

	const int totalWarps = (m / tileSize) * (n / tileSize);
	const int blockCount = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;
	wmmaGemmKernel<StorageType, FragmentType, tileK>
		<<<blockCount, warpsPerBlock * 32, 0, stream>>>(
			matrixA, matrixB, matrixC, m, n, k);
	CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launchWmmaGemm(const half* matrixA, const half* matrixB, float* matrixC,
					int m, int n, int k, cudaStream_t stream) {
	launch<half, half, 16>(matrixA, matrixB, matrixC, m, n, k, stream);
}

void launchWmmaTf32Gemm(const float* matrixA, const float* matrixB, float* matrixC,
						int m, int n, int k, cudaStream_t stream) {
	launch<float, wmma::precision::tf32, 8>(matrixA, matrixB, matrixC, m, n, k, stream);
}

void launchWmmaBf16Gemm(const __nv_bfloat16* matrixA, const __nv_bfloat16* matrixB,
						float* matrixC, int m, int n, int k, cudaStream_t stream) {
	launch<__nv_bfloat16, __nv_bfloat16, 16>(
		matrixA, matrixB, matrixC, m, n, k, stream);
}