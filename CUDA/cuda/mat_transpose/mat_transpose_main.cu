#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mat_transpose_kernel.cuh"

namespace {

constexpr int kTileRows = 16;
constexpr int kTileCols = 64;
constexpr int kWarmupIters = 10;
constexpr int kMeasureIters = 100;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error__ = (call);                                         \
    if (error__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error__));                                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

using Kernel = void (*)(float*, float*, int, int);

void fill_input(std::vector<float>& x, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // A non-symmetric pattern makes an accidental no-op easy to detect.
      x[row * cols + col] = static_cast<float>(row * cols + col) + 0.25f;
    }
  }
}

bool verify_transpose(const std::vector<float>& x, const std::vector<float>& y,
                      int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float expected = x[row * cols + col];
      const float actual = y[col * rows + row];
      if (std::fabs(expected - actual) > 1e-6f) {
        std::fprintf(stderr,
                     "mismatch at y[%d][%d]: expected %.8f, got %.8f\\n", col,
                     row, expected, actual);
        return false;
      }
    }
  }
  return true;
}

bool run_test(const char* name, Kernel kernel, const std::vector<float>& h_x,
              float* d_x, float* d_y, int rows, int cols) {
  const dim3 block(kTileRows, kTileRows);
  const dim3 grid(cols / kTileCols, rows / kTileRows);
  const size_t output_bytes =
      static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);

  CUDA_CHECK(cudaMemset(d_y, 0, output_bytes));
  for (int iter = 0; iter < kWarmupIters; ++iter) {
    kernel<<<grid, block>>>(d_x, d_y, rows, cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < kMeasureIters; ++iter) {
    kernel<<<grid, block>>>(d_x, d_y, rows, cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> h_y(static_cast<size_t>(rows) * cols);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, output_bytes, cudaMemcpyDeviceToHost));
  const bool passed = verify_transpose(h_x, h_y, rows, cols);
  std::printf("%-58s %s  %.4f ms/iter\\n", name,
              passed ? "PASS" : "FAIL", elapsed_ms / kMeasureIters);
  return passed;
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 512;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 256;
  if (rows <= 0 || cols <= 0 || rows % kTileRows != 0 ||
      cols % kTileCols != 0) {
    std::fprintf(stderr,
                 "Usage: %s [rows multiple of %d] [cols multiple of %d]\\n",
                 argv[0], kTileRows, kTileCols);
    return EXIT_FAILURE;
  }

  const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
  std::vector<float> h_x(static_cast<size_t>(rows) * cols);
  fill_input(h_x, rows, cols);

  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  std::printf("Testing transpose of %d x %d matrix (block: 16 x 16)\\n", rows,
              cols);
  bool passed = true;
  passed &= run_test("mat_transpose_f32x4_shared_col2row",
                     mat_transpose_f32x4_shared_col2row, h_x, d_x, d_y, rows,
                     cols);
  passed &= run_test("mat_transpose_ref_f32x4_shared_col2row2d_kernel",
                     mat_transpose_ref_f32x4_shared_col2row2d_kernel, h_x,
                     d_x, d_y, rows, cols);
  passed &= run_test("mat_transpose_f32x4_shared_bcf_col2row",
                     mat_transpose_f32x4_shared_bcf_col2row_kernel, h_x, d_x, d_y,
                     rows, cols);
  passed &= run_test("mat_transpose_ref_f32x4_shared_bcf_col2row2d_kernel",
                     mat_transpose_ref_f32x4_shared_bcf_col2row2d_kernel, h_x,
                     d_x, d_y, rows, cols);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
