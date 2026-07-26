#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include "mat_transpose_kernel.cuh"

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
const int M = x.size(0);                                                   \
const int N = x.size(1);                                                   \
dim3 block(WARP_SIZE);                                                     \
dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));                 \
mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
reinterpret_cast<element_type *>(x.data_ptr()),                        \
reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
}

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
n_element_row, n_element_col)            \
void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
const int M = x.size(0);                                                   \
const int N = x.size(1);                                                   \
dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                      \
dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),           \
(M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));          \
mat_transpose_##tag##2d_kernel<<<grid, block>>>(                           \
reinterpret_cast<element_type *>(x.data_ptr()),             \
reinterpret_cast<element_type *>(y.data_ptr()), M, N);      \
}

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
// 2d index. easier for diagonal
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 4, 1)

// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// shared memory
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1,
                              4)

// shared memory with bcf
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float,
                              1, 4)

TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_merge_write_row2col,
                              torch::kFloat32, float, 4, 1)

// // CuTe implentations
// extern void mat_transpose_cute_col2row_reg(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row2col_reg(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_col_smem(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row_smem(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_col_smem_swizzled(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row_smem_swizzled(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row_cvectorized(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row_rvectorized(torch::Tensor, torch::Tensor);
// extern void mat_transpose_cute_row_cvectorized_swizzled(torch::Tensor,
//                                                         torch::Tensor);
// extern void mat_transpose_cute_row_rvectorized_swizzled(torch::Tensor,
//                                                         torch::Tensor);
// extern void
//     mat_transpose_cute_row_rvectorized_swizzled_optimized(torch::Tensor,
//                                                           torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)
  // shared memory optimize with bcf
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(
      mat_transpose_f32x4_shared_bcf_merge_write_row2col2d)
  // // CuTe implentations
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col2row_reg)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row2col_reg)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_col_smem_swizzled)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_smem_swizzled)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_cvectorized_swizzled)
  // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_cute_row_rvectorized_swizzled)
  // TORCH_BINDING_COMMON_EXTENSION(
  //     mat_transpose_cute_row_rvectorized_swizzled_optimized)
}
