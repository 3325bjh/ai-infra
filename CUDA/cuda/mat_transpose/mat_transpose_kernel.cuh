#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__global__ void mat_transpose_f32_col2row_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32_row2col_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32x4_col2row_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32x4_row2col_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32_col2row2d_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y,
                                                   int row,
                                                   int col);

__global__ void mat_transpose_f32_diagonal2d_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_ref_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            int row,
                                                            int col);

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x,float *y,int M,int N);

__global__ void mat_transpose_ref_f32x4_shared_bcf_col2row2d_kernel(float *x,
                                                                float *y,
                                                                 int row,
                                                                 int col);

__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel(float *x,float *y,int M,int N);

