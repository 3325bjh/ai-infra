#pragma once
#include <cuda_runtime.h>
#include<cuda_fp16.h>

__global__  void elementwise_add_f32_kernel(
    float *a,
    float *b,
    float *c,
    int n
);

__global__  void elementwise_add_f32x4_kernel(
    float *a,
    float *b,
    float *c,
    int n
);

__global__  void elementwise_add_f16_kernel(
    half *a,
    half *b,
    half *c,
    int n
);

__global__ void elementwise_add_f16_kernel(
    half *a,
    half *b,
    half *c,
    int n
);

__global__ void elementwise_add_f16x2_kernel(
    half *a,
    half *b,
    half *c,
    int n
);

__global__ void elementwise_add_f16x8_kernel(
    half *a,
    half *b,
    half *c,
    int n
);

__global__ void elementwise_add_f16x8_pack_kernel(
    half *a,
    half *b,
    half *c,
    int n
);