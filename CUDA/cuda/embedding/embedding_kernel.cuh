#pragma once
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__global__ void embedding_f32_kernel(int* a, float* weight, float* o,int N,int emb_size);
__global__ void embedding_f32x4_kernel(int* a, float* weight, float* o,int N,int emb_size);
__global__ void embedding_f32x4_pack_kernel(int *a, float *weight, float *o,int N,int emb_size);
__global__ void embedding_f16_kernel(int* a, half* weight, half* o,int N,int emb_size);
__global__ void embedding_f16x8_kernel(int* a, half* weight, half* o,int N,int emb_size);
__global__ void embedding_f16x8_pack_kernel(int* a, half* weight, half* o,int N,int emb_size);