#include "embedding_kernel.cuh"
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&val)[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&val)[0])

__global__ void embedding_f32_kernel(int* a, float* weight, float* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x;
    int idx=a[row];
    o[row*emb_size+col]=weight[idx*emb_size+col];
}
__global__ void embedding_f32x4_kernel(int* a, float* weight, float* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x*4;
    int idx=a[row];
    o[row*emb_size+col]=weight[idx*emb_size+col];
    o[row*emb_size+col+1]=weight[idx*emb_size+col+1];
    o[row*emb_size+col+2]=weight[idx*emb_size+col+2];
    o[row*emb_size+col+3]=weight[idx*emb_size+col+3];
}

__global__ void embedding_f32x4_pack_kernel(int* a, float* weight, float* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x*4;
    int idx=a[row];
    FLOAT4(o[row*emb_size+col])=FLOAT4(weight[idx*emb_size+col]);
}

__global__ void embedding_f16_kernel(int* a, half* weight, half* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x;
    int idx=a[row];
    o[row*emb_size+col]=weight[idx*emb_size+col];
}
__global__ void embedding_f16x8_kernel(int* a, half* weight, half* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x*8;
    int idx=a[row];
    o[row*emb_size+col]=weight[idx*emb_size+col];
    o[row*emb_size+col+1]=weight[idx*emb_size+col+1];
    o[row*emb_size+col+2]=weight[idx*emb_size+col+2];
    o[row*emb_size+col+3]=weight[idx*emb_size+col+3];
    o[row*emb_size+col+4]=weight[idx*emb_size+col+4];
    o[row*emb_size+col+5]=weight[idx*emb_size+col+5];
    o[row*emb_size+col+6]=weight[idx*emb_size+col+6];
    o[row*emb_size+col+7]=weight[idx*emb_size+col+7];
}
__global__ void embedding_f16x8_pack_kernel(int* a, half* weight, half* o,int N,int emb_size) {
    int row=blockIdx.x;
    int col=threadIdx.x*8;
    int idx=a[row];
    LDST128BITS(o[row*emb_size+col])=LDST128BITS(weight[idx*emb_size+col]);
}

