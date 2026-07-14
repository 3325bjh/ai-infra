#include "elementwise_kernel.cuh"
#define WARP_SIZE 32
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&(val))[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val)))[0]


__global__ void elementwise_add_f32_kernel(
    float *a,
    float *b,
    float *c,
    int n
){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<n){
        c[idx]=a[idx]+b[idx];
    }
}

__global__ void elementwise_add_f32x4_kernel(
    float *a,
    float *b,
    float *c,
    int n
){
    int idx=4*(blockDim.x*blockIdx.x+threadIdx.x);
    if(idx+3<n){
        float4 reg_a=FLOAT4(a[idx]);
        float4 reg_b=FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x=reg_a.x+reg_b.x;
        reg_c.y=reg_a.y+reg_b.y;
        reg_c.w=reg_a.w+reg_b.w;
        reg_c.z=reg_a.z+reg_b.z;
        FLOAT4(c[idx])=reg_c;
    }else{
        for(int i=0;idx+i<n;i++){
            c[idx+i]=a[idx+i]+b[idx+i];
        }
    }
}

__global__  void elementwise_add_f16_kernel(
    half *a,
    half *b,
    half *c,
    int n
){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<n){
        c[idx]=__hadd(a[idx],b[idx]);
    }
}

__global__ void elementwise_add_f16x2_kernel(
    half *a,
    half *b,
    half *c,
    int n
){
    int idx=2*(blockDim.x*blockIdx.x+threadIdx.x);
    if(idx+1<n){
        half2 seg_a=HALF2(a[idx]);
        half2 seg_b=HALF2(b[idx]);
        half2 seg_c;
        seg_c.x=__hadd(seg_a.x,seg_b.x);
        seg_c.y=__hadd(seg_a.y,seg_b.y);
        HALF2(c[idx])=seg_c;
    }else{
        c[idx]=__hadd(a[idx],b[idx]);
    }
}

__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int n) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if ((idx + 7) < n) {
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    HALF2(c[idx + 0]) = reg_c_0;
    HALF2(c[idx + 2]) = reg_c_1;
    HALF2(c[idx + 4]) = reg_c_2;
    HALF2(c[idx + 6]) = reg_c_3;
  } else if (idx < n) {
    for (int i = 0; (idx + i) < n; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

__global__ void elementwise_add_f16x8_pack_kernel(
    half *a,
    half *b,
    half *c,
    int n
){
    int idx=8*(blockDim.x*blockIdx.x+threadIdx.x);
    if(idx+7<n){
        half pack_a[8],pack_b[8],pack_c[8];
        LDST128BITS(pack_a)=LDST128BITS(a[idx]);
        LDST128BITS(pack_b)=LDST128BITS(b[idx]);
#pragma unroll
        for(int i=0;i<8;i+=2){
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
        }
        LDST128BITS(c[idx])=LDST128BITS(pack_c[0]);
    }else if (idx<n)
    {
        for(int i=0;i+idx<n;i++){
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
    
}