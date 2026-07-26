#include "elu_kernel.cuh"
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&val)[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&val)[0])
#define ALPHA 1.0f


__device__  __forceinline__  float elu(float x) {
    return x>0.f?x:ALPHA*(expf(x)-1.f);
}

__device__ __forceinline__  half elu_half(half x) {
    return __hgt(x,__float2half(0.0f))?x:__hmul(__float2half(ALPHA), __hsub(hexp(x),__float2half(1.0f)));
}

// FP32
// elu x: N, y: N y=x>0?x:alpha*(exp(x)-1)
// grid(N/256), block(K=256)
__global__ void elu_f32_kernel(float *x, float *y, int N) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if (index<N) {
        y[index]=elu(x[index]);
    }
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void elu_f32x4_kernel(float *x, float *y, int N) {
    int index=(blockDim.x*blockIdx.x+threadIdx.x)*4;
    if (index+3<N) {
        float4 reg1;
        float4 reg2;
        reg1=FLOAT4(x[index]);
        reg2.x=elu(reg1.x);
        reg2.y=elu(reg1.y);
        reg2.z=elu(reg1.z);
        reg2.w=elu(reg1.w);
        FLOAT4(y[index])=reg2;
    }else {
        for (; index<N; index++) {
            y[index]=elu(x[index]);
        }
    }
}

//  FP16
__global__ void elu_f16_kernel(half *x, half *y, int N) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<N) {
        y[idx]=elu_half(x[idx]);
    }
}

__global__ void elu_f16x2_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*2;
    half zero=__float2half(0.0f);
    if (idx+1<N) {
        half2 reg1;
        half2 reg2;
        reg1=HALF2(x[idx]);
        reg2.x=elu_half(reg1.x);
        reg2.y=elu_half(reg1.y);
        HALF2(y[idx])=reg2;
    }else {
        for (;idx<N; idx++) {
            y[idx]=elu_half(x[idx]);
        }
    }
}

__global__ void elu_f16x8_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*8;
    half zero=__float2half(0.0f);
    half2 zero2=make_half2(zero,zero);
    if (idx+7<N) {
        half2 reg_x[4],reg_y[4];
        reg_x[0]=HALF2(x[idx]);
        reg_x[1]=HALF2(x[idx+2]);
        reg_x[2] =HALF2(x[idx+4]);
        reg_x[3]=HALF2(x[idx+6]);
        reg_y[0].x=elu_half(reg_x[0].x);
        reg_y[0].y=elu_half(reg_x[0].y);
        reg_y[1].x=elu_half(reg_x[1].x);
        reg_y[1].y=elu_half(reg_x[1].y);
        reg_y[2].x=elu_half(reg_x[2].x);
        reg_y[2].y=elu_half(reg_x[2].y);
        reg_y[3].x=elu_half(reg_x[3].x);
        reg_y[3].y=elu_half(reg_x[3].y);
        HALF2(y[idx])=reg_y[0];
        HALF2(y[idx+2])=reg_y[1];
        HALF2(y[idx+4])=reg_y[2];
        HALF2(y[idx+6])=reg_y[3];
    }else {
        for (;idx<N; idx++) {
            y[idx]=elu_half(x[idx]);
        }
    }
}

__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*8;
    half seg_x[8],seg_y[8];
    LDST128BITS(seg_x)=LDST128BITS(x[idx]);
#pragma unroll
    for (int i=0;i<8;i++) {
        seg_y[i]=elu_half(seg_x[i]);
    }
    if (idx+7<N) {
        LDST128BITS(y[idx])=LDST128BITS(seg_y[0]);
    }
    
}
