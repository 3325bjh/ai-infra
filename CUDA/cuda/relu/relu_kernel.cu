#include "relu_kernel.cuh"
#define FLOAT4(val) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&val)[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&val)[0])

// FP32
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float *x, float *y, int N) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if (index<N) {
        y[index]=fmaxf(x[index],0.0f);
    }
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
    int index=(blockDim.x*blockIdx.x+threadIdx.x)*4;
    if (index+3<N) {
        float4 reg1;
        float4 reg2;
        reg1=FLOAT4(x[index]);
        reg2.x=fmaxf(reg1.x,0.0f);
        reg2.y=fmaxf(reg1.y,0.0f);
        reg2.z=fmaxf(reg1.z,0.0f);
        reg2.w=fmaxf(reg1.w,0.0f);
        FLOAT4(y[index])=reg2;
    }else {
        for (; index<N; index++) {
            y[index]=fmaxf(x[index],0.0f);
        }
    }
}

//  FP16
__global__ void relu_f16_kernel(half *x, half *y, int N) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx<N) {
        y[idx]=__hmax(x[idx],__float2half(0.0f));
    }
}

__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*2;
    half zero=__float2half(0.0f);
    if (idx+1<N) {
        half2 reg1;
        half2 reg2;
        reg1=HALF2(x[idx]);
        reg2=__hmax2(reg1,make_half2(zero,zero));
        HALF2(y[idx])=reg2;
    }else {
        for (;idx<N; idx++) {
            y[idx]=__hmax(x[idx],zero);
        }
    }
}

__global__ void relu_f16x8_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*8;
    half zero=__float2half(0.0f);
    half2 zero2=make_half2(zero,zero);
    if (idx+7<N) {
        half2 reg_x[4],reg_y[4];
        reg_x[0]=HALF2(x[idx]);
        reg_x[1]=HALF2(x[idx+2]);
        reg_x[2] =HALF2(x[idx+4]);
        reg_x[3]=HALF2(x[idx+6]);
        reg_y[0]=__hmax2(reg_x[0],zero2);
        reg_y[1]=__hmax2(reg_x[1],zero2);
        reg_y[2]=__hmax2(reg_x[2],zero2);
        reg_y[3]=__hmax2(reg_x[3],zero2);
        HALF2(y[idx])=reg_y[0];
        HALF2(y[idx+2])=reg_y[1];
        HALF2(y[idx+4])=reg_y[2];
        HALF2(y[idx+6])=reg_y[3];
    }else {
        for (;idx<N; idx++) {
            y[idx]=__hmax(x[idx],zero);
        }
    }
}

__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*8;
    half seg_x[8],seg_y[8];
    half zero=__float2half(0.0f);
    half2 zero2=make_half2(zero,zero);
    LDST128BITS(seg_x)=LDST128BITS(x[idx]);
#pragma unroll
    for (int i=0;i<8;i+=2) {
        HALF2(seg_y[i])=__hmax2(HALF2(seg_x[i]),zero2);
    }
    if (idx+7<N) {
        LDST128BITS(y[idx])=LDST128BITS(seg_y[0]);
    }
    
}
