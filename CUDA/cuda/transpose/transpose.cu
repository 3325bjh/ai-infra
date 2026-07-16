#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
using namespace std;
class Perf {
public:
    Perf(const string &name) {
        m_name=name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }
    ~Perf() {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time=0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        cout<<m_name<<"elapse:"<<elapsed_time<<"ms"<<endl;
    }
private:
    string m_name;
    cudaEvent_t m_start,m_end;
};

__global__ void transpose(float * a, float * b,int M,int N) {
    int m=blockDim.y*blockIdx.y+threadIdx.y;
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    b[n*M+m]=a[m*N+n];
}
#define FLOAT4(val) reinterpret_cast<float4*>(&val)[0]
__global__ void transpose_4x4(float * a, float * b,int M,int N) {
    float src_transpose[4][4];
    float dst_transpose[4][4];
    float *a_start=a+blockIdx.y*blockDim.y*4*N+blockIdx.x*blockDim.x*4;
    for (int i=0;i<4;i++) {
        FLOAT4(src_transpose[i])=FLOAT4(a_start[(threadIdx.y*4+i)*N+threadIdx.x*4]);
    }
    FLOAT4(dst_transpose[0])=make_float4(src_transpose[0][0],src_transpose[1][0],src_transpose[2][0],src_transpose[3][0]);
    FLOAT4(dst_transpose[1])=make_float4(src_transpose[0][1],src_transpose[1][1],src_transpose[2][1],src_transpose[3][1]);
    FLOAT4(dst_transpose[2])=make_float4(src_transpose[0][2],src_transpose[1][2],src_transpose[2][2],src_transpose[3][2]);
    FLOAT4(dst_transpose[3])=make_float4(src_transpose[0][3],src_transpose[1][3],src_transpose[2][3],src_transpose[3][3]);
    float* b_start=b+blockIdx.x*blockDim.x*4*M+blockIdx.y*blockDim.y*4;
    for (int i=0;i<4;i++) {
        FLOAT4(b_start[(threadIdx.x*4+i)*M+threadIdx.y*4])=FLOAT4(dst_transpose[i][0]);
    }
}

void cpu_transpose(float *a,float *b,int M,int N) {
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            b[j*M+i]=a[i*N+j];
        }
    }
}
bool check(float *b,float *ground_truth,int M,int N) {
    for (int i=0;i<N;i++) {
        for (int j=0;j<M;j++) {
            if (b[i*M+j]!=ground_truth[i*M+j]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int M=1024,N=512;
    size_t n_bytes=sizeof(float)*M*N;
    float *a, *b,*ground_truth;
    float *a_device, *b_device;
    a=(float *)(malloc(n_bytes));
    b=(float *)(malloc(n_bytes));
    ground_truth=(float *)(malloc(n_bytes));
    cudaMalloc((void **)&a_device, n_bytes);
    cudaMalloc((void **)&b_device, n_bytes);
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            a[i*N+j]=2.0*(float)drand48()-1.0;
        }
    }
    cpu_transpose(a,ground_truth,M,N);
    cudaMemcpy(a_device,a,n_bytes,cudaMemcpyHostToDevice);
    for (int i=0;i<5;i++) {
        Perf perf("transpose_32_8");
        dim3 block(32,8);
        dim3 grid(N/32,M/8);
        transpose<<<grid,block>>>(a_device,b_device,M,N);
        cudaDeviceSynchronize();
    }
    for (int i=0;i<5;i++) {
        Perf perf("transpose_16_16");
        dim3 block(16,16);
        dim3 grid(N/16,M/16);
        transpose<<<grid,block>>>(a_device,b_device,M,N);
        cudaDeviceSynchronize();
    }
    for (int i=0;i<5;i++) {
        Perf perf("transpose_8_32");
        dim3 block(8,32);
        dim3 grid(N/8,M/32);
        transpose<<<grid,block>>>(a_device,b_device,M,N);
        cudaDeviceSynchronize();
    }
    for (int i=0;i<5;i++) {
        Perf perf("transpose_4x4");
        dim3 block(32,8);
        dim3 grid(N/128,M/32);
        transpose_4x4<<<grid,block>>>(a_device,b_device,M,N);
        cudaDeviceSynchronize();
    }
    for (int i=0;i<5;i++) {
        Perf perf("transpose_4x4");
        dim3 block(16,16);
        dim3 grid(N/64,M/64);
        transpose_4x4<<<grid,block>>>(a_device,b_device,M,N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(b,b_device,n_bytes,cudaMemcpyDeviceToHost);
    if (check(b,ground_truth,M,N)) {
        printf("pass\n");
    }
    free(a);
    free(b);
    free(ground_truth);
    cudaFree(a_device);
    cudaFree(b_device);
    return 0;
}
