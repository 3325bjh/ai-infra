#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "embedding_kernel.cuh"
#include <cuda_runtime.h>



void check_result(const float *host_ref, const float *dev_ref, int n,int m)
{
    const float eps = 1e-6f;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++) {
            if (std::fabs(host_ref[i*m+j] - dev_ref[i*m+j]) > eps)
            {
                std::printf("WRONG\n");
                return;
            }
        }
    }
    std::printf("MATCH\n");
}

void embedding_cpu(int *a,float *weight,float *output,int num,int emb_size) {
    for (int i=0;i<num;i++) {
        int idx=a[i];
        for (int j=0;j<emb_size;j++) {
            output[i*emb_size+j]=weight[idx*emb_size+j];
        }
    }
}

void init_weight(float *weight,int n,int emb_size) {
    for (int i=0;i<n;i++) {
        for (int j=0;j<emb_size;j++) {
            weight[i*emb_size+j]=drand48();
        }
    }
}

int main() {
    int*a;
    float*weight,*output,*truth;
    int* d_a;
    float*d_weight,*d_output;
    const int n = 1024;
    const int m = 512;
    const int num=2;
    const size_t weight_bytes= n*m*sizeof(float);
    const size_t a_bytes = num * sizeof(int);
    const size_t output_bytes= num*m*sizeof(float);
    a=(int *)malloc(a_bytes);
    weight=(float *)malloc(weight_bytes);
    output=(float *)malloc(output_bytes);
    truth=(float *)malloc(output_bytes);

    cudaMalloc(reinterpret_cast<void **>(&d_a), a_bytes);
    cudaMalloc(reinterpret_cast<void **>(&d_weight),weight_bytes);
    cudaMalloc(reinterpret_cast<void **>(&d_output),output_bytes);
    a[0]=0;
    a[1]=256;
    init_weight(weight,n,m);
    cudaMemcpy(d_a,a,a_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight,weight,weight_bytes,cudaMemcpyHostToDevice);
    embedding_cpu(a,weight,truth,num,m);
    dim3 grid(num);
    dim3 block(m);
    embedding_f32_kernel<<<grid, block>>>(d_a,d_weight,d_output,num,m);
    cudaError err=cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(output,d_output,output_bytes,cudaMemcpyDeviceToHost);
    check_result(output, truth, num,m);
    free(a);
    free(weight);
    free(output);
    free(truth);
    cudaFree(d_a);
    cudaFree(d_output);
    cudaFree(d_weight);
    return 0;
}

