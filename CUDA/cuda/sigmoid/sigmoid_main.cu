#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "sigmoid_kernel.cuh"
#include <cuda_runtime.h>


void check_result(const float *host_ref, const float *dev_ref, int n)
{
    const float eps = 1e-6f;
    for (int i = 0; i < n; i++)
    {
        if (std::fabs(host_ref[i] - dev_ref[i]) > eps)
        {
            std::printf("WRONG\n");
            return;
        }
    }
    std::printf("MATCH\n");
}

void sigmoid_cpu(float *x,float *y,int n) {
    for (int i=0;i<n;i++) {
        y[i]=1.0f/(1.0f+expf(-x[i]));
    }
}

int main() {
    float* h_x,*h_y,*truth;
    float* d_x,*d_y;
    const int n = 1024;
    const size_t nbytes = n * sizeof(float);
    h_x=static_cast<float *>(malloc(nbytes));
    h_y=static_cast<float *>(malloc(nbytes));
    truth=static_cast<float *>(malloc(nbytes));
    cudaMalloc(reinterpret_cast<void **>(&d_x), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_y),nbytes);
    for(int i=0;i<n;i++) {
        h_x[i]=1.0f;
    }
    cudaMemcpy(d_x,h_x,nbytes,cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 block(1024);
    sigmoid_f32_kernel<<<grid, block>>>(d_x,d_y,n);
    cudaError err=cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_y,d_y,nbytes,cudaMemcpyDeviceToHost);
    sigmoid_cpu(h_x,truth,n);
    check_result(h_y, truth, n);
    free(h_x);
    free(h_y);
    free(truth);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

