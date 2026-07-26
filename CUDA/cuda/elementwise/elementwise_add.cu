#include <cmath>
#include <cstdio>
#include <cstdlib>


void __global__ add1(float *x, float *y, float *z) {
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    z[n]=x[n]+y[n];
}

void __global__ add2(float *x, float *y, float *z) {
    int n=threadIdx.x+blockIdx.x*blockDim.x+1;
    z[n]=x[n]+y[n];
}

void __global__ add3(float *x, float *y, float *z) {
    int tid_permuted=threadIdx.x^0x1;
    int n=tid_permuted+blockIdx.x*blockDim.x+1;
    z[n]=x[n]+y[n];
}

void __global__ add4(float *x, float *y, float *z) {
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    int warp_idx=n/32;
    z[warp_idx]=x[warp_idx]+y[warp_idx];
}

void __global__ add5(float *x, float *y, float *z) {
    int n=(threadIdx.x+blockIdx.x*blockDim.x)*4;
    z[n]=x[n]+y[n];
}

void __global__ add6(float *x, float *y, float *z) {
    int n=threadIdx.x+blockIdx.x*blockDim.x;
    float a=x[n];
    float b=y[n];
    float c=0;
    for (int i=0;i<1000;i++) {
        c+=(a+b);
    }
    z[n]=c;
}


int main()
{

    const int n = 32*1024*1024;
    const size_t nbytes = n * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(nbytes));
    float *h_b = static_cast<float *>(std::malloc(nbytes));
    float *h_res = static_cast<float *>(std::malloc(nbytes));

    for (int i = 0; i < n; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_res = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_a), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_b), nbytes);
    cudaMalloc(reinterpret_cast<void **>(&d_res), nbytes);

    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);
    dim3 grid(n/1024);
    dim3 block(256);
    add1<<<grid,block>>>(d_a, d_b, d_res);
    // add2<<<grid,block>>>(d_a, d_b, d_res);
    // add3<<<grid,block>>>(d_a, d_b, d_res);
    // add4<<<grid,block>>>(d_a, d_b, d_res);
    // add5<<<grid,block>>>(d_a, d_b, d_res);
    add6<<<grid,block>>>(d_a, d_b, d_res);
    cudaGetLastError();
    cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    std::free(h_a);
    std::free(h_b);
    std::free(h_res);

    return 0;
}
