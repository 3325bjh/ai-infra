#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
__global__ void reduce(float *d_input,float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin=d_input+blockDim.x*blockIdx.x;
    shared[threadIdx.x]=input_begin[threadIdx.x];
    __syncthreads();
    for (int i=blockDim.x/2;i>0;i/=2) {
        if (threadIdx.x<i) {
            shared[threadIdx.x]+=shared[threadIdx.x+i];
        }
        __syncthreads();
    }
    if (threadIdx.x==0) {
        d_output[blockIdx.x]=shared[0];
    }

}

bool check(float *out,float *res,int n)
{
    for(int i=0;i<n;i++)
    {
        if(abs(out[i]-res[i])>1e-4)
            return false;
    }
    return true;
}


int main()
{
    const int N=32*1024*1024;
    float *input =(float *) malloc(N*sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N*sizeof(float));
    int block_num=N/THREAD_PER_BLOCK;
    float *output=(float *) malloc(block_num*sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num*sizeof(float));
    float *result=(float *) malloc(block_num*sizeof(float));
    for(int i=0;i<N;i++)
    {
        input[i]=2.0*(float)drand48()-1.0;
    }
    for(int i=0;i<block_num;i++){
        float cur=0;
        for(int j=0;j<THREAD_PER_BLOCK;j++)
        {
            cur+=input[i*THREAD_PER_BLOCK+j];
        }
        result[i]=cur;
    }
    cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(N/THREAD_PER_BLOCK);
    dim3 Block(THREAD_PER_BLOCK);
    reduce<<<Grid,Block>>>(d_input,d_output);
    cudaMemcpy(output, d_output, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    if(check(output,result,block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

