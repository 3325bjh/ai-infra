#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

template <unsigned int NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void reduce(float *d_input,float *d_output)
{
    int tid=threadIdx.x;
    float sum=0.f;
    float *input_begin=d_input+NUM_PER_BLOCK*blockIdx.x;
    for (int i=0;i<NUM_PER_THREAD;i++) {
        sum+=input_begin[tid+i*THREAD_PER_BLOCK];
    }
    sum+=__shfl_down_sync(0xffffffff,sum,16);
    sum+=__shfl_down_sync(0xffffffff,sum,8);
    sum+=__shfl_down_sync(0xffffffff,sum,4);
    sum+=__shfl_down_sync(0xffffffff,sum,2);
    sum+=__shfl_down_sync(0xffffffff,sum,1);

    __shared__ float shared[32];
    const int laneId=tid%32;
    const int warpId=tid/32;
    if(laneId==0) {
        shared[warpId]=sum;
    }
    __syncthreads();
    if (warpId==0) {
        sum=(laneId<blockDim.x/32)?shared[laneId]:0;
        sum+=__shfl_down_sync(0xffffffff,sum,16);
        sum+=__shfl_down_sync(0xffffffff,sum,8);
        sum+=__shfl_down_sync(0xffffffff,sum,4);
        sum+=__shfl_down_sync(0xffffffff,sum,2);
        sum+=__shfl_down_sync(0xffffffff,sum,1);
    }
    if (tid==0) {
        d_output[blockIdx.x]=sum;
    }
}

bool check(float *out,float *res,int n)
{
    for(int i=0;i<n;i++)
    {
        if(abs(out[i]-res[i])>1e-2)
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
    constexpr int block_num=1024;
    constexpr int num_per_block=N/block_num;
    constexpr int num_per_thread=num_per_block/THREAD_PER_BLOCK;
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
        for(int j=0;j<num_per_block;j++)
        {
            cur+=input[i*num_per_block+j];
        }
        result[i]=cur;
    }
    cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    reduce<num_per_block,num_per_thread><<<Grid,Block>>>(d_input,d_output);
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

