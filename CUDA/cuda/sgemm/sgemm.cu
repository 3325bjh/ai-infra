#include <cstdio>
#include "sgemm_kernel.cuh"
#define A(i,j) a[i*n+j]
#define B(i,j) b[i*n+j]
void random_matrix(int m,int n,float *a)
{
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            A(i,j)=2.0*(float)drand48()-1.0;
        }
    }
}

void cpu_sgemm(float* A_ptr,float* B_ptr,float *C_ptr,int M,int N,int K) {
    for (int m=0;m<M;m++) {
        for (int n=0;n<N;n++) {
            float temp=0.f;
            for (int k=0;k<K;k++) {
                temp+=A_ptr[m*K+k]*B_ptr[n+k*N];
            }
            C_ptr[N*m+n]=temp;
        }
    }
}

float compare_matrices(int m,int n,float *a,float *b) {
    float max_diff=0.0,diff;
    int printed=0;
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            diff=abs(A(i,j)-B(i,j));
            max_diff=diff>max_diff ? diff : max_diff;
            if (0==printed) {
                if (max_diff>0.5f||max_diff<-0.5f) {
                    printf("\n error: i %d j %d diff %f expect %f",i,j,A(i,j),B(i,j));
                    printed=1;
                }
            }
        }
    }
    return max_diff;
}

int main() {
    constexpr int m=1024;
    constexpr int n=1024;
    constexpr int k=1024;
    const size_t mem_size_A=m*k*sizeof(float);
    const size_t mem_size_B=k*n*sizeof(float);
    const size_t mem_size_C=m*n*sizeof(float);
    float *A_host=(float*)malloc(mem_size_A);
    float *B_host=(float*)malloc(mem_size_B);
    float *C_host=(float*)malloc(mem_size_C);
    float *C_host_gpu=(float*)malloc(mem_size_C);
    random_matrix(m,k,A_host);
    random_matrix(k,n,B_host);
    memset(C_host,0,mem_size_C);
    memset(C_host_gpu,0,mem_size_C);
    float *A_device,*B_device,*C_device;
    cudaMalloc((void **)&A_device,mem_size_A);
    cudaMalloc((void **)&B_device,mem_size_B);
    cudaMalloc((void **)&C_device,mem_size_C);
    cudaMemcpy(A_device,A_host,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(B_device,B_host,mem_size_B,cudaMemcpyHostToDevice);
    cpu_sgemm(A_host,B_host,C_host,m,n,k);
    // constexpr int BLOCK=16;
    // constexpr int STRIDE=2;
    // dim3 block(BLOCK,BLOCK);
    // dim3 grid(n/BLOCK/STRIDE,(m+BLOCK-1)/BLOCK/STRIDE);
    // cuda_sgemm<<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    // cuda_sgemm_shared_memory<BLOCK,k><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    // cuda_sgemm_tile<BLOCK><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    // cuda_sgemm_increase_work_of_per_thread<BLOCK,STRIDE><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    //.............
    // constexpr int M_NUM_PER_BLOCK=32;
    // constexpr int N_NUM_PER_BLOCK=32;
    // constexpr int K_NUM_PER_BLOCK=32;
    // constexpr int NUM_PER_THREAD=4;
    // dim3 block(8,32);
    // dim3 grid(n/N_NUM_PER_BLOCK,m/M_NUM_PER_BLOCK);
    // cuda_sgemm_register<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,NUM_PER_THREAD><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    //.............
    // constexpr int M_NUM_PER_BLOCK=64;
    // constexpr int N_NUM_PER_BLOCK=64;
    // constexpr int K_NUM_PER_BLOCK=64;
    // // constexpr int NUM_PER_THREAD=16;
    // constexpr int M_NUM_PER_THREAD=4;
    // constexpr int N_NUM_PER_THREAD=4;
    // constexpr int K_NUM_PER_THREAD=4;
    // dim3 block(16,16);
    // dim3 grid(n/N_NUM_PER_BLOCK,m/M_NUM_PER_BLOCK);
    // cuda_sgemm_transpose<M_NUM_PER_BLOCK,N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,M_NUM_PER_THREAD,N_NUM_PER_THREAD,K_NUM_PER_THREAD><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    //..............
    constexpr int BLOCK_SIZE_M=128;
    constexpr int BLOCK_SIZE_K=8;
    constexpr int BLOCK_SIZE_N=128;
    const int THREAD_SIZE_X=8;
    const int THREAD_SIZE_Y=8;
    dim3 block(16,16);
    dim3 grid(n/BLOCK_SIZE_N,m/BLOCK_SIZE_M);
    cuda_sgemm_double_buffer2<BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,THREAD_SIZE_Y,THREAD_SIZE_X><<<grid,block>>>(A_device,B_device,C_device,m,n,k);
    cudaMemcpy(C_host_gpu,C_device,mem_size_C,cudaMemcpyDeviceToHost);
    float diff=compare_matrices(m,n,C_host_gpu,C_host);
    if (diff>0.5f||diff<-0.5f) {
        printf("diff too big!\n");
        exit(-1);
    }else {
        printf("right\n");
    }
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    free(A_host);
    free(B_host);
    free(C_host_gpu);
    free(C_host);
    return 0;

}
