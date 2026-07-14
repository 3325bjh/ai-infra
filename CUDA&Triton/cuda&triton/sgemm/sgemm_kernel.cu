#include "sgemm_kernel.cuh"
#define FLOAT4(val) reinterpret_cast<float4*>(&val)[0]
__global__ void cuda_sgemm(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    const int x=blockDim.x*blockIdx.x+threadIdx.x;
    const int y=blockDim.y*blockIdx.y+threadIdx.y;
    float *A_ptr_start=A_ptr+blockDim.y*blockIdx.y*K;
    float *B_ptr_start=B_ptr+blockDim.x*blockIdx.x;
    float temp=0.f;
    for (int k=0;k<K;k++) {
        temp+=A_ptr_start[threadIdx.y*K+k]*B_ptr_start[k*N+threadIdx.x];
    }
    C_ptr[y*N+x]=temp;
}

template<unsigned int BLOCK_SIZE,unsigned int K_>
__global__ void cuda_sgemm_shared_memory(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    const int x=blockDim.x*blockIdx.x+threadIdx.x;
    const int y=blockDim.y*blockIdx.y+threadIdx.y;
    float *A_ptr_start=A_ptr+blockDim.y*blockIdx.y*K;
    float *B_ptr_start=B_ptr+blockDim.x*blockIdx.x;
    __shared__ float a_shared[BLOCK_SIZE][K_];
    __shared__ float b_shared[K_][BLOCK_SIZE];
    for (int s=0;s<K;s+=BLOCK_SIZE) {
        a_shared[threadIdx.y][s+threadIdx.x]=A_ptr_start[threadIdx.y*K+(s+threadIdx.x)];
        b_shared[s+threadIdx.y][threadIdx.x]=B_ptr_start[(s+threadIdx.y)*N+(threadIdx.x)];
    }
    __syncthreads();

    float temp=0.f;
    for (int k=0;k<K;k++) {
        temp+=a_shared[threadIdx.y][k]*b_shared[k][threadIdx.x];
    }
    C_ptr[y*N+x]=temp;
}

template<unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm_tile(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    const int x=blockDim.x*blockIdx.x+threadIdx.x;
    const int y=blockDim.y*blockIdx.y+threadIdx.y;
    float *A_ptr_start=A_ptr+blockDim.y*blockIdx.y*K;
    float *B_ptr_start=B_ptr+blockDim.x*blockIdx.x;
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    float temp=0.f;
    for (int s=0;s<K;s+=BLOCK_SIZE) {
        a_shared[threadIdx.y][threadIdx.x]=A_ptr_start[threadIdx.y*K+(s+threadIdx.x)];
        b_shared[threadIdx.y][threadIdx.x]=B_ptr_start[(s+threadIdx.y)*N+(threadIdx.x)];
        __syncthreads();
        for (int k=0;k<BLOCK_SIZE;k++) {
            temp+=a_shared[threadIdx.y][k]*b_shared[k][threadIdx.x];
        }
        __syncthreads();
    }

    C_ptr[y*N+x]=temp;
}

template __global__ void cuda_sgemm_tile<16u>(
    float*, float*, float*, int, int, int
);


template<unsigned int BLOCK_SIZE,unsigned int STRIDE>
__global__ void cuda_increase_work_of_per_thread(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    constexpr int STEP=BLOCK_SIZE*STRIDE;
    float *A_ptr_start=A_ptr+STEP*blockIdx.y*K;
    float *B_ptr_start=B_ptr+STEP*blockIdx.x;
    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];
    float temp[STRIDE][STRIDE]={0.f};
    for (int s=0;s<K;s+=STEP) {
       for (int i=0;i<STRIDE;i++) {
           for (int j=0;j<STRIDE;j++) {
                a_shared[i*BLOCK_SIZE+ty][j*BLOCK_SIZE+tx]=A_ptr_start[(i*BLOCK_SIZE+ty)*K+(j*BLOCK_SIZE+tx+s)];
                b_shared[i*BLOCK_SIZE+ty][j*BLOCK_SIZE+tx]=B_ptr_start[(s+i*BLOCK_SIZE+ty)*N+(j*BLOCK_SIZE+tx)];
           }
       }
        __syncthreads();
        for (int i=0;i<STRIDE;i++) {
            for (int j=0;j<STRIDE;j++) {
                for (int k=0;k<STEP;k++) {
                    temp[i][j]+=a_shared[i*BLOCK_SIZE+ty][k]*b_shared[k][j*BLOCK_SIZE+tx];
                }
            }
        }
        __syncthreads();
    }
    for (int i=0;i<STRIDE;i++) {
        for (int j=0;j<STRIDE;j++) {
          C_ptr[(STEP*blockIdx.y+i*BLOCK_SIZE+ty)*N+(STEP*blockIdx.x+j*BLOCK_SIZE+tx)] =temp[i][j];
        }
    }
}

template __global__ void cuda_increase_work_of_per_thread<16u,2u>(
    float*, float*, float*, int, int, int
);


template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_float4(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float *A_ptr_start=A_ptr+M_NUM_PER_BLOCK*blockIdx.y*K;
    float *B_ptr_start=B_ptr+N_NUM_PER_BLOCK*blockIdx.x;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float temp[NUM_PER_THREAD]={0.f};

    for (int s=0;s<K;s+=K_NUM_PER_BLOCK) {
        FLOAT4(a_shared[ty][tx*NUM_PER_THREAD])=FLOAT4(A_ptr_start[ty*K+s+tx*NUM_PER_THREAD]);
        FLOAT4(b_shared[ty][tx*NUM_PER_THREAD])=FLOAT4(B_ptr_start[(ty+s)*N+tx*NUM_PER_THREAD]);
        __syncthreads();
        for (int i=0;i<NUM_PER_THREAD;i++) {
            for (int k=0;k<K_NUM_PER_BLOCK;k++) {
                temp[i]+=a_shared[ty][k]*b_shared[k][tx*NUM_PER_THREAD+i];
            }
        }
        __syncthreads();
    }

    float *C_ptr_start=C_ptr+N*(blockIdx.y*M_NUM_PER_BLOCK)+blockIdx.x*N_NUM_PER_BLOCK;
    for (int i=0;i<NUM_PER_THREAD;i++) {
        C_ptr_start[ty*N+tx*NUM_PER_THREAD+i]=temp[i];
    }
}

template __global__ void cuda_sgemm_float4<32u,32u,32u,4u>(
    float*, float*, float*, int, int, int
);

template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_register(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int tid=ty*blockDim.x+tx;
    int ctx=tid%16;
    int cty=tid/16;
    float *A_ptr_start=A_ptr+M_NUM_PER_BLOCK*blockIdx.y*K;
    float *B_ptr_start=B_ptr+N_NUM_PER_BLOCK*blockIdx.x;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    constexpr int REG_NUM=NUM_PER_THREAD/2;
    float a_reg[REG_NUM]={0.f};
    float b_reg[REG_NUM]={0.f};
    float temp[REG_NUM][REG_NUM]={0.f};

    for (int s=0;s<K;s+=K_NUM_PER_BLOCK) {
        FLOAT4(a_shared[ty][tx*NUM_PER_THREAD])=FLOAT4(A_ptr_start[ty*K+s+tx*NUM_PER_THREAD]);
        FLOAT4(b_shared[ty][tx*NUM_PER_THREAD])=FLOAT4(B_ptr_start[(ty+s)*N+tx*NUM_PER_THREAD]);
        __syncthreads();
        for (int k=0;k<K_NUM_PER_BLOCK;k++) {
            a_reg[0]=a_shared[cty*REG_NUM][k];
            a_reg[1]=a_shared[cty*REG_NUM+1][k];
            b_reg[0]=b_shared[k][ctx*REG_NUM];
            b_reg[1]=b_shared[k][ctx*REG_NUM+1];
            for (int i=0;i<REG_NUM;i++) {
                for (int j=0;j<REG_NUM;j++) {
                    temp[i][j]+=a_reg[i]*b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *C_ptr_start=C_ptr+N*(blockIdx.y*M_NUM_PER_BLOCK)+blockIdx.x*N_NUM_PER_BLOCK;
    for (int i=0;i<REG_NUM;i++) {
        for (int j=0;j<REG_NUM;j++) {
            C_ptr_start[(cty*REG_NUM+i)*N+ctx*REG_NUM+j]=temp[i][j];
        }
    }
}

template __global__ void cuda_sgemm_register<32u,32u,32u,4u>(
    float*, float*, float*, int, int, int
);


template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int M_NUM_PER_THREAD,unsigned int N_NUM_PER_THREAD,unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_register_float4(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float *A_ptr_start=A_ptr+blockIdx.y*M_NUM_PER_BLOCK*K;
    float *B_ptr_start=B_ptr+blockIdx.x*N_NUM_PER_BLOCK;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float a_reg[M_NUM_PER_THREAD]={0.f};
    float b_reg[N_NUM_PER_THREAD]={0.f};
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD]={0.f};
    for (int s=0;s<K;s+=K_NUM_PER_BLOCK) {
        for (int i=0;i<M_NUM_PER_THREAD;i++) {
            FLOAT4(a_shared[ty*M_NUM_PER_THREAD+i][tx*K_NUM_PER_THREAD])=FLOAT4(A_ptr_start[K*(ty*M_NUM_PER_THREAD+i)+(s+tx*K_NUM_PER_THREAD)]);
        }
        for (int i=0;i<K_NUM_PER_THREAD;i++) {
            FLOAT4(b_shared[ty*K_NUM_PER_THREAD+i][tx*N_NUM_PER_THREAD])=FLOAT4(B_ptr_start[N*(s+ty*K_NUM_PER_THREAD+i)+tx*N_NUM_PER_THREAD]);
        }
        __syncthreads();
        for (int k=0;k<K_NUM_PER_BLOCK;k++) {
            a_reg[0]=a_shared[ty*M_NUM_PER_THREAD][k];
            a_reg[1]=a_shared[ty*M_NUM_PER_THREAD+1][k];
            a_reg[2]=a_shared[ty*M_NUM_PER_THREAD+2][k];
            a_reg[3]=a_shared[ty*M_NUM_PER_THREAD+3][k];
            FLOAT4(b_reg[0])=FLOAT4(b_shared[k][tx*N_NUM_PER_THREAD]);
            for (int i=0;i<M_NUM_PER_THREAD;i++) {
                for (int j=0;j<N_NUM_PER_THREAD;j++) {
                    temp[i][j]+=a_reg[i]*b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    float* C_ptr_start=C_ptr+N*(blockIdx.y*M_NUM_PER_BLOCK)+blockIdx.x*N_NUM_PER_BLOCK;
    for (int i=0;i<M_NUM_PER_THREAD;i++) {
        FLOAT4(C_ptr_start[(ty*M_NUM_PER_THREAD+i)*N+tx*N_NUM_PER_THREAD])=FLOAT4(temp[i][0]);
    }
}

template __global__ void cuda_sgemm_register_float4<64u,64u,64u,4u,4u,4u>(
    float*, float*, float*, int, int, int
);


template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int M_NUM_PER_THREAD,unsigned int N_NUM_PER_THREAD,unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_transpose(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float *A_ptr_start=A_ptr+blockIdx.y*M_NUM_PER_BLOCK*K;
    float *B_ptr_start=B_ptr+blockIdx.x*N_NUM_PER_BLOCK;
    __shared__ float a_shared[K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float a_reg[M_NUM_PER_THREAD]={0.f};
    float b_reg[N_NUM_PER_THREAD]={0.f};
    float a_load_reg[K_NUM_PER_THREAD]={0.f};
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD]={0.f};
    for (int s=0;s<K;s+=K_NUM_PER_BLOCK) {
        for (int i=0;i<M_NUM_PER_THREAD;i++) {
            FLOAT4(a_load_reg[0])=FLOAT4(A_ptr_start[K*(ty*M_NUM_PER_THREAD+i)+(s+tx*K_NUM_PER_THREAD)]);
            a_shared[tx*K_NUM_PER_THREAD][ty*M_NUM_PER_THREAD+i]=a_load_reg[0];
            a_shared[tx*K_NUM_PER_THREAD+1][ty*M_NUM_PER_THREAD+i]=a_load_reg[1];
            a_shared[tx*K_NUM_PER_THREAD+2][ty*M_NUM_PER_THREAD+i]=a_load_reg[2];
            a_shared[tx*K_NUM_PER_THREAD+3][ty*M_NUM_PER_THREAD+i]=a_load_reg[3];
        }
        for (int i=0;i<K_NUM_PER_THREAD;i++) {
            FLOAT4(b_shared[ty*K_NUM_PER_THREAD+i][tx*N_NUM_PER_THREAD])=FLOAT4(B_ptr_start[N*(s+ty*K_NUM_PER_THREAD+i)+tx*N_NUM_PER_THREAD]);
        }
        __syncthreads();
        for (int k=0;k<K_NUM_PER_BLOCK;k++) {
            FLOAT4(a_reg[0])=FLOAT4(a_shared[k][ty*M_NUM_PER_THREAD]);
            FLOAT4(b_reg[0])=FLOAT4(b_shared[k][tx*N_NUM_PER_THREAD]);
            for (int i=0;i<M_NUM_PER_THREAD;i++) {
                for (int j=0;j<N_NUM_PER_THREAD;j++) {
                    temp[i][j]+=a_reg[i]*b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    float* C_ptr_start=C_ptr+N*(blockIdx.y*M_NUM_PER_BLOCK)+blockIdx.x*N_NUM_PER_BLOCK;
    for (int i=0;i<M_NUM_PER_THREAD;i++) {
        FLOAT4(C_ptr_start[(ty*M_NUM_PER_THREAD+i)*N+tx*N_NUM_PER_THREAD])=FLOAT4(temp[i][0]);
    }
}

template __global__ void cuda_sgemm_transpose<64u,64u,64u,4u,4u,4u>(
    float*, float*, float*, int, int, int
);

template<const int BLOCK_SIZE_M,const int BLOCK_SIZE_N,const int BLOCK_SIZE_K,const int THREAD_SIZE_Y,const int THREAD_SIZE_X>
__global__ void cuda_sgemm_double_buffer(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    const int tid=ty*blockDim.x+tx;
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X]={0.f};
    float reg_a[THREAD_SIZE_Y]={0.f};
    float reg_b[THREAD_SIZE_X]={0.f};
    float ldg_a_reg[4]={0.f};
    float *A_ptr_start=A_ptr+blockIdx.y*BLOCK_SIZE_M*K;
    float *B_ptr_start=B_ptr+blockIdx.x*BLOCK_SIZE_N;
    const int A_tile_thread_per_row=BLOCK_SIZE_K/4;
    const int B_tile_thread_per_row=BLOCK_SIZE_N/4;
    const int A_tile_tid_x=tid%A_tile_thread_per_row;
    const int A_tile_tid_y=tid/A_tile_thread_per_row;
    const int B_tile_tid_x=tid%B_tile_thread_per_row;
    const int B_tile_tid_y=tid/B_tile_thread_per_row;

    FLOAT4(ldg_a_reg[0])=FLOAT4(A_ptr_start[K*A_tile_tid_y+A_tile_tid_x*4]);
    a_shared[0][A_tile_tid_x*4][A_tile_tid_y]=ldg_a_reg[0];
    a_shared[0][A_tile_tid_x*4+1][A_tile_tid_y]=ldg_a_reg[1];
    a_shared[0][A_tile_tid_x*4+2][A_tile_tid_y]=ldg_a_reg[2];
    a_shared[0][A_tile_tid_x*4+3][A_tile_tid_y]=ldg_a_reg[3];
    FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x*4])=FLOAT4(B_ptr_start[B_tile_tid_y*N+B_tile_tid_x*4]);
    __syncthreads();
    int write_stage_idx=1;
    for (int s=BLOCK_SIZE_K;s<K;s+=BLOCK_SIZE_K)
    {
        FLOAT4(ldg_a_reg[0])=FLOAT4(A_ptr_start[K*A_tile_tid_y+A_tile_tid_x*4+s]);
        a_shared[write_stage_idx][A_tile_tid_x*4][A_tile_tid_y]=ldg_a_reg[0];
        a_shared[write_stage_idx][A_tile_tid_x*4+1][A_tile_tid_y]=ldg_a_reg[1];
        a_shared[write_stage_idx][A_tile_tid_x*4+2][A_tile_tid_y]=ldg_a_reg[2];
        a_shared[write_stage_idx][A_tile_tid_x*4+3][A_tile_tid_y]=ldg_a_reg[3];
        FLOAT4(b_shared[write_stage_idx][B_tile_tid_y][B_tile_tid_x*4])=FLOAT4(B_ptr_start[(B_tile_tid_y+s)*N+B_tile_tid_x*4]);
        write_stage_idx=write_stage_idx^1;
        for (int k=0;k<BLOCK_SIZE_K;k++) {
            FLOAT4(reg_a[0])=FLOAT4(a_shared[write_stage_idx][k][ty*THREAD_SIZE_Y]);
            FLOAT4(reg_a[4])=FLOAT4(a_shared[write_stage_idx][k][ty*THREAD_SIZE_Y+4]);
            FLOAT4(reg_b[0])=FLOAT4(b_shared[write_stage_idx][k][tx*THREAD_SIZE_X]);
            FLOAT4(reg_b[4])=FLOAT4(b_shared[write_stage_idx][k][tx*THREAD_SIZE_X+4]);
            for (int i=0;i<THREAD_SIZE_Y;i++) {
                for (int j=0;j<THREAD_SIZE_X;j++) {
                    accum[i][j]+=reg_a[i]*reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    write_stage_idx=write_stage_idx^1;
    for (int k=0;k<BLOCK_SIZE_K;k++) {
        FLOAT4(reg_a[0])=FLOAT4(a_shared[write_stage_idx][k][ty*THREAD_SIZE_Y]);
        FLOAT4(reg_a[4])=FLOAT4(a_shared[write_stage_idx][k][ty*THREAD_SIZE_Y+4]);
        FLOAT4(reg_b[0])=FLOAT4(b_shared[write_stage_idx][k][tx*THREAD_SIZE_X]);
        FLOAT4(reg_b[4])=FLOAT4(b_shared[write_stage_idx][k][tx*THREAD_SIZE_X+4]);
        for (int i=0;i<THREAD_SIZE_Y;i++) {
            for (int j=0;j<THREAD_SIZE_X;j++) {
                accum[i][j]+=reg_a[i]*reg_b[j];
            }
        }
    }
    float* C_ptr_start=C_ptr+N*(by*BLOCK_SIZE_M)+bx*BLOCK_SIZE_N;
    for (int i=0;i<THREAD_SIZE_Y;i++) {
        FLOAT4(C_ptr_start[(ty*THREAD_SIZE_Y+i)*N+tx*THREAD_SIZE_X])=FLOAT4(accum[i][0]);
        FLOAT4(C_ptr_start[(ty*THREAD_SIZE_Y+i)*N+tx*THREAD_SIZE_X+4])=FLOAT4(accum[i][4]);
    }

}

template __global__ void cuda_sgemm_double_buffer<128, 128, 8, 8, 8>(
    float*, float*, float*, int, int, int
);


template<const int BLOCK_SIZE_M,const int BLOCK_SIZE_N,const int BLOCK_SIZE_K,const int THREAD_SIZE_Y,const int THREAD_SIZE_X>
__global__ void cuda_sgemm_double_buffer2(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K) {
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    __shared__ float a[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X]={0.f};
    float *A_Block_ptr=A_ptr+by*BLOCK_SIZE_M*K;
    float *B_Block_ptr=B_ptr+bx*BLOCK_SIZE_N;
    int tid=threadIdx.y*blockDim.x+threadIdx.x;
    int k_threads=BLOCK_SIZE_K/4;
    int n_threads=BLOCK_SIZE_N/4;
    int tx_adj_a=tid%k_threads;
    int ty_adj_a=tid/k_threads;
    int tx_adj_b=tid%n_threads;
    int ty_adj_b=tid/n_threads;
    float seg_a[4]={0.f};
    float seg_y[THREAD_SIZE_Y]={0.f};
    float seg_x[THREAD_SIZE_X]={0.f};

    FLOAT4(seg_a[0])=FLOAT4(A_Block_ptr[ty_adj_a*K+tx_adj_a*4]);
    a[0][tx_adj_a*4][ty_adj_a]=seg_a[0];
    a[0][tx_adj_a*4+1][ty_adj_a]=seg_a[1];
    a[0][tx_adj_a*4+2][ty_adj_a]=seg_a[2];
    a[0][tx_adj_a*4+3][ty_adj_a]=seg_a[3];
    FLOAT4(b[0][ty_adj_b][tx_adj_b*4])=FLOAT4(B_Block_ptr[ty_adj_b*N+tx_adj_b*4]);
    int write_positon=1;
    __syncthreads();
    for (int s=BLOCK_SIZE_K;s<K;s+=BLOCK_SIZE_K) {
        FLOAT4(seg_a[0])=FLOAT4(A_Block_ptr[ty_adj_a*K+tx_adj_a*4+s]);
        a[write_positon][tx_adj_a*4][ty_adj_a]=seg_a[0];
        a[write_positon][tx_adj_a*4+1][ty_adj_a]=seg_a[1];
        a[write_positon][tx_adj_a*4+2][ty_adj_a]=seg_a[2];
        a[write_positon][tx_adj_a*4+3][ty_adj_a]=seg_a[3];
        FLOAT4(b[write_positon][ty_adj_b][tx_adj_b*4])=FLOAT4(B_Block_ptr[(ty_adj_b+s)*N+tx_adj_b*4]);
        write_positon=1^write_positon;
        for (int k=0;k<BLOCK_SIZE_K;k++) {
                FLOAT4(seg_y[0])=FLOAT4(a[write_positon][k][ty*THREAD_SIZE_Y]);
                FLOAT4(seg_y[4])=FLOAT4(a[write_positon][k][ty*THREAD_SIZE_Y+4]);
                FLOAT4(seg_x[0])=FLOAT4(b[write_positon][k][tx*THREAD_SIZE_X]);
                FLOAT4(seg_x[4])=FLOAT4(b[write_positon][k][tx*THREAD_SIZE_X+4]);
            for (int i=0;i<THREAD_SIZE_Y;i++) {
                for (int j=0;j<THREAD_SIZE_X;j++) {
                    accum[i][j]+=seg_y[i]*seg_x[j];
                }
            }
        }
        __syncthreads();
    }
    write_positon=1^write_positon;
    for (int k=0;k<BLOCK_SIZE_K;k++) {
        FLOAT4(seg_y[0])=FLOAT4(a[write_positon][k][ty*THREAD_SIZE_Y]);
        FLOAT4(seg_y[4])=FLOAT4(a[write_positon][k][ty*THREAD_SIZE_Y+4]);
        FLOAT4(seg_x[0])=FLOAT4(b[write_positon][k][tx*THREAD_SIZE_X]);
        FLOAT4(seg_x[4])=FLOAT4(b[write_positon][k][tx*THREAD_SIZE_X+4]);
        for (int i=0;i<THREAD_SIZE_Y;i++) {
            for (int j=0;j<THREAD_SIZE_X;j++) {
                accum[i][j]+= seg_y[i]*seg_x[j];
            }
        }
    }

    float *C_Block_ptr=C_ptr+by*BLOCK_SIZE_M*N+bx*BLOCK_SIZE_N;
    for (int i=0;i<THREAD_SIZE_Y;i++) {
        FLOAT4(C_Block_ptr[(ty*THREAD_SIZE_Y+i)*N+tx*THREAD_SIZE_X])=FLOAT4(accum[i][0]);
        FLOAT4(C_Block_ptr[(ty*THREAD_SIZE_Y+i)*N+tx*THREAD_SIZE_X+4])=FLOAT4(accum[i][4]);
    }

}

template __global__ void cuda_sgemm_double_buffer2<128, 128, 8, 8, 8>(
    float*, float*, float*, int, int, int
);
