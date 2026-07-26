#include "mat_transpose_kernel.cuh"

#define FLOAT4(val) reinterpret_cast<float4*>(&val)[0]
#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1

// col2row means read x[row][col] and write y[col][row]
__global__ void mat_transpose_f32_col2row_kernel(float *x,float *y,int M,int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row=idx/N;
    int col=idx%N;
    if(idx<M*N) {
        y[col*M+row]=x[idx];
    }
}

// col2row means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_row2col_kernel(float *x,float *y,int M,int N) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row=idx/M;
    int col=idx%M;
    if(idx<N*M) {
        y[idx]=x[col*N+row];
    }

}


__global__ void mat_transpose_f32x4_col2row_kernel(float *x,float *y,int M,int N) {
    int idx=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if (idx+3<M*N) {
        int row=idx/N;
        int col=idx%N;
        float4 reg=FLOAT4(x[idx]);
        y[col*M+row]=reg.x;
        y[(col+1)*M+row]=reg.y;
        y[(col+2)*M+row]=reg.z;
        y[(col+3)*M+row]=reg.w;
    }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float *x,float *y,int M,int N) {
    int idx=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if (idx+3<M*N) {
        int row=idx/M;
        int col=idx%M;
        float4 x_val=make_float4(x[col*N+row],x[(col+1)*N+row],x[(col+2)*N+row],x[(col+3)*N+row]);
        FLOAT4(y[idx])=x_val;
    }
}


__global__ void mat_transpose_f32_col2row2d_kernel(float *x,float *y,int M,int N) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<M&&col<N) {
        y[col*M+row]=x[row*N+col];
    }
}


__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x,float *y,int M,int N) {
    int row=(blockIdx.y*blockDim.y+threadIdx.y)*4;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (row+3<M&&col<N) {
        float4 x_val;
        x_val.x=x[row*N+col];
        x_val.y=x[(row+1)*N+col];
        x_val.z=x[(row+2)*N+col];
        x_val.w=x[(row+3)*N+col];
        FLOAT4(y[col*M+row])=x_val;
    }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y,
                                                   int row,
                                                   int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_y < col && global_x < row) {
        y[global_y * row + global_x] = x[global_x * col + global_y];
    }
}

__global__ void mat_transpose_f32_diagonal2d_kernel(float *x,float *y,int M,int N) {
    int row=blockIdx.x*blockDim.y+threadIdx.y;
    int col=((blockIdx.x+blockIdx.y)%gridDim.x)*blockDim.x+threadIdx.x;
    if (row<M&&col<N) {
        y[col*M+row]=x[row*N+col];
    }
}

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x,float *y,int M,int N) {
    __shared__ float shared_data[WARP_SIZE_S][WARP_SIZE_S*4];
    if (blockIdx.y*blockDim.y+threadIdx.y<M && (blockIdx.x*blockDim.x+threadIdx.x)*4+3<N) {
        float *x_start=x+blockIdx.y*blockDim.y*N+blockDim.x*blockIdx.x*4;
        float *y_start=y+(blockDim.x*blockIdx.x*4)*M+blockIdx.y*blockDim.y;
        FLOAT4(shared_data[threadIdx.y][threadIdx.x*4])=FLOAT4(x_start[threadIdx.y*N+threadIdx.x*4]);
        __syncthreads();
        int idx=threadIdx.y*WARP_SIZE_S+threadIdx.x;
        int num_row=WARP_SIZE_S/4;
        int x_adj=idx%num_row;
        int y_adj=idx/num_row;
        float4 seg;
        seg.x=shared_data[x_adj*4][y_adj];
        seg.y=shared_data[x_adj*4+1][y_adj];
        seg.z=shared_data[x_adj*4+2][y_adj];
        seg.w=shared_data[x_adj*4+3][y_adj];
        FLOAT4(y_start[y_adj*M+x_adj*4])=seg;

    }
}

__global__ void mat_transpose_ref_f32x4_shared_col2row2d_kernel(float *x, float *y,
                                                            int row,
                                                            int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];
    if (global_x * 4 + 3 < col + 3 && global_y < row) {
        // load value from x to shared memory
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y =
            tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z =
            tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w =
            tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        // map index n*n to (n/4)*(n*4)
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x,float *y,int M,int N) {
    __shared__ float shared_data[WARP_SIZE_S][WARP_SIZE_S*4+1];
    if (blockIdx.y*blockDim.y+threadIdx.y<M && (blockIdx.x*blockDim.x+threadIdx.x)*4+3<N) {
        float *x_start=x+blockIdx.y*blockDim.y*N+blockDim.x*blockIdx.x*4;
        float *y_start=y+(blockDim.x*blockIdx.x*4)*M+blockIdx.y*blockDim.y;
        // Padding changes the row stride to 65 floats, so row starts after
        // row 0 are not 16-byte aligned.  Avoid float4 shared-memory stores.
        const float4 input = FLOAT4(x_start[threadIdx.y*N+threadIdx.x*4]);
        shared_data[threadIdx.y][threadIdx.x*4] = input.x;
        shared_data[threadIdx.y][threadIdx.x*4+1] = input.y;
        shared_data[threadIdx.y][threadIdx.x*4+2] = input.z;
        shared_data[threadIdx.y][threadIdx.x*4+3] = input.w;
        __syncthreads();
        int idx=threadIdx.y*WARP_SIZE_S+threadIdx.x;
        int num_row=WARP_SIZE_S/4;
        int x_adj=idx%num_row;
        int y_adj=idx/num_row;
        float4 seg;
        seg.x=shared_data[x_adj*4][y_adj];
        seg.y=shared_data[x_adj*4+1][y_adj];
        seg.z=shared_data[x_adj*4+2][y_adj];
        seg.w=shared_data[x_adj*4+3][y_adj];
        FLOAT4(y_start[y_adj*M+x_adj*4])=seg;
    }
}

__global__ void mat_transpose_ref_f32x4_shared_bcf_col2row2d_kernel(float *x,
                                                                float *y,
                                                                int row,
                                                                int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
    if (global_x * 4 + 3 < col + 3 && global_y < row) {
        // load value from x to shared memory
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        tile[local_y][local_x * 4] = x_val.x;
        tile[local_y][local_x * 4 + 1] = x_val.y;
        tile[local_y][local_x * 4 + 2] = x_val.z;
        tile[local_y][local_x * 4 + 3] = x_val.w;
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y =
            tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z =
            tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w =
            tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        // map index n*n to (n/4)*(n*4)
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel(float *x,float *y,int M,int N) {
    __shared__ float shared_data[WARP_SIZE_S*4][WARP_SIZE_S+1];
    float *x_start=x+blockIdx.y*blockDim.y*4*N+blockIdx.x*blockDim.x;
    shared_data[threadIdx.y*4][threadIdx.x]=x_start[threadIdx.y*4*N+threadIdx.x];
    shared_data[threadIdx.y*4+1][threadIdx.x]=x_start[(threadIdx.y*4+1)*N+threadIdx.x];
    shared_data[threadIdx.y*4+2][threadIdx.x]=x_start[(threadIdx.y*4+2)*N+threadIdx.x];
    shared_data[threadIdx.y*4+3][threadIdx.x]=x_start[(threadIdx.y*4+3)*N+threadIdx.x];
    __syncthreads();
    float4 seg;
    seg.x=shared_data[threadIdx.x*4][threadIdx.y];
    seg.y=shared_data[threadIdx.x*4+1][threadIdx.y];
    seg.z=shared_data[threadIdx.x*4+2][threadIdx.y];
    seg.w=shared_data[threadIdx.x*4+3][threadIdx.y];
    float *y_start=y+blockIdx.x*blockDim.x*M+blockIdx.y*blockDim.y*4;
    FLOAT4(y_start[threadIdx.y*M+threadIdx.x*4])=seg;
}