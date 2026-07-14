#pragma once
__global__ void cuda_sgemm(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);


template<unsigned int BLOCK_SIZE,unsigned int K_>
__global__ void cuda_sgemm_shared_memory(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);

template<unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm_tile(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);


template<unsigned int BLOCK_SIZE,unsigned int STRIDE>
__global__ void cuda_sgemm_increase_work_of_per_thread(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);

template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_float4(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);

template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_register(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);


template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int M_NUM_PER_THREAD,unsigned int N_NUM_PER_THREAD,unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_register_float4(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);


template<unsigned int M_NUM_PER_BLOCK,unsigned int N_NUM_PER_BLOCK,unsigned int K_NUM_PER_BLOCK,unsigned int M_NUM_PER_THREAD,unsigned int N_NUM_PER_THREAD,unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_transpose(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);

template<const int BLOCK_SIZE_M,const int BLOCK_SIZE_N,const int BLOCK_SIZE_K,const int THREAD_SIZE_Y,const int THREAD_SIZE_X>
__global__ void cuda_sgemm_double_buffer(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);

template<const int BLOCK_SIZE_M,const int BLOCK_SIZE_N,const int BLOCK_SIZE_K,const int THREAD_SIZE_Y,const int THREAD_SIZE_X>
__global__ void cuda_sgemm_double_buffer2(float *A_ptr,float *B_ptr,float *C_ptr,const int M,const int N,const int K);