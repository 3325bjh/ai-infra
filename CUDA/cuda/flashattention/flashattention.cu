#include <float.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

void init(float *q,float *k,float *v,int B,int H,int S,int Hidden) {
    for (int b=0;b<B;b++) {
        for (int h=0;h<H;h++) {
            for (int s=0;s<S;s++) {
                for (int hidden=0;hidden<Hidden;hidden++) {
                    q[b*H*S*Hidden+h*S*Hidden+s*Hidden+hidden]=2*(float)drand48()-1;
                    k[b*H*S*Hidden+h*S*Hidden+s*Hidden+hidden]=2*(float)drand48()-1;
                    v[b*H*S*Hidden+h*S*Hidden+s*Hidden+hidden]=2*(float)drand48()-1;
                }
            }
        }
    }

}

void safe_softmax(float *qk, int seq_len, bool causal) {
    for (int s = 0; s < seq_len; s++) {
        float max_val = -FLT_MAX;
        if (causal) {
            for (int j = s + 1; j < seq_len; ++j) {
                qk[s * seq_len + j] = -FLT_MAX;
            }
        }

        for (int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, qk[s * seq_len + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float value = expf(qk[s * seq_len + j] - max_val);
            qk[s * seq_len + j] = value;
            sum += value;
        }

        for (int j = 0; j < seq_len; j++) {
            qk[s * seq_len + j] /= sum;
        }
    }
}

void qk_gemm(const float *q, const float *k,
             int M, int N, int K, float *output, float scale) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int d = 0; d < K; d++) {
                sum += q[i * K + d] * k[j * K + d];
            }
            output[i * N + j] = sum * scale;
        }
    }
}

void gemm(const float *a, const float *b,
          int M, int N, int K, float *output) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            output[i * N + j] = sum;
        }
    }
}

void attention(const float *q, const float *k, const float *v,
               int S, int Hidden, float *output,bool causal) {
    float *qk = (float *)malloc((size_t)S * S * sizeof(*qk));
    if (!qk) return;  // 实际项目建议返回错误码

    float scale = 1.0f / sqrtf((float)Hidden);

    // qk = Q * K^T / sqrt(Hidden)
    qk_gemm(q, k, S, S, Hidden, qk, scale);
    safe_softmax(qk, S,causal);

    // output = softmax(qk) * V
    gemm(qk, v, S, Hidden, S, output);

    free(qk);
}

void cpu_attention(const float *q, const float *k, const float *v,
                   int B, int H, int S, int Hidden, float *output,bool causal) {
    size_t head_stride = (size_t)S * Hidden;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            size_t offset = ((size_t)b * H + h) * head_stride;

            attention(q + offset, k + offset, v + offset,
                      S, Hidden, output + offset,causal);
        }
    }
}

template<int BR,int BC>
__global__ void flash_attention_kernel(float *q,float *k,float *v,int seq_len,int hidden_dim,float *output,float *m,float *sum) {
    int bh = blockIdx.y * gridDim.x + blockIdx.x;
    float *q_start = q + (size_t)bh * seq_len * hidden_dim;
    float *k_start = k + (size_t)bh * seq_len * hidden_dim;
    float *v_start = v + (size_t)bh * seq_len * hidden_dim;
    float *output_start = output + (size_t)bh * seq_len * hidden_dim;
    float *m_start = m + (size_t)bh * seq_len;
    float *sum_start = sum + (size_t)bh * seq_len;
    int tx=threadIdx.x;
    int bx=blockDim.x;
    float dk = rsqrtf((float)hidden_dim);
    extern __shared__ float shared_mem[];
    // __shared__ float shared_q[BR][hidden_dim];
    float *shared_q=shared_mem;
    float qk[BC];
    // __shared__ float shared_k[BC][hidden_dim];
    float *shared_k=shared_q+BR*hidden_dim;
    // __shared__ float shared_v[BC][hidden_dim];
    float *shared_v=shared_k+BC*hidden_dim;
    for (int s = tx; s < seq_len; s += blockDim.x) {
        m_start[s] = -FLT_MAX;
        sum_start[s] = 0.0f;

        for (int h = 0; h < hidden_dim; h++) {
            output_start[s * hidden_dim + h] = 0.0f;
        }
    }
    __syncthreads();
    for (int i=0;i<seq_len;i+=BC) {
        for (int t=tx;t<BC;t+=bx) {
            for (int h=0;h<hidden_dim;h++) {
                shared_k[t*hidden_dim+h]=k_start[(i+t)*hidden_dim+h];
                shared_v[t*hidden_dim+h]=v_start[(i+t)*hidden_dim+h];
            }
        }
        __syncthreads();
        for (int j=0;j<seq_len;j+=BR) {
            for (int t=tx;t<BR;t+=bx) {
                for (int h=0;h<hidden_dim;h++) {
                    shared_q[t*hidden_dim+h]=q_start[(j+t)*hidden_dim+h];
                }
            }
            __syncthreads();
            float max=-FLT_MAX;
            for (int c = 0; c < BC; c++) {
                float dot = 0.0f;

                for (int h = 0; h < hidden_dim; h++) {
                    dot += shared_q[tx * hidden_dim + h] *
                           shared_k[c * hidden_dim + h];
                }

                qk[c] = dot * dk;
                max = fmaxf(max, qk[c]);
            }
            float preMax=m_start[j+tx];
            float preSum=sum_start[j+tx];
            max=fmaxf(max,preMax);
            float curSum=preSum*expf(preMax-max);
            for (int c=0;c<BC;c++) {
                qk[c]=expf(qk[c]-max);
                curSum+=qk[c];
            }
            for (int h=0;h<hidden_dim;h++) {
                float numerator =
                      expf(preMax-max) * preSum * output_start[(j+tx)*hidden_dim+h];
                for (int c=0;c<BC;c++) {
                    numerator+=shared_v[c*hidden_dim+h]*qk[c];
                }
                output_start[(j+tx)*hidden_dim+h]=numerator/curSum;
            }
            m_start[j+tx]=max;
            sum_start[j+tx]=curSum;
            __syncthreads();
        }
    }
}

template<int BR,int BC,int D>
__global__ void flash_attention2_kernel(float *q,float *k,float *v,int headNum,int seq_len,int hidden_dim,float *output, bool causal) {
    float *q_start=q+(blockIdx.z*headNum+blockIdx.y)*seq_len*hidden_dim+blockIdx.x*BR*hidden_dim;
    float *k_start=k+(blockIdx.z*headNum+blockIdx.y)*seq_len*hidden_dim;
    float *v_start=v+(blockIdx.z*headNum+blockIdx.y)*seq_len*hidden_dim;
    float *output_start=output+(blockIdx.z*headNum+blockIdx.y)*seq_len*hidden_dim+blockIdx.x*BR*hidden_dim;
    extern __shared__ float shared_mem[];
    float* shared_q=shared_mem;
    float* shared_k=shared_q+BR*hidden_dim;
    float* shared_v=shared_k+BC*hidden_dim;
    float o[D]={};
    float S[BC];
    float premax=-FLT_MAX;
    float l=0.0f;
    float dk = rsqrtf((float)hidden_dim);
    for (int i=0;i<hidden_dim;i++) {
        shared_q[threadIdx.x*hidden_dim+i]=q_start[threadIdx.x*hidden_dim+i];
    }
    __syncthreads();
    for (int j=0;j<seq_len;j+=BC) {
        for (int i=threadIdx.x;i<BC;i+=BR) {
            for (int h=0;h<hidden_dim;h++) {
                shared_k[i*hidden_dim+h]=k_start[(j+i)*hidden_dim+h];
                shared_v[i*hidden_dim+h]=v_start[(j+i)*hidden_dim+h];
            }
        }
        __syncthreads();
        float max=-FLT_MAX;
        const int q_pos = blockIdx.x * BR + threadIdx.x;
        for (int col = 0; col < BC; ++col) {
            const int k_pos = j + col;

            if (k_pos >= seq_len || (causal && k_pos > q_pos)) {
                S[col] = -FLT_MAX;
            } else {
                float score = 0.0f;
                for (int d = 0; d < hidden_dim; ++d) {
                    score += shared_q[threadIdx.x * hidden_dim + d] *
                             shared_k[col * hidden_dim + d];
                }
                S[col] = score * dk;
            }

            max = fmaxf(max, S[col]);
        }
        float curmax=fmax(max,premax);
        l=l*expf(premax-curmax);

        for (int col=0;col<BC;col++) {
            S[col]=expf(S[col]-curmax);
            l+=S[col];
        }
        for (int h=0;h<hidden_dim;h++) {
            o[h]=o[h]*expf(premax-curmax);
            for (int bc=0;bc<BC;bc++) {
                o[h]+=S[bc]*shared_v[bc*hidden_dim+h];
            }
        }
        premax=curmax;
        __syncthreads();
    }
    for (int h=0;h<hidden_dim;h++) {
        o[h]/=l;
        output_start[threadIdx.x*hidden_dim+h]=o[h];
    }
}


// Compares the GPU result against cpu_attention's output.  The tolerance rule
// is: abs(gpu - cpu) <= atol + rtol * abs(cpu).
bool validate_attention_output(const float *cpu_output, const float *d_output,
                               int batch_size, int head_num,
                               int seq_len, int hidden_dim,
                               float atol = 1e-3f, float rtol = 1e-3f) {
    const size_t element_count =
        (size_t)batch_size * head_num * seq_len * hidden_dim;
    std::vector<float> gpu_output(element_count);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "flash attention kernel failed: %s\n",
                cudaGetErrorString(status));
        return false;
    }

    status = cudaMemcpy(gpu_output.data(), d_output,
                        element_count * sizeof(float),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "copying attention output failed: %s\n",
                cudaGetErrorString(status));
        return false;
    }

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t max_error_index = 0;
    size_t failed_count = 0;

    for (size_t idx = 0; idx < element_count; ++idx) {
        const float reference = cpu_output[idx];
        const float actual = gpu_output[idx];
        const float abs_error = fabsf(actual - reference);
        const float rel_error = abs_error / fmaxf(fabsf(reference), 1e-6f);

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_error_index = idx;
        }
        max_rel_error = fmaxf(max_rel_error, rel_error);

        if (!isfinite(actual) || abs_error > atol + rtol * fabsf(reference)) {
            ++failed_count;
        }
    }

    const size_t head_stride = (size_t)seq_len * hidden_dim;
    const size_t bh = max_error_index / head_stride;
    const size_t token_and_dim = max_error_index % head_stride;
    const int batch = (int)(bh / head_num);
    const int head = (int)(bh % head_num);
    const int token = (int)(token_and_dim / hidden_dim);
    const int dim = (int)(token_and_dim % hidden_dim);

    printf("attention validation: %s\n",
           failed_count == 0 ? "PASS" : "FAIL");
    printf("  max_abs_error = %.8g, max_rel_error = %.8g\n",
           max_abs_error, max_rel_error);
    printf("  max error at [batch=%d, head=%d, token=%d, dim=%d]: "
           "cpu=%.8g, gpu=%.8g\n",
           batch, head, token, dim,
           cpu_output[max_error_index], gpu_output[max_error_index]);
    printf("  tolerance failures = %zu / %zu (atol=%g, rtol=%g)\n",
           failed_count, element_count, atol, rtol);

    return failed_count == 0;
}

int main() {
    const int batch_size=4;
    const int head_num=5;
    const int seq_len=1024;
    constexpr int hidden_dim=16;
    float* q,*k,*v,*output;
    constexpr bool causal = true;
    float* d_q,*d_k,*d_v,*d_output,*d_m,*d_sum;
    size_t size=batch_size*head_num*seq_len*hidden_dim*sizeof(float);
    size_t size_m=batch_size*head_num*seq_len*sizeof(float);
    size_t size_sum=batch_size*head_num*seq_len*sizeof(float);
    q=(float*)malloc(size);
    k=(float*)malloc(size);
    v=(float*)malloc(size);
    output=(float*)malloc(size);
    init(q,k,v,batch_size,head_num,seq_len,hidden_dim);
    cpu_attention(q,k,v,batch_size,head_num,seq_len,hidden_dim,output,causal);
    cudaMalloc((void **)&d_q,size);
    cudaMalloc((void **)&d_k,size);
    cudaMalloc((void **)&d_v,size);
    cudaMalloc((void **)&d_output,size);
    cudaMalloc((void **)&d_m,size_m);
    cudaMalloc((void **)&d_sum,size_sum);
    cudaMemcpy(d_q,q,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_k,k,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,v,size,cudaMemcpyHostToDevice);
    constexpr int br=32;
    constexpr int bc=64;
    // size_t shared_bytes =
    // (br + 2 * bc) * hidden_dim * sizeof(float);
    // dim3 grid(head_num,batch_size);
    // dim3 block(br);
    // flash_attention_kernel<br,bc><<<grid,block,shared_bytes>>>(d_q,d_k,d_v,seq_len,hidden_dim,d_output,d_m,d_sum);
    size_t shared_bytes =
    (br + 2 * bc) * hidden_dim * sizeof(float);
    dim3 grid(seq_len/br,head_num,batch_size);
    dim3 block(br);
    flash_attention2_kernel<br,bc,hidden_dim><<<grid,block,shared_bytes>>>(d_q,d_k,d_v,head_num,seq_len,hidden_dim,d_output,causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    bool passed = validate_attention_output(
        output, d_output, batch_size, head_num, seq_len, hidden_dim);
    return passed ? 0 : 1;

}
