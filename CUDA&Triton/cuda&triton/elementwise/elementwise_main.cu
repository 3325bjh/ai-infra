#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "elementwise_kernel.cuh"

#define CUDA_CHECK(call)                                                       \
    do                                                                        \
    {                                                                         \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                     \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                     \
    } while (0)

void add_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

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

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    const int n = 1024;
    const size_t nbytes = n * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(nbytes));
    float *h_b = static_cast<float *>(std::malloc(nbytes));
    float *h_res = static_cast<float *>(std::malloc(nbytes));
    float *ground_truth = static_cast<float *>(std::malloc(nbytes));

    for (int i = 0; i < n; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), nbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), nbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), nbytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice));
    dim3 grid(1);
    dim3 block(1024);
    elementwise_add_f32_kernel<<<grid,block>>>(d_a, d_b, d_res, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost));

    add_cpu(h_a, h_b, ground_truth, n);
    check_result(ground_truth, h_res, n);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_res));
    std::free(h_a);
    std::free(h_b);
    std::free(h_res);
    std::free(ground_truth);

    return 0;
}
