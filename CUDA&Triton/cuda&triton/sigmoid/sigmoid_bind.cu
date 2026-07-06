#include <torch/extension.h>
#include <torch/types.h>
#include "sigmoid_kernel.cuh"

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    {                                                               \
        if (T.options().dtype() != (th_type))                       \
        {                                                           \
            std::cout << "Torch Info:" << T.options() << std::endl; \
            throw std::runtime_error("values must be" #th_type);    \
        }                                                           \
    }

#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)  \
  void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      sigmoid_##packed_type##_kernel<<<grid, block>>>(                         \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }



TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}
