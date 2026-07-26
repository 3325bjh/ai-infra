#include <torch/extension.h>
#include <torch/types.h>
#include "embedding_kernel.cuh"

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

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
n_elements)                                    \
void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
torch::Tensor o) {                              \
CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
\
const int N = a.size(0);                                                   \
const int emb_size = weight.size(1);                                       \
dim3 block(emb_size / n_elements);                                         \
dim3 grid(N);                                                              \
embedding_##packed_type##_kernel<<<grid, block>>>(                         \
reinterpret_cast<int *>(a.data_ptr()),                                 \
reinterpret_cast<element_type *>(weight.data_ptr()),                   \
reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
}


TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(embedding_f32)
    TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack)
    TORCH_BINDING_COMMON_EXTENSION(embedding_f16)
    TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack)
}
