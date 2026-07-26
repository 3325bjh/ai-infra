import time
from functools import partial
from typing import Optional
import triton
import triton.language as tl
import torch
from torch.nn.functional import embedding
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE":512})
    ],key=["emb_size"]
)


@triton.jit
def embedding_kernel(a_ptr,weight_ptr,output_ptr,n,emb_size,BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    row=tl.load(a_ptr+pid)
    offset=tl.arange(0,BLOCK_SIZE)
    weight_start=weight_ptr+row*emb_size
    output_start = output_ptr + pid * emb_size
    for col in tl.range(0,emb_size,BLOCK_SIZE):
        mask=offset+col<emb_size
        weight=tl.load(weight_start+offset+col,mask=mask)
        tl.store(output_start+offset+col,weight,mask=mask)

def triton_embedding(a:torch.Tensor,weight:torch.Tensor,output:torch.Tensor):
    N=a.size(0)
    emb_size=weight.size(1)
    grid=(N,)
    embedding_kernel[grid](a,weight,output,N,emb_size)
    return output

# Load the CUDA kernel as a python module
lib = load(
    name="embedding",
    sources=["embedding_bind.cu","embedding_kernel.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>23}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


Ms = [1024, 4096]  # max value of token_ids
Ns = [2048, 4096]  # seqlen
Ks = [512, 1024]  # embedding size
MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
for M, N, K in MNKs:
    print("-" * 110)
    print(" " * 45 + f"MaxV={M}, SeqLen={N}, EmbSize={K}")
    i = torch.randint(0, M, size=(N,)).cuda().int().contiguous()
    weight = torch.randn((M, K)).float().cuda().contiguous()
    o = torch.zeros((N, K)).float().cuda().contiguous()

    run_benchmark(lib.embedding_f32, i, weight, "f32", o)
    run_benchmark(lib.embedding_f32x4, i, weight, "f32x4", o)
    run_benchmark(lib.embedding_f32x4_pack, i, weight, "f32x4_pack", o)
    run_benchmark(partial(embedding), i, weight, "f32_th")
    run_benchmark(triton_embedding,i,weight,"triton",o)

    print("-" * 110)
    weight_f16 = torch.randn((M, K)).half().cuda().contiguous()
    o_f16 = torch.zeros((N, K)).half().cuda().contiguous()
    run_benchmark(lib.embedding_f16, i, weight_f16, "f16", o_f16)
    run_benchmark(lib.embedding_f16x8, i, weight_f16, "f16x8", o_f16)
    run_benchmark(lib.embedding_f16x8_pack, i, weight_f16, "f16x8_pack", o_f16)
    run_benchmark(partial(embedding), i, weight_f16, "f16_th")
    print("-" * 110)
