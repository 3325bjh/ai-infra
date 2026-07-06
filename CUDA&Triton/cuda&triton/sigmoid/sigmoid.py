from functools import partial

import torch
import triton
import triton.language as tl
from typing import Optional
from torch.utils.cpp_extension import load

lib=load(
    name="sigmoid_lib",
    sources=["sigmoid_bind.cu","sigmoid_kernel.cu"],
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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['n']
)

@triton.jit
def sigmoid_kernel(x_ptr,output_ptr,n,BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    block_start=pid*BLOCK_SIZE
    offsets=block_start+tl.arange(0,BLOCK_SIZE)
    mask=offsets<n
    x=tl.load(x_ptr+offsets,mask=mask)
    y=1.0/(1+tl.exp(-x))
    tl.store(output_ptr+offsets,y,mask=mask)


def sigmoid(x:torch.Tensor,output:torch.Tensor):
    assert x.is_cuda and output.is_cuda and x.is_contiguous()
    n=x.numel()
    grid=lambda meta:(triton.cdiv(n,meta["BLOCK_SIZE"]),)
    sigmoid_kernel[grid](x,output,n)
    return output


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x,out)
    else:
        for i in range(warmup):
            _=perf_func(x)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    if out is not None:
        for i in range(iters):
            perf_func(x,out)
    else:
        for i in range(iters):
            out = perf_func(x)
    end.record()
    torch.cuda.synchronize()
    time=start.elapsed_time(end)
    mean_time = time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

if __name__ == '__main__':
    Ss=[1024,2048,4096]
    Ks=[1024,2048,4096]
    SKs=[(S,K) for S in Ss for K in Ks]
    for S,K in SKs:
        print("-" * 85)
        print(" " * 40 + f"S={S}, K={K}")
        x=torch.randn((S,K)).cuda().float().contiguous()
        y=torch.zeros_like(x).cuda().float().contiguous()
        run_benchmark(partial(torch.sigmoid, out=y) ,x, "f32_th")
        run_benchmark(sigmoid,x,"triton",y)
        run_benchmark(lib.sigmoid_f32, x, "f32", y)
        run_benchmark(lib.sigmoid_f32x4, x, "f32x4", y)

        print("-" * 85)
        x_f16=x.half().contiguous()
        y_f16 = y.half().contiguous()
        run_benchmark(lib.sigmoid_f16, x_f16, "f16", y_f16)
        run_benchmark(lib.sigmoid_f16x2, x_f16, "f16x2", y_f16)
        run_benchmark(lib.sigmoid_f16x8, x_f16, "f16x8", y_f16)
        run_benchmark(lib.sigmoid_f16x8_pack, x_f16, "f16x8pack", y_f16)
        run_benchmark(partial(torch.sigmoid, out=y_f16), x_f16, "f16_th")
        print("-" * 85)