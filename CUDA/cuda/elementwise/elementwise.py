import time
from array import array
from functools import partial
from typing import Optional
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE':256},num_warps=4),
        triton.Config({'BLOCK_SIZE':512},num_warps=4),
        triton.Config({'BLOCK_SIZE':1024},num_warps=4),
        triton.Config({'BLOCK_SIZE':2048},num_warps=4),
        triton.Config({'BLOCK_SIZE':256},num_warps=8),
        triton.Config({'BLOCK_SIZE':512},num_warps=8),
        triton.Config({'BLOCK_SIZE':1024},num_warps=8),
        triton.Config({'BLOCK_SIZE':2048},num_warps=8),
        triton.Config({'BLOCK_SIZE':256},num_warps=16),
        triton.Config({'BLOCK_SIZE':512},num_warps=16),
        triton.Config({'BLOCK_SIZE':1024},num_warps=16),
        triton.Config({'BLOCK_SIZE':2048},num_warps=16),
    ],
    key=['n']
)


@triton.jit
def elementwise_kernel(x_ptr,y_ptr,output_ptr,n,BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    block_start=pid*BLOCK_SIZE
    offsets=block_start+tl.arange(0,BLOCK_SIZE)
    mask=offsets<n
    x=tl.load(x_ptr+offsets,mask=mask)
    y=tl.load(y_ptr+offsets,mask=mask)
    output=x+y
    tl.store(output_ptr+offsets,output,mask)


def elementwise(x:torch.Tensor,y:torch.Tensor,output:torch.Tensor):
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n=x.numel()
    grid=lambda meta:(triton.cdiv(n,meta["BLOCK_SIZE"]),)
    elementwise_kernel[grid](x,y,output,n)
    return output


lib=load(
    name="elementwise_lib",
    sources=["elementwise_bind.cu", "elementwise.cu"],
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

def run_benchmark2(
        func:callable,
        a:torch.Tensor,
        b:torch.Tensor,
        tag:str,
        out:Optional[torch.Tensor]=None,
        warmup:int=10,
        iters:int=1000,
        show_all:bool=False
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            func(a,b,out)
    else:
        for i in range(warmup):
            _=func(a,b)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    if out is not None:
        for i in range(iters):
            func(a, b, out)
    else:
        for i in range(iters):
            out = func(a, b)
    end.record()
    torch.cuda.synchronize()
    total_time=start.elapsed_time(end)
    mean_time=total_time/iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    N=a.numel()
    bytes_total=3*N*4
    bw=(bytes_total/1e9)/(mean_time/1000.0)
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms,bw:{bw:.2f}GB/s")
    if show_all:
        print(out)
    return out, mean_time




def run_benchmark(
        func:callable,
        a:torch.Tensor,
        b:torch.Tensor,
        tag:str,
        out:Optional[torch.Tensor]=None,
        warmup:int=10,
        iters:int=1000,
        show_all:bool=False
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            func(a,b,out)
    else:
        for i in range(warmup):
            _=func(a,b)
    torch.cuda.synchronize()
    start=time.time()
    if out is not None:
        for i in range(iters):
            func(a,b,out)
    else:
        for i in range(iters):
            out=func(a,b)
    torch.cuda.synchronize()
    end=time.time()
    total_time=(end-start)*1000
    mean_time=total_time/iters
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
    for S in Ss:
        for K in Ks:
            print("-" * 85)
            print(" " * 40 + f"S={S}, K={K}")
            a=torch.randn((S,K)).cuda().float().contiguous()
            b=torch.randn((S,K)).cuda().float().contiguous()
            c=torch.zeros((S,K)).cuda().float().contiguous()
            run_benchmark2(lib.elementwise_add_f32,a,b,"f32",c)
            run_benchmark2(lib.elementwise_add_f32x4,a,b,"f32x4",c)
            run_benchmark2(partial(torch.add,out=c),a,b,"native")
            run_benchmark2(elementwise,a,b,"triton",c)
            print("-" * 85)
            a_f16 = a.half().contiguous()
            b_f16 = b.half().contiguous()
            c_f16 = c.half().contiguous()
            run_benchmark2(lib.elementwise_add_f16, a_f16, b_f16, "f16", c_f16)
            run_benchmark2(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", c_f16)
            run_benchmark2(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
            run_benchmark2(
                lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16
            )
            run_benchmark2(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_th")
            run_benchmark2(elementwise,a_f16,b_f16,"triton",c_f16)
            print("-" * 85)
