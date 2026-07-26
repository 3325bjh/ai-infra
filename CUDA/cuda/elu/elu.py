import torch
from typing import Optional

from torch.utils.cpp_extension import load
import triton
import triton.language as tl

DEVICE=torch.device(f'cuda:{torch.cuda.current_device()}')
ALPHA = tl.constexpr(1.0)
lib=load(
    "elu",
    sources=["elu_bind.cu","elu_kernel.cu"],
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
    [
        triton.Config({"BLOCK_SIZE":256}),
        triton.Config({"BLOCK_SIZE":512}),
    ],
    key=['N']
)

@triton.jit
def elu_kernel(x_ptr,y_ptr,N,BLOCK_SIZE:tl.constexpr):
    pid=tl.program_id(0)
    offset=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    mask=offset<N
    x=tl.load(x_ptr+offset,mask=mask)
    y=tl.where(x>0,x,ALPHA*(tl.exp(x)-1))
    tl.store(y_ptr+offset,y,mask=mask)


def elu(
        x:torch.Tensor,y:torch.Tensor
):
    assert x.is_contiguous() and y.is_contiguous()
    num=x.numel()
    grid=lambda meta:(triton.cdiv(num,meta["BLOCK_SIZE"]),)
    elu_kernel[grid](x,y,num)
    return y

def run_benchmark(
    fun:callable,
    x:torch.Tensor,
    tag:str,
    out:Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            fun(x,out)
    else:
        for i in range(warmup):
            _=fun(x)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    if out is not None:
        for i in range(iters):
            fun(x,out)
    else:
        for i in range(iters):
            out=fun(x)
    end.record()
    torch.cuda.synchronize()
    time=start.elapsed_time(end)
    mean_time = time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time




if __name__ == '__main__':
    Ss=[256,512,1024,2048]
    Ks=[256,512,1024,2048]
    SKs=[(S,K) for S in Ss for K in Ks]
    for S,K in SKs:
        print("-"*85)
        x=torch.randn((S,K),dtype=torch.float32,device=DEVICE).contiguous()
        y=torch.zeros_like(x,dtype=torch.float32,device=DEVICE).contiguous()
        out_ref,meantime=run_benchmark(torch.nn.functional.elu,x,"torch")
        out,_=run_benchmark(lib.elu_f32, x, "f32", y)
        torch.testing.assert_close(out_ref,out)
        out,_=run_benchmark(lib.elu_f32x4, x, "f32x4", y)
        torch.testing.assert_close(out_ref, out)
        out, _ = run_benchmark(elu, x, "triton", y)
        torch.testing.assert_close(out_ref, out)
        print("-" * 85)
        x_f16 = x.half().contiguous()
        y_f16 = y.half().contiguous()
        out_ref,meantime=run_benchmark(lib.elu_f16, x_f16, "f16", y_f16)
        out,_=run_benchmark(lib.elu_f16x2, x_f16, "f16x2", y_f16)
        torch.testing.assert_close(out_ref, out)
        out,_=run_benchmark(lib.elu_f16x8, x_f16, "f16x8", y_f16)
        torch.testing.assert_close(out_ref, out)
        out,_=run_benchmark(lib.elu_f16x8_pack, x_f16, "f16x8pack", y_f16)
        torch.testing.assert_close(out_ref, out)
        out,_=run_benchmark(torch.nn.functional.elu, x_f16, "f16_th")
        print("-" * 85)

