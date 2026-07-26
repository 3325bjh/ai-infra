import torch
import triton
import triton.language as tl

DEVICE=torch.device(f'cuda:{torch.cuda.current_device()}')

# import os
# os.environ["TRITON_INTERPRET"]="1"

autotune_config=[
    triton.Config({"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":64,"GROUP_SIZE":8},num_stages=3,num_warps=8),
    triton.Config({"BLOCK_SIZE_M":64,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=4,num_warps=4),
    triton.Config({"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=4,num_warps=4),
    triton.Config({"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=4,num_warps=4),
    triton.Config({"BLOCK_SIZE_M":64,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=4,num_warps=4),
    triton.Config({"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=4,num_warps=4),
    triton.Config({"BLOCK_SIZE_M":64,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=5,num_warps=2),
    triton.Config({"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":32,"GROUP_SIZE":8},num_stages=5,num_warps=2),
]


@triton.autotune(configs=autotune_config,key=["M","N","K"])
@triton.jit
def _matmul_kernel(
        a_ptr,b_ptr,c_ptr,
        M,N,K,
        stride_a_M,stride_a_K,
        stride_b_K,stride_b_N,
        stride_c_M,stride_c_N,
        BLOCK_SIZE_M:tl.constexpr,
        BLOCK_SIZE_N:tl.constexpr,
        BLOCK_SIZE_K:tl.constexpr,
        GROUP_SIZE:tl.constexpr
):
    PID=tl.program_id(axis=0)
    num_PID_aLONG_M=tl.cdiv(M,BLOCK_SIZE_M)
    num_PID_along_N=tl.cdiv(N,BLOCK_SIZE_N)
    num_PID_in_group=GROUP_SIZE*num_PID_along_N
    group_id=PID//num_PID_in_group
    first_PID_in_group_along_M=group_id*GROUP_SIZE
    group_size_adj=min(num_PID_aLONG_M-first_PID_in_group_along_M,GROUP_SIZE)
    '''
    [0, 2,| 4, 6]
    [1, 3,| 5, 7]
    ------|------
    [8,10,|12,14]
    [9,11,|13,15]
    '''
    PID_M=first_PID_in_group_along_M+((PID)%num_PID_in_group)%group_size_adj
    PID_N=(PID%num_PID_in_group)//group_size_adj
    offsets_M=PID_M*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offsets_N=PID_N*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    offsets_K=tl.arange(0,BLOCK_SIZE_K)
    a_offsets=offsets_M[:,None]*stride_a_M+offsets_K[None,:]*stride_a_K
    b_offsets=offsets_K[:,None]*stride_b_K+offsets_N[None,:]*stride_b_N
    accumulator=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        mask=offsets_K<K-k*BLOCK_SIZE_K
        a=tl.load(a_ptr+a_offsets,mask=mask[None,:],other=0.0)
        b=tl.load(b_ptr+b_offsets,mask=mask[:,None],other=0.0)
        accumulator=tl.dot(a,b,acc=accumulator)
        a_offsets+=BLOCK_SIZE_K*stride_a_K
        b_offsets+=BLOCK_SIZE_K*stride_b_K

    c_offsets=offsets_M[:,None]*stride_c_M+offsets_N[None,:]*stride_c_N
    c_mask=(offsets_M[:,None]<M)&(offsets_N[None,:]<N)
    tl.store(c_ptr+c_offsets,accumulator.to(tl.float16),mask=c_mask)


configs=[triton.testing.Benchmark(
        x_names=["M","N","K"],
        x_vals=[128*i for i in range(2,33,5)],
        line_arg='provider',
        line_vals=['torch','triton'],
        line_names=["Pytorch","Triton"],
        styles=[("green","-"),("blue","-")],
        ylabel="GB/S",
        plot_name="matmul-performance",
        args={},
    )]

@triton.testing.perf_report(configs)
def benchmark(M,N,K,provider):
    a=torch.randn((M,K),device=DEVICE,dtype=torch.float16)
    b=torch.randn((K,N),device=DEVICE,dtype=torch.float16)
    quantiles=[0.5,0.05,0.95]
    if provider=='torch':
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:torch.matmul(a,b),quantiles=quantiles)
    else:
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:matmul(a,b),quantiles=quantiles)
    perf=lambda ms:3*M*N*K*1e-12/(ms*1e-3)
    return perf(ms),perf(max_ms),perf(min_ms)


def matmul(a,b):
    assert a.ndim==b.ndim==2
    assert a.shape[1]==b.shape[0]
    (M,K),(_,N)=a.shape,b.shape
    c=torch.empty((M,N),device=DEVICE,dtype=torch.float16)
    grid=lambda meta:(triton.cdiv(M,meta["BLOCK_SIZE_M"])*triton.cdiv(N,meta["BLOCK_SIZE_N"]),)
    _matmul_kernel[grid](
        a,b,c,
        M,N,K,
        a.stride(0),a.stride(1),
        b.stride(0),b.stride(1),
        c.stride(0),c.stride(1),
    )
    return c





def matmul_kernel_test(size:tuple,atol=1e-2,rtol=1e-1,device=DEVICE):
    torch.manual_seed(0)
    assert type(size)==tuple and len(size)==2
    a=torch.randn(size,device=DEVICE,dtype=torch.float16)
    b=torch.randn(size,device=DEVICE,dtype=torch.float16)
    c_tri=matmul(a,b)
    c_ref=torch.matmul(a,b)
    torch.testing.assert_close(c_tri,c_ref,atol=atol,rtol=rtol)
    print("PASSED")

if __name__ == '__main__':
    matmul_kernel_test(size=(512,512))
    # print("Running benchmark...")
    benchmark.run(show_plots=True, print_data=False, save_path='./')
