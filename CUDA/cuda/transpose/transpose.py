import triton
import torch
import triton.language as tl

DEVICE=torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_M":128,"BLOCK_SIZE_N":128},num_stages=3,num_warps=8),
    ],key=["M","N"]
)

@triton.jit
def transpose_kernel(
input_ptr,output_ptr,
M,N,
stride_input_M,stride_input_N,
stride_output_N,stride_output_M,
BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr):
    pid=tl.program_id(0)
    num_pid_m=tl.cdiv(M,BLOCK_SIZE_M)
    num_pid_n=tl.cdiv(N,BLOCK_SIZE_N)
    m=pid//num_pid_n
    n=pid%num_pid_n
    offset_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    input_offset_m=offset_m[:,None]*stride_input_M
    input_offset_n=offset_n[None,:]*stride_input_N
    mask_m=offset_m<M
    mask_n=offset_n<N
    input=tl.load(input_ptr+input_offset_m+input_offset_n,mask=mask_m[:,None]&mask_n[None,:],other=0)
    output=tl.trans(input)
    output_offset_n=offset_n[:,None]*stride_output_N
    output_offset_m=offset_m[None,:]*stride_output_M
    tl.store(output_ptr+output_offset_n+output_offset_m,output,mask=mask_n[:,None]&mask_m[None,:])


def transpose(input:torch.Tensor):
    assert input.is_contiguous()
    assert input.ndim==2
    output = torch.empty_like(input, device=DEVICE, dtype=torch.float32)
    M,N=input.shape
    grid=lambda meta:(triton.cdiv(M,meta["BLOCK_SIZE_M"])*triton.cdiv(N,meta["BLOCK_SIZE_N"]),)
    transpose_kernel[grid](
        input,output,
        M,N,
        input.stride(0),input.stride(1),
        output.stride(0),output.stride(1),
    )
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M","N"],
        x_vals=[128*i for i in range(2,30,4)],
        line_arg='provider',
        line_vals=["triton","pytorch"],
        line_names=["Pytorch","Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/S",
        plot_name="transpose-performance",
        args={},
    )
)
def benchmark(M,N,provider):
    input=torch.randn((M,N),device=DEVICE,dtype=torch.float32)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.transpose(input,0,1), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: transpose(input), quantiles=quantiles)
    perf = lambda ms: 2 * M * N  * 1e-12 / (ms * 1e-3)
    return perf(ms)


def transpose_kernel_test(size:tuple,atol=1e-2,rtol=1e-1,device=DEVICE):
    input=torch.randn(size,device=DEVICE,dtype=torch.float32)
    c_tri=transpose(input)
    c_ref=torch.transpose(input,0,1)
    torch.testing.assert_close(c_ref,c_tri,atol=atol,rtol=rtol)
    print("PASSED")

if __name__ == '__main__':
    transpose_kernel_test((1024,1024))
    benchmark.run(show_plots=True,print_data=False, save_path='.')





