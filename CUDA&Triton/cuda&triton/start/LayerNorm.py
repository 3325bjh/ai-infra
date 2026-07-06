import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X,Y,W,B,Mean,Rstd,stride,N,eps,BLOCK_SIZE: tl.constexpr):
    row=tl.program_id(0)
    Y+=row*stride
    X+=row*stride
    _mean=tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    for off in range(0,N,BLOCK_SIZE):
        cols=off+tl.arange(0,BLOCK_SIZE)
        a=tl.load(X+cols,mask=cols<N,other=0.).to(tl.float32)
        _mean+=a
    mean=tl.sum(_mean,axis=0)/N

    _var=tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    for off in range(0,N,BLOCK_SIZE):
        cols=off+tl.arange(0,BLOCK_SIZE)
        x=tl.load(X+cols,mask=cols<N,other=0.).to(tl.float32)
        x=tl.where(cols<N,x,0.)
        _var+=x*x
    var=tl.sum(_var,axis=0)/N
    rstd=1/tl.sqrt(var+eps)

    tl.store(Mean+row,mean)
    tl.store(Rstd+row,rstd)

    for off in range(0,N,BLOCK_SIZE):
        cols=off+tl.arange(0,BLOCK_SIZE)
        mask=cols<N
        w=tl.load(W+cols,mask=mask)
        b=tl.load(B+cols,mask=mask)
        x=tl.load(X+cols,mask=mask,other=0.).to(tl.float32)
        x_hat=(x-mean)*rstd
        y=x_hat*w+b
        tl.store(Y+cols,y,mask=mask)


def layer_norm(x,wight,bias,eps):
    y=torch.empty_like(x)
    x_arg=x.reshape(-1,x.shape[-1])
    M,N=x_arg.shape
    mean=torch.empty((M,),dtype=torch.float32,device=x.device)
    rstd=torch.empty((M,),dtype=torch.float32,device=x.device)

    MAX_FUSED_SIEZE=65536//x.element_size()
    BLOCK_SIZE=min(MAX_FUSED_SIEZE,triton.next_power_of_2(N))
    if N>BLOCK_SIZE:
        raise RuntimeError(f'LayerNorm: N={N} is too large for BLOCK_SIZE={BLOCK_SIZE}. Please reduce the size of the last dimension of x.')

    _layer_norm_kernel[(M,)](x_arg,y,wight,bias,mean,rstd,x_arg.stride(0),N,eps,BLOCK_SIZE=BLOCK_SIZE)

    return y

if __name__ == '__main__':
    M=1151
    N=8192
    dtype=torch.float16
    eps=1e-5
    device='cuda'
    x_shape=(M,N)
    w_shape=(x_shape[-1],)
    weight=torch.rand(w_shape,dtype=dtype,device=device,requires_grad=True)
    bias=torch.rand(w_shape,dtype=dtype,device=device,requires_grad=True)

    x=-2.3+0.5*torch.rand(x_shape,dtype=dtype,device=device)
    y_tri=layer_norm(x,weight,bias,eps)
    print(y_tri)