import torch
import triton
import triton.language as tl
import math


DEVICE=torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _attn_fwd_inner(
        Q,O,L,M,
        K_ptr,V_ptr,
        K_T_offsets,V_offsets,
        block_index_QO,scale,
        stride_K_N,stride_V_N,
        N,
        BLOCK_SIZE_QO:tl.constexpr,BLOCK_SIZE_KV:tl.constexpr,
        DIAGONAL:tl.constexpr,
        offsets_QO_N:tl.constexpr,offsets_KV_N:tl.constexpr,
        Dh:tl.constexpr,
):
    if DIAGONAL:
        lo=block_index_QO*BLOCK_SIZE_QO
        hi=(block_index_QO+1)*BLOCK_SIZE_QO
        lo=tl.multiple_of(lo,BLOCK_SIZE_QO)

    else:
        lo,hi=0,block_index_QO*BLOCK_SIZE_QO
    K_T_offsets+=lo*stride_K_N
    V_offsets+=lo*stride_V_N
    offsets_KV_N+=lo
    for start_KV in range(lo,hi,BLOCK_SIZE_KV):
        start_KV=tl.multiple_of(start_KV,BLOCK_SIZE_KV)
        mask_KV_N=offsets_KV_N<N
        K_T=tl.load(K_ptr+K_T_offsets,mask=mask_KV_N[None,:],other=0.)
        S=tl.dot(Q,K_T)*scale
        if DIAGONAL:
            causal_mask=offsets_QO_N[:,None]>=offsets_KV_N[None,:]
            S=tl.where(causal_mask,S,-1.0e6)

        M_new=tl.maximum(M,tl.max(S,axis=1))
        S-=M_new[:,None]
        P=tl.exp2(S)
        L_new=tl.sum(P,axis=1)
        alpha=tl.exp2(M-M_new)
        L=L*alpha+L_new
        V=tl.load(V_ptr+V_offsets,mask=mask_KV_N[:,None],other=0.)
        O*=alpha[:,None]
        O=tl.dot(P,V,acc=O)
        M=M_new
        K_T_offsets+=BLOCK_SIZE_KV*stride_K_N
        V_offsets+=BLOCK_SIZE_KV*stride_V_N
        offsets_KV_N+=BLOCK_SIZE_KV
    return O,L,M


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_QO":BLOCK_SIZE_QO,"BLOCK_SIZE_KV":BLOCK_SIZE_KV},num_stages=num_stages,num_warps=num_warps,
        )for BLOCK_SIZE_QO in [16]#,32,64,128]
         for BLOCK_SIZE_KV in [16]#,32,64,128]
         for num_stages in [3,5]
         for num_warps in [4,8,16]
    ],
    key=["H","Dh"]
)

@triton.jit
def attn_fwd(
        Q_ptr,K_ptr,V_ptr,O_ptr,
        LSE_ptr,
        scale,
        stride_Q_B,stride_Q_H,stride_Q_N,stride_Q_Dh,
        stride_K_B,stride_K_H,stride_K_N,stride_K_Dh,
        stride_V_B,stride_V_H,stride_V_N,stride_V_Dh,
        stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
        stride_LSE_B,stride_LSE_H,stride_LSE_N,
        B,H,N,Dh:tl.constexpr,
        BLOCK_SIZE_QO:tl.constexpr,BLOCK_SIZE_KV:tl.constexpr
):
    rln2:tl.constexpr=1.4426950408889634
    """
        e^x=2^(x*1/ln2)
    """
    scale*=rln2
    tl.static_assert(BLOCK_SIZE_KV<=Dh)

    block_index_QO=tl.program_id(axis=0)
    index_BH=tl.program_id(axis=1)
    index_B=index_BH//H
    index_H=index_BH%H
    Q_ptr+=index_B*stride_Q_B+index_H*stride_Q_H
    K_ptr+=index_B*stride_K_B+index_H*stride_K_H
    V_ptr+=index_B*stride_V_B+index_H*stride_V_H
    O_ptr+=index_B*stride_O_B+index_H*stride_O_H
    offsets_QO_N=block_index_QO*BLOCK_SIZE_QO+tl.arange(0,BLOCK_SIZE_QO)
    offsets_KV_N=tl.arange(0,BLOCK_SIZE_KV)
    offsets_DH=tl.arange(0,Dh)
    Q_offests=offsets_QO_N[:,None]*stride_Q_N+offsets_DH[None,:]*stride_Q_Dh #shape (BLOCK_SIZE_QO,DH)
    K_T_offsets=offsets_DH[:,None]*stride_K_Dh+offsets_KV_N[None,:]*stride_K_N
    V_offsets=offsets_KV_N[:,None]*stride_V_N+offsets_DH[None,:]*stride_V_Dh

    mask_QO_N=offsets_QO_N<N
    Q=tl.load(Q_ptr+Q_offests,mask=mask_QO_N[:,None],other=0.)

    M=tl.full(shape=[BLOCK_SIZE_QO],value=-1e6,dtype=tl.float32)
    L=tl.full(shape=[BLOCK_SIZE_QO],value=1.0,dtype=tl.float32)
    O=tl.zeros([BLOCK_SIZE_QO,Dh],dtype=tl.float32)
    O,L,M=_attn_fwd_inner(
        Q,O,L,M,K_ptr,V_ptr,K_T_offsets,V_offsets,
        block_index_QO,
        scale,
        stride_K_N,stride_V_N,
        N,
        BLOCK_SIZE_QO,BLOCK_SIZE_KV,
        False,
        offsets_QO_N,offsets_KV_N,
        Dh
    )
    O,L,M=_attn_fwd_inner(
        Q,O,L,M,K_ptr,V_ptr,K_T_offsets,V_offsets,
        block_index_QO,
        scale,
        stride_K_N,stride_V_N,
        N,
        BLOCK_SIZE_QO,BLOCK_SIZE_KV,
        True,
        offsets_QO_N,offsets_KV_N,
        Dh
    )
    O=O/L[:,None]
    LSE=M+tl.math.log2(L)
    """
    softmax(x_i)=exp(x_i-m_i)/l_i
                =exp(x_i-m_i)/exp(log(l_i))
                =exp(x_i-(m_i+log(l_i)))
    """
    LSE_offsets=index_BH*stride_LSE_H+offsets_QO_N
    LSE_mask=block_index_QO*BLOCK_SIZE_QO+tl.arange(0,BLOCK_SIZE_QO)<N
    tl.store(LSE_ptr+LSE_offsets,LSE,mask=LSE_mask)
    O_offsets=offsets_QO_N[:,None]*stride_O_N+offsets_DH[None,:]*stride_O_Dh
    tl.store(O_ptr+O_offsets,O,mask=mask_QO_N[:,None])


@triton.autotune(
    [
        triton.Config(
            {"PRE_BLOCK_SIZE_ROW":PRE_BLOCK_SIZE_ROW},num_stages=num_stages,num_warps=num_warps,
        )for PRE_BLOCK_SIZE_ROW in [16]
         for num_stages in [3,5]
         for num_warps in [4,8,16]
    ],
    key=["Dh"]
)

@triton.jit
def attn_backward_preprocess(
        O_ptr,dLdO_ptr,Delta_ptr,
        stride_O_B,stride_O_H,stride_O_N,stride_O_Dh,
        stride_dLdO_B,stride_dLdO_H,stride_dLdO_N,stride_dLdO_Dh,
        stride_Delta_B,stride_Delta_H,stride_Delta_N,
        N,Dh:tl.constexpr,
        PRE_BLOCK_SIZE_ROW:tl.constexpr
):
    index_BH=tl.program_id(axis=1)
    row=tl.program_id(axis=0)
    row_offsets=row*PRE_BLOCK_SIZE_ROW+tl.arange(0,PRE_BLOCK_SIZE_ROW)
    col_offsets=tl.arange(0,Dh)
    mask=row_offsets<N
    O_ptr+=index_BH*stride_O_H
    O_offsets=row_offsets[:,None]*stride_O_N+col_offsets[None,:]*stride_O_Dh
    O=tl.load(O_ptr+O_offsets,mask=mask[:,None],other=0.)
    dLdO_ptr+=index_BH*stride_O_H
    dLdO_offsets=row_offsets[:,None]*stride_dLdO_N+col_offsets[None,:]*stride_dLdO_Dh
    dLdO=tl.load(dLdO_ptr+dLdO_offsets,mask=mask[:,None],other=0.)
    Delta=tl.sum(dLdO*O,axis=1)
    Delta_ptr+=index_BH*stride_Delta_H
    tl.store(Delta_ptr+row_offsets,Delta,mask=mask)

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_MACRO":BLOCK_SIZE_MACRO,"BLOCK_SIZE_MICRO":BLOCK_SIZE_MICRO},num_stages=num_stages,num_warps=num_warps,
        )for BLOCK_SIZE_MACRO in [32]
         for BLOCK_SIZE_MICRO in [16]
         for num_stages in [3,]
         for num_warps in [4]
    ],
    key=["Dh"]
)
@triton.jit
def attn_backward(
        Q_ptr,K_ptr,V_ptr,
        dLdO_ptr,dLdQ_ptr,dLdK_ptr,dLdV_ptr,
        LSE_ptr,Delta_ptr,
        scale,
        stride_B,stride_H,stride_N,stride_Dh,
        H,N,Dh:tl.constexpr,
        BLOCK_SIZE_MACRO:tl.constexpr,BLOCK_SIZE_MICRO:tl.constexpr,
):
    ln2:tl.constexpr=0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    idx_BH=tl.program_id(axis=1)
    idx_B=idx_BH//H
    idx_H=idx_BH%H
    BH_jump=idx_B*stride_B+idx_H*stride_H
    Q_ptr+=BH_jump
    K_ptr+=BH_jump
    V_ptr+=BH_jump
    dLdO_ptr+=BH_jump
    dLdQ_ptr+=BH_jump
    dLdK_ptr+=BH_jump
    dLdV_ptr+=BH_jump
    BH_jump=idx_BH*N
    LSE_ptr+=BH_jump
    Delta_ptr+=BH_jump
    tl.static_assert(BLOCK_SIZE_MACRO%BLOCK_SIZE_MICRO==0)
    BLOCK_SIZE_ROW_1:tl.constexpr=BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1:tl.constexpr=BLOCK_SIZE_MACRO

    pid=tl.program_id(0)
    start_COL=pid*BLOCK_SIZE_COL_1
    start_ROW=start_COL
    num_steps=BLOCK_SIZE_COL_1//BLOCK_SIZE_ROW_1

    offsets_COL_1=start_COL+tl.arange(0,BLOCK_SIZE_COL_1)
    offsets_Dh=tl.arange(0,Dh)
    KV_offsets=offsets_COL_1[:,None]*stride_N+offsets_Dh[None,:]*stride_Dh
    KV_mask=offsets_COL_1[:,None]<N
    K=tl.load(K_ptr+KV_offsets,mask=KV_mask,other=0.)
    V=tl.load(V_ptr+KV_offsets,mask=KV_mask,other=0.)
    K*=scale*rln2
    dLdK=tl.zeros([BLOCK_SIZE_COL_1,Dh],dtype=tl.float32)
    dLdV=tl.zeros([BLOCK_SIZE_COL_1,Dh],dtype=tl.float32)
    dLdK,dLdV=_attn_backward_KV(
        K,V,dLdK,dLdV,
        Q_ptr,dLdO_ptr,LSE_ptr,Delta_ptr,
        stride_N,stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1,BLOCK_SIZE_COL_1,
        start_ROW,start_COL,num_steps,
        scale,ln2,rln2,
        MASK=True
    )
    start_ROW += BLOCK_SIZE_MACRO
    N_adj=tl.cdiv(N,BLOCK_SIZE_COL_1)*BLOCK_SIZE_COL_1
    num_steps=(N_adj-start_ROW)//BLOCK_SIZE_ROW_1
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H,N,Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )
    dLdK*=scale*rln2
    tl.store(dLdK_ptr+KV_offsets,dLdK,mask=KV_mask)
    tl.store(dLdV_ptr+KV_offsets,dLdV,mask=KV_mask)

    BLOCK_SIZE_ROW_2:tl.constexpr=BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2:tl.constexpr=BLOCK_SIZE_MICRO

    start_ROW=pid*BLOCK_SIZE_ROW_2
    start_COL=start_ROW
    num_steps=BLOCK_SIZE_ROW_2//BLOCK_SIZE_COL_2

    offsets_ROW=start_ROW+tl.arange(0,BLOCK_SIZE_ROW_2)
    QO_offsets=offsets_ROW[:,None]*stride_N+offsets_Dh[None,:]*stride_Dh
    mask_ROW=offsets_ROW<N
    Q=tl.load(Q_ptr+QO_offsets,mask=mask_ROW[:,None],other=0.)
    Q*=scale*rln2
    dLdO=tl.load(dLdO_ptr+QO_offsets,mask=mask_ROW[:,None],other=0.)
    LSE=tl.load(LSE_ptr+offsets_ROW,mask=mask_ROW,other=0.)[:,None]
    dLdQ=tl.zeros([BLOCK_SIZE_ROW_2,Dh],dtype=tl.float32)
    dLdQ=_attn_backward_Q(
        dLdQ,Q,dLdO,LSE,
        K_ptr,V_ptr,Delta_ptr,
        stride_N,stride_Dh,
        H,N,Dh,
        BLOCK_SIZE_ROW_2,BLOCK_SIZE_COL_2,
        start_ROW,start_COL,num_steps,
        scale,ln2,rln2,
        MASK=True
    )

    end_COL=start_COL
    start_COL=0
    num_steps=end_COL//BLOCK_SIZE_COL_2

    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dLdQ*=scale*rln2
    tl.store(dLdQ_ptr+QO_offsets,dLdQ,mask=mask_ROW[:,None])


@triton.jit
def _attn_backward_KV(
        K,V,dLdK,dLdV,
        Q_ptr,dLdO_ptr,LSE_ptr,Delta_ptr,
        stride_N,strdie_Dh,
        H,N,Dh:tl.constexpr,
        BLOCK_SIZE_ROW:tl.constexpr,BLOCK_SIZE_COL:tl.constexpr,
        start_ROW,start_COL,num_steps,
        scale,ln2:tl.constexpr,rln2:tl.constexpr,
        MASK:tl.constexpr
):
    offsets_ROW=start_ROW+tl.arange(0,BLOCK_SIZE_ROW)
    offsets_COL=start_COL+tl.arange(0,BLOCK_SIZE_COL)
    offsets_Dh=tl.arange(0,Dh)
    Q_T_offsets=offsets_Dh[:,None]*strdie_Dh+offsets_ROW[None,:]*stride_N
    dLdO_offsets=offsets_ROW[:,None]*stride_N+offsets_Dh[None,:]*strdie_Dh
    for block_idx in range(num_steps):
        mask_N=offsets_ROW<N
        Q_T=tl.load(Q_ptr+Q_T_offsets,mask=mask_N[None,:],other=0.)
        LSE=tl.load(LSE_ptr+offsets_ROW,mask=mask_N,other=0.)
        dLdO=tl.load(dLdO_ptr+dLdO_offsets,mask=mask_N[:,None],other=0.)
        Delta=tl.load(Delta_ptr+offsets_ROW,mask=mask_N,other=0.)
        S_T=tl.dot(K,Q_T) #(BLOCK_SIZE_COL,BLOCK_SIZE_ROW)
        P_T=tl.exp2(S_T-LSE[None,:]) #(BLOCK_SIZE_COL,BLOCK_SIZE_ROW)
        if MASK:
            mask=(offsets_COL[:,None]<=offsets_ROW[None,:])
            P_T=tl.where(mask,P_T,0.)
        dLdV=tl.dot(P_T,dLdO,acc=dLdV)
        dLdP_T=tl.dot(V,tl.trans(dLdO))
        dLdS_T=(P_T*(dLdP_T-Delta[None,:])*ln2)
        dLdK=tl.dot(dLdS_T,tl.trans(Q_T),acc=dLdK)

        offsets_ROW+=BLOCK_SIZE_ROW
        Q_ptr+=BLOCK_SIZE_ROW*stride_N
        dLdO_ptr+=BLOCK_SIZE_ROW*stride_N
    return dLdK,dLdV


@triton.jit
def _attn_backward_Q(
        dLdQ,Q,dLdO,LSE,
        K_ptr,V_ptr,Delta_ptr,
        stride_N,stride_Dh,
        H,N,Dh:tl.constexpr,
        BLOCK_SIZE_ROW:tl.constexpr,
        BLOCK_SIZE_COL:tl.constexpr,
        start_ROW,start_COL,num_steps,
        scale,ln2:tl.constexpr,rln2:tl.constexpr,
        MASK:tl.constexpr
):
    offsets_ROW=start_ROW+tl.arange(0,BLOCK_SIZE_ROW)
    offsets_COL=start_COL+tl.arange(0,BLOCK_SIZE_COL)
    offsets_Dh=tl.arange(0,Dh)
    K_and_V_T_offsets=offsets_Dh[:,None]*stride_Dh+offsets_COL[None,:]*stride_N
    Delta=tl.load(Delta_ptr+offsets_ROW,mask=offsets_ROW<N,other=0.)
    for block_idx in range(num_steps):
        KV_mask=(offsets_COL<N)[None,:]
        K_T=tl.load(K_ptr+K_and_V_T_offsets,mask=KV_mask,other=0.)
        V_T=tl.load(V_ptr+K_and_V_T_offsets,mask=KV_mask,other=0.)
        S=tl.dot(Q,K_T) #(BLOCK_SIZE_ROW,BLOCK_SIZE_COL)
        P=tl.exp2(S-LSE)
        if MASK:
            P=tl.where(offsets_ROW[:,None]>=offsets_COL[None,:],P,0.)
        dLdP=tl.dot(dLdO,V_T) #(BLOCK_SIZE_ROW,BLOCK_SIZE_COL)
        dLdS=(P*(dLdP-Delta[:,None])*ln2)
        dLdQ=tl.dot(dLdS,tl.trans(K_T),acc=dLdQ)

        offsets_COL+=BLOCK_SIZE_COL
        K_ptr+=BLOCK_SIZE_COL*stride_N
        V_ptr+=BLOCK_SIZE_COL*stride_N

    return dLdQ






class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,scale):
        assert q.shape==k.shape==v.shape
        assert q.shape[-1]<=128,\
            f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
        assert q.device==k.device and q.device==v.device
        assert q.dtype==k.dtype==v.dtype==torch.float32
        B,H,N,Dh=q.shape
        O=torch.empty_like(q)
        LSE=torch.empty((B,H,N),device=q.device,dtype=torch.float32)
        grid=lambda args:(
            triton.cdiv(N,args["BLOCK_SIZE_QO"]),
            B*H,
        )
        attn_fwd[grid](
            q,k,v,O,LSE,
            scale,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            k.stride(0),k.stride(1),k.stride(2),k.stride(3),
            v.stride(0),v.stride(1),v.stride(2),v.stride(3),
            O.stride(0),O.stride(1),O.stride(2),q.stride(3),
            LSE.stride(0),LSE.stride(1),LSE.stride(2),
            B,H,N,Dh
        )

        ctx.save_for_backward(q,k,v,O,LSE)
        ctx.grid=grid
        ctx.B,ctx.H,ctx.N,ctx.Dh=B,H,N,Dh
        ctx.scale=scale
        return O
    @staticmethod
    def backward(ctx,dLdO):
        q,k,v,O,LSE=ctx.saved_tensors
        grid=ctx.grid
        scale=ctx.scale
        B,H,N,Dh=ctx.B,ctx.H,ctx.N,ctx.Dh
        dLdq=torch.empty_like(q)
        dLdk=torch.empty_like(k)
        dLdv=torch.empty_like(v)
        dLdO=dLdO.contiguous()
        assert q.stride()==k.stride()==v.stride()==O.stride()==dLdO.stride()
        Delta=torch.empty_like(LSE)
        pre_grid=lambda meta:(triton.cdiv(N,meta["PRE_BLOCK_SIZE_ROW"]),B*H)
        attn_backward_preprocess[pre_grid](
            O,dLdO,Delta,
            O.stride(0),O.stride(1),O.stride(2),O.stride(3),
            dLdO.stride(0), dLdO.stride(1), dLdO.stride(2), dLdO.stride(3),
            Delta.stride(0),Delta.stride(1),Delta.stride(2),
            N,Dh
        )
        grid=lambda meta:(triton.cdiv(N,meta["BLOCK_SIZE_MACRO"]),B*H)
        attn_backward[grid](
            q,k,v,
            dLdO,dLdq,dLdk,dLdv,
            LSE,Delta,
            scale,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
            H,N,Dh,
        )
        return dLdq,dLdk,dLdv,None




triton_attention=_flashattention.apply

configs=[]
for mode in ["fwd"]:  #,"bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[512*i for i in range(1,5)],
            line_arg="provider",
            line_vals=["torch","triton"],
            line_names=[
                "torch.nn.functional.scaled_dot_product_attention",
                "Triton"
            ],
            styles=[("red","-"),("blue","-")],
            ylabel="TFLOPS",
            plot_name=f"attention-performance-{mode}",
            args={"mode":mode}
        )
    )

@triton.testing.perf_report(configs)
def bench_mark(SEQ_LEN,mode,provider,device=DEVICE):
    assert mode in ["fwd","bwd"]
    dtype=torch.float32
    BATCH,N_HEADS=32,4
    HEAD_DIM=128
    q=torch.randn((BATCH,N_HEADS,SEQ_LEN,HEAD_DIM),dtype=dtype,device=device,requires_grad=True)
    k=torch.randn((BATCH,N_HEADS,SEQ_LEN,HEAD_DIM),dtype=dtype,device=device,requires_grad=True)
    v=torch.randn((BATCH,N_HEADS,SEQ_LEN,HEAD_DIM),dtype=dtype,device=device,requires_grad=True)
    sm_scale=1/math.sqrt(HEAD_DIM)
    if provider=="torch":
        fn=lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    if provider=="triton":
        fn=lambda:triton_attention(q,k,v,sm_scale)
    if mode=="bwd":
        O=fn()
        dLdO=torch.randn_like(O)
        fn=lambda: O.backward(dLdO,retain_graph=True)
    ms=triton.testing.do_bench(fn)
    flops_per_matmul=2.0*BATCH*N_HEADS*SEQ_LEN*SEQ_LEN*HEAD_DIM
    total_flops=2*flops_per_matmul*0.5
    if mode=="bwd":
        total_flops*=2.5
    return total_flops*1e-12/(ms*1e-3)



def falshattention_kernel_test(B,H,N,Dh,device=DEVICE,atol=5e-3):
    q=torch.randn((B,H,N,Dh),dtype=torch.float32,device=device,requires_grad=True)
    k=torch.randn((B,H,N,Dh),dtype=torch.float32,device=device,requires_grad=True)
    v=torch.randn((B,H,N,Dh),dtype=torch.float32,device=device,requires_grad=True)
    scale=1/math.sqrt(Dh)
    tri_out=triton_attention(q,k,v,scale)
    ref_out=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    torch.testing.assert_close(tri_out,ref_out,atol=atol,rtol=0)
    print("passed fwd")
    dLdout=0.1*torch.randn_like(q)
    tri_out.backward(dLdout,retain_graph=True)
    dLdq_tri,dLdk_tri,dLdv_tri=[_.grad.clone() for _ in [q,k,v]]
    q.grad,k.grad,v.grad=None,None,None
    ref_out.backward(dLdout,retain_graph=True)
    dLdq_ref,dLdk_ref,dLdv_ref=[_.grad.clone() for _ in [q,k,v]]
    q.grad,k.grad,v.grad=None,None,None
    torch.testing.assert_close(dLdq_tri,dLdq_ref,atol=atol,rtol=0)
    torch.testing.assert_close(dLdk_tri,dLdk_ref,atol=atol,rtol=0)
    torch.testing.assert_close(dLdv_tri,dLdv_ref,atol=atol,rtol=0)
    print("passed bwd")


if __name__ == '__main__':
    falshattention_kernel_test(1,1,128,32)
    falshattention_kernel_test(1,1,128,64)
    falshattention_kernel_test(1,1,128,128)
    falshattention_kernel_test(38,8,69,128)
    # bench_mark.run(save_path=".",print_data=False)