import sys
import pytest
import torch
import math

import triton
import triton.language as tl

from flash_attn_custom import flash_attn_func_custom

DEVICE = torch.cuda.current_device()

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse, TMP, sm_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, grp_size,
    seqlen, seqlen_rounded,
    D: tl.constexpr, 
    CACHE_KEY_SEQLEN, 
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):  
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_h_kv = off_h // grp_size
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    #init pointers
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h_kv * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h_kv * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    if EVEN_M & EVEN_N:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    # loop over k, v and update accumulator
    end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen)     # as causal
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # --- compute qk ---
        if EVEN_N & EVEN_M:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q,tl.trans(k))
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen, 0, float("-inf"))
        # causal mask
        qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1) * sm_scale, lse_i)  
        p = tl.exp(qk * sm_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M: 
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen,
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # store output
    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    offs_d = tl.arange(0, D)
    out_ptrs = (
        Out 
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen)


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, nheads_kv, _ = k.shape
        assert seqlen_q == seqlen_k 
        assert nheads >= nheads_kv and nheads % nheads_kv == 0
        grp_size = nheads // nheads_kv
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        sm_scale = sm_scale or 1.0 / math.sqrt(d)

        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            tmp,
            sm_scale,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            o.stride(0), o.stride(2), o.stride(1),
            nheads, grp_size,
            seqlen_q, seqlen_q_rounded,
            d, 
            seqlen_q // 32, 
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, num_warps=num_warps, num_stages=1
        )
        ctx.save_for_backward(q, k, v, o, lse)
        return o
    
    @staticmethod
    def backward(ctx, do):
        return None, None, None, None
    
fa_func = FlashAttnFunc.apply

# utility fn for baseline
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# @pytest.mark.parametrize("Z, H, H_kv, N_CTX, HEAD_DIM", [(1, 16, 2, 1024, 128)])
# def test_fwd(Z, H, H_kv, N_CTX, HEAD_DIM, dtype=torch.bfloat16):
#     torch.manual_seed(20)
#     q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
#     k = (torch.empty((Z, H_kv, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
#     v = (torch.empty((Z, H_kv, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
#     sm_scale = 0.5
#     dout = torch.randn_like(q)
#     sm_scale = math.sqrt(HEAD_DIM)
#     # reference implementation
#     k_rep = repeat_kv(k, H // H_kv)
#     v_rep = repeat_kv(v, H // H_kv)
#     M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
#     p = torch.matmul(q, k_rep.transpose(2, 3)) * sm_scale
#     p[:, :, M == 0] = float("-inf")
#     p = torch.softmax(p.float(), dim=-1).to(dtype)
#     ref_out = torch.matmul(p, v_rep)
#     # triton implementation
#     out = fa_func(q, k, v, sm_scale)
#     # compare
#     assert torch.allclose(out, ref_out, atol=1e-2, rtol=0)

@pytest.mark.parametrize("Z, H, H_kv, N_CTX, HEAD_DIM", [(1, 16, 16, 1024, 128)])
def test_fwd(Z, H, H_kv, N_CTX, HEAD_DIM, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (torch.empty((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, N_CTX, H_kv, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, N_CTX, H_kv, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    dout = torch.randn_like(q)
    sm_scale = math.sqrt(HEAD_DIM)
    # reference implementation
    ref_out = flash_attn_func_custom(q, k, v, None, True, sm_scale)
    # triton implementation
    out = fa_func(q, k, v, sm_scale)
    import pdb; pdb.set_trace()
    # compare
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=0)

if __name__ == '__main__':
    exit_code = pytest.main([__file__])
    sys.exit(exit_code)