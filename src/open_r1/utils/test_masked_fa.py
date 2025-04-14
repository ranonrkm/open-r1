import torch
from torch.nn import functional as F
from einops import rearrange
try:
    from .flag_attn import masked_attention, attention
except:
    from flag_attn import masked_attention, attention

def ref_masked_attention(q, k, v, block_mask, block_size, sm_scale=1.):
    B, H, M, D = q.shape
    attn_scores = torch.einsum('bhmd,bhnd->bhmn', q, k) * sm_scale
    causal_mask = torch.ones(M, M, device=q.device, dtype=torch.bool).tril_(0)[None, None, :, :].expand(B, H, -1, -1)
    block_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, -1, block_size)
    block_mask = block_mask.reshape(B, H, M, -1)
    causal_mask = causal_mask & block_mask
    attn_scores.masked_fill_(causal_mask.logical_not(), float('-inf'))
    S = torch.softmax(attn_scores, dim=-1)
    out = torch.einsum('bhmn,bhnd->bhmd', S, v)
    return out

def repeat_kv(k, v, nrep):
    B, H, M, D = k.shape
    k_rep = k.unsqueeze(2).expand(B, H, nrep, M, D).reshape(B, H * nrep, M, D)
    v_rep = v.unsqueeze(2).expand(B, H, nrep, M, D).reshape(B, H * nrep, M, D)
    return k_rep, v_rep

def test_masked_attention():
    B, H, H_kv, M, D = 1, 16, 4, 2048, 128
    block_size = 16
    q = torch.randn(B, H, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    block_mask = torch.randint(0, 2, (B, H_kv, M, M)).gt(0).cuda()
    block_mask.diagonal(dim1=-2, dim2=-1).fill_(1)
    block_mask = rearrange(block_mask, 'b h m (n r) -> b h m n r', r=block_size)[..., 0]
    block_mask[..., 0] = 1
    sm_scale = D**-0.5
    
    k_rep, v_rep = repeat_kv(k, v, H // H_kv)
    block_mask_rep = block_mask.unsqueeze(2).expand(-1, -1, H // H_kv, -1, -1).reshape(B, H, M, -1)
    # out_ref = ref_masked_attention(q, k_rep, v_rep, block_mask_rep, block_size, sm_scale)
    # out_ref.sum().backward()
    
    # copy gradients and set them to None
    # ref_q_grad = q.grad.clone()
    # ref_k_grad = k.grad.clone()
    # ref_v_grad = v.grad.clone()
    # q.grad = None
    # k.grad = None
    # v.grad = None
    
    out = masked_attention(q, k, v, block_mask.contiguous(), causal=True, sm_scale=sm_scale, dropout_p=0.0)
    # assert torch.allclose(out, out_ref, atol=1e-2)
    out.sum().backward()
    
    dq = q.grad.clone()
    dk = k.grad.clone()
    dv = v.grad.clone()
    q.grad = None
    k.grad = None
    v.grad = None
    
    
    attn_mask = block_mask_rep.unsqueeze(-1).expand(-1, -1, -1, -1, 16).reshape(B, H, M, -1)
    causal_mask = torch.ones(M, M, device=q.device, dtype=torch.bool).tril_(0)[None, None, :, :].expand(B, H, -1, -1)
    attn_mask = attn_mask & causal_mask
    out_sdpa = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0)
    out_sdpa.sum().backward()
    
    import pdb; pdb.set_trace()
    assert torch.allclose(out_sdpa, out, atol=1e-2)
    assert torch.allclose(q.grad, dq, atol=1e-2)
    assert torch.allclose(k.grad, dk, atol=1e-2)
    assert torch.allclose(v.grad, dv, atol=1e-2)
    

test_masked_attention()