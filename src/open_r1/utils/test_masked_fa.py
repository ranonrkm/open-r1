import torch
from torch.nn import functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
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

BLOCK_SIZE = 16
@torch.no_grad()
def create_custom_block_mask(query, key, local, offset, topk):
    # create key blocks from key given block_size
    B, H, N, D = key.shape
    blocked_len = (N // BLOCK_SIZE) * BLOCK_SIZE
    key_blocks = rearrange(key[..., :blocked_len, :], 
                           "b h (n r) d -> b h n r d", 
                           r=BLOCK_SIZE).mean(dim=-2)  # B H N_blk D
    
    local_mask = torch.ones(N, N, device=key.device, dtype=torch.bool).tril_(0)
    assert offset >= local, "offset should be greater than or equal to local"
    dynamic_mask = torch.ones(N - offset, N, device=key.device, dtype=torch.bool).tril_(offset - local)
    local_mask[offset:, :].logical_xor_(dynamic_mask)
    
    # block level masks
    local_mask = rearrange(local_mask, "m (n r) -> m n r", r=BLOCK_SIZE).any(dim=-1)
    dynamic_mask = rearrange(dynamic_mask, "m (n r) -> m n r", r=BLOCK_SIZE).all(dim=-1)
    
    w = torch.einsum("bhmd,bhnd->bhmn", query.narrow(2, offset, N - offset), key_blocks)
    w.masked_fill_(dynamic_mask.logical_not(), -float("inf"))
    topk_blocks = w.topk(topk, dim=-1).indices
    topk_mask = torch.zeros_like(w, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_blocks, 1)
    topk_mask.logical_and_(dynamic_mask)
    
    local_mask = local_mask[None, None, ...].expand(B, H, -1, -1)
    attn_mask = torch.cat(
        [local_mask[..., :offset, :], topk_mask | local_mask[..., offset:, :]],
        dim=-2
    )
    return attn_mask

def flex_attn(q, k, v, block_mask, block_size, sm_scale=1.):
    B, H, M, D = q.shape
    H_kv = k.shape[1]
    grp_size = H // H_kv
    q = rearrange(q, "b (h r) m d -> b h (r m) d", r=grp_size)
    import pdb; pdb.set_trace()
    def flexify_block_mask(b, h, q_idx, k_idx):
        return block_mask[b, h, q_idx // grp_size, k_idx // block_size] & (q_idx // grp_size >= k_idx)
    flex_block_mask = create_block_mask(flexify_block_mask, B, H_kv, M * grp_size, M, BLOCK_SIZE=[block_size, grp_size])
    out = flex_attention(q, k, v, block_mask=flex_block_mask)
    return out
    
def test_masked_attention():
    B, H, H_kv, M, D = 1, 28, 4, 4096, 128
    block_size = 16
    q = torch.randn(B, H, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    group_mean_query = rearrange(q, "b (h r) m d -> b h r m d", r=H // H_kv).mean(dim=2)
    block_mask = create_custom_block_mask(group_mean_query, k, 256, 1024, 16)
    sm_scale = D**-0.5
    
    # k_rep, v_rep = repeat_kv(k, v, H // H_kv)
    # block_mask_rep = block_mask.unsqueeze(2).expand(-1, -1, H // H_kv, -1, -1).reshape(B, H, M, -1)
    # out_ref = ref_masked_attention(q, k_rep, v_rep, block_mask_rep, block_size, sm_scale)
    # out_ref.sum().backward()
    
    # copy gradients and set them to None
    # ref_q_grad = q.grad.clone()
    # ref_k_grad = k.grad.clone()
    # ref_v_grad = v.grad.clone()
    # q.grad = None
    # k.grad = None
    # v.grad = None
    
    out = flex_attn(q, k, v, block_mask, block_size, sm_scale)
    # out = masked_attention(q, k, v, block_mask.contiguous(), causal=True, sm_scale=sm_scale, dropout_p=0.0)
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
    
    assert torch.allclose(out_sdpa, out, atol=1e-2)
    assert torch.allclose(q.grad, dq, atol=1e-2)
    assert torch.allclose(k.grad, dk, atol=1e-2)
    assert torch.allclose(v.grad, dv, atol=1e-2)
    

test_masked_attention()