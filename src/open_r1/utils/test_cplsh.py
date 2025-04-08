import torch
try:
    from .flag_attn import cplsh_attention, attention
except:
    from flag_attn import cplsh_attention, attention

def hash_attention(q, k, v, q_codes, k_codes, sm_scale=1., local_size=4):
    B, H, M, D = q.shape
    attn_scores = torch.einsum('bhmd,bhnd->bhmn', q, k) * sm_scale
    causal_mask = torch.ones(M, M, device=q.device).tril_(0)[None, None, :, :].expand(B, H, -1, -1)
    matches = (q_codes[..., local_size:, None, :] == k_codes[..., None, :, :]).any(dim=-1)   # B H M N 
    dynamic_mask = torch.ones(M - local_size, M, device=q.device).tril_(0)[None, None, :, :].expand(B, H, -1, -1)
    topk_ids = torch.topk(attn_scores[..., local_size:, :].masked_fill_(dynamic_mask.logical_not(), float('-inf')), 10, dim=-1).indices 
    matches.logical_and_(dynamic_mask)
    recall = matches.gather(-1, topk_ids).sum(dim=-1).float().mean()  # B H M
    mask = causal_mask[..., local_size:, :].logical_xor(dynamic_mask)
    mask.logical_or_(matches)
    mask = torch.cat([causal_mask[..., :local_size, :], mask], dim=-2)
    cnts = mask.sum(dim=-1)
    attn_scores.masked_fill_(mask.logical_not(), float('-inf'))
    S = torch.softmax(attn_scores, dim=-1)
    out = torch.einsum('bhmn,bhnd->bhmd', S, v)
    return out, recall, cnts


# ----- unit tests -----
def test_cplsh_attention():
    B, H, M, N, D = 1, 8, 2048, 2048, 128
    q = torch.randn(B, H, M, D).cuda().half()
    k = torch.randn(B, H, N, D).cuda().half()
    v = torch.randn(B, H, N, D).cuda().half()
    L, K = 32, 256
    q_codes = torch.randint(-128, 127, (B, H, M, L), dtype=torch.int32).cuda()
    k_codes = torch.randint(-128, 127, (B, H, N, L), dtype=torch.int32).cuda()
    
    out_ref, recall, cnt_ref = hash_attention(q, k, v, q_codes, k_codes, sm_scale=D**-0.5, local_size=4)
    out, cnt = cplsh_attention(q, k, v, q_codes, k_codes, causal=True, sm_scale=D**-0.5, dropout_p=0.0, local_size=4)
    out_full = attention(q, k, v, causal=True, sm_scale=D**-0.5, dropout_p=0.0)

    import pdb; pdb.set_trace()
    assert not torch.isnan(out).any()
    # assert torch.allclose(out, out_full, atol=1e-2, rtol=0.0)

def test_backward():
    B, H, M, N, D = 1, 8, 2048, 2048, 128
    q = torch.randn(B, H, M, D).cuda().half()
    k = torch.randn(B, H, N, D).cuda().half()
    v = torch.randn(B, H, N, D).cuda().half()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    L, K = 32, 256
    q_codes = torch.randint(0, K, (B, H, M, L), dtype=torch.int32).cuda()
    k_codes = torch.randint(0, K, (B, H, N, L), dtype=torch.int32).cuda()

    out = cplsh_attention(q, k, v, q_codes, k_codes, causal=True, sm_scale=D**-0.5, dropout_p=0.0, local_size=4)
    loss = out.norm()
    loss.backward()
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()

if __name__ == '__main__':
    test_cplsh_attention()
    # test_backward()
    print("Test passed.")