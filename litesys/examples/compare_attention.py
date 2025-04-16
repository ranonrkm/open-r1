import sys
sys.path.append("..")
import torch
from litesys.attention.cache_utils import attention_forward, attention_forward_torch
import random
import pytest


@pytest.mark.parametrize("seq_len", [32, 100, 4096, 8192])
@pytest.mark.parametrize("max_batch_size", [1,4,8])
@pytest.mark.parametrize("num_query_heads", [16, 24, 32])
@pytest.mark.parametrize("num_kv_heads", [2, 4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_triton_attention(
    seq_len,
    max_batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim):
    
    dtype = torch.bfloat16
    batch_size = random.randint(a=1, b=max_batch_size)
    key = torch.randn(max_batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    query = torch.randn(batch_size, num_query_heads, 1, head_dim, device='cuda', dtype=dtype)
    value = torch.randn(max_batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    batch_idx = torch.randperm(max_batch_size, device='cuda')[:batch_size].int()
    offset = torch.randint(low=1, high=seq_len, size=(batch_size,), device='cuda')
    

    triton_output = attention_forward(query, key, value, batch_idx, offset)
    torch_output = attention_forward_torch(query, key,value, batch_idx, offset, dtype)

    
    correct = torch.allclose(triton_output.float(), torch_output.float(), atol=1e-1, rtol=1e-2)
    max_diff = (triton_output.float() - torch_output.float()).abs().max().item()
    
    print(f"Final Attention Value correctness: {'✅' if correct else '❌'} (max diff: {max_diff:.4f})")
    assert correct