from qk import attention_score_triton, attention_score_kernel_optimized
from softmax import softmax_triton, softmax_kernel, softmax_kernel_inplace
from sv import attention_value_triton_fast, attention_value_kernel_fast
import torch
import triton
import random

def attention_forward(key, query, value, batch_idx, offset):
    
    BLOCK_SEQ = 128
    BLOCK_HEAD_DIM = 128
    
    batch_size, num_query_heads, _, head_dim = query.shape
    max_batch_size, num_key_value_heads, max_seq_len, _ = key.shape

    logits_triton = torch.full((batch_size, num_query_heads, max_seq_len), float('-inf'), device=query.device, dtype=torch.float32)
    
    output = torch.zeros((batch_size, num_query_heads, head_dim), device=logits_triton.device, dtype=logits_triton.dtype)
    
    
    grid = (batch_size, num_key_value_heads, triton.cdiv(max_seq_len, BLOCK_SEQ))
    
    attention_score_kernel_optimized[grid](
        key, query, batch_idx, offset, logits_triton,
        max_batch_size,
        num_key_value_heads,
        num_query_heads,
        max_seq_len,
        head_dim,
        BLOCK_SEQ=BLOCK_SEQ
    )
    
    grid = (batch_size, num_query_heads)

    softmax_kernel_inplace[grid](
        logits_triton, batch_idx, offset,
        num_heads=num_query_heads, max_seq_len=max_seq_len,
        BLOCK_SIZE=triton.next_power_of_2(max_seq_len)
    )

    attention_value_kernel_fast[grid](
        logits_triton, value, output, batch_idx, offset,
        num_query_heads=num_query_heads,
        num_kv_heads=num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM
    )

    return output
    


def attention_forward_torch(key, query, value, batch_idx, offset, dtype):
    batch_size, num_query_heads, _, head_dim = query.shape
    num_kv_heads = key.shape[1]
    max_seq_len = key.shape[2]

    output = torch.zeros((batch_size, num_query_heads, head_dim), device=key.device, dtype=dtype)

    group_size = num_query_heads // num_kv_heads
    scale = head_dim ** -0.5

    for b in range(batch_size):
        b_idx = batch_idx[b]
        seq_len = offset[b]
        for kvh in range(num_kv_heads):
            k = key[b_idx, kvh, :seq_len, :]
            v = value[b_idx, kvh, :seq_len, :]
            for qh_in_group in range(group_size):
                qh = kvh * group_size + qh_in_group
                q = query[b, qh, 0, :]

                scores = (k @ q) * scale
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
                output[b, qh, :] = probs @ v

    return output

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
    

    triton_output = attention_forward(key, query, value, batch_idx, offset)
    torch_output = attention_forward_torch(key, query, value, batch_idx, offset, dtype)

    
    correct = torch.allclose(triton_output.float(), torch_output.float(), atol=1e-1, rtol=1e-2)
    max_diff = (triton_output.float() - torch_output.float()).abs().max().item()
    
    print(f"Final Attention Value correctness: {'✅' if correct else '❌'} (max diff: {max_diff:.4f})")
    assert correct

def benchmark_triton(seq_len, batch_size=8, num_query_heads=32, num_kv_heads=8, head_dim=128, dtype=torch.bfloat16):
    key_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    value_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    query = torch.randn(batch_size, num_query_heads, 1, head_dim, device='cuda', dtype=dtype)
    batch_idx = torch.tensor(list(range(batch_size)), device='cuda')
    offset = torch.tensor([seq_len for _ in range(batch_size)], device='cuda')

    # Triton benchmark (warmup自动完成)
    ms = triton.testing.do_bench(lambda: attention_forward(key_cache, query, value_cache, batch_idx, offset))
    total_bytes = key_cache.numel() * 4
    bandwidth = total_bytes / (1024 * 1024 * 1024) / (ms / 1000)
    print(f"Triton kernel seq_len={seq_len}: {ms:.4f} ms  {bandwidth:.2f} GB/s")
    

if __name__ == "__main__":
    test_triton_attention(669, 2, 32, 8, 128)
    test_triton_attention(1024, 2, 32, 8, 128)
    test_triton_attention(2048, 2, 32, 8, 128)
    test_triton_attention(4096, 2, 32, 8, 128)
    test_triton_attention(8192, 2, 32, 8, 128)
    test_triton_attention(16384, 2, 32, 8, 128)
    test_triton_attention(32768, 2, 32, 8, 128)
    test_triton_attention(100, 2, 32, 8, 128)
    test_triton_attention(100, 2, 32, 8, 64)
    for seq_len in [32, 1024, 4096, 8192, 16384, 32768]:
        benchmark_triton(seq_len)

    

    

    
