import triton
import triton.language as tl
import torch
from einops import rearrange

@triton.jit
def copy_to_cache_kernel(
    key_cache_ptr, key_ptr, batch_id_ptr, offset_ptr,
    num_heads, head_dim,
    stride_kc_b, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    BLOCK_HEAD_DIM: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_dim = tl.arange(0, BLOCK_HEAD_DIM)

    batch_id = tl.load(batch_id_ptr + batch_idx)
    offset = tl.load(offset_ptr + batch_idx)

    key_offset = (
        batch_idx * stride_k_b +
        head_idx * stride_k_h +
        0 * stride_k_s +  # 因为 key 的 seq_len 固定为 1
        offs_dim * stride_k_d
    )

    cache_offset = (
        batch_id * stride_kc_b +
        head_idx * stride_kc_h +
        offset * stride_kc_s +
        offs_dim * stride_kc_d
    )

    mask = offs_dim < head_dim

    key_vals = tl.load(key_ptr + key_offset, mask=mask, other=0)
    tl.store(key_cache_ptr + cache_offset, key_vals, mask=mask)


def copy_to_cache(key_cache, key, batch_id, offset):
    batch_size, num_heads, _, head_dim = key.shape

    grid = (batch_size, num_heads)

    copy_to_cache_kernel[grid](
        key_cache, key, batch_id, offset,
        num_heads, head_dim,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim)
    )
    

@triton.jit
def attention_score_kernel_optimized(
    key_cache_ptr, query_ptr, batch_idx_ptr, offset_ptr, output_ptr,
    max_batch_size: tl.constexpr,
    num_key_value_heads: tl.constexpr,
    num_query_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    seq_block = tl.program_id(2)

    seq_offsets = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    valid_mask = seq_offsets < tl.load(offset_ptr + batch_id)
    batch_idx = tl.load(batch_idx_ptr + batch_id)

    key_offset = batch_idx * num_key_value_heads * max_seq_len * head_dim \
                 + kv_head_id * max_seq_len * head_dim
    key_ptrs = key_cache_ptr + key_offset + \
               (seq_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :])

    key = tl.load(key_ptrs, mask=valid_mask[:, None], other=0.0)

    group_size = num_query_heads // num_key_value_heads
    for qh_in_group in range(group_size):
        query_head_id = kv_head_id * group_size + qh_in_group
        query_offset = batch_id * num_query_heads * head_dim + query_head_id * head_dim
        query = tl.load(query_ptr + query_offset + tl.arange(0, head_dim))

        attn_score = tl.sum(key * query[None, :], axis=1)
        attn_score *= (1.0 / (head_dim ** 0.5))
        attn_score = tl.where(valid_mask, attn_score, float('-inf'))

        output_offset = batch_id * num_query_heads * max_seq_len \
                        + query_head_id * max_seq_len
        tl.store(output_ptr + output_offset + seq_offsets,
                 attn_score,
                 mask=seq_offsets < max_seq_len)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, batch_idx_ptr, offset_ptr,
    num_heads: tl.constexpr, max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load offset and batch_idx
    seq_len = tl.load(offset_ptr + batch_id)
    batch_idx = tl.load(batch_idx_ptr + batch_id)

    seq_offsets = tl.arange(0, BLOCK_SIZE)
    mask = seq_offsets < seq_len

    # Pointers
    input_offset = batch_id * num_heads * max_seq_len + head_id * max_seq_len
    input_ptrs = input_ptr + input_offset + seq_offsets

    # Load inputs
    logits = tl.load(input_ptrs, mask=mask, other=float('-inf'))

    # Compute max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max

    # Compute exponentials and sum
    exp_logits = tl.where(mask, tl.exp(logits), 0.0)
    exp_sum = tl.sum(exp_logits, axis=0)

    # Compute softmax output
    softmax = exp_logits / exp_sum

    # Store results
    output_offset = input_offset
    output_ptrs = output_ptr + output_offset + seq_offsets
    tl.store(output_ptrs, softmax, mask=mask)


@triton.jit
def softmax_kernel_inplace(
    ptr, batch_idx_ptr, offset_ptr,
    num_heads: tl.constexpr, max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load offset
    seq_len = tl.load(offset_ptr + batch_id)

    seq_offsets = tl.arange(0, BLOCK_SIZE)
    mask = seq_offsets < seq_len

    # Pointers
    offset = batch_id * num_heads * max_seq_len + head_id * max_seq_len
    ptrs = ptr + offset + seq_offsets

    # Load inputs
    logits = tl.load(ptrs, mask=mask, other=float('-inf'))

    # Compute max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max

    # Compute exponentials and sum
    exp_logits = tl.where(mask, tl.exp(logits), 0.0)
    exp_sum = tl.sum(exp_logits, axis=0)

    # Compute softmax in-place
    softmax = exp_logits / exp_sum

    # Store results in-place
    tl.store(ptrs, softmax, mask=mask)


@triton.jit
def attention_value_kernel(
    score_ptr, value_ptr, output_ptr, batch_idx_ptr, offset_ptr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr
):
    batch_id = tl.program_id(0)
    query_head_id = tl.program_id(1)

    seq_offsets = tl.arange(0, BLOCK_SEQ)
    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)

    batch_idx = tl.load(batch_idx_ptr + batch_id)
    seq_len = tl.load(offset_ptr + batch_id)

    kv_head_id = query_head_id // (num_query_heads // num_kv_heads)

    score_offset = batch_id * num_query_heads * max_seq_len + query_head_id * max_seq_len
    value_offset = batch_idx * num_kv_heads * max_seq_len * head_dim + kv_head_id * max_seq_len * head_dim

    acc = tl.zeros((BLOCK_HEAD_DIM,), dtype=tl.float32)

    for seq_start in range(0, max_seq_len, BLOCK_SEQ):
        seq_pos = seq_start + seq_offsets
        mask = seq_pos < seq_len

        # Load scores
        scores = tl.load(score_ptr + score_offset + seq_pos, mask=mask, other=0.0)

        for head_start in range(0, head_dim, BLOCK_HEAD_DIM):
            # Load value once per KV head
            value = tl.load(
                value_ptr + value_offset + (seq_pos[:, None] * head_dim + head_start + dim_offsets[None, :]),
                mask=mask[:, None], other=0.0
            )
            # Accumulate
            acc += tl.sum(scores[:, None] * value, axis=0)

    # Write result to output
    output_offset = batch_id * num_query_heads * head_dim + query_head_id * head_dim
    tl.store(output_ptr + output_offset + dim_offsets, acc)


def attention_forward_orig(query, key, value, batch_idx, offset):
    
    batch_size, num_query_heads, _, head_dim = query.shape
    max_batch_size, num_key_value_heads, max_seq_len, _ = key.shape

    BLOCK_SEQ = 128
    BLOCK_HEAD_DIM = 128 if head_dim > 128 else head_dim
    
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

    attention_value_kernel[grid](
        logits_triton, value, output, batch_idx, offset,
        num_query_heads=num_query_heads,
        num_kv_heads=num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM
    )

    return output.to(query.dtype)



def attention_forward(query, key, value, batch_idx, offset, attn_local=0, attn_topk=0, nsample=0, attn_topk_block=0):
    assert sum([attn_local > 0, attn_topk > 0, nsample > 0, attn_topk_block > 0]) <= 1, "can not have multiple sparsity strategies"

    sink = 4
    local = 1024    # default used for topk or sampling

    batch_size, num_query_heads, _, head_dim = query.shape
    max_batch_size, num_key_value_heads, max_seq_len, _ = key.shape
    grp_size = num_query_heads // num_key_value_heads

    BLOCK_SEQ = 128
    BLOCK_HEAD_DIM = 128 if head_dim > 128 else head_dim
    
    logits_triton = torch.full((batch_size, num_query_heads, max_seq_len), float('-inf'), device=query.device, dtype=torch.float32)

    output = torch.zeros((batch_size, num_query_heads, head_dim), device=logits_triton.device, dtype=logits_triton.dtype)
    
    
    grid = (batch_size, num_key_value_heads, triton.cdiv(max_seq_len, BLOCK_SEQ))
    
    attention_score_kernel_optimized[grid](
        key, query, batch_idx, offset, logits_triton, # dynamic_mask,
        max_batch_size,
        num_key_value_heads,
        num_query_heads,
        max_seq_len,
        head_dim,   #sink, local, topk,
        BLOCK_SEQ=BLOCK_SEQ
    )
    
    if attn_topk > 0:
        topk = attn_topk
        n_full = sink + local + topk    
        for batch_id in range(batch_size):
            seq_len = offset[batch_id]
            if seq_len > n_full:
                logits = logits_triton[batch_id, :, sink:seq_len - local]   # dynamic part
                mean_logits = rearrange(logits, '(h r) s -> h r s', h=num_key_value_heads).mean(dim=1, keepdim=True)    # average over the query group (h 1 s)
                topk_ids = torch.topk(mean_logits, topk, dim=-1).indices # topk indices based on logits avged over query group (h 1 k)
                mask = torch.zeros_like(mean_logits, dtype=torch.bool)  # (h 1 s)
                mask.scatter_(-1, topk_ids, True)    # mask for topk (h 1 s)
                mask = mask.expand(-1, grp_size, -1).reshape(num_query_heads, -1)   # expand to all query heads in a query group
                logits_triton[batch_id, :, sink:seq_len - local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part

    elif attn_topk_block > 0:
        BLOCK_SIZE = 16
        n_full = sink + local + attn_topk_block * BLOCK_SIZE
        n_full = (n_full + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        for batch_id in range(batch_size):
            seq_len = offset[batch_id]
            if seq_len > n_full:
                _n_dynamic = (seq_len - local - sink) // BLOCK_SIZE * BLOCK_SIZE
                _local = seq_len - _n_dynamic - sink
                logits = logits_triton[batch_id, :, sink:seq_len - _local]   # dynamic part
                mean_logits = rearrange(logits, '(h r) s -> h r s', h=num_key_value_heads).mean(dim=1, keepdim=True)    # average over the query group (h 1 s)
                mean_logits = rearrange(mean_logits, 'h 1 (n b) -> h 1 n b', b = BLOCK_SIZE).mean(dim=-1)   # average over blocks (h 1 n)
                block_topk_ids = torch.topk(mean_logits, attn_topk_block, dim=-1).indices # topk indices based on logits avged over query group (h 1 k)
                mask = torch.zeros_like(mean_logits, dtype=torch.bool)  # (h 1 n)
                mask.scatter_(-1, block_topk_ids, True)    # mask for topk (h 1 n)
                mask = mask.unsqueeze(-1).expand(-1, -1, -1, BLOCK_SIZE)  # expand to blocks (h 1 s)
                mask = mask.reshape(*mask.shape[:-2], -1)
                mask = mask.expand(-1, grp_size, -1)   # expand to all query heads in a query group
                mask = mask.reshape(num_query_heads, -1)
                logits_triton[batch_id, :, sink:seq_len - _local] = torch.where(mask, logits, float('-inf'))   # apply mask to dynamic part
                
    elif attn_local > 0:
        local = attn_local
        for batch_id in range(batch_size):
            seq_len = offset[batch_id]
            if seq_len > sink + local:
                logits_triton[batch_id, :, sink:seq_len - local].fill_(float('-inf'))   # mask dynamic part

    grid = (batch_size, num_query_heads)

    softmax_kernel_inplace[grid](
        logits_triton, batch_idx, offset,
        num_heads=num_query_heads, max_seq_len=max_seq_len,
        BLOCK_SIZE=triton.next_power_of_2(max_seq_len)
    )

    # sample using torch multinomial
    if nsample > 0:
        n_full = sink + local + nsample    
        for batch_id in range(batch_size):
            seq_len = offset[batch_id]
            if seq_len > n_full:
                probs = logits_triton[batch_id, :, sink:seq_len - local]   # dynamic part
                dynamic_prob_sum = probs.sum(dim=-1, keepdim=True)
                head_mask = dynamic_prob_sum < 1e-4
                uniform_probs = torch.full_like(probs, 1.0 / probs.shape[-1])   # uniform distribution
                probs = torch.where(head_mask, uniform_probs, probs)   # replace 0 with uniform distribution
                # renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True) #[H, kv_len]
                assert not torch.isnan(probs).any(), "NaN in probs"
                sample_ids = torch.multinomial(probs, nsample)
                logits_triton[batch_id, :, sink:seq_len - local].zero_()
                logits_triton[batch_id, :, sink:seq_len - local].scatter_add_(-1, 
                                                                              sample_ids, 
                                                                              torch.ones_like(sample_ids, dtype=logits_triton.dtype))  
                # renormalize by nsample
                logits_triton[batch_id, :, sink:seq_len - local].mul_(dynamic_prob_sum / nsample)

    attention_value_kernel[grid](
        logits_triton, value, output, batch_idx, offset,
        num_query_heads=num_query_heads,
        num_kv_heads=num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM
    )

    return output.to(query.dtype)


def attention_forward_torch(query, key, value, batch_idx, offset, dtype):
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