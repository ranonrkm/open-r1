import triton
import triton.language as tl
import torch

@triton.jit
def copy_new_tokens_kernel(
    slot_tokens_ptr,  # [max_batch_size, max_seq_len]
    new_tokens_ptr,   # [batch_size]
    offset_ptr,       # [batch_size]
    batch_idx_ptr,    # [batch_size]
    max_seq_len: tl.constexpr,
    batch_size: tl.constexpr,
):
    pid = tl.program_id(0)  # 每个 program 处理一个 new_token

    if pid >= batch_size:
        return

    # Load indices
    offset = tl.load(offset_ptr + pid)
    batch_idx = tl.load(batch_idx_ptr + pid)
    new_token = tl.load(new_tokens_ptr + pid)

    # Compute flat index into slot_tokens
    out_ptr = slot_tokens_ptr + batch_idx * max_seq_len + offset

    # Write
    tl.store(out_ptr, new_token)


def copy_new_tokens(slot_tokens, new_tokens, offset, batch_idx):
    # slot_tokens: [max_batch_size, max_seq_len]
    # new_tokens, offset, batch_idx: [batch_size]
    assert new_tokens.shape == offset.shape == batch_idx.shape
    batch_size = new_tokens.shape[0]
    max_batch_size, max_seq_len = slot_tokens.shape

    grid = (batch_size,)  # 每个 thread 处理一个 token

    copy_new_tokens_kernel[grid](
        slot_tokens, new_tokens, offset, batch_idx,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
