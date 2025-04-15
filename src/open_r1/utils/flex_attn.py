from einops import rearrange
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)

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

dtype = torch.float16
BATCH = 1
H = 4
H_kv = 4
grp_size = H // H_kv
N_CTX = 8192
HEAD_DIM = 128
device = torch.device("cuda")
q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((BATCH, H_kv, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((BATCH, H_kv, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

# q_mean = rearrange(q, "b (h r) n d -> b h r n d", r=7).mean(dim=2)
block_mask = create_custom_block_mask(q, k, 256, 512, 16)   # B x H_kv x N_CTX x (N_CTX // KV_block_size)

def flexify_block_mask(b, h, q_idx, k_idx):
    return block_mask[b, h, q_idx, k_idx // BLOCK_SIZE]

flex_block_mask = create_block_mask(flexify_block_mask, BATCH, H, N_CTX, N_CTX, BLOCK_SIZE=[BLOCK_SIZE, 1], device="cuda")

out = flex_attention(q, k, v, block_mask=flex_block_mask)

print(out)

