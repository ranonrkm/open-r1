from typing import Optional, Tuple
import torch
from .flag_attn import attention

def flash_attn_triton_interface(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)
    attn_output = attention(
        query,
        key,
        value,
        True,
        scaling
    )
    attn_output = attn_output.transpose(1, 2)

    return attn_output, None