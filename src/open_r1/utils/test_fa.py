from types import MethodType
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
from contextlib import contextmanager
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM
from flash_attn_tri import flash_attn_triton_interface

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        sliding_window = None

        attention_interface = flash_attn_triton_interface

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

@contextmanager
def swap_attention_fn(model: nn.Module):
    orig_fns = []
    for block in model.model.layers:
        orig_fns.append(block.self_attn.forward)
        block.self_attn.forward = MethodType(attn_forward, block.self_attn)
    yield
    for block, orig_fn in zip(model.model.layers, orig_fns):
        block.self_attn.forward = orig_fn
            

def check_error_norms(
        model,
        input_ids: torch.Tensor,
):
    model.eval()
    grad_o = torch.randn((1, 1024, model.config.vocab_size), device="cuda:0", dtype=torch.float32)

    grads_a = []
    outputs_a = []

    def save_outs_a(module, input, output):
        outputs_a.append(output[0].clone().detach())

    def save_grads_a(module, grad_input, grad_output):
        grads_a.append(grad_input[0].detach())

    hooks_a = []
    for block in model.model.layers:
        hooks_a.append(block.self_attn.register_forward_hook(save_outs_a))
        hooks_a.append(block.self_attn.register_backward_hook(save_grads_a))

    logits = model(input_ids, ).logits
    logits.backward(grad_o)

    model.zero_grad()
    for hook in hooks_a:
        hook.remove()

    torch.cuda.empty_cache()

    outputs_b = []
    grads_b = []

    def save_outs_b(module, input, output):
        outputs_b.append(output[0].detach())

    def save_grads_b(module, grad_input, grad_output):
        grads_b.append(grad_input[0].detach())

    with swap_attention_fn(model):
        for block in model.model.layers:
            block.self_attn.register_forward_hook(save_outs_b)
            block.self_attn.register_backward_hook(save_grads_b)

        logits = model(input_ids).logits
        logits.backward(grad_o)
        
    # calculate errors
    fwd_errors = []
    bwd_errors = []
    for i, (out_a, out_b) in enumerate(zip(outputs_a, outputs_b)):
        fwd_errors.append(torch.norm(out_a - out_b).item() / input_ids.shape[1]**0.5)

    for i, (grad_a, grad_b) in enumerate(zip(grads_a, grads_b)):
        bwd_errors.append(torch.norm(grad_a - grad_b).item() / input_ids.shape[1]**0.5)
    
    # plot errors
    plt.figure(figsize=(12, 6))
    plt.plot(fwd_errors, label='Forward Errors')
    # plt.plot(bwd_errors, label='Backward Errors')
    plt.xlabel('Layer')
    plt.ylabel('Error Norm')
    plt.title('Error Norms')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('error_norms_fwd.png')

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model = model.to("cuda:0")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 1024), device="cuda:0", dtype=torch.long)
    check_error_norms(model, input_ids)



    