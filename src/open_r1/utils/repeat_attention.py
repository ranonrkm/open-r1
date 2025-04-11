from typing import Optional, Union, Tuple, Callable
import copy
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2DecoderLayer, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPast
from tqdm import tqdm
import os

logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q_0, q_1, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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
    q_embed_0 = (q_0 * cos) + (rotate_half(q_0) * sin)
    q_embed_1 = (q_1 * cos) + (rotate_half(q_1) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed_0, q_embed_1, k_embed
    

class RepeatedAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.sink = 0
        self.local = config.sliding_window
        assert layer_idx > 0, "First layer is not repeated"
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, 2 * config.num_attention_heads * self.head_dim, bias=True)  # every query head is repeated
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(2 * config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # self.gamma = nn.Parameter(torch.ones(1) * 1e-6, requires_grad=True)  # learnable parameter for sparse attention - zero init for initialization stability
        self.gamma = nn.Parameter(torch.empty(config.num_attention_heads), requires_grad=True)    # every head has a learnable parameter for sparse attention
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        query_0, query_1 = torch.tensor_split(query_states, 2, dim=-1)  
        query_0 = query_0.view(hidden_shape).transpose(1, 2)
        query_1 = query_1.view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_0, query_1, key_states = apply_rotary_pos_emb(query_0, query_1, key_states, cos, sin)

        local_window_size = getattr(self, "local", 0)
        if local_window_size > 0:
             sliding_window = local_window_size
        else:
             sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output_0, attn_weights_0 = attention_interface(
            self,
            query_0,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )   # B, L, H, D
        
        attn_output_1, attn_weights_1 = attention_interface(
            self,
            query_1,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )   # B, L, H, D
        attn_output_1 = F.tanh(self.gamma)[None, None, :, None] * attn_output_1   # tanh(gamma) * attn_output_1
        
        attn_output_0 = attn_output_0.reshape(*input_shape, -1).contiguous()    # B, L, D
        attn_output_1 = attn_output_1.reshape(*input_shape, -1).contiguous()     # B, L, D
        attn_output = torch.cat([attn_output_0, attn_output_1], dim=-1)   # B, L, 2D
        attn_output = self.o_proj(attn_output)   # B, L, D
        return attn_output, None    


class RepeatedAttentionWithGating(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.sink = 0
        self.local = config.sliding_window
        assert layer_idx > 0, "First layer is not repeated"
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, 2 * config.num_attention_heads * self.head_dim, bias=True)  # every query head is repeated
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(2 * config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # create gates for each head pair (dense and its local windowed version). the gate is a projection of the hidden states
        self.attn_gate = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        query_0, query_1 = torch.tensor_split(query_states, 2, dim=-1)  
        query_0 = query_0.view(hidden_shape).transpose(1, 2)
        query_1 = query_1.view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        gate_states = self.attn_gate(hidden_states)
        gate_states = F.sigmoid(gate_states)  # B, L, H

        cos, sin = position_embeddings
        query_0, query_1, key_states = apply_rotary_pos_emb(query_0, query_1, key_states, cos, sin)

        local_window_size = getattr(self, "local", 0)
        if local_window_size > 0:
             sliding_window = local_window_size
        else:
             sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output_0, attn_weights_0 = attention_interface(
            self,
            query_0,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )   # B, L, H, D
        

        attn_output_1, attn_weights_1 = attention_interface(
            self,
            query_1,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )   # B, L, H, D
        
        attn_output_0 = gate_states.unsqueeze(-1) * attn_output_0
        attn_output_1 = (1 - gate_states).unsqueeze(-1) * attn_output_1
        
        attn_output_0 = attn_output_0.reshape(*input_shape, -1).contiguous()    # B, L, D
        attn_output_1 = attn_output_1.reshape(*input_shape, -1).contiguous()     # B, L, D
        attn_output = torch.cat([attn_output_0, attn_output_1], dim=-1)   # B, L, 2D
        attn_output = self.o_proj(attn_output)   # B, L, D
        return attn_output, None      
 
class RepeatedDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = RepeatedAttention(config=config, layer_idx=layer_idx) if not getattr(config, "gated_attention", False) else RepeatedAttentionWithGating(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
 
class RepeatedModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, 0)] + \
            [RepeatedDecoderLayer(config, layer_idx) for layer_idx in range(1, config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert past_key_values is None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()
    
class RepeatedForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = RepeatedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    

# create a new checkpoint with the new model class from existing Qwen2 checkpoint
def convert_checkpoint(model_name: str, output_path: str, use_gated_attention: bool = False):
    # Load the config first
    config = Qwen2ForCausalLM.config_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    config.gated_attention = use_gated_attention
    
    # Create the new model with the same config
    model = RepeatedForCausalLM(config).bfloat16()
    
    # Load the state dict from the reference model
    ref_state_dict = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).state_dict()
    
    # Create a new state dict for the model
    new_state_dict = {}
    
    # Process each parameter in the reference state dict
    for name, param in tqdm(ref_state_dict.items()):
        # Skip q_proj and o_proj in self_attn layers as they need special handling
        if 'self_attn.q_proj' in name or 'self_attn.o_proj' in name:
            continue
            
        # Copy all other parameters directly
        if name in model.state_dict():
            new_state_dict[name] = param
    
    # Now handle the RepeatedAttention layers specifically
    # We need to find all the self_attn layers in the model
    for i in tqdm(range(config.num_hidden_layers)):  # Skip the first layer
        # Get the q_proj and o_proj from the reference model
        q_proj_weight = ref_state_dict[f'model.layers.{i}.self_attn.q_proj.weight']
        q_proj_bias = ref_state_dict[f'model.layers.{i}.self_attn.q_proj.bias']
        o_proj_weight = ref_state_dict[f'model.layers.{i}.self_attn.o_proj.weight']
        
        if i == 0:
            new_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = q_proj_weight
            new_state_dict[f'model.layers.{i}.self_attn.q_proj.bias'] = q_proj_bias
            new_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = o_proj_weight
        
        else:
            # Replicate q_proj weights and biases
            new_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = torch.cat([q_proj_weight, q_proj_weight.clone()], dim=0)
            new_state_dict[f'model.layers.{i}.self_attn.q_proj.bias'] = torch.cat([q_proj_bias, q_proj_bias.clone()], dim=0)
            # Replicate o_proj weights
            new_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = torch.cat([o_proj_weight, o_proj_weight.clone()], dim=1)
    
    # Clear reference state dict to free memory
    del ref_state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load the new state dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    # Clear new state dict to free memory
    del new_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    # Ensure model is on CPU before saving
    model = model.cpu()
    
    # initialize the gate weights
    for name, param in model.named_parameters():
        if name.endswith("attn_gate.weight"):
            nn.init.xavier_uniform_(param)
        elif "gamma" in name:
            nn.init.uniform_(param, -1, 1)
    
    # Save the model config first
    config.save_pretrained(output_path)
    
    # Save the model weights in a more controlled way
    # Save the model weights as safetensors and shard into 4 files
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save the tokenizer if it exists
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    # Clear model to free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return None  # Return None since we've saved the model

if __name__ == "__main__":
    # convert_checkpoint("/project/flame/beidic/rsadhukh/ET/open-r1/base_ckpt/Qwen/Qwen2.5-Math-7B-Instruct", 
    #                    "/project/flame/beidic/rsadhukh/ET/open-r1/base_ckpt/Qwen/Qwen2.5-Math-7B-Instruct-repeat-gated",
    #                    use_gated_attention=True)
    convert_checkpoint("InfiniAILab/Qwen2.5-Math-7B-Instruct-32k",
                       "/sensei-fs/users/xuhuang/rsadhukh/open-r1/Qwen2.5-Math-7B-Instruct-repeat-gated",
                       use_gated_attention=True)
    convert_checkpoint("InfiniAILab/Qwen2.5-Math-7B-Instruct-32k",
                       "/sensei-fs/users/xuhuang/rsadhukh/open-r1/Qwen2.5-Math-7B-Instruct-repeat",
                       use_gated_attention=False)