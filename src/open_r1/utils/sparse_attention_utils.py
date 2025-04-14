from typing import Optional, Union, Tuple, Callable
import torch
from torch import nn
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Attention, Qwen2Model, Qwen2DecoderLayer, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPast
from tqdm import tqdm
try:
    from .flag_attn import flash_attn_triton_interface
    from .flag_attn import cplsh_attn_triton_interface, masked_attn_triton_interface
except ImportError:
    from flag_attn import flash_attn_triton_interface, cplsh_attn_triton_interface, masked_attn_triton_interface

logger = logging.get_logger(__name__)

BLOCK_SIZE = 16
MIN_FULL_ATTN_SIZE = 2048

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
    
def local_attn_forward_orig(
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        local_window_size = getattr(self, "local", 0)
        if local_window_size > 0:
             sliding_window = local_window_size
        else:
             sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

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

    
def local_attn_forward(
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        local_window_size = getattr(self, "local", 0)
        if local_window_size > 0:
             sliding_window = local_window_size
        else:
             sliding_window = None

        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        query_0, query_1 = torch.tensor_split(query_states, [self.num_key_value_groups], dim=1)
        key_0, key_1 = torch.tensor_split(key_states, [1], dim=1)
        value_0, value_1 = torch.tensor_split(value_states, [1], dim=1)

        attn_output_0, attn_weights_0 = attention_interface(
            self,
            query_0,
            key_0,
            value_0,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )

        attn_output_1, attn_weights_1 = attention_interface(
            self,
            query_1,
            key_1,
            value_1,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        attn_output = torch.cat([attn_output_0, attn_output_1], dim=2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

# divide query_states, key_states, value_states into two parts - first head (or head group) for dense attention or rest for sparse attention

@torch.no_grad()
def create_block_mask(query, key, local, offset, topk):
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

def nsa_attn_forward(
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

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    seq_len = hidden_states.size(1)
    local = self.local
    topk = max(1, self.topk // BLOCK_SIZE)
    n_full_attn = max(MIN_FULL_ATTN_SIZE, local)
    attn_mask = None

    assert self.layer_idx > 0, "layer_idx should be greater than 0"
    if seq_len > n_full_attn:
        with torch.no_grad():
            group_mean_query = rearrange(
                query_states, 
                "b (h r) m d -> b h r m d", 
                r=self.num_key_value_groups
            ).mean(dim=2)   # B H M D
            
            attn_mask = create_block_mask(group_mean_query, key_states, local, n_full_attn, topk)
        
        attention_interface = masked_attn_triton_interface  
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attn_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,  # main diff with Llama
        **kwargs,
    )
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights  

def headwise_nsa_attn_forward(
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

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    seq_len = hidden_states.size(1)
    local = self.local
    topk = max(1, self.topk // BLOCK_SIZE)
    n_full_attn = max(MIN_FULL_ATTN_SIZE, local)
    attn_mask = None

    assert self.layer_idx > 0, "layer_idx should be greater than 0"
    if seq_len > n_full_attn:
        query_0, query_1 = torch.tensor_split(query_states, [self.num_key_value_groups], dim=1)
        key_0, key_1 = torch.tensor_split(key_states, [1], dim=1)
        value_0, value_1 = torch.tensor_split(value_states, [1], dim=1)
        
        # dense attention
        dense_attn_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        attn_output_0, attn_weights = dense_attn_interface(
            self,
            query_0,
            key_0,
            value_0,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )
        
        # sparse attention
        with torch.no_grad():
            group_mean_query = rearrange(
                query_1, 
                "b (h r) m d -> b h r m d", 
                r=self.num_key_value_groups
            ).mean(dim=2)   # B H M D
            
            attn_mask = create_block_mask(group_mean_query, key_1, local, n_full_attn, topk)
          
        attn_output_1, attn_weights = masked_attn_triton_interface(
            self,
            query_1,
            key_1,
            value_1,
            attn_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )
        
        attn_output = torch.cat([attn_output_0, attn_output_1], dim=2)
        
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,  # main diff with Llama
            **kwargs,
        )
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights  

class CPLSHAttention(Qwen2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.sink = 0
        self.local = 2048

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotation: torch.Tensor,  
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H N D
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)      # B H_kv N D
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H_kv N D

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        seq_len = hidden_states.size(1)
        sink = self.sink
        local = self.local
        n_full_attn = sink + local
        attn_mask = None

        assert self.layer_idx > 0, "layer_idx should be greater than 0"
        if seq_len > n_full_attn:
            with torch.no_grad():
                #_q_norma = query_states.narrow(2, n_full_attn, seq_len - n_full_attn) 
                #_q_norma = _q_norma / _q_norma.norm(dim=-1, keepdim=True)  
                _q_norma = query_states / query_states.norm(dim=-1, keepdim=True)  # B H N D
                _q_norma = rearrange(_q_norma, "b (h r) m d -> b h r m d", r=self.num_key_value_groups)
                _key_norma = key_states / key_states.norm(dim=-1, keepdim=True)  # B H_kv N D

                query_codes = []
                key_codes = []
                chunk_size = 1024
                num_chunks = seq_len // chunk_size
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = seq_len if i == num_chunks - 1 else (i + 1) * chunk_size
                    _q_norma_chunk = _q_norma[..., start:end, :]  # B H r M D
                    _key_norma_chunk = _key_norma[..., start:end, :]  # B H_kv N D
                    _query_codes = torch.einsum("bhrnd,hlkd->bhrnlk", 
                                                _q_norma_chunk, 
                                                rotation).argmax(dim=-1)  # B H r M L
                    _key_codes = torch.einsum("bhnd,hlkd->bhnlk",
                                               _key_norma_chunk,
                                               rotation).argmax(dim=-1)
                    query_codes.append(_query_codes)
                    key_codes.append(_key_codes)

                query_codes = torch.cat(query_codes, dim=-2)    # B H r M L
                key_codes = torch.cat(key_codes, dim=-2)        # B H_kv N L
                query_codes = rearrange(query_codes, "b h r m l -> b (h r) m l")  # B H M L
                
            # query key hash matches moved to kernel, otherwise OOM
            attn_output, attn_weights = cplsh_attn_triton_interface(
                self,
                query_states,
                key_states,
                value_states,
                q_codes=query_codes,    # pass query codes
                k_codes=key_codes,        # pass key codes
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.local,  # main diff with Llama
                **kwargs,
            )
                
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=None,  # main diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights           
 
class CPLSHForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        config.L = 48
        config.K = 256
        self.model = CPLSHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class CPLSHModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, 0)] + \
            [CPLSHDecoderLayer(config, layer_idx) for layer_idx in range(1, config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        rotation = torch.rand(config.num_key_value_heads, config.L, config.K, config.hidden_size // config.num_attention_heads)
        rotation = rotation / rotation.norm(dim=-1, keepdim=True)
        self.rotation = nn.Parameter(rotation, requires_grad=False)

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

        # assert past_key_values is None

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

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if layer_idx == 0:
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
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        self.rotation,
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
                        rotation=self.rotation,
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
    
class CPLSHDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = CPLSHAttention(config=config, layer_idx=layer_idx)
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
        rotation: torch.Tensor = None,  # necessary
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
            rotation=rotation,
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
    
def profile_attn_forward(method: str):
    # write a profiling script for nsa_attn_forward
    
    B, H, H_kv, M, D = 1, 28, 4, 32768, 128
    block_size = 16
    q = torch.randn(B, H, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    k = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    v = torch.randn(B, H_kv, M, D, requires_grad=True, device="cuda:0", dtype=torch.float16)
    block_mask = torch.randint(0, 2, (B, H_kv, M, M)).gt(0).cuda()
    block_mask.diagonal(dim1=-2, dim2=-1).fill_(1)
    block_mask = rearrange(block_mask, 'b h m (n r) -> b h m n r', r=block_size)[..., 0]
    block_mask[..., 0] = 1
    sm_scale = D**-0.5
    
    
    # warmup
    print("Warming up...")
    for _ in tqdm(range(10)):
        if method == "nsa":
            out = masked_attn_triton_interface(
                None,
                q,
                k,
                v,
                block_mask,
                dropout=0.0,
                scaling=sm_scale,
            )[0]   
        elif method == "flag":
            out = flash_attn_triton_interface(
                None,
                q,
                k,
                v,
                attention_mask=None, 
                dropout=0.0,
                scaling=sm_scale,
            )[0]   
        else:
            attn_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            out = attn_interface(
                None,
                q,
                k,
                v,
                attention_mask=None, 
                dropout=0.0,
                scaling=sm_scale,
                sliding_window=None,
            )[0]
        
        out.sum().backward()
        q.grad = None
        k.grad = None
        v.grad = None
    
    # Force CUDA synchronization before profiling
    torch.cuda.synchronize()
    
    # Use a simpler profiling approach focused on getting a Chrome trace
    print("Starting profiling...")
    
    # Create a custom context manager to handle profiler cleanup
    class ProfilerContext:
        def __init__(self):
            self.prof = None
            
        def __enter__(self):
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                record_shapes=True
            )
            self.prof.__enter__()
            return self.prof
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.prof is not None:
                self.prof.__exit__(exc_type, exc_val, exc_tb)
                
                # Safely export chrome trace
                try:
                    print("Exporting Chrome trace...")
                    self.prof.export_chrome_trace(f"logs/{method}_attn_forward.json")
                    print(f"Chrome trace exported successfully to logs/{method}_attn_forward.json")
                except Exception as e:
                    print(f"Chrome trace export error: {e}")
    
    # Use our custom context manager
    with ProfilerContext() as prof:
        for _ in tqdm(range(10)):
            if method == "nsa":
                out = masked_attn_triton_interface(
                    None,
                    q,
                    k,
                    v,
                    block_mask,
                    dropout=0.0,
                    scaling=sm_scale,
                )[0]  
            elif method == "flag":
                out = flash_attn_triton_interface(
                    None,
                    q,
                    k,
                    v,
                    attention_mask=None, 
                    dropout=0.0,
                    scaling=sm_scale,
                )[0]   
            else:
                attn_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
                out = attn_interface(
                    None,
                    q,
                    k,
                    v,
                    attention_mask=None, 
                    dropout=0.0,
                    scaling=sm_scale,
                    sliding_window=None,
                )[0]
            out.sum().backward()
            torch.cuda.synchronize()
            q.grad = None
            k.grad = None
            v.grad = None
        
        # Make sure CUDA is synchronized before exiting the profiler context
        torch.cuda.synchronize()
    
    # Print a message about viewing the trace
    print("\nTo view the Chrome trace:")
    print("1. Open Chrome/Chromium browser")
    print("2. Navigate to chrome://tracing")
    print(f"3. Click 'Load' and select logs/{method}_attn_forward.json")
    
def profile_attn_module(method: str):
    # profile nsa_attn_forward
    from transformers import AutoConfig
    from types import MethodType
    from tqdm import tqdm
    
    # method = "nsa"
    method = "flash"
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    attn_module = Qwen2Attention(config=config, layer_idx=1)
    attn_module.eval()
    attn_module = attn_module.to("cuda:0", dtype=torch.float16)
    if method == "nsa":
        attn_module.local = 1024
        attn_module.topk = 1024
        attn_module.forward = MethodType(nsa_attn_forward, attn_module)
    
    hidden_states = torch.randn(1, 32768, 3584, requires_grad=True, device="cuda:0", dtype=torch.float16)
    position_embeddings = (torch.randn(1, 32768, 128, device="cuda:0", dtype=torch.float16),
                           torch.randn(1, 32768, 128, device="cuda:0", dtype=torch.float16))
    
    # profile
    print("warming up...")
    for _ in tqdm(range(10)):
        out = attn_module(
            hidden_states,
            position_embeddings,
            attention_mask=None,
        )[0]
        out.sum().backward()
        hidden_states.grad = None
            
    torch.cuda.synchronize()
        
    print("profiling...")
    # Use a simpler profiling approach focused on getting a Chrome trace
    print("Starting profiling...")
    
    # Create a custom context manager to handle profiler cleanup
    class ProfilerContext:
        def __init__(self):
            self.prof = None
            
        def __enter__(self):
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                record_shapes=True
            )
            self.prof.__enter__()
            return self.prof
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.prof is not None:
                self.prof.__exit__(exc_type, exc_val, exc_tb)
                
                # Safely export chrome trace
                try:
                    print("Exporting Chrome trace...")
                    self.prof.export_chrome_trace(f"logs/{method}_attn_module.json")
                    print(f"Chrome trace exported successfully to logs/{method}_attn_module.json")
                except Exception as e:
                    print(f"Chrome trace export error: {e}")
                    
    with ProfilerContext() as prof:
        for _ in tqdm(range(10)):
            out = attn_module(
                hidden_states,
                position_embeddings,
                attention_mask=None,
            )[0]
            out.sum().backward()
            hidden_states.grad = None
            
        torch.cuda.synchronize()
        
    print("profiling done")
    
    prof.export_chrome_trace(f"logs/{method}_attn_module.json")
    

if __name__ == "__main__":
    profile_attn_forward("nsa")
    # profile_attn_forward("flag")
    # profile_attn_module("flash")