import torch
from transformers import AutoConfig
import flashinfer
from .cache_utils import copy_to_cache, attention_forward


class BatchKVManager:
    def __init__(self,
        config :AutoConfig,
        max_batch_size: int,
        max_seq_len: int,
        device: str,
        dtype: torch.dtype,
        attn_local: int = 0,
        attn_topk: int = 0,
        attn_nsample: int = 0,
        attn_topk_block: int = 0
        ):
        
        
        self.config = config
        self.max_length = max_seq_len
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.attn_local = attn_local
        self.attn_topk = attn_topk
        self.attn_nsample = attn_nsample
        self.attn_topk_block = attn_topk_block
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        
        self.num_layers = config.num_hidden_layers
        self.hidden_size= config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        self.k_cache = [torch.zeros(
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)]

        self.v_cache = [torch.zeros(
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_length,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)]
        
        self.kv_offset = [torch.zeros(
            self.max_batch_size,
            device=self.device,
            dtype=torch.int32
        ) for _ in range(self.num_layers)]

        
    
    
    def append_key_value_cache(self, 
        new_key_cache: torch.Tensor,
        new_value_cache: torch.Tensor,
        batch_idx: torch.LongTensor,
        layer_idx:int):
        
        copy_to_cache(self.k_cache[layer_idx], new_key_cache, batch_idx, self.kv_offset[layer_idx][batch_idx])
        copy_to_cache(self.v_cache[layer_idx], new_value_cache, batch_idx, self.kv_offset[layer_idx][batch_idx])
        self.kv_offset[layer_idx][batch_idx] += 1
        
    def compute_attention(self, 
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: torch.Tensor,
        batch_idx: torch.LongTensor):
        
        seq_len = query_states.shape[2]
        
        if seq_len > 1:
            return self.prefill_or_append(query_states, key_states, value_states, layer_idx, batch_idx)

        else:
            
            bsz, seq_len = query_states.shape[0], query_states.shape[2]
            
            self.append_key_value_cache(key_states, value_states, batch_idx, layer_idx)

            output =  attention_forward(query_states, self.k_cache[layer_idx], self.v_cache[layer_idx], batch_idx, self.kv_offset[layer_idx][batch_idx], attn_local=self.attn_local, attn_topk=self.attn_topk, nsample=self.attn_nsample, attn_topk_block=self.attn_topk_block)
            
            
            output = output.reshape(bsz, seq_len, self.hidden_size)
            return output
        
    def prefill_or_append(self, 
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: torch.Tensor,
        batch_idx: torch.LongTensor):
        
        bsz, seq_len = query_states.shape[0], query_states.shape[2]
        batch_id = batch_idx.item()
        
        
        offset = self.kv_offset[layer_idx][batch_id]

        self.k_cache[layer_idx][batch_id][:,offset:offset+seq_len] = key_states
        self.v_cache[layer_idx][batch_id][:,offset:offset+seq_len] = value_states
        
        self.kv_offset[layer_idx][batch_id] += seq_len
        
        key_cache = self.k_cache[layer_idx][batch_id][:,:self.kv_offset[layer_idx][batch_id]]
        value_cache = self.v_cache[layer_idx][batch_id][:,:self.kv_offset[layer_idx][batch_id]]
        
        
        hidden_states = flashinfer.single_prefill_with_kv_cache(
                    q = query_states[0].transpose(0,1),
                    k = key_cache,
                    v = value_cache,
                    kv_layout="HND",
                    allow_fp16_qk_reduction=False,
                    causal=True
                )
        
        hidden_states = hidden_states.reshape(bsz, seq_len, self.hidden_size)
        
        return hidden_states

    def clear_cache(self, batch_idx: int):
    
        for i in range(self.num_layers):
            self.k_cache[i][batch_idx].zero_()
            self.v_cache[i][batch_idx].zero_()
            self.kv_offset[i][batch_idx].zero_()