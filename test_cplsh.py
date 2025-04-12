from typing import Optional, Tuple, Callable
import json
import functools
import torch
import numpy as np
from tqdm import tqdm
import gc
from einops import rearrange
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from src.open_r1.utils.sparse_attention_utils import CPLSHForCausalLM, CPLSHAttention, apply_rotary_pos_emb

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

BLOCK_SIZE = 16

class CPLSHCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.key_codes = []
        self.key_means = []
        self.n_full = 1
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        if layer_idx == 0:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        rotation = cache_kwargs["rotation"]
        
        if len(self.key_means) <= layer_idx - self.n_full:
            self.key_means.append(key_states[..., 4:, :].mean(dim=-2, keepdim=True))
            key_means = self.key_means[-1]
        else:
            key_means = self.key_means[layer_idx - self.n_full]
            
        _key_norma = (key_states - key_means) 
        _key_norma = _key_norma / _key_norma.norm(dim=-1, keepdim=True)
        _key_codes = torch.einsum("bhnd,hlkd->bhnlk",
                                   _key_norma,
                                   rotation).argmax(dim=-1)
        if len(self.key_codes) <= layer_idx - self.n_full:
            self.key_codes.append(_key_codes)
        else:
            self.key_codes[layer_idx - self.n_full] = torch.cat([self.key_codes[layer_idx - self.n_full], _key_codes], dim=-2)
            
        key_states, value_states = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states, self.key_codes[layer_idx - self.n_full]
    
    def reset(self):
        self.key_codes = []
        self.key_means = []
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        
class CPLSHChunkedCache(CPLSHCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        if layer_idx == 0:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        rotation = cache_kwargs["rotation"]
        
        if len(self.key_means) <= layer_idx - self.n_full:
            self.key_means.append(key_states[..., 4:, :].mean(dim=-2, keepdim=True))
            key_means = self.key_means[-1]
        else:
            key_means = self.key_means[layer_idx - self.n_full]
            
        # divide the key_states into blocks of 16 and take the mean of each block
        key_blocks = torch.split(key_states, BLOCK_SIZE, dim=-2)
        block_centroids = torch.cat([block.mean(dim=-2, keepdim=True) for block in key_blocks], dim=-2)
        block_centroids = block_centroids - key_means
        block_centroids = block_centroids / block_centroids.norm(dim=-1, keepdim=True)
        block_codes = torch.einsum("bhnd,hlkd->bhnlk", block_centroids, rotation).argmax(dim=-1)
        
        if len(self.key_codes) <= layer_idx - self.n_full:
            self.key_codes.append(block_codes)
        else:
            self.key_codes[layer_idx - self.n_full] = torch.cat([self.key_codes[layer_idx - self.n_full], block_codes], dim=-2)
        
        # call update of super of parent class
        key_states, value_states = super(CPLSHCache, self).update(key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states, self.key_codes[layer_idx - self.n_full]
      
class CPLSHVotingCache(CPLSHCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        if layer_idx == 0:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        rotation = cache_kwargs["rotation"]
        
        if len(self.key_means) <= layer_idx - self.n_full:
            self.key_means.append(key_states[..., 4:, :].mean(dim=-2, keepdim=True))
            key_means = self.key_means[-1]
        else:
            key_means = self.key_means[layer_idx - self.n_full]
            
        _key_norma = (key_states - key_means) 
        _key_norma = _key_norma / _key_norma.norm(dim=-1, keepdim=True)
        _key_codes = torch.einsum("bhnd,hlkd->bhnlk",
                                   _key_norma,
                                   rotation).argmax(dim=-1)
        B, H, N, L = _key_codes.shape
        K = rotation.shape[-2]
        N_blk = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        _key_codes = _key_codes.transpose(-1, -2)
        
        # Split into blocks of size BLOCK_SIZE
        _block_codes = torch.split(_key_codes, BLOCK_SIZE, dim=-1)  # List of N_blk [B H L blk] tensors, last block might have less than BLOCK_SIZE
        
        # Process each block to get majority vote to get a single code per block. Final dim: [B H N_blk L]
        block_codes = []
        for _block_code in _block_codes:
            _block_count = torch.zeros((B, H, L, K), device=_key_codes.device, dtype=torch.int32)
            _block_count.scatter_add_(dim=-1, index=_block_code, src=torch.ones_like(_block_count))
            block_codes.append(_block_count.argmax(dim=-1)) # B H L
        block_codes = torch.stack(block_codes, dim=-2)    # B H N_blk L
        
        if len(self.key_codes) <= layer_idx - self.n_full:
            self.key_codes.append(block_codes)
        else:
            self.key_codes[layer_idx - self.n_full] = torch.cat([self.key_codes[layer_idx - self.n_full], block_codes], dim=-2)
        
        key_states, value_states = super(CPLSHCache, self).update(key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states, self.key_codes[layer_idx - self.n_full]

def hash_matches(q_codes, key_codes):
    chunk_size = 128
    M = q_codes.size(-2)
    N = key_codes.size(-2)
    num_chunks = (N + chunk_size - 1) // chunk_size
    matches = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, N)
        matches.append((q_codes.unsqueeze(-2) == key_codes[:, :, None, None, :, :]).sum(dim=-1).gt(1))
    return torch.cat(matches, dim=-1)

def attn_forward(
        self,
        hidden_states: torch.Tensor,
        rotation: torch.Tensor,  
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H N D
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)      # B H_kv N D
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H_kv N D

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "rotation": rotation}
            key_states, value_states, key_codes = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        seq_len = key_states.size(2)
        sink = 0
        local = 256
        n_full_attn = 2048
        attn_mask = None
        M = query_states.size(-2)

        assert self.layer_idx > 0, "layer_idx should be greater than 0"
        if seq_len > n_full_attn:
            query_states_reshaped = rearrange(query_states, "b (h r) m d -> b h r m d", r=self.num_key_value_groups)    # B H r M D
            q_codes = torch.einsum("bhrnd,hlkd->bhrnlk", 
                                    query_states_reshaped, 
                                    rotation).argmax(dim=-1)  # B H r M L
            matches = (q_codes.unsqueeze(-2) == key_codes[:, :, None, None, :, :]).sum(dim=-1).gt(1)  # B H r M N
            
            w = torch.einsum("bhrmd,bhnd->bhrmn", query_states_reshaped, key_states)  # B H r M D
            dynamic_mask = torch.ones(w.shape[-2:], device=w.device).tril_(seq_len - M - local)
            matches.logical_and_(dynamic_mask)
            w.masked_fill_(dynamic_mask.logical_not(), float("-inf"))
            topk = torch.topk(w, k=10, dim=-1).indices  # B H r M K
            recall = matches.gather(dim=-1, index=topk).float().sum(dim=-1)  # B H r M
            recall = recall.view(-1, recall.size(-1)).mean(dim=0)   # M
            seq_lens = torch.arange(seq_len - M, seq_len, device=w.device)
            sparsity = matches.float().sum(dim=-1)
            sparsity = sparsity.view(-1, sparsity.size(-1)).mean(dim=0)
            sparsity = sparsity / seq_lens
            meta_data = torch.stack([recall, sparsity], dim=-1)
        else:
            meta_data = None
            
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

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
        return attn_output, meta_data  

def attn_forward_chunked(
        self,
        hidden_states: torch.Tensor,
        rotation: torch.Tensor,  
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H N D
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)      # B H_kv N D
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # B H_kv N D

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "rotation": rotation}
            key_states, value_states, key_codes = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        seq_len = key_states.size(2)
        sink = 0
        local = 1024
        n_full_attn = 2048
        attn_mask = None
        M = query_states.size(-2)
        
        assert self.layer_idx > 0, "layer_idx should be greater than 0"
        if seq_len > n_full_attn and M < seq_len:  # do not enter during prefilling
            query_states_reshaped = rearrange(query_states, "b (h r) m d -> b h r m d", r=self.num_key_value_groups)    # B H r M D
            q_codes = torch.einsum("bhrnd,hlkd->bhrnlk", 
                                    query_states_reshaped, 
                                    rotation).argmax(dim=-1)  # B H r M L
            matches = (q_codes.unsqueeze(-2) == key_codes[:, :, None, None, :, :]).sum(dim=-1).gt(0)     # B H r M N_blk
            matches = matches.unsqueeze(-1).expand(-1, -1, -1, -1, -1, BLOCK_SIZE)  # B H r M blk N_blk
            matches = matches.reshape(*matches.shape[:-2], -1)  # B H r M N
            
            w = torch.einsum("bhrmd,bhnd->bhrmn", query_states_reshaped, key_states)  # B H r M D
            dynamic_mask = torch.ones(w.shape[-2:], device=w.device).tril_(seq_len - M - local)
            matches = matches[..., :dynamic_mask.size(-1)]
            matches.logical_and_(dynamic_mask)
            w.masked_fill_(dynamic_mask.logical_not(), float("-inf"))
            topk = torch.topk(w, k=10, dim=-1).indices  # B H r M K
            recall = matches.gather(dim=-1, index=topk).float().sum(dim=-1)  # B H r M
            recall = recall.view(-1, recall.size(-1)).mean(dim=0)   # M
        
            sparsity = matches.float().sum(dim=-1)
            sparsity = sparsity.view(-1, sparsity.size(-1)).mean(dim=0)
            seq_lens = torch.arange(seq_len - M, seq_len, device=w.device)
            sparsity = sparsity / seq_lens
            meta_data = torch.stack([recall, sparsity], dim=-1)
        else:
            meta_data = None
            
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

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
        return attn_output, meta_data 
    
    
if __name__ == "__main__":
    CPLSHAttention.forward = attn_forward_chunked
    # CPLSHAttention.forward = attn_forward
    tokenizer = AutoTokenizer.from_pretrained("InfiniAILab/OpenR1-Qwen-7B-Math-dense-packing")
    model = CPLSHForCausalLM.from_pretrained("InfiniAILab/OpenR1-Qwen-7B-Math-dense-packing", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")

    model.eval()
    model.to("cuda")

    # past_key_values = CPLSHChunkedCache()
    # past_key_values = CPLSHCache()
    past_key_values = CPLSHVotingCache()
    recall_stats = np.zeros((model.model.config.num_hidden_layers - 1, 32768 // 128))
    sparsity_stats = np.zeros((model.model.config.num_hidden_layers - 1, 32768 // 128))
    
    fname = "data/amc23_2025-04-09_15-50-11.jsonl"
    num_samples = np.zeros(recall_stats.shape[1])
    num_example = 0
    with open(fname, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if "query" not in data:
                continue
            if num_example > 10:
                break
            query = data["query"]
            prediction = data["prediction"]
            # apply chat template of qwen2
            conversation = [
                {"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=query)},
                {"role": "assistant", "content": prediction[0]}
            ]
            input_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")
    
            chunk_size = 1024
            seq_len = input_ids.size(1)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            recalls = []
            sparsity = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, seq_len)
                input_ids_chunk = input_ids[:, start:end]
                
                with torch.no_grad():
                    _stats = model(input_ids = input_ids_chunk, output_attentions=True, past_key_values=past_key_values).attentions[1:]
                
                if _stats[0] is None:
                    continue
                _stats = torch.stack(_stats, dim=0) # layer x seq_len
                recalls.append(_stats[:, :, 0])
                sparsity.append(_stats[:, :, 1])
            past_key_values.reset()
            gc.collect()
            torch.cuda.empty_cache()
            
            if len(recalls):
                recalls = torch.cat(recalls, dim=1) # layer x seq_len
                sparsity = torch.cat(sparsity, dim=1) # layer x seq_len 
                # chunk recalls into groups of 128 in dim 1
                block_id = 0
                for j in range(0, recalls.size(1), 128):
                    _recalls = recalls[:, j:j+128].mean(dim=-1)
                    recall_stats[:, block_id] += _recalls.cpu().numpy()
                    _sparsity = sparsity[:, j:j+128].mean(dim=-1)
                    sparsity_stats[:, block_id] += _sparsity.cpu().numpy()
                    num_samples[block_id] += 1
                    block_id += 1
                num_example += 1
        
    max_nonzero_block = np.where(num_samples > 0)[0].max()
    print(f"max seq len: {min((max_nonzero_block + 1) * 128 + 2048, 32768)}")
    print(f"num examples: {num_example}")
    recall_stats = recall_stats[:, :max_nonzero_block]
    num_samples = num_samples[:max_nonzero_block]
    recall_stats = recall_stats / num_samples[None, :]
    sparsity_stats = sparsity_stats[:, :max_nonzero_block]
    sparsity_stats = sparsity_stats / num_samples[None, :]
    
    np.save("recall_stats_amc23_voting.npy", recall_stats)
    np.save("sparsity_stats_amc23_voting.npy", sparsity_stats)
    
    '''
    import matplotlib.pyplot as plt
    cplsh_recall = np.load("recall_stats_amc23.npy")
    cplsh_chunked_recall = np.load("recall_stats_amc23_avg.npy")
    
    plt.plot(cplsh_recall.mean(axis=0)[16:], label="CPLSH")
    plt.plot(cplsh_chunked_recall.mean(axis=0)[16:], label="CPLSH Chunked")
    plt.xlabel("seq block id (128 tokens, starting from 2k)")
    plt.ylabel("recall@10")
    plt.legend()
    plt.savefig("recall_stats_amc23.png")
    
    cplsh_sparsity = np.load("sparsity_stats_amc23.npy")
    cplsh_chunked_sparsity = np.load("sparsity_stats_amc23_avg.npy")
    plt.plot(cplsh_sparsity.mean(axis=0)[16:], label="CPLSH")
    plt.plot(cplsh_chunked_sparsity.mean(axis=0)[16:], label="CPLSH Chunked")
    plt.xlabel("seq block id (128 tokens, starting from 2k)")
    plt.ylabel("sparsity")
    plt.legend()
    plt.savefig("sparsity_stats_amc23.png")
    
    '''