import torch
from ..models.qwen import Qwen
from ..attention.batch_cache import BatchKVManager
from transformers import AutoConfig, GenerationConfig, AutoTokenizer
from .utils import copy_new_tokens
import copy
from ..logging_config import setup_logger  
from tqdm import tqdm
import time

logger = setup_logger()

class LLM:
    def __init__(self, 
    model_name: str, 
    dtype: torch.dtype, 
    device: str, 
    max_seq_len: int,
    max_batch_size: int,
    top_p: float,
    temperature: float,
    use_chat_template: bool,
    attn_topk: int = 0):
    
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.max_seq_len = max_seq_len
        
        self.top_p = top_p
        self.temperature = temperature

        self.max_batch_size = max_batch_size
        
        self.model_executor = Qwen(self.model_name, self.max_seq_len, self.device, self.dtype)
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.kv_cache = BatchKVManager(self.model_config, self.max_batch_size, self.max_seq_len, self.device, self.dtype, attn_topk=attn_topk)
        self.model_executor.alloc()
        
        self.slots_occupy_status = [False for _ in range(self.max_batch_size)]
        self.slots_offsets = torch.zeros((self.max_batch_size,), dtype=torch.long, device=self.device)
        self.slots_prompt_len = torch.zeros((self.max_batch_size,), dtype=torch.long, device=self.device)
        self.slots_tokens = torch.zeros((self.max_batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.eos_tensor = self.generation_config.eos_token_id if (isinstance(self.generation_config.eos_token_id, list)) else [self.generation_config.eos_token_id]
        self.eos_tensor = torch.tensor(self.eos_tensor, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.use_chat_template = use_chat_template
        
        logger.info("LLM Initialized:")
        logger.info(f"  Model Name      : {self.model_name}")
        logger.info(f"  Device          : {self.device}")
        logger.info(f"  Dtype           : {self.dtype}")
        logger.info(f"  Max Seq Length  : {self.max_seq_len}")
        logger.info(f"  Max Batch Size  : {self.max_batch_size}")
        logger.info(f"  Top-p           : {self.top_p}")
        logger.info(f"  Temperature     : {self.temperature}")
        logger.info(f"  Chat-template   : {self.use_chat_template}")
        logger.info(f"  Attention Topk  : {attn_topk}")
        
    def clear_request(self, batch_idx: int):
        
        self.slots_occupy_status[batch_idx] = False
        self.slots_offsets[batch_idx] = 0
        self.slots_prompt_len[batch_idx] = 0
        self.slots_tokens[batch_idx].zero_()
        self.kv_cache.clear_cache(batch_idx)
       
    def prefill(self, batch_idx: int, input_ids: torch.LongTensor):
        
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        self.slots_tokens[batch_idx][:seq_len].copy_(input_ids[0])
        self.slots_offsets[batch_idx] = seq_len
        self.slots_prompt_len[batch_idx] = seq_len
        self.slots_occupy_status[batch_idx] = True
        
        batch_idx = torch.tensor([batch_idx], device=self.device)
        input_ids = input_ids.to(self.device)
        
        logits = self.model_executor.inference(input_ids, position_ids, batch_idx, self.kv_cache)[:,-1:,:]
        next_tokens = self.sample_next_tokens(logits)
        
        self.slots_tokens[batch_idx.item()][seq_len] = next_tokens.item()
        
        
    def decode(self, batch_idx: torch.LongTensor):
        
        offset = self.slots_offsets[batch_idx].unsqueeze(1)
        
        input_ids = self.slots_tokens[batch_idx].gather(dim=-1, index=offset)
        
        logits = self.model_executor.inference(input_ids, offset, batch_idx, self.kv_cache)
        next_tokens = self.sample_next_tokens(logits)    
        
        self.slots_offsets[batch_idx] += 1
        copy_new_tokens(self.slots_tokens, next_tokens.squeeze(1), self.slots_offsets[batch_idx], batch_idx)
        
        is_eos = (next_tokens == self.eos_tensor.view(1, -1)).any(dim=1)
        
        return is_eos
        
    def sample_next_tokens(self, logits: torch.Tensor):
        """
        logits: Tensor of shape [batch_size, 1, vocab_size]
        returns: Tensor of shape [batch_size, 1]
        """
        
        if self.temperature < 0.02:
            return logits.argmax(dim=-1)
        

        logits = logits.squeeze(1)

        probs = torch.softmax(logits / self.temperature, dim=-1)  # [batch_size, vocab_size]

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # [batch_size, vocab_size]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > self.top_p

        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = 0

    
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)  # [batch_size, 1]

        next_tokens = sorted_indices.gather(dim=1, index=sampled_indices)  # [batch_size, 1]

        return next_tokens


    def offline_exec(self, requests: list, max_new_tokens: int): 
        
        processed_requests = []
        request_idx = 0
        batch_size = self.max_batch_size
        total_requests = len(requests)

        logger.critical("Offline JOB Started:")
        logger.critical(f"  Total Requests      : {total_requests}")
        logger.critical(f"  Max New Tokens      : {max_new_tokens}")
        
        gen_token_counts = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        # Track metadata for each slot
        slot_to_request_meta = [None for _ in range(batch_size)]
        total_generate_tokens = 0
        torch.cuda.synchronize()
        t1 = time.time()
        with tqdm(total=total_requests, desc="Processing Requests", ncols=100) as pbar:
            # Initialize the batch
            for i in range(min(batch_size, total_requests)):
                req = requests[request_idx]
                self.preprocess_request(req)
                input_ids = torch.tensor(req["input_ids"]).unsqueeze(0)
                self.prefill(i, input_ids)
                gen_token_counts[i] = 1
                slot_to_request_meta[i] = req
                request_idx += 1

            while any(self.slots_occupy_status):
                active_slots = [i for i, status in enumerate(self.slots_occupy_status) if status]
                active_batch_idx = torch.tensor(active_slots, device=self.device)

                is_eos = self.decode(active_batch_idx)

                for i, b in enumerate(active_slots):
                    gen_token_counts[b] += 1

                    if is_eos[i] or gen_token_counts[b] >= max_new_tokens:
                        seq_len = self.slots_offsets[b].item()
                        prompt_len = self.slots_prompt_len[b].item()
                        tokens = self.slots_tokens[b, prompt_len:seq_len].tolist()
                        
                        # Prepare output dict with metadata
                        total_generate_tokens += seq_len - prompt_len
                        result = copy.deepcopy(slot_to_request_meta[b])
                        result["output_tokens"] = tokens
                        self.postprocess_request(result)
                        processed_requests.append(result)

                        # Clear and reuse
                        self.clear_request(b)
                        slot_to_request_meta[b] = None
                        pbar.update(1)
                        if request_idx < total_requests:
                            req = requests[request_idx]
                            self.preprocess_request(req)
                            input_ids = torch.tensor(req["input_ids"]).unsqueeze(0)
                            self.prefill(b, input_ids)
                            gen_token_counts[b] = 1
                            slot_to_request_meta[b] = req
                            request_idx += 1

        torch.cuda.synchronize()
        t2 = time.time()
        
        logger.info("Total Generated Tokens {:.2f} | Throughput {:.2f} TPS".format(total_generate_tokens, total_generate_tokens/(t2-t1)))
        
        return processed_requests

    
    def preprocess_request(self, req):
        if "input_ids" in req:
            return
        
        prompt = self.tokenizer.apply_chat_template(
                req["conversations"],
                tokenize=False,
                add_generation_prompt=True
            )

        tokens = self.tokenizer.encode(prompt)
        
        req["input_ids"] = tokens
    
    def postprocess_request(self, req):
        req["output_text"] = self.tokenizer.decode(req["output_tokens"])

                    
                
                
                
                