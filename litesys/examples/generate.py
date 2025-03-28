import sys
sys.path.append("..")
from litesys.models.qwen import Qwen
from transformers import AutoTokenizer
import torch
from litesys.attention.batch_cache import BatchKVManager, AutoConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda:0"
max_new_tokens = 64

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = Qwen(model_name=model_name, max_length=4096, device=device, dtype=torch.bfloat16)
llm.alloc()
config = AutoConfig.from_pretrained(model_name)
kv_cache = BatchKVManager(config=config, max_batch_size=4, max_seq_len=100, device=device, dtype=torch.bfloat16)

position_ids = torch.arange(0, 4096, device=device)

text1 = "Tell me about Reinforcement Learning in 200 words."
text2 = "Tell me about vLLM."

input_ids1 = tokenizer.encode(text1, return_tensors="pt").to(device)
input_ids2 = tokenizer.encode(text2, return_tensors="pt").to(device)

logits1 = llm.inference(
input_ids=input_ids1, 
position_ids=position_ids[:input_ids1.shape[1]].unsqueeze(0),
batch_idx=torch.tensor([0], device=device, dtype=torch.int32),
kv_cache=kv_cache)[:,-1:,:]


logits2 = llm.inference(
input_ids=input_ids2, 
position_ids=position_ids[:input_ids2.shape[1]].unsqueeze(0),
batch_idx=torch.tensor([2], device=device, dtype=torch.int32),
kv_cache=kv_cache)[:,-1:,:]

logits = torch.cat([logits1, logits2], dim=0)

input_ids1_list = input_ids1[0].tolist()
input_ids2_list = input_ids2[0].tolist()
for i in range(max_new_tokens):
    new_input_ids = logits.argmax(dim=-1)
    input_ids1_list.append(new_input_ids[0].item())
    input_ids2_list.append(new_input_ids[1].item())
    new_position_ids = torch.tensor([[input_ids1.shape[1] + i], [input_ids2.shape[1] + i]]).to(device)
    
    batch_idx=torch.tensor([0,2], device=device, dtype=torch.int32)

    logits = llm.inference(
        input_ids=new_input_ids, 
        position_ids=new_position_ids,
        batch_idx=batch_idx,
        kv_cache=kv_cache)
    
    
text1 = tokenizer.decode(input_ids1_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)
text2 = tokenizer.decode(input_ids2_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(text1)
print(text2)
