import sys
sys.path.append("..")
from litesys.models.qwen import Qwen
from transformers import AutoTokenizer, Qwen2ForCausalLM
import torch
from litesys.attention.batch_cache import BatchKVManager, AutoConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda:0"
max_new_tokens = 64
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device, _attn_implementation="eager")
llm.eval()
config = AutoConfig.from_pretrained(model_name)
kv_cache = BatchKVManager(config=config, max_batch_size=4, max_seq_len=4096, device=device, dtype=torch.bfloat16)

position_ids = torch.arange(0, 4096, device=device)

text1 = "Tell me about Reinforcement Learning in 200 words."
text2 = "Tell me about Reinforcement Learning in 200 words."

inputs = tokenizer([text1, text2], return_tensors="pt", padding="longest", padding_side='left').to(device)

input_ids1_list = inputs["input_ids"][0].tolist()
input_ids2_list = inputs["input_ids"][1].tolist()

with torch.no_grad():
    output = llm(**inputs, use_cache=True)
    for i in range(max_new_tokens):
        past_key_values = output.past_key_values
        logits = output.logits[:,-1:,:]

        new_input_ids = logits.argmax(dim=-1)
        input_ids1_list.append(new_input_ids[0].item())
        input_ids2_list.append(new_input_ids[1].item())
        output = llm(new_input_ids, use_cache=True, past_key_values=past_key_values)

   
text1 = tokenizer.decode(input_ids1_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)
text2 = tokenizer.decode(input_ids2_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(text1)
print(text2)