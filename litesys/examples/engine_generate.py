import sys
sys.path.append("..")
from transformers import AutoTokenizer
import torch
from litesys.attention.batch_cache import AutoConfig
from litesys.engine.engine import LLM
import json
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name=model_name, max_seq_len=4096, max_batch_size=4,
device=device, dtype=torch.bfloat16, top_p=0.95, temperature=0.6, use_chat_template=True)

requests = []
texts = [
r"Tell me about Reinforcement Learning in 200 words.", 
r"Tell me about ResNet.",
r"The duration of a process used to manufacture components is known to be normally distributed with a mean of 30 minutes and a standard deviation of 4 minutes. What is the probability of a time greater than 33 minutes being recorded?",
r"If the original price of a shirt is $25, a discount of 20% is applied. How much will you pay for the shirt after the discount?",
r"Every day, a certain number of guests arrive at a hotel. On the first day, 2 guests arrive. On the second day, 3 guests arrive. On the third day, 5 guests arrive. And on the fourth day, 8 guests arrive. If this pattern continues, on which day will 101 guests arrive?"]
for text in texts:
    requests.append(
        {
            "conversations": [{"role": "user", "content": text}]
        }
    )

processed_request = llm.offline_exec(requests, max_new_tokens=2048)

with open("output.jsonl", "w", encoding="utf-8") as f:
    for item in processed_request:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")


# llm.prefill(0, input_ids1)
# llm.prefill(2, input_ids2)

# for _ in range(256):
#     llm.decode(torch.tensor([0,2]).long().cuda())

# offset1 = llm.slots_offsets[0]
# offset2 = llm.slots_offsets[2]

# text1 = tokenizer.decode(llm.slots_tokens[0][:offset1+1].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
# text2 = tokenizer.decode(llm.slots_tokens[2][:offset2+1].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

# print(text1)
# print(text2)