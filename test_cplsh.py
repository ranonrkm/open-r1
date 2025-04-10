import torch
from transformers import AutoTokenizer
from src.open_r1.utils.sparse_attention_utils import CPLSHForCausalLM

tokenizer = AutoTokenizer.from_pretrained("InfiniAI/OpenR1-Qwen-7B-Math-dense-packing")
model = CPLSHForCausalLM.from_pretrained("InfiniAI/OpenR1-Qwen-7B-Math-dense-packing", torch_dtype=torch.bfloat16)

model.eval()



