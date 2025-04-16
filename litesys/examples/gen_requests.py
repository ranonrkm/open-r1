import sys
sys.path.append("..")
from litesys.helper import generate_requests
from datasets import load_dataset

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

aime = load_dataset("HuggingFaceH4/aime_2024", split="train")
generate_requests(aime, "problem", MATH_QUERY_TEMPLATE)