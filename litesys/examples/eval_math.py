import sys
sys.path.append("..")
from litesys.helper import generate_requests
from litesys.engine.engine import LLM
from datasets import load_dataset
import os
import argparse
import torch

import torch.multiprocessing as mp
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from datetime import datetime
from lm_eval.models.utils import (
    stop_sequences_criteria,
)

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

import random
import numpy as np


DATAMAP = {
    "amc23": "math-ai/amc23",
    "aime24": "HuggingFaceH4/aime_2024",
    "gsm8k": "InfiniAILab/gsm8k",
    "math500": "HuggingFaceH4/MATH-500"
}


SPLITMAP = {
    "amc23": "test",
    "aime24": "train",
    "gsm8k": "test",
    "math500": "test"
}

COLUMNMAP = {
    "amc23": "question",
    "aime24": "problem",
    "gsm8k": "question",
    "math500": "problem"
}

ANSWERMAP = {
    "amc23": "answer",
    "aime24": "answer",
    "gsm8k": "answer",
    "math500": "solution"
}

GENLENMAP = {
    "amc23": 8192,
    "aime24": 30768,
    "gsm8k": 8192,
    "math500": 30768
}

MAXLENMAP = {
    "amc23": 9116,
    "aime24": 32768,
    "gsm8k": 9116,
    "math500": 32768
}

JUDGEMAP = {
    "amc23": "expr",
    "aime24": "expr",
    "gsm8k": "expr",
    "math500": "latex"
}
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 适用于多 GPU 训练

    # 让 cuDNN 采用确定性算法（保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker(rank, world_size, args, shared_list):
    
    torch.cuda.set_device(rank)
    
    data = DATAMAP[args.task]
    split = SPLITMAP[args.task]
    column = COLUMNMAP[args.task]
    gen_len = GENLENMAP[args.task]
    max_len = MAXLENMAP[args.task]
    answer_column = ANSWERMAP[args.task]
    judge = JUDGEMAP[args.task]
    
    llm = LLM(model_name=args.model, dtype=torch.bfloat16, device=f"cuda:{rank}",
    max_seq_len=args.max_len, max_batch_size=args.batch_size, top_p=0.95, temperature=0.6, use_chat_template=True, attn_topk=args.topk)
    dataset = load_dataset(data, split=split)
    all_processed_request = []
    
    for i in range(args.trial):
        set_seed(42 + i)
        requests = generate_requests(dataset=dataset, field_name=column, data_format=MATH_QUERY_TEMPLATE, trial=args.trial, rank=rank, world_size=world_size)
        processed_request = llm.offline_exec(requests, args.gen_len)
        all_processed_request.extend(processed_request)
    
    if judge == "expr":
        gold_metric = multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )
    
    else:
        gold_metric = multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )

    for data in all_processed_request:
        golds = [data[answer_column]]
        target = Doc(query=data[column],choices=golds, gold_index=0)
        predictions = data["output_text"]
        try:
            result = gold_metric.compute(golds=golds,predictions=[predictions],formatted_doc=target)
        except:
            result = {"extractive_match": 0.0}
            
        shared_list.append(
                    {   
                        "score": result["extractive_match"] * 100,
                        "prediction": [predictions],
                        "choices": golds,
                        "query": data[column]
                    }
                )
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name')
    parser.add_argument('--task', type=str, default="niah_single_3", help='Task identifier')
    parser.add_argument('--nproc', type=int, default=8, help='Number of processes to launch')
    parser.add_argument('--batch_size', type=int, default=16, help='Max Batch Size on GPUs')
    parser.add_argument('--max_len', type=int, default=32768, help='Max Length')
    parser.add_argument('--gen_len', type=int, default=30768, help='Max Generation Length')
    parser.add_argument('--trial', type=int, default=1, help='Number of trials')
    parser.add_argument('--topk', type=int, default=0, help='Attn TopK')
    args = parser.parse_args()

    output_dir = args.model.split("/")[1]
    
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"{args.task}_{current_time}.jsonl")

    # Use mp.Manager to create a shared list for process-safe communication
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    shared_list = manager.list()

    # Launch multiple processes using mp.spawn; each process handles a subset of the dataset
    world_size = args.nproc
    mp.spawn(worker, args=(world_size, args, shared_list), nprocs=world_size, join=True)

    # Gather results and compute global accuracy (optional)
    total_accuracy = 0.0
    count = 0
    results = []
    for item in shared_list:
            results.append(item)
            total_accuracy += item['score']
            count += 1

    global_summary = {
        'model': args.model,
        'task': args.task,
        'accuracy': total_accuracy / count if count > 0 else 0,
        'total_example': count
    }
    results.insert(0, global_summary)

    # Save the aggregated results to a single JSONLines file
    with jsonlines.open(output_file, "w") as f:
        f.write_all(results)

    print("Results saved to", output_file)

if __name__ == "__main__":
    main()