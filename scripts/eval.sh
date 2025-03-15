#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
#SBATCH --array=0-7

NUM_GPUS=4
# MODEL=data/OpenR1-Qwen-7B-SFT-completiononly
# MODEL=Qwen/Qwen2.5-3B-Instruct
# MODEL=${HOME}/InfiniAI/rsadhukh/inference_engines/MagicDec/checkpoints/open-r1/OpenR1-Qwen-7B
# MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft/ckpt_11000
MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft_sparse/ckpt_10000
# MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft_sparse_local1k_top256/ckpt_7500
# MODEL=flyingbugs/OpenR1-Qwen-7B-SFT
# MODEL_NAME=open-r1/OpenR1-Qwen-7B
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,data_parallel_size=${NUM_GPUS},gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:30720,temperature:0.6,top_p:0.95}"
# tensor_parallel_size=$NUM_GPUS,
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,top_k=256,generation_parameters={max_new_tokens:30720,temperature:0.6,top_p:0.95}"
# MODEL_ARGS=test_configs/qwen3b.yaml
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

job_id=$SLURM_ARRAY_TASK_ID
start_id=$((job_id * 50 + 100))

TASK=math_500
accelerate launch --multi_gpu --num_processes=${NUM_GPUS} --main_process_port 29800 -m lighteval accelerate $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --override-batch-size 1 \
    --start-sample $start_id \
    --max-samples 50 \
    --save-details

# MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --max-samples 100 \
#     --save-details

# # GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --save-details

# # LiveCodeBench
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

# AMC 2023
# TASK=amc23
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# TASK=math_500
# lighteval accelerate $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --max-samples 20 