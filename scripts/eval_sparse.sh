#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/dense_train_sparse_eval_%A_%a.out
#SBATCH --error=logs/dense_train_sparse_eval_%A_%a.err
#SBATCH --array=2-2

NUM_GPUS=8

models=("Rano23/OpenR1-qwen-7b-lora16-sft-sparse256-local64-step10k" "Rano23/OpenR1-qwen-7b-lora16-sft-sparse256-local64-step10k" "Rano23/OpenR1-qwen-7b-lora16-sft-step11k")
start_ids=(50 200 200)

MODEL=${models[$SLURM_ARRAY_TASK_ID]}
start_id=${start_ids[$SLURM_ARRAY_TASK_ID]}
# port=$((29800 + $SLURM_ARRAY_TASK_ID * 10))

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,sparse_attn=true,sparse_attn_topk=256,sink=4,local=512,generation_parameters={max_new_tokens:30720,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

TASK=math_500
accelerate launch --multi_gpu --num_processes=${NUM_GPUS} -m lighteval accelerate $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --override-batch-size 1 \
    --start-sample $start_id \
    --max-samples 50 \
    --save-details