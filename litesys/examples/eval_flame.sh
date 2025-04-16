#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/openr1_nsa_%A_%a.out
#SBATCH --error=logs/openr1_nsa_%A_%a.err
#SBATCH --array=0-2

ROOT=/project/flame/rsadhukh
models=(
    InfiniAILab/OpenR1-Qwen-7B-SFT-Instruct
)
BUDGETS=(
    1024
)
TASKS=(
    "amc23"
    "aime24"
    "math500"
)
# trials specific for each task - amc23: 3, aime24: 3, math500: 1
trials=(
    3
    3
    1
)
job_id=$SLURM_ARRAY_TASK_ID
num_models=${#models[@]}
num_tasks=${#TASKS[@]}
num_budgets=${#BUDGETS[@]}

model_id=$((job_id % num_models))
task_id=$((job_id / num_models % num_tasks))
budget_id=$((job_id / num_models / num_tasks % num_budgets))
MODEL=${models[$model_id]}
TASK=${TASKS[$task_id]}
BUDGET=${BUDGETS[$budget_id]}
TRIAL=${trials[$task_id]}
echo "Running task: $TASK with model: $MODEL and budget: $BUDGET"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
    --model ${MODEL} \
    --task ${TASK} \
    --nproc 8 \
    --trial ${TRIAL} \
    --batch_size 1 \
    --max_len 32768 \
    --gen_len 30768 \
    --topk ${BUDGET}