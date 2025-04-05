#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_7b_local_%A_%a.out
#SBATCH --error=logs/sft_7b_local_%A_%a.err
#SBATCH --array=0-3

CACHE=/mnt/localssd/rsadhukh
mkdir -p $CACHE

MODELS=(
    base_ckpt/Qwen/Qwen2.5-Math-7B-Instruct
    Qwen/Qwen2.5-7B-Instruct
)
# NOTE: changes in ckpt names
ckpt_names=(
    OpenR1-Qwen-7B-Math-local-layerwise
    OpenR1-Qwen-7B-local-layerwise
)
budgets=(
    2048
    4096
)
model_id=$((SLURM_ARRAY_TASK_ID % 2))
budget_id=$((SLURM_ARRAY_TASK_ID / 2))
model=${MODELS[model_id]}
ckpt_name=${ckpt_names[model_id]}
budget=${budgets[budget_id]}

export WANDB_PROJECT=openr1
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path  ${model} \
    --model_revision main \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name open-r1/OpenR1-Math-220k \
    --dataset_num_proc 48 \
    --max_length 32768 \
    --weight_decay 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --learning_rate 5.0e-05 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --num_train_epochs 3 \
    --bf16 \
    --do_eval false \
    --use_liger_kernel true \
    --use_liger true \
    --eval_strategy no \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --log_level info \
    --logging_steps 5 \
    --logging_strategy steps \
    --packing true \
    --output_dir ${CACHE}/${ckpt_name}-B${budget} \
    --overwrite_output_dir \
    --push_to_hub \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 42 \
    --sparse_training \
    --sparse_attn local \
    --local ${budget} #\
    # --full_attn_layers 0 1 5 10 15 20 25

    # NOTE: changed from 3 to 5 epochs
    # NOTE: changed from true to false

rm -rf ${CACHE}/${ckpt_name}-B${budget}/checkpoint-*
mv ${CACHE}/${ckpt_name}-B${budget} data/