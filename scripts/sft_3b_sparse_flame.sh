#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_3b_sparse.out
#SBATCH --error=logs/sft_3b_sparse.err

export WANDB_PROJECT=openr1

topk=$1
local=$2
ctx_len=$3

accelerate launch --config_file=recipes/accelerate_configs/zero2.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --model_revision main \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name open-r1/OpenR1-Math-220k \
    --dataset_num_proc 32 \
    --max_length $ctx_len \
    --weight_decay 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --num_train_epochs 5 \
    --bf16 \
    --do_eval false \
    --use_liger_kernel \
    --use_liger \
    --eval_strategy no \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
    --log_level info \
    --logging_steps 10 \
    --logging_strategy steps \
    --packing false \
    --output_dir data/OpenR1-Qwen-3B-Instruct-SFT-sparse-local${local}-topk${topk}-ctx${ctx_len} \
    --overwrite_output_dir \
    --push_to_hub \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 42 \
    --sparse_training \
    --local ${local} \
    --sparse_attn_topk ${topk} 