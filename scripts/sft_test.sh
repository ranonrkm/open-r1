#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_7b_test.out
#SBATCH --error=logs/sft_7b_test.err

model=Qwen/Qwen2.5-7B-Instruct
ckpt_name=OpenR1-Qwen-7B-Math-dense-test
budget=128

export WANDB_PROJECT=sparsity
export WANDB_API_KEY=43249e11b6dc61a3e85f8385185eecc99d122c3d
# accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 src/open_r1/sft.py \
    --model_name_or_path  ${model} \
    --model_revision main \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name open-r1/OpenR1-Math-220k \
    --dataset_num_proc 48 \
    --max_length 8192 \
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
    --output_dir data/${ckpt_name}-B${budget} \
    --overwrite_output_dir \
    --push_to_hub false \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 42 \
    --sparse_training true \
    --sparse_attn nsa \
    --sparse_attn_topk 256 \
    --local 256 #\
    # --headwise true
    # --full_attn_layers 0 1 5 10 15 20 25