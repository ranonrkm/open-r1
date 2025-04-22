#!/bin/bash

budget=1024
model_path=InfiniAILab/OpenR1-Qwen-7B-Math-Instruct
nepochs=1

export WANDB_PROJECT=sparsity
export WANDB_API_KEY=43249e11b6dc61a3e85f8385185eecc99d122c3d
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path ${model_path} \
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
    --learning_rate 2.0e-05 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --num_train_epochs ${nepochs} \
    --bf16 true \
    --do_eval false \
    --use_liger_kernel true \
    --use_liger true \
    --eval_strategy no \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --log_level info \
    --logging_steps 5 \
    --logging_strategy steps \
    --packing true \
    --output_dir data/OpenR1-Qwen-7B-Math-Instruct-budget${budget}-postTrainE${nepochs} \
    --overwrite_output_dir \
    --push_to_hub true \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 42 \
    --sparse_training true \
    --sparse_attn nsa \
    --sparse_attn_topk ${budget} \
    --local 1024 \
    --headwise false 

mv data/OpenR1-Qwen-7B-Math-Instruct-budget${budget}-postTrainE${nepochs} /sensei-fs/users/xuhuang/rsadhukh/