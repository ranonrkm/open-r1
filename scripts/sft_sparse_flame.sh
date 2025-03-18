#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_7b_sparse_lora.out
#SBATCH --error=logs/sft_7b_sparse_lora.err

export WANDB_ENTITY=infini-lab
export WANDB_PROJECT=openr1
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
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
    --use_liger_kernel \
    --use_liger \
    --eval_strategy no \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --hub_model_id OpenR1-Qwen-7B-Instruct-SFT-sparse-local1024-topk256 \
    --hub_strategy every_save \
    --log_level info \
    --logging_steps 5 \
    --logging_strategy steps \
    --packing false \
    --output_dir data/OpenR1-Qwen-7B-Instruct-SFT-sparse-local1024-topk256 \
    --overwrite_output_dir \
    --push_to_hub \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --seed 42 \
    --sparse_training \
    --sparse_attn_topk 256 
