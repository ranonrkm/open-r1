#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/sft_7b_nopack.out
#SBATCH --error=logs/sft_7b_nopack.err

export WANDB_ENTITY=infini-lab
export WANDB_PROJECT=openr1
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
 --config recipes/Qwen-7B-Instruct/sft/config.yaml