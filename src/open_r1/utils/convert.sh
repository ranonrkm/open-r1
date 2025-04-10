#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/convert.out
#SBATCH --error=logs/convert.err

python repeat_attention.py