#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/eval_7b_%a.out
#SBATCH --error=logs/eval_7b_%a.err
#SBATCH --array=0-2

# MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft_sparse_ctx16k_local256_top2048_min512_iter
MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft_sparse_ctx16k_local512_top256
# MODEL=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output/qwen7b_lora_sft_16k/ckpt_11000
# LMF_DIR=${HOME}/InfiniAI/rsadhukh/LLaMA-Factory/output
# models=(
# ${LMF_DIR}/qwen7b_lora_sft_sparse_ctx16k_local256_top2048_min512_iter
# ${LMF_DIR}/qwen7b_lora_sft_sparse_ctx16k_local512_top256
# ${LMF_DIR}/qwen7b_lora_sft_16k/ckpt_11000
# )
# MODEL=${models[${SLURM_ARRAY_TASK_ID}]}
# MODEL=Qwen/Qwen2.5-7B-Instruct
echo "MODEL: ${MODEL}"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
#     --model ${MODEL} \
#     --task math500 \
#     --nproc 8 \
#     --trial 1 \
#     --batch_size 8 \
#     --max_len 32768 \
#     --gen_len 30768

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
    --model ${MODEL} \
    --task amc23 \
    --nproc 8 \
    --trial 1 \
    --batch_size 10 \
    --max_len 32768 \
    --gen_len 30768 \
    --topk 1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
#     --model ${MODEL} \
#     --task aime24 \
#     --nproc 4 \
#     --trial 3 \
#     --batch_size 8 \
#     --max_len 32768 \
#     --gen_len 30768


