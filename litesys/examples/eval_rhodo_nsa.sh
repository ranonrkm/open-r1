#!/bin/bash

MODEL=$1
topk_block=$2
echo "MODEL: ${MODEL}"

# get the base name of the model
MODEL_BASE_NAME=$(basename ${MODEL})



mkdir -p /sensei-fs/users/xuhuang/rsadhukh/litesys/eval/${MODEL_BASE_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task math500 \
        --nproc 8 \
        --trial 1 \
        --batch_size 8 \
        --max_len 32768 \
        --gen_len 30768 \
        --topk_block ${topk_block}

for file in ${MODEL_BASE_NAME}/*.jsonl; do
    head -n 1 $file
    cp $file /sensei-fs/users/xuhuang/rsadhukh/litesys/eval/${MODEL_BASE_NAME}/
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task amc23 \
        --nproc 8 \
        --trial 3 \
        --batch_size 10 \
        --max_len 32768 \
        --gen_len 30768 \
        --topk_block ${topk_block}
        
for file in ${MODEL_BASE_NAME}/*.jsonl; do
    head -n 1 $file
    cp $file /sensei-fs/users/xuhuang/rsadhukh/litesys/eval/${MODEL_BASE_NAME}/
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task aime24 \
        --nproc 4 \
        --trial 3 \
        --batch_size 8 \
        --max_len 32768 \
        --gen_len 30768 \
        --topk_block ${topk_block}

for file in ${MODEL_BASE_NAME}/*.jsonl; do
    head -n 1 $file
    cp $file /sensei-fs/users/xuhuang/rsadhukh/litesys/eval/${MODEL_BASE_NAME}/
done