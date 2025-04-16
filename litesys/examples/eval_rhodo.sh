#!/bin/bash

MODEL=$1
echo "MODEL: ${MODEL}"

if [ "$task" == "math" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task math500 \
        --nproc 8 \
        --trial 1 \
        --batch_size 8 \
        --max_len 32768 \
        --gen_len 30768 
elif [ "$task" == "amc23" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task amc23 \
        --nproc 8 \
        --trial 3 \
        --batch_size 10 \
        --max_len 32768 \
        --gen_len 30768 
elif [ "$task" == "aime24" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_math.py \
        --model ${MODEL} \
        --task aime24 \
        --nproc 4 \
        --trial 3 \
        --batch_size 8 \
        --max_len 32768 \
        --gen_len 30768 
fi