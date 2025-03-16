#!/bin/bash

start_id=$1
DATA_DIR=/sensei-fs/users/xuhuang/rsadhukh/data/evals
mkdir -p $DATA_DIR
NUM_GPUS=8
MODEL=$2
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,sparse_attn=true,sparse_attn_topk=256,sink=4,local=512,generation_parameters={max_new_tokens:30720,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=${DATA_DIR}/$MODEL

# AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

# MATH-500
TASK=math_500
accelerate launch --multi_gpu --num_processes=${NUM_GPUS} -m lighteval accelerate $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --override-batch-size 1 \
    --start-sample $start_id \
    --max-samples 50 \
    --save-details
