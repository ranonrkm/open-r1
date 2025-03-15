#!/bin/bash

DATA_DIR=/sensei-fs/users/xuhuang/rsadhukh/data/evals
mkdir -p $DATA_DIR
NUM_GPUS=4
MODEL=open-r1/OpenR1-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,data_parallel_size=${NUM_GPUS},gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:30720,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=${DATA_DIR}/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
