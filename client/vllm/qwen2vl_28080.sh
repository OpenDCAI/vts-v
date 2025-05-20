#!/bin/bash

# set your own model path
MODEL_PATH=your_model_path_here

# set your own port
MODEL_PORT=28080

CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL_PATH \
    --tensor-parallel-size 2 \
    --port $MODEL_PORT \
    --host 0.0.0.0 \
    --dtype float16 \
    --max-model-len 65536 \
    --limit-mm-per-prompt image=30,video=0 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.6 \
    --block-size 16
