#!/bin/bash


export DASHSCOPE_API_KEY=sk-573b81427b2a4ee4b18931a44139a8b8

QWEN_PATH=/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/model/Qwen2-VL-7B-Instruct

QWEN_PORT=28080

CUDA_VISIBLE_DEVICES=0,1 vllm serve $QWEN_PATH \
    --tensor-parallel-size 2 \
    --port $QWEN_PORT \
    --host 0.0.0.0 \
    --dtype float16 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=30,video=0 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.6 \
    --block-size 16
