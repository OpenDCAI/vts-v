#!/bin/bash

cd ../

QWEN_PATH="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct"
QWEN_PORT=28080

TASK_NAME="figureqa(cauldron,llava_format)"

OUTPUT_DIR="./outputs_dir"


VLLM_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/vllm_log/vllm_${TASK_NAME}.log"
RESULT_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/result_log/result_${TASK_NAME}.output"

> "$VLLM_LOG"
> "$RESULT_LOG"

echo "vLLM log will be written to: $VLLM_LOG"
echo "Test results log will be written to: $RESULT_LOG"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $QWEN_PATH \
    --tensor-parallel-size 4 \
    --port $QWEN_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --limit-mm-per-prompt image=30,video=4 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --block-size 16 > "$VLLM_LOG" 2>&1