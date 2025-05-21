#!/bin/bash

cd ../


QWEN_PATH="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct"
QWEN_PORT=28080


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VTS_INFERENCE_DIR=$(dirname "$SCRIPT_DIR")


TASK_NAME="figureqa(cauldron,llava_format)"

VLLM_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/vllm_log/vllm_${TASK_NAME}.log"
RESULT_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/result_log/result_${TASK_NAME}.output"

> "$VLLM_LOG"
> "$RESULT_LOG"

echo "vLLM log will be written to: $VLLM_LOG"
echo "Test] results log will be written to: $RESULT_LOG"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $QWEN_PATH \
    --tensor-parallel-size 4 \
    --port $QWEN_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --limit-mm-per-prompt image=30,video=4 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --block-size 16 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "vLLM server started (PID: $VLLM_PID), waiting 300 seconds for initialization..."
sleep 300

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "Error: vLLM server failed to start! Check log: $VLLM_LOG" | tee -a "$RESULT_LOG"
    exit 1
fi

GPUS=(4 5 6 7)
PROC_PER_GPU=3
WORLD_SIZE=$(( ${#GPUS[@]} * $PROC_PER_GPU ))

WORKER_PIDS=()


## It's best to write your own absolute path
OUTPUT_DIR="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/outputs_dirs/vts_inference_output_dirs"

SCRIPTS=vts_inference.py

for (( gpu_idx=0; gpu_idx<${#GPUS[@]}; gpu_idx++ )); do
    GPU=${GPUS[$gpu_idx]}
    for (( local_rank=0; local_rank<$PROC_PER_GPU; local_rank++ )); do
        RANK=$(( $gpu_idx * $PROC_PER_GPU + $local_rank ))
        
        echo "Starting rank $RANK on GPU $GPU" | tee -a "$RESULT_LOG"
        CUDA_VISIBLE_DEVICES=$GPU python -u $SCRIPTS \
            --task_name $TASK_NAME \
            --reasoner_type qwen-vl \
            --reasoner_model_path $QWEN_PATH \
            --reasoner_port $QWEN_PORT \
            --sample_ratio 0.3 \
            --output_dir $OUTPUT_DIR \
            --rank $RANK \
            --world_size $WORLD_SIZE >> "$RESULT_LOG" 2>&1 &
        WORKER_PIDS+=($!)
        echo "Started worker PID: $!"
    done
done


echo "Waiting for all worker processes to complete..." | tee -a "$RESULT_LOG"
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid
done

echo "All worker processes completed." | tee -a "$RESULT_LOG"

echo "Stopping vLLM server (PID: $VLLM_PID)..." | tee -a "$RESULT_LOG"
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "vLLM server stopped." | tee -a "$RESULT_LOG"

exit 0
