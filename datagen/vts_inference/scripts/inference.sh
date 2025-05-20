#!/bin/bash

cd ../

QWEN_PATH="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct"
QWEN_PORT=28080

TASK_NAME="figureqa(cauldron,llava_format)"


VLLM_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/vllm_log/vllm_${TASK_NAME}.log"
RESULT_LOG="/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/log/result_log/result_${TASK_NAME}.output"


> "$RESULT_LOG"


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

