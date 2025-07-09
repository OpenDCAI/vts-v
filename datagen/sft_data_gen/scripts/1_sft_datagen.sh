#!/bin/bash



# set your own model path and port here
MODEL_PATH=your-model-path-here
MODEL_PORT=28080


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATAGEN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))


VLLM_LOG="$DATAGEN_DIR/sft_data_gen/log/vllm_log/vllm.log"
RESULT_LOG="$DATAGEN_DIR/sft_data_gen/log/result_log/result.log"

> "$VLLM_LOG"
> "$RESULT_LOG"

echo "vLLM log will be written to: $VLLM_LOG"
echo "Test] results log will be written to: $RESULT_LOG"

# launch Qwen2.5-VL-72B-Instruct using vLLM
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_PATH \
    --tensor-parallel-size 4 \
    --port $MODEL_PORT \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --limit-mm-per-prompt image=30,video=4 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --block-size 16 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "vLLM server started (PID: $VLLM_PID), waiting 500 seconds for initialization..."
sleep 500

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "Error: vLLM server failed to start! Check log: $VLLM_LOG" | tee -a "$RESULT_LOG"
    exit 1
fi



INPUT_DIR=$DATAGEN_DIR/vts_inference/outputs_dirs/merge_trace_output_dirs
OUTPUT_DIR=$DATAGEN_DIR/sft_data_gen/output_dirs/sft_datagen_output_dirs

SCRIPTS=$DATAGEN_DIR/sft_data_gen/sft_datagen.py

WORKER_PIDS=()
python $SCRIPTS \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_PATH \
    --model_port $MODEL_PORT \
    --worker_id 0 \
    --num_workers 1 &
WORKER_PIDS+=($!)

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


