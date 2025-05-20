#!/bin/bash

MODEL_PATH=/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct
MODEL_PROT=28080

INPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/outputs_dirs/merge_trace_output_dirs
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/sft_data_gen/output_dirs/sft_datagen_output_dirs

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/sft_data_gen/sft_datagen.py

python $SCRIPTS \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_PATH \
    --model_port $MODEL_PROT \
    --worker_id 0 \
    --num_workers 1

echo "All worker Done!"
