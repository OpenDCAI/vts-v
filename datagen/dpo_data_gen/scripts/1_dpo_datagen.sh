#!/bin/bash

cd ../

QWEN_PATH="/fs-computility/llmit_d/shared/baitianyi/model/Qwen2.5-VL-72B-Instruct"
QWEN_PORT=28080

TASK_NAME="figureqa(cauldron,llava_format)"


## It's best to write your own absolute path
INPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/outputs_dirs/merge_trace_output_dirs
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/dpo_data_gen.py

CUDA_VISIBLE_DEVICES=4,5,6,7 python $SCRIPTS \
    --task $TASK_NAME \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --reasoner_type qwen-vl \
    --reasoner_model_path $QWEN_PATH \
    --reasoner_port $QWEN_PORT \
    --max_retries 5 \
    --rank 0 \
    --world_size 1