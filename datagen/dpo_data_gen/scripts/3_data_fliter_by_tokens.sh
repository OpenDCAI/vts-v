#!/bin/bash


MODEL_PATH=/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/model/Qwen2.5-VL-7B-Instruct
INPUT_FILE=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs/dpo_format_gen_output_dirs/llavaov_dpo_train_merged.json
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs/data_filter_by_tokens_output_dirs

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/data_fliter_by_tokens.py

python $SCRIPTS --model $MODEL_PATH --input_file $INPUT_FILE --output_dir $OUTPUT_DIR --threshold 10000
