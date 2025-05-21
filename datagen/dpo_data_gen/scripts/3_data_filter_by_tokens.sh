#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAGEN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

INPUT_FILE=$DATAGEN_DIR/dpo_data_gen/output_dirs/dpo_format_gen_output_dirs/llavaov_dpo_train_merged.json
OUTPUT_DIR=$DATAGEN_DIR/dpo_data_gen/output_dirs/data_filter_by_tokens_output_dirs

SCRIPTS=$DATAGEN_DIR/dpo_data_gen/data_filter_by_tokens.py

python $SCRIPTS --model $MODEL_PATH --input_file $INPUT_FILE --output_dir $OUTPUT_DIR --threshold 10000
