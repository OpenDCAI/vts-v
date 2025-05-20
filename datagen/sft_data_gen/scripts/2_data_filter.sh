#!/bin/bash


INPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/sft_data_gen/output_dirs/sft_datagen_output_dirs
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/sft_data_gen/output_dirs/data_fliter_output_dirs

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/sft_data_gen/merge_and_filt.py

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

