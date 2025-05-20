#!/bin/bash

INPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs/data_gen_output_dirs
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs/dpo_format_gen_output_dirs

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/dpo_format_gen.py

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR