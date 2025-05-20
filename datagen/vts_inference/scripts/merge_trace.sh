#!/bin/bash



SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/merge_trace_and_depulicate.py

INPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/outputs_dirs/vts_inference_output_dirs
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/vts_inference/outputs_dirs/merge_trace_output_dirs

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR