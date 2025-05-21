#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VTS_INFERENCE_DIR=$(dirname "$SCRIPT_DIR")


SCRIPTS=$VTS_INFERENCE_DIR/merge_trace_and_depulicate.py

INPUT_DIR=$VTS_INFERENCE_DIR/outputs_dirs/vts_inference_output_dirs
OUTPUT_DIR=$VTS_INFERENCE_DIR/outputs_dirs/merge_trace_output_dirs

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR