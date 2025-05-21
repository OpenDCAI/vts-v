#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATAGEN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

INPUT_DIR=$DATAGEN_DIR/sft_data_gen/output_dirs/sft_datagen_output_dirs
OUTPUT_DIR=$DATAGEN_DIR/sft_data_gen/output_dirs/data_fliter_output_dirs

SCRIPTS=$DATAGEN_DIR/sft_data_gen/merge_and_filt.py

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

