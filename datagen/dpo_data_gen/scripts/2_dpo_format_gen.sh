#!/bin/bash



SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATAGEN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))



INPUT_DIR=$DATAGEN_DIR/dpo_data_gen/output_dirs/data_gen_output_dirs
OUTPUT_DIR=$DATAGEN_DIR/dpo_data_gen/output_dirs/dpo_format_gen_output_dirs

SCRIPTS=$DATAGEN_DIR/dpo_data_gen/dpo_format_gen.py

python $SCRIPTS --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR