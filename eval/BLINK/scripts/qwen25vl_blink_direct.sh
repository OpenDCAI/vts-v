#!/bin/bash



# default 
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=your_dashscope_api_key_here
export DASHSCOPE_MODEL="qwen-max"




# Set your own model path and port here
MODEL_PATH=your_model_path_here
MODEL_PORT=28080



SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BLINK_DIR=$(dirname "$SCRIPT_DIR")

# set output directory path
OUTPUT_DIR="$BLINK_DIR/output_dirs/qwen25vl_blink_direct"

# Set the path for the benchmark testing code
SCRIPTS=$BLINK_DIR/main.py

USING_VTS=False
USING_VERIFIER=False

# Set the task name, you can choosen from BLINK's subset name.
# Additionally, you can set the TASK_NAME variable to "all" to execute all subsets in the BLINK benchmark.
TASK_NAME=Visual_Similarity

python -u $SCRIPTS \
    --task_name $TASK_NAME \
    --using_vts $USING_VTS \
    --reasoner_type qwen-vl \
    --reasoner_model_path $MODEL_PATH \
    --reasoner_port $MODEL_PORT \
    --using_verifier $USING_VERIFIER \
    --dpo_model_type qwen-vl \
    --dpo_model_name_or_path None \
    --ref_model_type qwen-vl \
    --ref_model_name_or_path None \
    --verifier_threshold_type logits_score \
    --verifier_threshold 600 \
    --output_dir $OUTPUT_DIR


