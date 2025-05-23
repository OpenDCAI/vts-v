#!/bin/bash



# default 
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=your-own-dashscope-api-key
export DASHSCOPE_MODEL="qwen-max"


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MATHVISTA_DIR=$(dirname "$SCRIPT_DIR")



MODEL_PATH=your-model-path

MODEL_PORT=28080


DPO_MODEL_PATH=your-own-path

REF_MODEL_PATH=your-own-path


OUTPUT_DIR="$MATHVISTA_DIR/output_dirs/"



SCRIPTS=$MATHVISTA_DIR/main.py

USING_VTS=False
USING_VERIFIER=False
TASK_NAME=testmini

python -u $SCRIPTS \
    --task_name $TASK_NAME \
    --using_vts $USING_VTS \
    --reasoner_type qwen-vl \
    --reasoner_model_path $MODEL_PATH \
    --reasoner_port $MODEL_PORT \
    --using_verifier $USING_VERIFIER \
    --dpo_model_type qwen-vl \
    --dpo_model_name_or_path $DPO_MODEL_PATH \
    --ref_model_type qwen-vl \
    --ref_model_name_or_path $REF_MODEL_PATH \
    --verifier_threshold_type logits_score \
    --verifier_threshold 600 \
    --output_dir $OUTPUT_DIR


