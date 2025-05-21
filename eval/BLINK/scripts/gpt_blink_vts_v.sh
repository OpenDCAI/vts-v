#!/bin/bash



# By default, we use Qwen-max as the LLM-as-a-judge evaluation model. You can utilize any OpenAI-API-compatible model by configuring the corresponding base-url, api-key, and model-name parameters. However, please don't modify the variable names DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, and DASHSCOPE_MODEL.
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=your_dashscope_api_key
export DASHSCOPE_MODEL="qwen-max"


## Set your own gpt-4o api base-url and api-key here
export OPENAI_BASE_URL=https://api.openai.com/v1 # default value
export OPENAI_API_KEY=your-openai-api-key-here


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BLINK_DIR=$(dirname "$SCRIPT_DIR")

# set output directory path
OUTPUT_DIR="$BLINK_DIR/output_dirs/gpt_blink_direct"

# Set the path for the benchmark testing code
SCRIPTS=$BLINK_DIR/main.py

# Set the reasoning mode
USING_VTS=True
USING_VERIFIER=True

# Set the task name, you can choosen from BLINK's subset name.
# Additionally, you can set the TASK_NAME variable to "all" to execute all subsets in the BLINK benchmark.
TASK_NAME=Visual_Similarity

# set dpo model path and ref model path
DPO_MODEL_PATH=your_own_dpo_model_path_here
REF_MODEL_PATH=your_own_ref_model_path_here



python -u $SCRIPTS \
    --task_name $TASK_NAME \
    --using_vts $USING_VTS \
    --reasoner_type gpt-4o \
    --reasoner_model_path None \
    --reasoner_port None \
    --using_verifier $USING_VERIFIER \
    --dpo_model_type qwen-vl \
    --dpo_model_name_or_path None \
    --ref_model_type qwen-vl \
    --ref_model_name_or_path None \
    --verifier_threshold_type logits_score \
    --verifier_threshold 600 \
    --output_dir $OUTPUT_DIR


