#!/bin/bash



# By default, we use Qwen-max as the LLM-as-a-judge evaluation model. You can utilize any OpenAI-API-compatible model by configuring the corresponding base-url, api-key, and model-name parameters. However, please refrain from modifying the variable names DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, and DASHSCOPE_MODEL.
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=your_dashscope_api_key
export DASHSCOPE_MODEL="qwen-max"


DPO_MODEL_PATH=your_model_path_here
REF_MODEL_PATH=your_model_path_here

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/eval/BLINK/main.py

USING_VTS=True
USING_VERIFIER=True
TASK_NAME=Visual_Similarity

python -u $SCRIPTS \
    --task_name $TASK_NAME \
    --using_vts $USING_VTS \
    --reasoner_type qwen-vl \
    --reasoner_model_path None \
    --reasoner_port None \
    --using_verifier $USING_VERIFIER \
    --dpo_model_type qwen-vl \
    --dpo_model_name_or_path $DPO_MODEL_PATH \
    --ref_model_type qwen-vl \
    --ref_model_name_or_path $REF_MODEL_PATH \
    --verifier_threshold_type logits_score \
    --verifier_threshold 600 \
    --output_dir $OUTPUT_DIR


