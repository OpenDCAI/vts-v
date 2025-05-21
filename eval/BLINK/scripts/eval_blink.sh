#!/bin/bash



# default 
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=sk-573b81427b2a4ee4b18931a44139a8b8
export DASHSCOPE_MODEL="qwen-max"




REASONER_MODEL_PATH="/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/model/Qwen2-VL-7B-Instruct"

REASONER_PORT=28080


DPO_MODEL_PATH="/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/output_dirs/internvl8b_dpo_301k_trl_8machines_v2/checkpoint-12225"

REF_MODEL_PATH="/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/model/Qwen2.5-VL-7B-Instruct"


OUTPUT_DIR="/fs-computility/llmit_d/shared/baitianyi/vts_v/eval/BLINK/output_dirs/test_code_qwen2vl_original"



SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/eval/BLINK/main.py

USING_VTS=False
USING_VERIFIER=False
TASK_NAME=Visual_Similarity

python -u $SCRIPTS \
    --task_name $TASK_NAME \
    --using_vts $USING_VTS \
    --reasoner_type qwen-vl \
    --reasoner_model_path $REASONER_MODEL_PATH \
    --reasoner_port $REASONER_PORT \
    --using_verifier $USING_VERIFIER \
    --dpo_model_type qwen-vl \
    --dpo_model_name_or_path $DPO_MODEL_PATH \
    --ref_model_type qwen-vl \
    --ref_model_name_or_path $REF_MODEL_PATH \
    --verifier_threshold_type logits_score \
    --verifier_threshold 600 \
    --output_dir $OUTPUT_DIR


