#!/bin/bash



# export WANDB_API_KEY=[your wandb_api_key here]
export WANDB_API_KEY=34a4634ba3972e92e7a2e6115c90f604708f06e3

CONFIG_FILE=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/dpo/config/zero3_dpo_8.yaml

SCRIPTS=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/dpo/trl_dpo_train.py

# DATASET_NAME_OR_PATH=/fs-computility/llmit_d/shared/baitianyi/vts_v/datagen/dpo_data_gen/output_dirs/data_filter_by_tokens_output_dirs/llavaov_dpo_train_merged_less_than_10000_12.json
DATASET_NAME_OR_PATH=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/dpo/data/old_select_1000.json
MODEL_NAME_OR_PATH=/fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/model/Qwen2.5-VL-7B-Instruct
OUTPUT_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/dpo/output_dirs


accelerate launch --config_file /fs-computility/llmit_d/shared/baitianyi/vts_v/train/dpo/config/zero3_dpo_8.yaml \
    /fs-computility/llmit_d/shared/baitianyi/vts/train/dpo/trl_dpo_vlm.py \
    --dataset_name $DATASET_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataset_num_proc 64 \
    --bf16 \
    --learning_rate 5e-6  \
    --lr_scheduler_type 'cosine' \
    --logging_step 10 \
    --save_steps 1000 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing True \
    --ddp_timeout 180000000 \
    --warmup_ratio 0.03 \
    --report_to wandb \
    --run_name test_code_dpo
