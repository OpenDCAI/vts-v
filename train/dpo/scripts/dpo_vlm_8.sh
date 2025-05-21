#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))



CONFIG_FILE=$TRAIN_DIR/dpo/config/zero3_dpo_8.yaml
SCRIPTS=$TRAIN_DIR/dpo/trl_dpo_train.py
OUTPUT_DIR=$TRAIN_DIR/dpo/output_dirs


## TODO: you should specify you own dataset and model path
DATASET_NAME_OR_PATH=your-dataset-path
MODEL_NAME_OR_PATH=your-model-path




accelerate launch --config_file $CONFIG_FILE \
    $SCRIPTS\
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
