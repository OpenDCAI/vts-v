#!/bin/bash



export DISABLE_VERSION_CHECK=1


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_DIR=$(dirname $(dirname "$SCRIPT_DIR"))



CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


LOG_FILE="$TRAIN_DIR/sft/scripts/llama_vision/log/training_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output (standard output and error output) to the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Start training, with logs saved to: $LOG_FILE"
echo "Execution time: $(date)"


## TODO
# You should set your own model path 
MODEL_NAME_OR_PATH=your-own-model-name-or-path




OUTPUT_DIR="$TRAIN_DIR/sft/LLaMA-Factory/saves/LLaMA_3_2_Vision_Instruct_11B/lora"
LLAMAFACTORY_DATA_DIR=$TRAIN_DIR/sft/LLaMA-Factory/data
DEEPSPEED_CONFIG_PATH=$TRAIN_DIR/sft/LLaMA-Factory/cache/ds_z3_config.json



# Basic commands and parameters
CMD="llamafactory-cli train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --optim adamw_torch \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --dataset_dir $LLAMAFACTORY_DATA_DIR \
    --dataset vts_sft_data \
    --template mllama \
    --cutoff_len 131072 \
    --max_samples 10000000 \
    --include_num_input_tokens_seen True \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 4 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --learning_rate 3.0e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --deepspeed $DEEPSPEED_CONFIG_PATH"



# Locate valid checkpoints
VALID_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    # Retrieve all checkpoint folders and sort them by version
    CHECKPOINTS=($(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*checkpoint*" | sort -V))
    
    # Start checking from the latest checkpoint and find the first one that contains `trainer_state.json`
    for ((i=${#CHECKPOINTS[@]}-1; i>=0; i--)); do
        CHECKPOINT="${CHECKPOINTS[i]}"
        if [ -f "$CHECKPOINT/trainer_state.json" ]; then
            VALID_CHECKPOINT="$CHECKPOINT"
            echo "Valid checkpoint found: $VALID_CHECKPOINT"
            break
        else
            echo "checkpoint $CHECKPOINT don't have trainer_state.json, Try the previous checkpoint."
        fi
    done
    
    if [ -z "$VALID_CHECKPOINT" ]; then
        echo "No valid checkpoint found. Training will start from scratch."
    fi
else
    echo "The output directory does not exist. Training will start from scratch."
fi

# start training
echo "Start training the model using command-line parameters..."


# Determine how to start training based on whether there is a valid checkpoint.
if [ -n "$VALID_CHECKPOINT" ]; then
    echo "Training will resume from the valid checkpoint.: $VALID_CHECKPOINT"
    $CMD --resume_from_checkpoint "$VALID_CHECKPOINT"
else
    echo "No valid checkpoint found. Training will start from scratch:"
    $CMD
fi

echo "The training task has been completed."
echo "Completion time: $(date)"

exit 0