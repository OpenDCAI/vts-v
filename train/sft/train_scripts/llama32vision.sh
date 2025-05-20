#!/bin/bash



export DISABLE_VERSION_CHECK=1


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


LOG_FILE="/fs-computility/llmit_d/shared/baitianyi/vts/train/sft/scripts/llama_vision/log/training_$(date +%Y%m%d_%H%M%S).log"

# 重定向所有输出（标准输出和错误输出）到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "开始训练，日志保存到: $LOG_FILE"
echo "执行时间: $(date)"


OUTPUT_DIR="/fs-computility/llmit_d/shared/baitianyi/vts_v/train/sft/LLaMA-Factory/saves/LLaMA_3_2_Vision_Instruct_11B/lora"


MODEL_NAME_OR_PATH=/fs-computility/llmit_d/shared/baitianyi/dl/unsloth/Llama-3.2-11B-Vision-Instruct
LLAMAFACTORY_DATA_DIR=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/sft/LLaMA-Factory/data
DEEPSPEED_CONFIG_PATH=/fs-computility/llmit_d/shared/baitianyi/vts_v/train/sft/LLaMA-Factory/cache/ds_z3_config.json



# 基本命令和参数
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



# 查找有效的检查点
VALID_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    # 获取所有checkpoint文件夹，按版本排序
    CHECKPOINTS=($(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*checkpoint*" | sort -V))
    
    # 从最新的检查点开始检查，找到第一个包含trainer_state.json的检查点
    for ((i=${#CHECKPOINTS[@]}-1; i>=0; i--)); do
        CHECKPOINT="${CHECKPOINTS[i]}"
        if [ -f "$CHECKPOINT/trainer_state.json" ]; then
            VALID_CHECKPOINT="$CHECKPOINT"
            echo "找到有效的检查点: $VALID_CHECKPOINT"
            break
        else
            echo "检查点 $CHECKPOINT 中没有找到 trainer_state.json，尝试上一个检查点"
        fi
    done
    
    if [ -z "$VALID_CHECKPOINT" ]; then
        echo "未找到有效的检查点，将从头开始训练"
    fi
else
    echo "输出目录不存在，将从头开始训练"
fi

# 开始训练模型
echo "开始训练模型，使用命令行参数方式..."


# 根据是否有有效检查点决定如何启动训练
if [ -n "$VALID_CHECKPOINT" ]; then
    echo "将从有效检查点继续训练: $VALID_CHECKPOINT"
    # 使用检查点继续训练
    $CMD --resume_from_checkpoint "$VALID_CHECKPOINT"
else
    echo "没有找到有效检查点，从头开始训练"
    # 首次训练，不传递resume_from_checkpoint参数
    $CMD
fi

echo "训练任务已完成"
echo "完成时间: $(date)"

exit 0