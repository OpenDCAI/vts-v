# Training

In this work, we conducted supervised fine-tuning (SFT) using our custom-built VTS-SFT dataset on three models: [Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), and [LLaMA-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct). Additionally, we performed DPO training on [Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) using our constructed VTS-DPO dataset.

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning (SFT) training and utilize the [TRL](https://huggingface.co/docs/trl/index) library for Direct Preference Optimization (DPO) training.

# SFT Training

## Step 1: Install Requirements

```bash
cd train/sft
conda create -n vts_v_sft python=3.10
pip install -r requirements.txt
conda activate vts_v_sft
```



## Step 2: Dataset Preparation

First, you need to copy the generated VTS-SFT dataset to the `train/sft/LLaMA-Factory/data` folder according to LLaMA-Factory's requirements.

The SFT dataset generated during the [Dataset Construction](../datagen/README.md#step-2-1-sft-dataset-generation) step should be stored at `datagen/sft_data_gen/output_dirs/data_fliter_output_dirs/llavaov_sft_train.json`

Then, add the following entry in the `train/sft/LLaMA-Factory/data/dataset_info.json` file:
```json
[
    "vts_sft_data" : { 
        "file_name": "llavaov_sft_train.json",
        "formatting": "sharegpt",
        "columns": {
            "images" :"images",
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag" : "system"
        }
    },
]
```

## Step 3: Performing SFT Training with LoRA

In our experiments, we conducted training on three models:[Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), and [LLaMA-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct). The corresponding training scripts are located at [qwen25vl.sh](./sft/train_scripts/qwen25vl.sh), [qwen2vl.sh](./sft/train_scripts/qwen2vl.sh) and [llama32vision.sh](./sft/train_scripts/llama32vision.sh).

You should specify your own model paths in these training scripts:
```bash
# TODO: You should set your own model path 
MODEL_NAME_OR_PATH=your-own-model-name-or-path
```

Then run the following command in the terminal:
```
cd train/sft/train_scripts
conda activate vts_v_sft
bash qwen25vl.sh  # or qwen2vl.sh, or llama32vision.sh
```

## Step 4: Merge Model

The LoRA checkpoints generated during SFT training with LLaMA-Factory need to be merged with the original base model for deployment.

The configuration files for merging are located at [llamafactory_merge_qwen25vl.yaml](./sft/merge_model/scripts/qwen25vl/llamafactory_merge_qwen25vl.yaml), [llamafactory_merge_qwen2vl.yaml](./sft/merge_model/scripts/qwen2vl/llamafactory_merge_qwen2vl.yaml) and [llamafactory_merge_llama32vision.yaml](./sft/merge_model/scripts/llama32vision/llamafactory_merge_llama32vision.yaml).


First, you need to specify the `model_name_or_path`, `adapter_name_or_path`, and `export_dir` with your own paths in these YAML configuration files:
```yaml
model_name_or_path: your-original-model-name-or-path
adapter_name_or_path: your-lora-checkpoint-path
export_dir: your-export-dir
```

Then execute the following command in the terminal (taking Qwen2.5-VL as an example):
```bash
conda activate vts_v_sft
cd train/sft/merge_model/scripts/qwen25vl
bash merge_qwen25vl.sh
```


# DPO Training

## Step 1: Install Requirements

```bash
cd train/dpo
conda create -n vts_v_dpo python=3.10
pip install -r requirements.txt
```

## Step 2: Start Training

You can run the script [dpo_vlm_8.sh](./dpo/scripts/dpo_vlm_8.sh) to start the training.

First, you need to copy the path of the dataset constructed in the [Dataset Construct Step](../datagen/README.md#step-2-2-dpo-dataset-generation), and then set the addresses for the dataset and model in the training script:
```bash
## TODO: you should specify you own dataset and model path
DATASET_NAME_OR_PATH=your-dataset-path
MODEL_NAME_OR_PATH=your-model-path
```

Then execute the following command in the terminal to start the training:
```bash
conda activate vts_v_dpo
cd scripts
bash dpo_vlm_8.sh
```



