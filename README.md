# VTS-V: Multi-step Visual Reasoning with Visual Tokens Scaling and Verification

This repository is the official implementation of [Multi-step Visual Reasoning with Visual Tokens Scaling and Verification]

## Requirements

To install requirements:

```setup
conda create -n vts_v python=3.10
conda activate vts_v
pip install -r requirements.txt
```



## Evaluation

### Step-1: Launch model deployment

You can use closed-source APIs and open-source models to run our method VTS-V.

#### Closed-source APIs
For closed-source APIs, the system must support OpenAI's chat/completions functionality. 

In our experiments, we employed GPT-4o as the reasoning model.

#### Open-source Models

For open-source models, they need to be deployed in OpenAI API format using vLLM or SGLang.

In our approach, we employed the [Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), and [LLaMA-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) models, along with our corresponding fine-tuned versions. For the Qwen2.5VL and Qwen2-VL series models, we utilized vLLM for deployment, while for the LLaMA-3.2-Vision series models, we adopted SGLang for deployment.

For vLLM, you can see [qwen2vl_28080.sh](./client/vllm/qwen2vl_28080.sh) and [qwen25vl_28080.sh](./client/vllm/qwen25vl_28080.sh) as examples

For SGLang, you can see [llama32vision.sh](./client/sglang/llama32vision_28080.sh) as an example


In the [qwen2vl_28080.sh](./client/vllm/qwen2vl_28080.sh), [qwen25vl_28080.sh](./client/vllm/qwen25vl_28080.sh) and [llama32vision.sh](./client/sglang/llama32vision_28080.sh) scripts, you need to specify `MODEL_PATH` with your own model directory or the corresponding HuggingFace model ID, and specify `MODEL_PORT` with your chosen port number. Note that these `MODEL_PATH` and `MODEL_PORT` values must remain consistent with what you set here when running subsequent test scripts.

```
# set your own model path
MODEL_PATH=your model path or name here
# set your own port
MODEL_PORT=28080
```

### Step-2: Using VTS-V Inference and Evaluation

Currently, our supported benchmarks for testing include [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), and [Vstar](https://huggingface.co/datasets/craigwu/vstar_bench).

Our method operates in three modes: Direct, VTS, and VTS-V.

By default, we use `Qwen-max` as the LLM-as-a-judge evaluation model. You can utilize any OpenAI-API-compatible model by configuring the corresponding base-url, api-key, and model-name parameters. However, please refrain from modifying the variable names `DASHSCOPE_BASE_URL`, `DASHSCOPE_API_KEY`, and `DASHSCOPE_MODEL`.
```
# default 
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY=your-dashscope-api-key-here
export DASHSCOPE_MODEL="qwen-max"
```


#### Step2-1: Direct 


#### Step2-2: Using VTS Inference




| Mode | using_vts | using_verifier | reasoner_model| dpo_model_path | ref_model_path|
|------| --------- |----------------| --------------|----------------| ---------------|
|Direct| False     |   False        | None          | None           | None           |
|VTS   | True      | False          |

#### Step2-3: Using VTS-V inference


## Data Construction




## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Results


## Contributing
