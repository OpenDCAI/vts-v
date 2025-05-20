#!/bin/bash


# Disabling the proxy is necessary; otherwise, you may encounter a 404 Forbidden error.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY no_proxy

# set your model path
MODEL_PATH=your_llama-3.2-vision_model_here

# set your port
MODEL_PORT=28080


CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 127.0.0.1 \
    --port $MODEL_PORT \
    --mem-fraction-static 0.8 \
    --chat-template=llama_3_vision




