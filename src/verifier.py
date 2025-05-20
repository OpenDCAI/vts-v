
from copy import deepcopy
from qwen_vl_utils import process_vision_info
import json
import os
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)

from openai import OpenAI



class Verifier:
    def __init__(self, dpo_model_type, dpo_model_name_or_path,  ref_model_type, ref_model_name_or_path, threshold_type : str = "l2_norm", threshold: float = 0.5):

        self.dpo_model_type = dpo_model_type
        self.dpo_model_name_or_path = dpo_model_name_or_path
        self.ref_model_type = ref_model_type
        self.ref_model_name_or_path = ref_model_name_or_path
        self.threshold_type = threshold_type
        self.threshold = threshold

        if self.dpo_model_type == "qwen-vl" and self.dpo_model_name_or_path != "None":
            self.dpo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.dpo_model_name_or_path, torch_dtype="auto", device_map="auto"
            )
            self.dpo_processor = AutoProcessor.from_pretrained(self.dpo_model_name_or_path)
        if self.ref_model_type == "qwen-vl" and self.ref_model_name_or_path != "None":
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.ref_model_name_or_path, torch_dtype="auto", device_map="auto"
            )
            self.ref_processor = AutoProcessor.from_pretrained(self.ref_model_name_or_path)

    def get_reward(self, message_list, retry=3, timeout=1):
        verifier_messagelist = deepcopy(message_list)

        # Process the OpenAI-format message list and convert it into a message list format supported by Qwen-VL.
        for message in verifier_messagelist:
            if message["role"] == "user":
                contents = message["content"]
                if isinstance(contents, list):
                    for item in contents:
                        if item.get("type") == "image_url" and "image_url" in item:
                            base64image = item["image_url"]["url"]
                            item["image_url"] = base64image

        if self.dpo_model_type == "qwen-vl" and self.ref_model_type == "qwen-vl":
            # DPO model generates logits
            dpo_text = self.dpo_processor.apply_chat_template(
                verifier_messagelist,
                tokenize=False,
                add_generation_prompt=True
            )
            dpo_image_inputs, dpo_ = process_vision_info(verifier_messagelist)
            dpo_inputs = self.dpo_processor(
                text=[dpo_text],
                images=dpo_image_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            dpo_input_ids = dpo_inputs.input_ids
            dpo_input_length = dpo_input_ids.shape[1]
            self.dpo_model.eval()
            with torch.no_grad():
                dpo_outputs = self.dpo_model(**dpo_inputs)
                dpo_logits = dpo_outputs.logits
                dpo_generated_logits = dpo_logits[:, dpo_input_length - 1:, :]
            torch.cuda.empty_cache()


            ## Ref model generates logits
            ref_text = self.ref_processor.apply_chat_template(
                verifier_messagelist,
                tokenize=False,
                add_generation_prompt=True
            )
            ref_image_inputs, ref_ = process_vision_info(verifier_messagelist)
            ref_inputs = self.ref_processor(
                text=[ref_text],
                images=ref_image_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            ref_input_ids = ref_inputs.input_ids
            ref_input_length = ref_input_ids.shape[1]
            self.ref_model.eval()
            with torch.no_grad():
                ref_outputs = self.ref_model(**ref_inputs)
                ref_logits = ref_outputs.logits
                ref_generated_logits = ref_logits[:, ref_input_length - 1:, :]
            torch.cuda.empty_cache()


            return dpo_generated_logits, ref_generated_logits
        
            
