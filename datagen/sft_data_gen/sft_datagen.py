import json
from pathlib import Path
from typing import Dict, List, Any
import re
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import math
import hashlib
import os
from llm_as_a_judge import llm_as_a_judge

def extract_image_indices_and_replace(text):
    """
    Extract all Image Indexes from the text and replace the corresponding positions with the <image> tag.
    Return: (Modified text, List of extracted image indexes)
    """
    image_indices = []
    pattern = r"Image Index:\s*(\d+)"
    matches = re.finditer(pattern, text)
    for match in matches:
        image_indices.append(int(match.group(1)))
    modified_text = re.sub(pattern, "<image>", text)
    
    return modified_text, image_indices if image_indices else None

def check_image_placeholders(data):
    images_count = len(data["images"])
    placeholder_count = 0
    for message in data["messages"]:
        if "<image>" in message["content"]:
            placeholder_count += message["content"].count("<image>")
    
    return images_count == placeholder_count


def check_imagepaths_exist(imagepaths: list[str]):
    for file in imagepaths:
        if not os.path.exists(file):
            return False
    return True


def transform_to_llama_factory_sft_format(original_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the original format to llama-factory SFT format
    """
    transformed = {
        "images": [],
        "messages": []
    }

    # Add system message
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant, and your goal is to solve the question provided. You can either rely on your own capabilities or perform actions with external tools to help you. You can use these actions: GroundingAction, DepthAction, ZoomInAction, VisualSearchAction, SegmentAction, CropAction, OCRAction, TextToImagesSimilarityAction, ImageToTextsSimilarityAction, ImageToImagesSimilarityAction, OverlayAction, TerminateAction. Answer the question using step-by-step reasoning and enclose your final answer with TerminateAction"
    }
    transformed["messages"].append(system_message)
    
    # Get all image paths from chosen response (assuming it's the correct one)
    image_paths = original_entry.get("traces", {}).get("images_saved_paths", [])
    if image_paths:
        transformed["images"].append(image_paths[0])  # Add the original image
    
    # Process conversation
    conversation = original_entry.get("conversation", [])
    if conversation:
        # User's first message (question)
        human_message = conversation[0]
        if human_message["from"] == "human":
            user_content = human_message["value"]
            # Remove <image> tag and keep the rest
            if "<image>" in user_content:
                user_content = user_content.replace("<image>", "").strip()
            
            transformed["messages"].append({
                "role": "user",
                "content": f"<image>{user_content}"
            })
    
    # Process chosen response's message_list for assistant and follow-up messages
    message_list = original_entry.get("traces", {}).get("message_list", [])
    
    for msg in message_list[2:]:  # Skip the first system message
        if msg["role"] == "assistant":
            transformed["messages"].append({
                "role": "assistant",
                "content": msg["content"]
            })
        elif msg["role"] == "user":
            if isinstance(msg["content"], list):
                response_content = ""
                for content in msg["content"]:
                    if content["type"] == "text":
                        text = content["text"]
                        modified_text, indices = extract_image_indices_and_replace(text)
                        response_content += modified_text
                        if indices:
                            for idx in indices:
                                if idx <= len(image_paths):
                                    transformed["images"].append(image_paths[idx-1])
                
                if response_content.strip():
                    transformed["messages"].append({
                        "role": "user",
                        "content": response_content.strip()
                    })
    
    return transformed



def process_task_chunk(task_dirs_chunk, output_dir, model_name, model_port, worker_id=0, num_workers=1):
    """Process a task shard."""
    chunk_output = []
    processed_count = 0

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{worker_id}_{num_workers}.json"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # Create a separate progress bar for each worker.
    position = int(worker_id) if worker_id else 0   
    pbar = tqdm(task_dirs_chunk, desc=f"Worker {worker_id}" if worker_id else "Worker", position=position)
    
    for task_dir in pbar:
        json_files = list(task_dir.glob("merged_trace*.json"))
        
        if not json_files:
            continue
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    data = [data]
                
                for entry in tqdm(data):
                    try:
                        source  = entry["data_source"]
                        idx = entry["id"]
                        question = entry["conversation"][0]["value"]
                        gold_answer = entry["gold_answer"]
                        prediction = entry["prediction_response"]
                        is_correct = llm_as_a_judge(question, gold_answer, prediction, model_name, model_port) == "Correct"
                        transformed_entry = transform_to_llama_factory_sft_format(entry)
                        image_placeholders_equal = check_image_placeholders(transformed_entry)
                        images_or_messages_not_empty = len(transformed_entry["images"]) > 0 and len(transformed_entry["messages"]) > 0
                        imagepaths_exist = check_imagepaths_exist(transformed_entry["images"])

                        data_result = {
                            "source" : source,
                            "id" : idx,
                            "question" : question,
                            "gold_answer" : gold_answer,
                            "prediction" : prediction,
                            "is_correct" : is_correct,
                            "image_placeholders_equal" : image_placeholders_equal,
                            "images_or_messages_not_empty" : images_or_messages_not_empty,
                            "imagepaths_exist" : imagepaths_exist,
                            "sft_train_data" : transformed_entry
                        }

                        chunk_output.append(data_result)
                        processed_count += 1
                        if processed_count % 10000 == 0:
                            with open(output_filepath, 'w', encoding='utf-8') as f:
                                json.dump(chunk_output, f, indent=4)
                    except Exception as e:
                        print(e)
                        
            except Exception as e:
                print(e)
    

    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk_output, f, indent=4)
    print(f"chunk_output len {len(chunk_output)}")
    return len(chunk_output)


def main():
    parser = argparse.ArgumentParser(description="Parallel data processing for LLaMA-Factory SFT format conversion")
    parser.add_argument("--input_dir", required=True, help="Root directory containing task directories")
    parser.add_argument("--output_dir", required=True, help="Final output JSON file path")
    parser.add_argument("--model_name", type=str, default=None, help="model_name")
    parser.add_argument("--model_port", type=int, default=None, help="model_path")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for distributed processing (optional)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()

    root_path = Path(args.input_dir)
    task_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print(f"No task directories found in {args.input_dir}")
        return
    
    print(f"Found {len(task_dirs)} task directories to process with {args.num_workers} workers")
    

    chunk_size = math.ceil(len(task_dirs) / args.num_workers)
    start = args.worker_id * chunk_size
    end = min(start + chunk_size, len(task_dirs))
    if start <= end:
        task_chunk = task_dirs[start:end]
        
        print(f"Worker {args.worker_id} processing {len(task_chunk)} directories (items {start}-{end})")
        process_task_chunk(task_chunk, args.output_dir, args.model_name, args.model_port, args.worker_id, args.num_workers)
    
if __name__ == "__main__":
    main()