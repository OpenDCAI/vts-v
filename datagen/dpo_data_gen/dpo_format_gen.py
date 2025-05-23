import json
from pathlib import Path
from typing import Dict, List, Any
from copy import deepcopy
from tqdm import tqdm
import os
import argparse

def extract_image_indices(text):
    """
    Extract all Image Indexes and their associated text from the document.
    Supported formats include:
    1. "Here's...Image Index: 1\n"
    2. "Image Index: 0\n"
    3. "Image Index: 12\nHere's the image marked...\nPay attention.\n"
    
    return: list of (index, associated_text)
    """
    results = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Image Index:" in line:
            # Extract the index values.
            try:
                idx_str = line.split("Image Index:")[1].strip().split()[0]
                index = int(idx_str)
                
                # Collect the associated text (current line + subsequent lines that are not Image Index lines).
                associated_lines = [line]
                j = i + 1
                while j < len(lines) and "Image Index:" not in lines[j]:
                    associated_lines.append(lines[j])
                    j += 1
                
                associated_text = '\n'.join(associated_lines)
                results.append((index, associated_text))
                i = j  # Skip the lines that have already been processed
            except (ValueError, IndexError):
                i += 1
        else:
            i += 1
    
    return results if results else None


def check_imagepaths_exist(imagepaths: list[str]):
    for file in imagepaths:
        if not os.path.exists(file):
            return False
    return True





def transform_entry(original_entry: Dict[str, Any]) -> Dict[str, Any]:
    transformed = {
        "source": original_entry.get("source", ""),
        "id": original_entry.get("id", ""),
        "prompt_image_path": [],
        "chosen_image_path": [],
        "rejected_image_path": [],
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    # Retrieve all image paths
    chosen_image_paths = original_entry.get("chosen", {}).get("images_saved_paths", [])

    # Set the prompt image path (original image).
    if chosen_image_paths:
        transformed["prompt_image_path"].append(chosen_image_paths[0])

    # Construct the prompt section - keep it as-is.
    system_prompt = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant, and your goal is to solve the question provided. You can either rely on your own capabilities or perform actions with external tools to help you. You can use these actions: GroundingAction, DepthAction, ZoomInAction, VisualSearchAction, SegmentAction, CropAction, OCRAction, TextToImagesSimilarityAction, ImageToTextsSimilarityAction, ImageToImagesSimilarityAction, OverlayAction, TerminateAction. Answer the question using step-by-step reasoning and enclose your final answer with TerminateAction."
        }]
    }
    
    # user prompt
    user_prompt_content = []
    conversation = original_entry.get("conversation", [])
    if conversation and conversation[0]["from"] == "human":
        user_value = conversation[0]["value"]
        # Check if there is an image marker.
        if "<image>" in user_value:
            user_prompt_content.append({
                "type": "image"
            })
            # Remove the image marker and retain the remaining text.
            text_content = user_value.replace("<image>", "").strip()
            if text_content:
                user_prompt_content.append({
                    "type": "text",
                    "text": text_content
                })
        else:
            user_prompt_content.append({
                "type": "text",
                "text": user_value
            })
    
    user_prompt = {
        "role": "user",
        "content": user_prompt_content
    }
    
    transformed["prompt"] = [system_prompt, user_prompt]
    
    ## chosen
    chosen_message_list = original_entry.get("chosen", {}).get("message_list", [])
    
    for msg in chosen_message_list[2:]:
        if msg["role"] == "assistant":
            transformed["chosen"].append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": msg["content"]
                }]
            })
        elif msg["role"] == "user":
            user_content = []
            
            if isinstance(msg["content"], list):
                for content in msg["content"]:
                    if content["type"] == "text":
                        text = content["text"]
                        indices_data = extract_image_indices(text)
                        
                        if indices_data:
                            for idx, full_text in indices_data:
                                # print(f"idx: {idx}  full_text: {full_text} chosen_image_path_len: {len(chosen_image_paths)}")
                                if idx <= len(chosen_image_paths):
                                    # print("idx < len chosen image paths")
                                    transformed["chosen_image_path"].append(chosen_image_paths[idx-1])
                                    user_content.append({
                                        "type": "text",
                                        "text": full_text
                                    })
                                    user_content.append({
                                        "type": "image"
                                    })
                        else:
                            user_content.append({
                                "type": "text",
                                "text": text
                            })
            
            if user_content:
                transformed["chosen"].append({
                    "role": "user",
                    "content": user_content
                })

    ## rejected
    rejected_message_list = original_entry.get("rejected", {}).get("message_list", [])
    rejected_image_paths = original_entry.get("rejected", {}).get("images_saved_paths", [])

    
    for msg in rejected_message_list[2:]:
        if msg["role"] == "assistant":
            transformed["rejected"].append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": msg["content"]
                }]
            })
        elif msg["role"] == "user":
            user_content = []
            
            if isinstance(msg["content"], list):
                for content in msg["content"]:
                    if content["type"] == "text":
                        text = content["text"]
                        
                        indices_data = extract_image_indices(text)
                        
                        if indices_data:
                            for idx, full_text in indices_data:
                                # print(f"idx: {idx}  full_text: {full_text} rejected_image_path_len: {len(rejected_image_paths)}")
                                if idx <= len(rejected_image_paths):
                                    # print("idx < len rejected image paths")
                                    transformed["rejected_image_path"].append(rejected_image_paths[idx-1])
                                    user_content.append({
                                        "type": "text",
                                        "text": full_text
                                    })
                                    user_content.append({
                                        "type": "image"
                                    })
                        else:
                            user_content.append({
                                "type": "text",
                                "text": text
                            })
            
            if user_content:
                transformed["rejected"].append({
                    "role": "user",
                    "content": user_content
                })
    
    return transformed


def process_all_datasets(root_dir: str, output_dir: str):
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # Ensure that the output directory exists.
    output_path.mkdir(parents=True, exist_ok=True)
    
    existing_data = []
    existing_ids = set()
    
    # Check if there is a merged main output file.
    main_output_file = output_path / "llavaov_dpo_train_merged.json"
    if main_output_file.exists():
        try:
            with open(main_output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_ids = {entry["source"] + entry["id"] for entry in existing_data}
            print(f"Loaded existing merged output with {len(existing_data)} entries")
        except Exception as e:
            print(f"Warning: Could not load existing merged output file: {str(e)}")
    
    all_transformed_data = deepcopy(existing_data)
    processed_files = 0
    total_new_entries = 0
    
    for subdir in tqdm(root_path.iterdir()):
        if subdir.is_dir():
            # Create corresponding output directories for each subdirectory.
            subdir_output_dir = output_path / subdir.name
            subdir_output_dir.mkdir(exist_ok=True)
            
            subdir_transformed_data = []
            subdir_existing_ids = set()
            
            # Check if there are already output files in the subdirectory.
            subdir_output_file = subdir_output_dir / f"{subdir.name}_processed.json"
            if subdir_output_file.exists():
                try:
                    with open(subdir_output_file, 'r', encoding='utf-8') as f:
                        subdir_existing_data = json.load(f)
                        subdir_existing_ids = {entry["source"] + entry["id"] for entry in subdir_existing_data}
                        subdir_transformed_data = deepcopy(subdir_existing_data)
                    print(f"Loaded existing output for {subdir.name} with {len(subdir_existing_data)} entries")
                except Exception as e:
                    print(f"Warning: Could not load existing output file for {subdir.name}: {str(e)}")
            
            # Process the files in the subdirectory.
            subdir_processed_files = 0
            subdir_new_entries = 0
            
            for input_file in subdir.glob("processed_output*.json"):
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        data = [data]
                    
                    for entry in tqdm(data, desc=f"Processing {input_file.name}"):
                        try:
                            ## `is_correct` and `build_preference_data` are in the original data.
                            identifier = entry["source"] + entry["id"]
                            is_correct = entry.get("is_correct", False)
                            build_preference_success = entry.get("build_preference_data", False)
                            
                            transformed_entry = transform_entry(entry)


                            prompt_image_path_exist = check_imagepaths_exist(transformed_entry["prompt_image_path"])
                            chosen_image_path_exist = check_imagepaths_exist(transformed_entry["chosen_image_path"])
                            rejected_image_path_exist = check_imagepaths_exist(transformed_entry["rejected_image_path"])
                            
                           
                            if (identifier not in existing_ids and identifier not in subdir_existing_ids and is_correct and build_preference_success and prompt_image_path_exist and chosen_image_path_exist and rejected_image_path_exist):
                                subdir_transformed_data.append(transformed_entry)
                                all_transformed_data.append(transformed_entry)
                                subdir_existing_ids.add(identifier)
                                existing_ids.add(identifier)
                                subdir_new_entries += 1
                                total_new_entries += 1
                        except Exception as e:
                            print(f"Error processing entry {entry.get('id', 'unknown')} from {input_file}: {str(e)}")
                            continue
                    
                    subdir_processed_files += 1
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing file {input_file}: {str(e)}")
                    continue
            
            # Save the processing results for the subdirectory.
            if subdir_transformed_data:
                with open(subdir_output_file, 'w', encoding='utf-8') as f:
                    json.dump(subdir_transformed_data, f, indent=4, ensure_ascii=False)
                print(f"Saved {len(subdir_transformed_data)} entries for {subdir.name} to {subdir_output_file}")
    
    ## Convert to the format supported by TRL:

        
    converted_dataset = []

    for data in tqdm(all_transformed_data):
        converted_data = {
            "prompt" : data["prompt"],
            "chosen" : data["chosen"],
            "rejected" : data["rejected"],
            "image_path" : data["prompt_image_path"][0],
            "source" : data["source"],
            "id" : data["id"]
        }
        converted_dataset.append(converted_data)

    # Save the merged main output file.
    if converted_dataset:
        with open(main_output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_dataset, f, indent=4, ensure_ascii=False)
        print(f"Saved merged output with {len(converted_dataset)} entries to {main_output_file}")
    


    print(f"\nProcessing completed:")
    print(f"- Processed {processed_files} files")
    print(f"- Added {total_new_entries} new entries")
    print(f"- Total entries in merged output: {len(converted_dataset)}")


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="DPO data format generation.")
    parser.add_argument("--input_dir", type=str, default="/fs-computility/llmit_d/shared/baitianyi/vts/datagen/outputs_dir/outputs1", help="Base directory containing task folders")
    parser.add_argument("--output_dir", type=str, default="/fs-computility/llmit_d/shared/baitianyi/vts/datagen/dpo_data_gen/output_dis", help="Base directory to save processed outputs")
    
    args = parser.parse_args()


    process_all_datasets(args.input_dir, args.output_dir)