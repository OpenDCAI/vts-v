import json
import os
from shutil import rmtree
import argparse

import re

def safe_slugify(text):
    """Convert text to a safe filename by removing special characters and spaces."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text


def check_image_paths_existing(image_paths_list: list[str]):
    """
    This function is designed to remove data where the `image_saved_path` does not actually exist. 
    This can happen because the data might be rerun, and the trajectory is saved twice. 
    One of the trajectories might have cleared the images, but during data deduplication, the data with non-existent image paths is retained.
    """
    for file in image_paths_list:
        if not os.path.exists(file):
            return False
    return True


def merge_trace_and_depulicate(root_dir, output_base_dir):
    """
    Collect all trace_x_y.json files from each task directory,
    process them to remove image URLs, merge them and remove duplicates.
    """
    os.makedirs(output_base_dir, exist_ok=True)


    # Walk through all task directories
    for task_name in os.listdir(root_dir):
        # print(f"Processing task {task_name}.....")
        

        total_count = 0
        depulicated_count = 0
        original_count = 0

        task_path = os.path.join(root_dir, task_name)

        task_output_dir = os.path.join(output_base_dir, task_name)
        if os.path.exists(task_output_dir):
            rmtree(task_output_dir)
        os.makedirs(task_output_dir, exist_ok=True)

        if not os.path.isdir(task_path):
            continue

        print(f"Processing task {task_name}.....")

        merged_data = []
        existing_ids = set()
        
        # Find all trace files
        for file_name in os.listdir(task_path):
            if file_name.startswith("trace_") and file_name.endswith(".json"):
                file_path = os.path.join(task_path, file_name)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                        
                    # Process each item in the trace file
                    if isinstance(data, list):
                        original_count += len(data)
                        for item in data:
                            identifier = item["data_source"] + item["id"]
                            image_paths_exist = check_image_paths_existing(item["traces"]["images_saved_paths"])
                            if identifier not in existing_ids and image_paths_exist:
                                existing_ids.add(identifier)
                                merged_data.append(item)
                                total_count += 1
                            else:
                                depulicated_count += 1
                    else:
                        original_count += 1
                        identifier = data["data_source"] + data["id"]
                        image_paths_exist = check_image_paths_existing(data["traces"]["images_saved_paths"])
                        if identifier not in existing_ids and image_paths_exist:
                            existing_ids.add(identifier)
                            merged_data.append(data)
                            total_count += 1
                        else:
                            depulicated_count += 1
                        
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

        print(f"Task {task_name} processed data count: {len(merged_data)}, depylicated_count: {depulicated_count}")
        output_file = os.path.join(task_output_dir, "merged_trace.json")
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=4) 
    
    print("All tasks processed successfully!")
    return output_base_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    merge_trace_and_depulicate(args.input_dir, args.output_dir)