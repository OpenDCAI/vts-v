import json
import argparse
import os
from tqdm import tqdm


def check_only_one_step(train_data: dict):
    if len(train_data["images"]) == 1 and len(train_data["messages"]) == 3:
        return True
    return False


def merge_and_filter(input_dir, output_dir):
    seen_ids = set()
    merged_data = []
    for file_name in tqdm(os.listdir(input_dir), desc="Processing json file: ", position=0):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r") as f:
                dataset = json.load(f)
            for data in tqdm(dataset, position=1):
                identifier = data["source"] + data["id"]
                is_correct = data.get("is_correct", False)
                image_placeholders_equal = data.get("image_placeholders_equal", False)
                images_or_messages_not_empty = data.get("images_or_messages_not_empty", False)
                imagepaths_exist = data.get("imagepaths_exist", False)
                only_one_step = check_only_one_step(data["sft_train_data"])


                if identifier not in seen_ids and is_correct and image_placeholders_equal and images_or_messages_not_empty and imagepaths_exist and not only_one_step:
                    merged_data.append(data["sft_train_data"])
                    seen_ids.add(identifier)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "llavaov_sft_train.json")
    print(f"Total Count: {len(merged_data)}")
    with open(output_file_path, "w") as f:
        json.dump(merged_data, f, indent=4)
    
    return len(merged_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Data fileter")
    parser.add_argument("--input_dir", required=True, help="Root directory containing task directories")
    parser.add_argument("--output_dir", required=True, help="Final output JSON file path")
    args = parser.parse_args()

    merge_and_filter(args.input_dir, args.output_dir)