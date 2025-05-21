import json
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import argparse
import os

def count_tokens(messages, tokenizer):
    """Count tokens in messages"""
    full_text = ""
    for msg in messages:
        if isinstance(msg["content"], str):
            full_text += msg["content"]
        elif isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    full_text += item["text"]
    return len(tokenizer.encode(full_text))

def process_example(example, tokenizer, threshold):
    """Process a single example and return token counts and classification"""
    try:
        prompt_tokens = count_tokens(example["prompt"], tokenizer)
        chosen_tokens = count_tokens(example["chosen"], tokenizer)
        rejected_tokens = count_tokens(example["rejected"], tokenizer)
        total_tokens = prompt_tokens + max(chosen_tokens, rejected_tokens)
        
        return {
            "example": example,
            "total_tokens": total_tokens,
            "below_threshold": total_tokens < threshold
        }
    except Exception as e:
        print(f"Error processing example: {e}")
        return {
            "example": example,
            "total_tokens": float('inf'),  # Mark as above threshold if error occurs
            "below_threshold": False
        }

def process_chunk(args):
    """Process a chunk of data"""
    chunk, tokenizer, threshold = args
    chunk_results = []
    for example in chunk:
        chunk_results.append(process_example(example, tokenizer, threshold))
    return chunk_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Filter dataset by token count using Qwen2.5-VL tokenizer")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",help="Path to the Qwen2.5-VL model directory")
    parser.add_argument("--input_file", type=str, required=True,help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, required=True,help="Directory to save filtered results")
    parser.add_argument("--threshold", type=int, default=10000,help="Token count threshold (default: 10000)")
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load data
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine optimal number of processes
    num_processes = max(1, cpu_count() - 1)
    chunk_size = math.ceil(len(data) / num_processes)
    print(f"Processing {len(data)} examples with {num_processes} processes (chunk size: {chunk_size})")
    
    # Split data into chunks and prepare arguments for each process
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    process_args = [(chunk, tokenizer, args.threshold) for chunk in chunks]
    
    # Process in parallel with progress bar
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, process_args),
            total=len(chunks),
            desc="Processing examples"
        ))
    
    # Combine and separate results
    less_than_results = []
    more_than_results = []
    
    for chunk in results:
        for result in chunk:
            if result["below_threshold"]:
                less_than_results.append(result["example"])
            else:
                more_than_results.append(result["example"])
    
    # Extract base filename without extension
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Save results
    less_than_output_path = os.path.join(
        args.output_dir,
        f"{input_basename}_less_than_{args.threshold}_{len(less_than_results)}.json"
    )
    more_than_output_path = os.path.join(
        args.output_dir,
        f"{input_basename}_more_than_{args.threshold}_{len(more_than_results)}.json"
    )
    
    print(f"Saving {len(less_than_results)} examples below threshold to {less_than_output_path}")
    with open(less_than_output_path, "w") as f:
        json.dump(less_than_results, f, indent=4)
    
    print(f"Saving {len(more_than_results)} examples above threshold to {more_than_output_path}")
    with open(more_than_output_path, "w") as f:
        json.dump(more_than_results, f, indent=4)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()