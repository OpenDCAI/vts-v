import argparse
from shutil import rmtree
import json
from datasets import load_dataset
from tqdm import tqdm
import traceback
import os
import sys
import re
import signal
from functools import wraps
import time
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference import vts_reasoner, vts_reasoner_verifier
from src.reasoner import Reasoner
from src.verifier import Verifier
from src.prompts import load_vts_system_prompt, load_vts_has_verifier_system_prompt


from query_model import query_gpt4o, query_qwenvl
from analyze_utils import analyze_answer

# Timeout mechanism
class TimeoutError(Exception):
    pass

def timeout(seconds=300, error_message="Function call timed out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(error_message)
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator

@timeout(300)  # 5 minutes timeout
def run_reasoner_with_timeout(args, reasoner, verifier, images, prompt, system_prompt, developer_prompt, image_save_dir):
    if not args.using_vts:
        if args.reasoner_type == "gpt-4o":
            return query_gpt4o(images, prompt)
        elif args.reasoner_type == "qwen-vl":
            return query_qwenvl(images, prompt, args.reasoner_model_path, args.reasoner_port)
        else:
            print("ERROR: don't have this model")
            return None, None
    else:
        if not args.using_verifier:
            return vts_reasoner(
                reasoner,
                images,
                prompt,
                system_prompt,
                developer_prompt,
                image_save_dir
            )
        else:
            return vts_reasoner_verifier(
                reasoner,
                verifier,
                images,
                prompt,
                system_prompt,
                developer_prompt,
                image_save_dir
            )

MMStar_data_path = "/fs-computility/llmit_d/shared/baitianyi/datasets_local/MMStar"

def load_existing_results(result_save_dir):
    """Load existing results if they exist"""
    answers_path = os.path.join(result_save_dir, "answer.json")
    traces_path = os.path.join(result_save_dir, "traces.json")
    
    if os.path.exists(answers_path) and os.path.exists(traces_path):
        with open(answers_path, 'r') as f:
            answers_output = json.load(f)
        with open(traces_path, 'r') as f:
            all_traces = json.load(f)
        return answers_output, all_traces
    return {'test': []}, []

def test_benchmark(args):
    result_save_dir = args.output_dir
    os.makedirs(result_save_dir, exist_ok=True)
    
    # Load existing results if they exist
    answers_output, all_traces = load_existing_results(result_save_dir)
    
    # Get set of already processed question IDs
    processed_ids = {item['idx'] for item in answers_output['test']}
    
    image_save_dir_parent = os.path.join(result_save_dir, "images")
    error_log_path = os.path.join(result_save_dir, "errors_mmstar.log")

    # Initialize models only if we have new items to process
    reasoner = None
    verifier = None
    system_prompt = None
    
    # Load dataset
    dataset = load_dataset(MMStar_data_path)['val']
    # dataset = dataset.select(range(5))
    
    if len(processed_ids) < len(dataset):
        reasoner = Reasoner(args.reasoner_type, args.reasoner_model_path, args.reasoner_port)
        if args.using_verifier:
            verifier = Verifier(args.dpo_model_type, args.dpo_model_name_or_path, 
                               args.ref_model_type, args.ref_model_name_or_path,
                               args.verifier_threshold_type, args.verifier_threshold)
        system_prompt = load_vts_system_prompt() if not args.using_verifier else load_vts_has_verifier_system_prompt()

    processed_count = len(processed_ids)
    skipped_count = 0
    new_processed_count = 0
    accuracy = {'test': sum(1 for item in answers_output['test'] if item['gold_answer'] == item['prediction'])}

    all_choices = ['(A)', '(B)', '(C)', '(D)']
    developer_prompt = "In the terminate_action, you should only include the selected choices (A), (B), (C) or (D) in the final response."

    for data in tqdm(dataset, desc="Processing MMStar dataset"):
        idx = data["index"]
        
        # Skip if already processed
        if idx in processed_ids:
            continue
            
        try:
            question = data["question"]
            image = data["image"]
            gold_answer = data["answer"]
            image_save_dir = os.path.join(image_save_dir_parent, f"{idx}")

            start_time = time.time()
            try:
                response, trace = run_reasoner_with_timeout(
                    args,
                    reasoner,
                    verifier,
                    [image],
                    question,
                    system_prompt,
                    developer_prompt,
                    image_save_dir
                )
            except Exception as e:
                skipped_count += 1
                error_type = "TimeoutError" if isinstance(e, TimeoutError) else type(e).__name__
                error_msg = f"Error processing data id {idx} ({error_type}): {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                with open(error_log_path, 'a') as f:
                    f.write(error_msg + "\n\n")
                continue
            
            processing_time = time.time() - start_time
            print(f"\nProcessed data id {idx} in {processing_time:.2f} seconds")
            
            prediction = analyze_answer(question, response, all_choices)
            if f"({gold_answer})" == prediction:
                accuracy['test'] += 1
            
            all_traces.append({
                "idx": idx,
                "question": question,
                "gold_answer": f"({gold_answer})", 
                "full_response": response,
                "prediction": prediction,
                "traces": trace
            })
            
            answers_output['test'].append({
                'idx': idx, 
                'question': question,
                'gold_answer': f"({gold_answer})", 
                'full_response': response,
                'prediction': prediction,
            })

            new_processed_count += 1
            if new_processed_count % 10 == 0:
                # Save intermediate results
                json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
                json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)

        except Exception as e:
            skipped_count += 1
            error_msg = f"Error processing data id {idx}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open(error_log_path, 'a') as f:
                f.write(error_msg + "\n\n")

    # Save final results
    json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
    json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)
    
    # Calculate and report accuracy
    print('-'*50)
    print(f"Total items in dataset: {len(dataset)}")
    print(f"Previously processed items: {processed_count}")
    print(f"Newly processed items: {new_processed_count}")
    print(f'MMStar test accuracy: {round(accuracy["test"]/len(answers_output["test"])*100, 2)}%')
    print(f'Total skipped items: {skipped_count} (including timeouts and other errors)')
    
    accuracy_json = [{
        "accuracy count": accuracy['test'],
        "len of dataset": len(answers_output['test']),
        "test accuracy": f"{round(accuracy['test']/len(answers_output['test'])*100, 2)}%",
        "skipped count": skipped_count,
        "error log": error_log_path,
        "previously processed": processed_count,
        "newly processed": new_processed_count
    }]
    json.dump(accuracy_json, open(os.path.join(result_save_dir, "accuracy.json"), 'w'), indent=4)
    return answers_output

def main(args):
    test_benchmark(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="MMStar")
    parser.add_argument("--using_vts", type=str, default="False")
    parser.add_argument("--reasoner_type", type=str, default="gpt-4o")
    parser.add_argument("--reasoner_model_path", type=str, default="None")
    parser.add_argument("--reasoner_port", type=str, default="None")
    parser.add_argument("--using_verifier", type=str, default="False")
    parser.add_argument("--dpo_model_type", type=str, default="None")
    parser.add_argument("--dpo_model_name_or_path", type=str, default="None")
    parser.add_argument("--ref_model_type", type=str, default="None")
    parser.add_argument("--ref_model_name_or_path", type=str, default="None")
    parser.add_argument("--verifier_threshold_type", type=str, default="logits_score")
    parser.add_argument("--verifier_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./output_dirs")
    
    args = parser.parse_args()
    args.using_vts = args.using_vts.lower() in ("true", "1", "yes", "y")
    args.using_verifier = args.using_verifier.lower() in ("true", "1", "yes", "y")

    main(args)