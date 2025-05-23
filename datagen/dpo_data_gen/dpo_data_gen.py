import json
from tqdm import tqdm
from llm_as_a_judge import llm_as_a_judge
import os
import sys
import argparse
from pathlib import Path
import traceback
from itertools import islice
import random
import numpy as np
import re
from shutil import rmtree

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference import vts_reasoner
from src.prompts import load_model_response_prompt, load_vts_system_prompt
from src.reasoner import Reasoner


def safe_slugify(text):
    """Convert text to a safe filename by removing special characters and spaces."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text


def load_existing_results(output_dir):
    existing_ids = set()
    existing_data = []

    existing_files = [f for f in os.listdir(output_dir) if f.startswith('processed_output') and f.endswith('.json')]
    
    print("existing_files", existing_files)

    for existing_file in existing_files:
        # Note: This loads all data shards from the task result directory, so deduplication is mandatory.
        try:
            with open(os.path.join(output_dir, existing_file), 'r') as f:
                data = json.load(f)

                for item in data:
                    item_id = str(item.get('id'))
                    item_is_correct = item.get("is_correct", False)
                    if item_is_correct and item_id not in existing_ids:
                        existing_data.append(item)
                        existing_ids.update(item_id)
            with open(os.path.join(output_dir, existing_file), 'w') as f:
                json.dump(existing_data, f, indent=4)
                    
        except Exception as e:
            print(f"Warning: Failed to load answer file {existing_file}: {str(e)}")
    
   
    return existing_ids, existing_data


def create_iterator(raw_iterator, rank, world_size, limit=None):
    """Create a sliced iterator for multi-GPU processing"""
    return islice(raw_iterator, rank, limit, world_size)



class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

def process_task(task_name, args):
    # Set up paths with rank-specific filenames
    safe_task_name = safe_slugify(task_name)
    input_path = Path(args.input_dir) / safe_task_name / "merged_trace.json"
    output_dir = Path(args.output_dir) / safe_task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"processed_output_{args.rank}_{args.world_size}.json"
    error_log_file = output_dir / f"errors_{args.rank}_{args.world_size}.log"
    image_saved_dirs = output_dir / "image_saved_dirs"
    image_saved_dirs.mkdir(exist_ok=True)
    
    # Load existing results
    existing_ids, existing_data = load_existing_results(output_dir)
    answer_output = existing_data.copy()
    
    # Load dataset
    try:
        with open(input_path, "r") as f:
            dataset = json.load(f)
    except Exception as e:
        error_msg = f"Error loading dataset for task {task_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(error_log_file, 'a') as f:
            f.write(error_msg + "\n\n")
        result =  {
            "task_name": task_name,
            "status": "failed",
            "error": error_msg
        }
        try:
            with open(output_dir / f"result_info_{args.rank}_{args.world_size}.json", "w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            error_msg = f"Error saving result_info.json for task {task_name}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
        return result

    # Create rank-specific data indices
    data_indices = list(create_iterator(range(0, len(dataset)), args.rank, args.world_size))


    reasoner = Reasoner(args.reasoner_type, args.reasoner_model_path, args.reasoner_port)
    total = len(dataset)
    accuracy_count = 0
    processed_count = 0

    for index in tqdm(data_indices, desc=f"Rank {args.rank} processing {task_name}"):
        data = dataset[index]

        idx = str(data["id"])
        
        # Skip if already processed
        if idx in existing_ids:
            continue
            
        try:
            # Prepare image directory
            image_saved_path = image_saved_dirs / safe_slugify(idx)
            if image_saved_path.exists():
                rmtree(image_saved_path)
            image_saved_path.mkdir()
            
            # Process question and answers
            question = data["conversation"][0]["value"]
            if "<image>" in question:
                question = question.replace("<image>", "").strip()
            gold_answer = data["gold_answer"]
            prediction_response = data["prediction_response"]
            
            # Judge the response with error handling
            try:
                judge = llm_as_a_judge(question, gold_answer, prediction_response, args.reasoner_model_path, args.reasoner_port)
                is_correct = judge == "Correct"
            except Exception as e:
                error_msg = f"Error judging response for ID {idx}: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                with open(error_log_file, 'a') as f:
                    f.write(error_msg + "\n\n")
                is_correct = False
            
            if is_correct:
                accuracy_count += 1
                orginal_image_path = data["traces"]["images_saved_paths"][0]
                rejected_developer_prompt = f"The correct answer to this question is: {gold_answer}\nBased on this information, generate an **incorrect** reasoning path and get a **wrong** answer. Don't reveal you know the answer in any way. After reasoning, provide your answer in a single word or phrase. And wrap it with <answer></answer>"
                
                # Generate incorrect answer with retry limit
                success = False
                response, trace = None, None
                
                for attempt in range(args.max_retries):
                    try:
                        response, trace = vts_reasoner(
                            reasoner, 
                            [orginal_image_path], 
                            question, 
                            load_vts_system_prompt(), 
                            rejected_developer_prompt, 
                            str(image_saved_path)
                        )
                        print(f"Attempt {attempt + 1}: response: {response}")
                        
                        try:
                            if llm_as_a_judge(question, gold_answer, response, args.reasoner_model_path, args.reasoner_port) != "Correct":
                                success = True
                                break
                        except Exception as e:
                            print(f"Error judging generated response for ID {idx}, attempt {attempt + 1}: {str(e)}")
                            continue
                            
                    except Exception as e:
                        print(f"Error generating response for ID {idx}, attempt {attempt + 1}: {str(e)}")
                        continue
                
                if success:
                    answer_output.append({
                        "source": data["data_source"],
                        "id": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "chosen_response": prediction_response,
                        "rejected_response": response, 
                        "is_correct": True,
                        "build_preference_data": True,
                        "conversation": data["conversation"],
                        "chosen": data["traces"],
                        "rejected": trace,
                    })
                else:
                    print(f"Failed to generate incorrect answer for ID {idx} after {args.max_retries} attempts")
                    answer_output.append({
                        "source": data["data_source"],
                        "id": idx,
                        "question": question,
                        "gold_answer": gold_answer,
                        "chosen_response": prediction_response,
                        "is_correct": True,
                        "build_preference_data": False,
                        "conversation": data["conversation"],
                    })
            else:

                answer_output.append({
                    "source": data["data_source"],
                    "id": idx,
                    "gold_answer": gold_answer,
                    "prediction_response": prediction_response,
                    "is_correct": False,
                    "conversation": data["conversation"],
                })
            
            processed_count += 1
            
            # Save progress periodically
            if processed_count % 10 == 0:
                try:
                    with open(output_file, "w") as f:
                        json.dump(answer_output, f, indent=4)
                except Exception as e:
                    error_msg = f"Error saving progress for ID {idx}: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    with open(error_log_file, 'a') as f:
                        f.write(error_msg + "\n\n")
        
        except Exception as e:
            error_msg = f"Error processing data ID {idx}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open(error_log_file, 'a') as f:
                f.write(error_msg + "\n\n")
            continue
    
    # Final save
    accuracy_percentage = round(accuracy_count / total * 100, 2) if total > 0 else 0
    print(f"Rank {args.rank} completed task {task_name}. Accuracy: {accuracy_percentage}%")
    
    try:
        with open(output_file, "w") as f:
            json.dump(answer_output, f, indent=4)
    except Exception as e:
        error_msg = f"Error saving final output: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(error_log_file, 'a') as f:
            f.write(error_msg + "\n\n")
    

    result = {
        "task_name": task_name,
        "rank": args.rank,
        "total_samples": total,
        "processed_samples": processed_count,
        "correct_samples": accuracy_count,
        "accuracy": accuracy_percentage,
        "output_path": str(output_file),
        "error_log_path": str(error_log_file)
    }
    try:
        with open(output_dir / f"result_info_{args.rank}_{args.world_size}.json", "w") as f:
            json.dump(result, f, indent=4)
    except Exception as e:
        error_msg = f"Error saving result_info.json for task {task_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
    return result
    

def main(args):
    
    # Determine which tasks to process
    if args.task == "all":
        tasks = [
            'CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 
            'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 
            'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VisualWebInstruct(filtered)', 
            'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 
            'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 
            'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 
            'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 
            'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 
            'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 
            'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 
            'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 
            'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 
            'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 
            'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 
            'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 
            'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 
            'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 
            'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 
            'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 
            'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 
            'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'ureader_cap', 
            'ureader_ie', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 
            'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 
            'websight(cauldron)'
        ]
    else:
        tasks = [args.task]
    
    # Process each task
    for task_name in tasks:
        print(f"Rank {args.rank} processing task {task_name}-------------------------------------------")
        result = process_task(task_name, args)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Rank: {result.get('rank', 'N/A')}")
        print(f"Task: {result['task_name']}")
        print(f"Total samples: {result['total_samples']}")
        print(f"Processed samples: {result.get('processed_samples', 'N/A')}")
        print(f"Correct samples: {result['correct_samples']}")
        print(f"Accuracy: {result['accuracy']}%")
        print(f"Output saved to: {result['output_path']}")
        if 'error_log_path' in result:
            print(f"Error log at: {result['error_log_path']}")
        if 'error' in result:
            print(f"Error occurred: {result['error']}")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process visual task datasets for DPO data generation.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to process or 'all1-4'")
    parser.add_argument("--input_dir", type=str, help="Base directory containing task folders")
    parser.add_argument("--output_dir", type=str, help="Base directory to save processed outputs")
    parser.add_argument("--reasoner_type", type=str, default="qwen-vl", help="Type of reasoner to use")
    parser.add_argument("--reasoner_model_path", type=str, default="none", help="Path to reasoner model")
    parser.add_argument("--reasoner_port", type=int, default=28080, help="Port for Qwen-VL service")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum attempts to generate incorrect answer")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes")
    
    args = parser.parse_args()

    main(args)
