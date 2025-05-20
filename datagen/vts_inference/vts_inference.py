import json
import os
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
import argparse
from shutil import rmtree
import traceback
import random
import numpy as np
import re


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference import vts_reasoner
from src.prompts import load_model_response_prompt, load_vts_system_prompt
from src.reasoner import Reasoner




DATASETS_NAME = "/fs-computility/llmit_d/shared/baitianyi/datasets_local/llavaov"


# Set the random seed for sampling to ensure consistent data retrieval across each sampling iteration.
random.seed(42)
np.random.seed(42)

def get_sampled_indices(dataset_size, sample_ratio=0.3):
    """获取随机采样的索引，保证可重现"""
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    sample_size = int(dataset_size * sample_ratio)
    return sorted(all_indices[:sample_size])  # 排序以便于调试


def safe_slugify(text):
    """Convert text to a safe filename by removing special characters and spaces."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text



def create_iterator(raw_iterator, rank, world_size, limit=None):
    """Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


def load_existing_results(result_dir):
    existing_ids = set()
    all_traces = []
    
    trace_files = [f for f in os.listdir(result_dir) if f.startswith('trace_') and f.endswith('.json')]

    for trace_file in trace_files:
        try:
            with open(os.path.join(result_dir, trace_file), 'r') as f:
                data = json.load(f)
                existing_ids.update(item['id'] for item in data)
                all_traces.extend(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load trace file {trace_file}: {str(e)}")
    
    return existing_ids, all_traces


class NoAnswerReachedError(Exception):
    def __init__(self, message, question_id=None):
        self.question_id = question_id
        self.error_code = 4001
        super().__init__(message)


def process_task(task_name, args, rank, world_size):

    reasoner = Reasoner(args.reasoner_type, args.reasoner_model_path, args.reasoner_port)
    
    task_name_safe = safe_slugify(task_name)
    result_save_dir = f"{args.output_dir}/{task_name_safe}"
    os.makedirs(result_save_dir, exist_ok=True)
    
    image_save_dir = os.path.join(result_save_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)
    
    trace_path = os.path.join(result_save_dir, f"trace_{rank}_{world_size}.json")
    
    existing_ids, all_traces = load_existing_results(result_save_dir)
    
    dataset = load_dataset(DATASETS_NAME, task_name)['train']
    
    dataset_size = len(dataset)
    sampled_indices = get_sampled_indices(dataset_size, args.sample_ratio)

    data_indices = list(create_iterator(sampled_indices, rank, world_size))
    data_indices = data_indices[0:10]

    processed_count = 0
    for idx in tqdm(data_indices, desc=f"Rank {rank}"):
        data = dataset[idx]
        id = str(data["id"])
        
        if id in existing_ids:
            continue
        
        try:
            image = data["image"].convert("RGB")
            conversation = data["conversations"]
            data_source = data["data_source"]
            
            image_save_path = os.path.join(image_save_dir, f"{safe_slugify(id)}")

            if os.path.exists(image_save_path):
                rmtree(image_save_path)
            os.makedirs(image_save_path, exist_ok=True)
            
            gold_answer = conversation[1]["value"]
            question = conversation[0]["value"]
            
            response, trace = vts_reasoner(
                reasoner,
                [image],
                question,
                load_vts_system_prompt(),
                "",
                image_save_path
            )
            if response == "No answer was reached.":
                raise NoAnswerReachedError(
                    f"Reasoning failed for question: {question}",
                    question_id=id
                )
            all_traces.append({
                "id": id,
                "data_source": data_source, 
                "gold_answer": gold_answer,
                "prediction_response": response,
                "conversation": conversation,
                "traces": trace
            })
            
            processed_count += 1
            if processed_count % 10 == 0:
                with open(trace_path, 'w') as f:
                    json.dump(all_traces, f, indent=4)
        
        except Exception as e:
            error_msg = f"Error processing data id {id}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            error_log_path = os.path.join(result_save_dir, f"errors_{rank}_{world_size}.log")
            with open(error_log_path, 'a') as f:
                f.write(error_msg + "\n\n")
    
    with open(trace_path, 'w') as f:
        json.dump(all_traces, f, indent=4)

def main(args):
    if args.task_name == "all":
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
        tasks = [args.task_name]
    
    for task_name in tasks:
        process_task(task_name, args, args.rank, args.world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="None")
    parser.add_argument("--using_vts", type=str, default="True")
    parser.add_argument("--reasoner_type", type=str, default="qwen-vl")
    parser.add_argument("--reasoner_port", type=str, default="28080")
    parser.add_argument("--reasoner_model_path", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./output_dirs")
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    args = parser.parse_args()
    args.using_vts = args.using_vts.lower() in ("true", "1", "yes", "t")
    
    main(args)