import argparse
from shutil import rmtree
import json
from datasets import load_dataset
from tqdm import tqdm
import traceback
import os
import sys
import re
from PIL import Image


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference import vts_reasoner, vts_reasoner_verifier
from src.reasoner import Reasoner
from src.verifier import Verifier
from src.prompts import load_vts_system_prompt, load_vts_has_verifier_system_prompt


from query_model import query_gpt4o, query_qwenvl
from analyze_utils import qwen_llm_as_a_judge


# you can also set your own local path 
mathvista_path = "AI4Math/MathVista"


def test_benchmark(args):
    result_save_dir = args.output_dir
    if os.path.exists(result_save_dir):
        rmtree(result_save_dir)
    os.makedirs(result_save_dir, exist_ok=True)
    image_save_dir_parent = os.path.join(result_save_dir, "images")
    answers_output = {'test':[]}
    all_traces = []

    dataset = load_dataset(mathvista_path)[args.task_name]
    # dataset = dataset.select(range(5))

    reasoner = Reasoner(args.reasoner_type, args.reasoner_model_path, args.reasoner_port)

    verifier = None
    if args.using_verifier:
        verifier = Verifier(args.dpo_model_type, args.dpo_model_name_or_path, args.ref_model_type, args.ref_model_name_or_path, args.verifier_threshold_type, args.verifier_threshold)

    processed_count = 0
    correct_count = 0
    for data in tqdm(dataset):
        idx = data["pid"]
        image = data["decoded_image"].convert("RGB")
        question = data["question"]
        hint_and_question = data["query"]
        gold_answer = data["answer"]
        image_save_dir = os.path.join(image_save_dir_parent, f"{idx}")
        
        try:
            if not args.using_vts:
                if args.reasoner_type == "gpt-4o":
                    response, trace = query_gpt4o(
                        [image],
                        hint_and_question
                    )
                elif args.reasoner_type == "qwen-vl":
                    response, trace = query_qwenvl(
                        [image],
                        hint_and_question,
                        args.reasoner_model_path,
                        args.reasoner_port
                    )
                else:
                    print("ERROR: don't have this model")
            else:
                if not args.using_verifier:
                    response, trace = vts_reasoner(
                        reasoner,
                        [image],
                        hint_and_question,
                        load_vts_system_prompt(),
                        "",
                        image_save_dir
                    )
                else:
                    response, trace = vts_reasoner_verifier(
                        reasoner,
                        verifier,
                        [image],
                        hint_and_question,
                        load_vts_has_verifier_system_prompt(),
                        "",
                        image_save_dir
                    )

            judge = qwen_llm_as_a_judge(question, gold_answer, response)
            is_correct = judge == "Correct"
            if is_correct:
                correct_count += 1

            all_traces.append({
                "idx": idx,
                "question": question,
                "answer": gold_answer, 
                "full_response": response,
                "llm_as_a_judge_result": judge,
                "is_correct": is_correct, 
                "traces": trace
            })
            
            answers_output['test'].append({
                "idx": idx, 
                "question": question,
                "gold_answer": gold_answer, 
                "full_response": response,
                "llm_as_a_judge_result": judge,
                "is_correct": is_correct, 
            })

            processed_count += 1
            if processed_count % 10:
                json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
                json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)
        except Exception as e:
            error_msg = f"Error processing data id {idx}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            error_log_path = os.path.join(result_save_dir, f"errors_vstar.log")
            with open(error_log_path, 'a') as f:
                f.write(error_msg + "\n\n")

    json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
    json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)
    
    print('-'*50)
    print(f'Mathvista {args.task_name} accuracy: {round(correct_count/len(answers_output["test"])*100, 2)}%')
    accuracy_json = []
    accuracy_json.append({
        "accuracy count": correct_count,
        "len of dataset": len(answers_output['test']),
        "test accuracy": f"{round(correct_count/len(answers_output['test'])*100, 2)}%"
    })
    json.dump(accuracy_json, open(os.path.join(result_save_dir, "accuracy.json"), 'w'), indent=4)
    return answers_output

def main(args):
    test_benchmark(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="testmini")
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