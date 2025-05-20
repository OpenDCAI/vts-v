import argparse
from shutil import rmtree
import json
from datasets import load_dataset
from tqdm import tqdm
import traceback
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference import vts_reasoner, vts_reasoner_verifier
from src.reasoner import Reasoner
from src.verifier import Verifier
from src.prompts import load_vts_system_prompt, load_vts_has_verifier_system_prompt


from query_model import query_gpt4o, query_qwenvl


from analyze_utils import analyze_answer

DATASET_NAME = "/fs-computility/llmit_d/shared/baitianyi/datasets_local/BLINK"

def test_benchmark(task_name, args):
	result_save_dir = f"{args.output_dir}/{task_name}"
	if os.path.exists(result_save_dir):
		rmtree(result_save_dir)
	os.makedirs(result_save_dir, exist_ok=True)
	image_save_dir_parent = os.path.join(result_save_dir, "images")
	answers_output = {'val':[], 'test':[]}
	all_traces = []

	reasoner = Reasoner(args.reasoner_type, args.reasoner_model_path, args.reasoner_port)

	verifier = None
	if args.using_verifier:
		verifier = Verifier(args.dpo_model_type, args.dpo_model_name_or_path, args.ref_model_type, args.ref_model_name_or_path, args.verifier_threshold_type, args.verifier_threshold)

	processed_count = 0
	for split in ['val']:
		test_data = load_dataset(DATASET_NAME, task_name)[split]
		# test_data = test_data.select(range(5))
		for data in tqdm(test_data):
			idx = data['idx']
			try:
				gold_answer = data['answer']
				all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(data['choices'])]
				task_prompt = data['prompt']
				images = []
				for k in ['image_1', 'image_2', 'image_3', 'image_4']:
					if data[k] is not None:
						images.append(data[k])
				developer_prompt = f"In the terminate_action, you should only include the selected choices (A, B, C, or D) in the final response."
				image_save_dir = os.path.join(image_save_dir_parent, f"{idx}")
				
				if not args.using_vts:
					if args.reasoner_type == "gpt-4o":
						response, trace = query_gpt4o(
							images,
							task_prompt
						)
					elif args.reasoner_type == "qwen-vl":
						response, trace = query_qwenvl(
							images,
							task_prompt,
							args.reasoner_model_path,
							args.reasoner_port
						)
					else:
						print("ERROR: don't have this model")
				else:
					if not args.using_verifier :
						response, trace = vts_reasoner(
							reasoner,
							images,
							task_prompt,
							load_vts_system_prompt(),
							developer_prompt,
							image_save_dir
						)
					else:
						response, trace = vts_reasoner_verifier(
							reasoner,
							verifier,
							images,
							task_prompt,
							load_vts_has_verifier_system_prompt(),
							developer_prompt,
							image_save_dir
						)
				all_traces.append({
					"idx": idx, 
					"answer": gold_answer, 
					"full_prediction": response, 
					"traces": trace
				})
				prediction = analyze_answer(data, response, all_choices)
				answers_output[split].append({
					'idx': idx, 
					'answer': gold_answer, 
					'full_prediction':response , 
					'prediction': prediction
				})

				processed_count += 1
				if processed_count % 10 :
					json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
					json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)
			except Exception as e:
				error_msg = f"Error processing data id {idx}: {str(e)}\n{traceback.format_exc()}"
				print(error_msg)
				
				error_log_path = os.path.join(result_save_dir, f"errors_{task_name}.log")
				with open(error_log_path, 'a') as f:
					f.write(error_msg + "\n\n")


	json.dump(answers_output, open(os.path.join(result_save_dir, "answer.json"), 'w'), indent=4)
	json.dump(all_traces, open(os.path.join(result_save_dir, "traces.json"), 'w'), indent=4)
	accuracy = {'val': 0, 'test': 0}
	for split in ['val', 'test']:
		for d in answers_output[split]:
			if d['answer'] == d['prediction']:
				accuracy[split] += 1
	print('-'*50)
	print(f'Task {task_name} Performance')
	for split in ['val']:
		print(f'{split} accuracy: {round(accuracy[split]/len(answers_output[split])*100, 2)}%')
	accuracy_json = []
	accuracy_json.append({
		"accuracy count": accuracy['val'],
		"len of dataset" : len(answers_output['val']),
		"val accuracy": f"{round(accuracy['val']/len(answers_output['val'])*100, 2)}%"
	})
	json.dump(accuracy_json, open(os.path.join(result_save_dir, "accuracy.json"), 'w'), indent=4)
	return answers_output






def main(args):
	if args.task_name == "all":
		subtasks = ['Visual_Similarity', 'Counting', 'Relative_Depth', 'Jigsaw', 'Art_Style', 'Functional_Correspondence', 'Semantic_Correspondence', 'Spatial_Relation', 'Object_Localization', 'Visual_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Forensic_Detection', 'IQ_Test']
	else:
		subtasks = [args.task_name]
	for task_name in subtasks:
		test_benchmark(task_name, args)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--task_name", type=str, default="Counting")
	parser.add_argument("--using_vts", type=str, default=False)
	parser.add_argument("--reasoner_type", type=str, default="gpt-4o")
	parser.add_argument("--reasoner_model_path", type=str, default="None")
	parser.add_argument("--reasoner_port", type=str, default="None")
	parser.add_argument("--using_verifier", type=str, default=False)
	parser.add_argument("--dpo_model_type", type=str, default="None")
	parser.add_argument("--dpo_model_name_or_path", type=str, default="None")
	parser.add_argument("--ref_model_type", type=str, default="None")
	parser.add_argument("--ref_model_name_or_path", type=str, default="None")
	parser.add_argument("--verifier_threshold_type", type=str, default="logits")
	parser.add_argument("--verifier_threshold", type=float, default=600)
	parser.add_argument("--output_dir", type=str, default="./output_dirs/default_output_dir")
	
	args = parser.parse_args()
	args.using_vts = args.using_vts.lower() in ("true", "1", "yes", "y")
	args.using_verifier = args.using_verifier.lower() in ("true", "1", "yes", "y")

	main(args)

