from openai import OpenAI
import os
import time
import json
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import argparse
import base64



def build_prompt(question, prediction):
	"""
	Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

	If the GPT-3.5 model is unable to find a match, it will output (Z).
	Also, if the original prediction does not clearly lean towards any of the options, it will output (Z).

	Parameters:
	- question: String, the question.
	- options: String, the options. E.g. ['(A)', '(B)']
	- prediction: String, the answer. E.g. '(B)'
	"""
	tmpl = (
		"You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
		"You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
		"If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
		"If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
		"Your should output one of the choices, (A),(B),(C),(D),(E) (if they are valid options), or (Z)\n"
		"Example 1: \n"
		"Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
		"Example 2: \n"
		"Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
		"Example 3: \n"
		"Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
		"Example 4: \n"
		"Question: {}  \nAnswer: {}\nYour output: "
	)
	return tmpl.format(question, prediction)



def match_multiple_choice(question, prediction):
	print("using match_multiple_choice()")
	prompt = build_prompt(question, prediction)
	retry_limit = 10

	dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
	dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
	dashscope_model = os.getenv("DASHSCOPE_MODEL")

	model = OpenAI(
		api_key=dashscope_api_key,
		base_url=dashscope_base_url,
	)

	for retry in range(retry_limit):
		try:
			response = model.chat.completions.create(
				model=dashscope_model,
				messages=[
					{"role": "user", "content": prompt},
				],
				temperature=0.0,
			)
			return response.choices[0].message.content
		except Exception as e:
			time.sleep(1)
	return '(Z) Failed to get answer'




def analyze_answer(question, gpt_answer, all_choices):
	"""
	extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

	Parameters:
	- d : data, the data containing the question and choices.
	- gpt_answer: String, the model output.
	- all_choices: List of strings, the list of all choices.

	Returns:
	- prediction, the extracted answer.
	"""
	try:
		intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
		intersect_last = list(set(all_choices).intersection(set(gpt_answer.split('\n\n')[-1].split())))
		if gpt_answer in ["A", "B", "C", "D", "E"]:
			prediction = "(" + gpt_answer + ")"
		elif gpt_answer in ['(A)', '(B)', '(C)', '(D)', '(E)']:
			prediction = gpt_answer
		elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
			choices = ['(A)', '(B)', '(C)', '(D)']
			extracted_answer = match_multiple_choice(question, gpt_answer)
			prediction = extracted_answer
		else:
			if len(intersect_last) == 1:
				intersect = intersect_last
				gpt_answer = gpt_answer.split('\n\n')[-1]
			prediction = intersect[0]
		return prediction
	except Exception as e:
		# pass
		print(e)

