

def build_prompt(question, gold_answer, prediction):
	prompt_template = """
	### Task Description
	You need to evaluate whether a model's prediction matches the given correct answer. The prediction should be considered correct if it conveys the same core meaning as the answer, even if the wording differs.
	You must respond ONLY with either "Correct" or "Incorrect" - no other words, explanations, or formatting.

	### Instructions
	1. Carefully read the original question to understand what is being asked
	2. Examine both the reference answer and model prediction
	3. Determine if the prediction:
	- Correctly answers the question (even if wording differs)
	- Contains the same key information as the reference answer
	- Doesn't introduce incorrect or irrelevant information

	### Input Format
	**Original Question:** [The exact question being asked]
	**Reference Answer:** [The correct answer to the question]
	**Model Prediction:** [The model's generated response]

	### Evaluation Rules
	- Answer "Correct" only if:
	- The prediction fully answers the original question
	- AND the core meaning matches the reference answer
	- AND there are no factual errors
	- Answer "Incorrect" if:
	- Any part of the prediction contradicts the reference answer
	- OR the prediction fails to answer the question
	- OR the prediction adds incorrect information

	### Required Output Format
	Correct/Incorrect
	
	### Evaluation Example
	**Original Question:** How many planets are in our solar system?
	**Reference Answer:** There are 8 planets.
	**Model Prediction:** The solar system contains 8 major planets.
	**Judgment:** Correct

	### Actual Evaluation
	**Original Question:** {}
	**Reference Answer:** {}
	**Model Prediction:** {}

	**Judgment:**
	"""

	return prompt_template.format(question, gold_answer, prediction)




import os
import time
from openai import OpenAI


def qwen_llm_as_a_judge(question, gold_answer, prediction_response,retry=5, timeout=1):
	dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
	dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
	dashscope_model = os.getenv("DASHSCOPE_MODEL")
	model = OpenAI(
		api_key=dashscope_api_key,
		base_url=dashscope_base_url,
	)

	messages = [
		{
			"role": "user",
			"content": build_prompt(question, gold_answer, prediction_response)
		}
	]
	for i in range(retry):
		try:
			response = model.chat.completions.create(
				model=dashscope_model,
				messages=messages,
				temperature=0.0,
			)
			return response.choices[0].message.content
		except Exception as e:
			print(f"Error when geting complement in {i}th tries: {e}")
			time.sleep(timeout)
	return 'Failed Judge'
		

