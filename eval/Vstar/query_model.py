from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import os
import time


SYSTEM_PROMPT = """
### Strict Output Format
You MUST respond with ONLY correct options for the question. 

### Rules:
1. DO NOT include any reasoning, explanations, or intermediate steps.
2. DO NOT use any formatting (no JSON, markdown, quotes, etc.).
3. DO NOT add any text outside the provided options.
4. ONLY output the single chosen option exactly as provided.

### Example 1:
Question: What is the capital of France?
Options:
(A) Paris
(B) London
(C) Berlin
(D) Rome
Correct Output: (A)

### Penalty:
If you include ANY extra text or deviate from the format, the response will be rejected.
"""


def encode_image(image: Image.Image) -> str:
	output_buffer = BytesIO()
	image.save(output_buffer, format="jpeg")
	byte_data = output_buffer.getvalue()
	base64_str = base64.b64encode(byte_data).decode("utf-8")
	return base64_str




def query_gpt4o(images, question, retry=5, timeout=1):
	openai_base_url = os.environ.get("OPENAI_BASE_URL")
	openai_api_key = os.environ.get("OPENAI_API_KEY")
	model = OpenAI(
		api_key=openai_api_key,
		base_url=openai_base_url
	)
	messages = [
		{
			"role": "system",
			"content" : SYSTEM_PROMPT,
		},
		{
			"role": "user",
			"content": [],
		}
	]
	for image in images:
		base64_image_str = encode_image(image)
		messages[-1]["content"].append({
			"type": "image_url",
			"image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}
		})
		
	messages[-1]["content"].append({
		"type": "text",
		"text": question
	})
	
	for i in range(retry):
		try:
			response = model.chat.completions.create(
				model="gpt-4o",
				messages=messages,
				temperature=0.0,
			)
			return response.choices[0].message.content, response.usage.model_dump_json()
		except Exception as e:
			print(f"Error when geting complement in {i}th tries: {e}")
			time.sleep(timeout)
	return 'No answer was reached.', None
		


def query_qwenvl(images, question, qwen_model_path, qwen_port, retry=5, timeout=1):
	os.environ["http_proxy"] = ""
	os.environ["https_proxy"] = ""
	qwen_openai_api_key = "EMPTY"
	qwen_openai_api_base = f"http://localhost:{qwen_port}/v1"

	qwen_client = OpenAI(
		api_key=qwen_openai_api_key,
		base_url=qwen_openai_api_base,
	)
	messages = [
		{
			"role": "system",
			"content" : SYSTEM_PROMPT,
		},
		{
			"role": "user",
			"content": [],
		}
	]
	for image in images:
		base64_image_str = encode_image(image)
		messages[-1]["content"].append({
			"type": "image_url",
			"image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}
		})
		
	messages[-1]["content"].append({
		"type": "text",
		"text": question
	})
	for i in range(retry):
		try:
			response = qwen_client.chat.completions.create(
				model=qwen_model_path,
				messages=messages,
				temperature=0.0,
			)
			return response.choices[0].message.content, response.usage.model_dump_json()
		except Exception as e:
			print(f"Error when geting complement in {i}th tries: {e}")
			time.sleep(timeout)
	return 'No answer was reached.', None

