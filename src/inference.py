import torch.nn.functional as F
from copy import deepcopy
from PIL import Image
import json
import os
from shutil import rmtree
import torch
import re


from .tools import (
	encode_image,
	get_depth, 
	zoom_in_image_by_bbox, 
	visual_search,  
	grounded_segmentation, 
	crop_image_action,
	segment_image,
	ocr_extract_text,
	overlay,
	calculate_text_to_images_similarity,
	calculate_image_to_texts_similarity,
	calculate_image_to_images_similarity,
	AllStep,
	TerminateStep,
	ActionStep,	 
	TERMINATE,
	GROUNDING,
	DEPTH,
	ZOOMIN,
	VISUALSEARCH,
	TEXT,
	OVERLAY,
	CROP,
	SEGMENT,
	OCR,
	IMAGE_TO_IMAGES_SIMILARITY,
	TEXT_TO_IMAGES_SIMILARITY,
	IMAGE_TO_TEXTS_SIMILARITY
)



from .prompts import (
	load_model_response_prompt,
	load_verifier_system_prompt,
)




def vts_reasoner(reasoner, image_path_list, task_prompt, system_prompt="", developer_prompt="", image_save_dir=None):
	if len(image_path_list) > 0 and isinstance(image_path_list[0], Image.Image):
		images = image_path_list
	else:
		images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]

	message_list = [
		{
			"role": "system",
			"content": system_prompt if len(system_prompt) else "You are a helpful assistant"
		}
	]
	if len(developer_prompt):
		message_list[-1]['content'] += "\n" + developer_prompt
	
	message_list[-1]["content"] += "\n" + load_model_response_prompt()

	message_list.append(
		{
			"role": "user",
			"content": []
		}
	)
	
	for i, image in enumerate(images):
		base64_image_str = encode_image(image)
		message_list[-1]["content"].append(
			{
				"type": "text",
				"text": f"Image Index: {i+1}\n"
			}
		)
		message_list[-1]["content"].append(
			{
				"type": "image_url",
				"image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}
			}
		)
	
	message_list[-1]["content"].append(
		{
			"type": "text",
			"text": task_prompt
		}
	)

	response = reasoner.get_completion(message_list, "all")


	
	call_stack = []
	completion_stack = []

	while True:
		content = []
		for retry in range(5):
			try:
				completion_message = response.choices[0].message.content
				print(f"\ncompletion_message: {completion_message}")
				json_str = re.search(r'\{.*\}', completion_message, re.DOTALL).group().strip()
				formatted_content = json.loads(json_str)
				action_name = formatted_content["action_name"]
				action_args = formatted_content["action"]
				break
			
			except Exception as e:
				print(f"{retry}th retry error in outer inference loop. Error: {e}")
				if retry == 4:
					response = ""
				else:
					response = reasoner.get_completion(message_list, "all")

		if response == "":
			print("Error during inference. End.")
			break
		
		message_list.append(
			json.loads(response.choices[0].message.model_dump_json())
		)
		completion_stack.append(response.usage.model_dump_json())
		call_stack.extend([{"name": action_name, "args": action_args}])
		
		if action_name == TERMINATE:
			break
		else:
			return_images = []
			if action_name == GROUNDING:
				return_images, boxes, labels = grounded_segmentation(
					image=images[action_args["image_index"]-1],
					labels=action_args["text"],
					polygon_refinement=True
				)
				if len(boxes):
					box_str = "\n".join([f"Label: {label}, Box: {box}" for label, box in zip(labels, boxes)])
					content.extend(
						[
							{
								"type": "text",
								"text": f"The bounding box coordinates are: {box_str}\n"  
							}
						]
					)
				for i, image in enumerate(return_images):
					images.append(image)
					if i == 0:
						content.extend(
							[ 
								{
									"type": "text",
									"text": f"Here's the image marked with masks and boxes: Image Index: {len(images)+1}\n"
								},
								{
									"type": "image_url",
									"image_url": {
										"url": f"data:image/jpeg;base64,{encode_image(image)}"
									}
								}
							]
						)
					elif i == 1:
						content.extend(
							[
								
								{
									"type": "text",
									"text": f"Here's the original image: Image Index: {len(images)}\n"
								},
								{
									"type": "image_url",
									"image_url": {
										"url": f"data:image/jpeg;base64,{encode_image(image)}"
									}
								}
							]
						)		
			
			elif action_name == DEPTH:
				return_images = get_depth(
					image=images[action_args["image_index"]-1]
				)
				for depth in return_images:
					images.append(depth)
					content.extend(
						[
							{
								"type": "text",
								"text": f"Here's the depth map: "  
							},
							{
								"type": "text",
								"text": f"Image Index: {len(images)}\n"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{encode_image(depth)}"
								}
							}
						]
					)
			elif action_name == ZOOMIN:
				return_images = zoom_in_image_by_bbox(
					image=images[action_args["image_index"]-1],
					bounding_box=action_args["bounding_box"]
				)
				for crop in return_images:
					images.append(crop)
					content.extend(
						[
							{
								"type": "text",
								"text": f"Here's the zoomed in image: "  
							},
							{
								"type": "text",
								"text": f"Image Index: {len(images)}\n"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{encode_image(crop)}"
								}
							}
						]
					)
					
			
			elif action_name == VISUALSEARCH:
				return_images, boxes_list, labels_list = visual_search(
					image=images[action_args["image_index"]-1],
					objects=action_args["objects"]
				)
				
				if len(boxes_list):
					for i, patch in enumerate(return_images):
						
						boxes = boxes_list[i]
						labels = labels_list[i]
						
						box_str = "\n".join([f"Label: {label}, Box: {box}" for label, box in zip(labels, boxes)])
						images.append(patch)
						content.extend(
							[
								{
									"type": "text",
									"text": f"The bounding box coordinates are: {box_str}"  
								},
								{
									"type": "text",
									"text": f"Image Index: {len(images)}\nHere's the image marked with masks and boxes.\nPay attention.\n"
								},
								{
									"type": "image_url",
									"image_url": {
										"url": f"data:image/jpeg;base64,{encode_image(patch)}"
									}
								}
							]
						)
			elif action_name == CROP:
				return_images = crop_image_action(
					image=images[action_args["image_index"]-1],
					bounding_box=action_args["bounding_box"],
					padding=action_args.get("padding", 0.05)
				)
				for crop in return_images:
					images.append(crop)
					content.extend([
						{
							"type": "text",
							"text": f"Here's the cropped image: Image Index: {len(images)}\n"
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(crop)}"
							}
						}
					])
			elif action_name == SEGMENT:
				return_images = segment_image(
					image=images[action_args["image_index"]-1],
					bounding_boxes=action_args["bounding_boxes"]
				)
				for segmented in return_images:
					images.append(segmented)
					content.extend([
						{
							"type": "text",
							"text": f"Here's the segmented image: Image Index: {len(images)}\n"
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(segmented)}"
							}
						}
					])
			elif action_name == OCR:
				ocr_result = ocr_extract_text(
					image=images[action_args["image_index"]-1],
					engine=action_args.get("engine", "easyocr")
				)
				
				content.append(
					{
						"type": "text",
						"text": f"Extracted text: {ocr_result['text']}\nImage Index: {action_args['image_index']}"
					}
				)
			elif action_name == OVERLAY:
				return_images = overlay(
					background_image=images[action_args["background_image_index"]-1],
					overlay_image=images[action_args["overlay_image_index"]-1],
					overlay_proportion=action_args["overlay_proportion"]
				)
				for result in return_images:
					images.append(result)
					content.extend([
						{
							"type": "text",
							"text": f"Here's the overlayed image: Image Index: {len(images)}\n"
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(result)}"
							}
						}
					])
			elif action_name == IMAGE_TO_IMAGES_SIMILARITY:
				selected_images = [images[idx-1] for idx in action_args["other_image_indices"]]
				result = calculate_image_to_images_similarity(
					reference_image=images[action_args["reference_image_index"]-1],
					other_images=selected_images
				)
				content.extend([
					{
						"type": "text",
						"text": f"Similarity scores: {result['similarity_scores']}\n"
							f"Best match is image {action_args['other_image_indices'][result['best_match_index']]}"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(result['best_match_image'])}"
						}
					}
				])
			if action_name != TEXT:
				if not len(return_images) > 0 and action_name != OCR and action_name != IMAGE_TO_IMAGES_SIMILARITY and action_name != IMAGE_TO_TEXTS_SIMILARITY and action_name != TEXT_TO_IMAGES_SIMILARITY:
					content.append(
						{
							"type": "text",
							"text": "Tool call successful! However, no image was returned"
						}
					)
				message_list.append(
					{
						"role": "user",
						"content": content
					}
				)
			
			response = reasoner.get_completion(message_list, "all")
	
	message_list_no_image = deepcopy(message_list)
	
	for ms in message_list_no_image:
		contents = ms["content"]
		if isinstance(contents, list):
			for content in contents:
				if content["type"] == "image_url":
					content["image_url"] = "<image>"
	
	if os.path.exists(image_save_dir):
		rmtree(image_save_dir)
	os.makedirs(image_save_dir, exist_ok=True)
	image_save_paths = []
	for i, image in enumerate(images):
		image_save_paths.append(os.path.join(image_save_dir, f'{i}.jpg'))
		image.save(os.path.join(image_save_dir, f'{i}.jpg'))
	
	try:
		# final_response = json.loads(completion_message.content)["action"]["final_response"]
		json_str = re.search(r'\{.*\}', completion_message, re.DOTALL).group().strip()
		final_response = json.loads(json_str)["action"]["final_response"]
	except Exception as e:
		final_response = "No answer was reached."
		
	traces = {
		"call_stack": call_stack,
		"completion_stack": completion_stack,
		"message_list": message_list_no_image,
		"images_saved_paths": image_save_paths
	}
	
						
	return final_response, traces


#####################################################################
#####################################################################
#####################################################################
############  vts_reasoner_verifier #################################


def vts_reasoner_verifier(reasoner, verifier, image_path_list, task_prompt, system_prompt="", developer_prompt="", image_save_dir=None):
	if len(image_path_list) > 0 and isinstance(image_path_list[0], Image.Image):
		images = image_path_list
	else:
		images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]

	message_list = [
		{
			"role": "system",
			"content": system_prompt if len(system_prompt) else "You are a helpful assistant"
		}
	]
	if reasoner.model_type == "qwen-vl":
		message_list[-1]["content"] += "\n" + load_model_response_prompt()
	# message_list[-1]["content"] += "\n" + load_model_response_prompt_action()
	
	if len(developer_prompt):
		message_list[-1]['content'] += "\n" + developer_prompt
	
	message_list.append(
		{
			"role": "user",
			"content": []
		}
	)
	
	verifier_message_list = [
		{
			"role" : "system",
			"content" : load_verifier_system_prompt()
		}
	]
	
	verifier_content = []


	for i, image in enumerate(images):
		base64_image_str = encode_image(image)
		message_list[-1]["content"].append(
			{
				"type": "text",
				"text": f"Image Index: {i+1}\n"
			}
		)
		message_list[-1]["content"].append(
			{
				"type": "image_url",
				"image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}
			}
		)
		verifier_content.append(
			{
				"type": "image_url",
				"image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}
			}
		)
		
	
	message_list[-1]["content"].append(
		{
			"type": "text",
			"text": task_prompt
		}
	)
	message_list[-1]["content"].append(
		{
			"type" : "text",
			"text" : "Next step you should use **tool actions**."
		}
	)
	verifier_content.append(
		{
			"type": "text",
			"text" : f"<question>{task_prompt + developer_prompt}</question>"
		}
	)

	response = reasoner.get_completion(message_list, "action")
	
	call_stack = []
	completion_stack = []

	verifier_message_list.append(
		{
			"role" : "user",
			"content" : verifier_content
		}
	)
	verifier_content = []

	dpo_logits, ref_logits = verifier.get_reward(verifier_message_list)
	dpo_prob = F.softmax(dpo_logits, dim=-1)
	ref_prob = F.softmax(ref_logits, dim=-1)

	dpo_log = torch.log(dpo_prob)
	ref_log = torch.log(ref_prob)
	
	current_reward = dpo_log - ref_log
	previous_reward = torch.zeros_like(current_reward)

	delta = torch.norm(current_reward - previous_reward, p=2)

	# print(f"\nprevious_reward: {previous_reward}")
	# print(f"current_reward: {current_reward}")
	# print(f"delta: {delta}")

	multi_turn = 0
	
	for i in range(10):
		content = []
		for retry in range(5):
			try:
				completion_message = response.choices[0].message.content

				# print(f"\ncompletion_message: {completion_message}")
				json_str = re.search(r'\{.*\}', completion_message, re.DOTALL).group().strip()
				formatted_content = json.loads(json_str)
				action_name = formatted_content["action_name"]
				action_args = formatted_content["action"]

				verifier_content.append(
					{
						"type" : "text",
						"text" : f"<reasoner>{completion_message}</reasoner>"
					}
				)
				
				break
			
			except Exception as e:
				print(f"{retry}th retry error in outer inference loop. Error: {e}")
				if retry == 4:
					response = ""
				else:
					response = reasoner.get_completion(message_list, "all")

		if response == "":
			print("Error during inference. End.")
			break
		
		message_list.append(
			json.loads(response.choices[0].message.model_dump_json())
		)
		completion_stack.append(response.usage.model_dump_json())
		call_stack.extend([{"name": action_name, "args": action_args}])
		
		
		return_images = []
		if action_name == GROUNDING:
			return_images, boxes, labels = grounded_segmentation(
				image=images[action_args["image_index"]-1],
				labels=action_args["text"],
				polygon_refinement=True
			)
			if len(boxes):
				box_str = "\n".join([f"Label: {label}, Box: {box}" for label, box in zip(labels, boxes)])
				content.extend(
					[
						{
							"type": "text",
							"text": f"The bounding box coordinates are: {box_str}\n"  
						}
					]
				)
				verifier_content.append(
					{
						"type" : "text",
						"text" : f"<observation>The bounding box coordinates are: {box_str}</observation>"
					}
				)
			for i, image in enumerate(return_images):
				images.append(image)
				if i == 0:
					content.extend(
						[ 
							{
								"type": "text",
								"text": f"Here's the image marked with masks and boxes: Image Index: {len(images)}\n"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{encode_image(image)}"
								}
							}
						]
					)
					verifier_content.append(
						{
							"type" : "text",
							"text" : f"<observation>Here's the image marked with masks and boxes: Image Index: {len(images)}</observation>"
						}
					)
					verifier_content.append(
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(image)}"
							}
						}
					)
				elif i == 1:
					content.extend(
						[
							
							{
								"type": "text",
								"text": f"Here's the original image: Image Index: {len(images)}\n"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{encode_image(image)}"
								}
							}
						]
					)
					verifier_content.append(
						{
							"type" : "text",
							"text" : f"<observation>Here's the original image: Image Index: {len(images)}</observation>"
						}
					)
					verifier_content.append(
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(image)}"
							}
						}
					)
		elif action_name == DEPTH:
			return_images = get_depth(
				image=images[action_args["image_index"]-1]
			)
			for depth in return_images:
				images.append(depth)
				content.extend(
					[
						{
							"type": "text",
							"text": f"Here's the depth map: "  
						},
						{
							"type": "text",
							"text": f"Image Index: {len(images)}\n"
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(depth)}"
							}
						}
					]
				)
				verifier_content.append(
					{
						"type": "text",
						"text": f"<observation> Image Index: {len(images)} Here's the depth map: </observation>"
					}
				)
				verifier_content.append(
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(depth)}"
						}
					}
				)
		elif action_name == ZOOMIN:
			return_images = zoom_in_image_by_bbox(
				image=images[action_args["image_index"]-1],
				bounding_box=action_args["bounding_box"]
			)
			for crop in return_images:
				images.append(crop)
				content.extend(
					[
						{
							"type": "text",
							"text": f"Here's the zoomed in image: "  
						},
						{
							"type": "text",
							"text": f"Image Index: {len(images)}\n"
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(crop)}"
							}
						}
					]
				)
				verifier_content.append(
					{
						"type": "text",
						"text": f"<observation> Image Index: {len(images)} Here's the zoomed in image: </observation>"
					}
				)
				verifier_content.append(
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(crop)}"
						}
					}
				)
		
		elif action_name == VISUALSEARCH:
			return_images, boxes_list, labels_list = visual_search(
				image=images[action_args["image_index"]-1],
				objects=action_args["objects"]
			)
			
			if len(boxes_list):
				for i, patch in enumerate(return_images):
					
					boxes = boxes_list[i]
					labels = labels_list[i]
					
					box_str = "\n".join([f"Label: {label}, Box: {box}" for label, box in zip(labels, boxes)])
					images.append(patch)
					content.extend(
						[
							{
								"type": "text",
								"text": f"The bounding box coordinates are: {box_str}"  
							},
							{
								"type": "text",
								"text": f"Image Index: {len(images)}\nHere's the image marked with masks and boxes.\nPay attention.\n"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{encode_image(patch)}"
								}
							}
						]
					)
					verifier_content.append(
						{
							"type": "text",
							"text": f"<observation>The bounding box coordinates are: {box_str}\nImage Index: {len(images)}\nHere's the image marked with masks and boxes.\nPay attention.\n </observation>"
						}
					)
					verifier_content.append(
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/jpeg;base64,{encode_image(patch)}"
							}
						}
					)
		elif action_name == CROP:
			return_images = crop_image_action(
				image=images[action_args["image_index"]-1],
				bounding_box=action_args["bounding_box"],
				padding=action_args.get("padding", 0.05)
			)
			for crop in return_images:
				images.append(crop)
				content.extend([
					{
						"type": "text",
						"text": f"Here's the cropped image: Image Index: {len(images)}\n"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(crop)}"
						}
					}
				])

				verifier_content.append(
					{
						"type": "text",
						"text": f"<observation>Here's the cropped image: Image Index: {len(images)}\n</observation>"
					}
				)
				verifier_content.append(
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(crop)}"
						}
					}
				)

		elif action_name == SEGMENT:
			return_images = segment_image(
				image=images[action_args["image_index"]-1],
				bounding_boxes=action_args["bounding_boxes"]
			)
			for segmented in return_images:
				images.append(segmented)
				content.extend([
					{
						"type": "text",
						"text": f"Here's the segmented image: Image Index: {len(images)}\n"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(segmented)}"
						}
					}
				])
				verifier_content.append(
					{
						"type": "text",
						"text": f"<observation>Here's the segmented image: Image Index: {len(images)}\n</observation>"
					}
				)
				verifier_content.append(
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(segmented)}"
						}
					}
				)
		
		elif action_name == OCR:
			ocr_result = ocr_extract_text(
				image=images[action_args["image_index"]-1],
				engine=action_args.get("engine", "easyocr")
			)
			
			content.append(
				{
					"type": "text",
					"text": f"Extracted text: {ocr_result['text']}\nImage Index: {action_args['image_index']}"
				}
			)
			verifier_content.append(
				{
					"type": "text",
					"text": f"<observation>Extracted text: {ocr_result['text']}\nImage Index: {action_args['image_index']}</observation>"
				},
			)

		elif action_name == OVERLAY:
			return_images = overlay(
				background_image=images[action_args["background_image_index"]-1],
				overlay_image=images[action_args["overlay_image_index"]-1],
				overlay_proportion=action_args["overlay_proportion"]
			)
			for result in return_images:
				images.append(result)
				content.extend([
					{
						"type": "text",
						"text": f"Here's the overlayed image: Image Index: {len(images)}\n"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(result)}"
						}
					}
				])
				verifier_content.append(
					{
						"type": "text",
						"text": f"<observation>Here's the overlayed image: Image Index: {len(images)}\n</observation>"
					}
				)
				verifier_content.append(
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encode_image(result)}"
						}
					}
				)

		elif action_name == IMAGE_TO_IMAGES_SIMILARITY:
			selected_images = [images[idx-1] for idx in action_args["other_image_indices"]]
			result = calculate_image_to_images_similarity(
				reference_image=images[action_args["reference_image_index"]-1],
				other_images=selected_images
			)
			content.extend([
				{
					"type": "text",
					"text": f"Similarity scores: {result['similarity_scores']}\n"
						f"Best match is image {action_args['other_image_indices'][result['best_match_index']]}"
				},
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{encode_image(result['best_match_image'])}"
					}
				}
			])
			verifier_content.append(
				{
					"type": "text",
					"text": f"<observation>Similarity scores: {result['similarity_scores']}\nBest match is image {action_args['other_image_indices'][result['best_match_index']]}</observation>"
				}
			)
			verifier_content.append(
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{encode_image(result['best_match_image'])}"
					}
				}
			)

		if action_name != TEXT:
			if not len(return_images) > 0 and action_name != OCR and action_name != IMAGE_TO_IMAGES_SIMILARITY and action_name != IMAGE_TO_TEXTS_SIMILARITY and action_name != TEXT_TO_IMAGES_SIMILARITY:
				content.append(
					{
						"type": "text",
						"text": "Tool call successful! However, no image was returned"
					}
				)
				verifier_content.append(
					{
						"type": "text",
						"text": "<observation>Tool call successful! However, no image was returned</observation>"
					}
				)
		
			message_list.append(
				{
					"role": "user",
					"content": content
				}
			)
		
		verifier_message_list.append(
			{
				"role" : "user",
				"content" : verifier_content
			}
		)
		verifier_content = []
		

		dpo_logits, ref_logits = verifier.get_reward(verifier_message_list)
		
		dpo_prob = F.softmax(dpo_logits, dim=-1)
		ref_prob = F.softmax(ref_logits, dim=-1)

		dpo_log = torch.log(dpo_prob)
		ref_log = torch.log(ref_prob)

		
		previous_reward = current_reward
		current_reward = dpo_log - ref_log
		delta = torch.norm(current_reward - previous_reward, p=2)

		# print(f"current_reward: {current_reward}")
		# print(f"previous_reward: {previous_reward}")
		# print(f"verifier_threshold: {verifier.threshold}")
		# print(f"delta: {delta}")

		if delta < verifier.threshold:
			message_list[-1]["content"].append(
				{
					"type": "text",
					"text": "Next step you **must** give the final answer using TerminateAction."
				}
			)
			
			response = reasoner.get_completion(message_list, "terminate")
			
			completion_message = response.choices[0].message.content
			# print(f"\ncompletion_message: {completion_message}")
			json_str = re.search(r'\{.*\}', completion_message, re.DOTALL).group().strip()
			formatted_content = json.loads(json_str)
			action_name = formatted_content["action_name"]
			action_args = formatted_content["action"]
			message_list.append(
				json.loads(response.choices[0].message.model_dump_json())
			)
			completion_stack.append(response.usage.model_dump_json())
			call_stack.extend([{"name": action_name, "args": action_args}])
		
			
			if formatted_content["action_name"] == TERMINATE:
				# final_response = formatted_terminate_content["action"]["final_response"]
				break
			else:
				message_list.append(
					{
						"role": "user",
						"content": "The verifier has already generated a stop signal. However, the action you produced was not a terminate action. Please generate a terminate action and provide the final response."
					}
				)
		else:
			message_list[-1]["content"].append(
				{
					"type": "text",
					"text": "Next step you should use tool actions to continue reasoning. Please do not execute TerminateAction."
				}
			)
			response = reasoner.get_completion(message_list, "action")

			
	message_list_no_image = deepcopy(message_list)
	for ms in message_list_no_image:
		contents = ms["content"]
		if isinstance(contents, list):
			for content in contents:
				if content["type"] == "image_url":
					content["image_url"] = "<image>"

	verifier_message_list_no_image = deepcopy(verifier_message_list)
	for ms in verifier_message_list_no_image:
		contents = ms["content"]
		if isinstance(contents, list):
			for content in contents:
				if content["type"] == "image_url":
					content["image_url"] = "<image>"
	
	if os.path.exists(image_save_dir):
		rmtree(image_save_dir)
	os.makedirs(image_save_dir, exist_ok=True)
	image_save_paths = []
	for i, image in enumerate(images):
		image_save_paths.append(os.path.join(image_save_dir, f'{i}.jpg'))
		image.save(os.path.join(image_save_dir, f'{i}.jpg'))
	
	try:
		json_str = re.search(r'\{.*\}', completion_message, re.DOTALL).group().strip()
		final_response = json.loads(json_str)["action"]["final_response"]
	except Exception as e:
		final_response = "No answer was reached."
		
	traces = {
		"call_stack": call_stack,
		"completion_stack": completion_stack,
		"message_list": message_list_no_image,
		"images_saved_paths": image_save_paths,
		"verfier_message_list" : verifier_message_list_no_image
	}
					
	return final_response, traces

