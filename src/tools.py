import base64
from dataclasses import dataclass
import inspect
import json
import os
import time

import cv2
import numpy as np
import supervision as sv
import torch

from ast import literal_eval
from io import BytesIO
from copy import deepcopy
from typing import Any, Optional, Tuple, Union, Literal, List, Dict

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from supervision.annotators.utils import resolve_color
from transformers import (
    GroundingDinoForObjectDetection, 
    GroundingDinoProcessor,
    
    AutoModelForDepthEstimation,
    AutoImageProcessor,
    
    SamModel,
    SamProcessor,
    
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    
    pipeline
)
from tqdm import tqdm



GROUNDING = "grounding_action"
DEPTH = "depth_action"
ZOOMIN = "zoomin_action"
VISUALSEARCH = "visual_search_action"
TEXT = "text_action"
OVERLAY = "overlay_action"
CROP = "crop_action"
SEGMENT = "segment_action"
OCR = "ocr_action"
TEXT_TO_IMAGES_SIMILARITY = "text_to_images_similarity_action"
IMAGE_TO_TEXTS_SIMILARITY = "image_to_texts_similarity_action"
IMAGE_TO_IMAGES_SIMILARITY = "image_to_images_similarity_action"
TERMINATE = "terminate_action"



class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ZoomInAction(BaseModel):
    """
    ZoomInAction crops and zooms into a specific region of an image based on a provided bounding box.
    This is particularly useful when:
    - Objects are too small to be detected clearly
    - You need to examine details within a specific region
    - The initial detection results are unclear

    The action will:
    1. Take the specified bounding box as the region of interest
    2. Apply default padding (5% of image dimensions) around the box
    3. Ensure the zoomed region maintains a minimum size (at least 10% of original image)
    4. Return the zoomed-in image while preserving aspect ratio
    3. zoomin_action: {"image_index": int, "bounding_box": {"x_min": int, "y_min": int, "x_max": int, "y_max": int}}
    
    Note:
    - The bounding box coordinates should be in [x, y, w, h] format
    - Coordinates are expected to be normalized (values between 0 and 1)
    - Padding and minimum zoom constraints are automatically handled
    """
    image_index: int
    bounding_box: BoundingBox

class GroundingAction(BaseModel):
    """
    GroundingAction leverages the Grounding DINO model and SegmentAnything to associate textual descriptions with objects in an image. 
    It first uses GroundingDINO to identify relevant bounding boxes according to the textual queries, and use the bounding boxes as prompts
    to segment the precise shapes of the queried objects.
    This functionality is particularly useful for visual spatial reasoning.

    The model takes a natural language query and identifies the corresponding regions in the image by returning:
        A PIL image marked by relevant bounding box and segmentation masks of the queried objects.

    While the Grounding DINO model is powerful in grounding natural objects (e.g., persons, animals, and furniture) to their 
    textual descriptions, it has certain limitations:
        - It may not handle complex or ambiguous queries well. Keep your input phrases simple and concise, 
          such as "cat", "red chair", or "man". Avoid using complex or reasoning-based phrases like "the object behind the person."
        - The detected bounding boxes might be inaccurate, partially out of the image boundaries, or miss objects altogether.
        - The output should be treated as a reference rather than ground truth. Always verify the results before relying on them.

    Args:
        image_index (int): The index of the image to process. For example, if multiple images are provided, 
                           the correct index should be specified as 0, 1, etc.
        text (str): A short text or phrase describing the objects you want to ground.
                    You can ground multiple objects in one phrase, for example, "[a dog, a tree]", or just a single object like "[red car]".

    Returns:
        - image (PIL.Image.Image): An Image object marked by the bounding boxes and segmentation masks returned by grounding dino and segment anything model.

    Note:
        The quality of the outputs depends on the clarity of the input query and the complexity of the image. 
        Double-check the outputs, especially for critical applications, as false positives and negatives are possible.
    """
    image_index: int
    text: List[str]
    

class DepthAction(BaseModel):
    """
    DepthAction provides depth estimation for an input image using the DepthAnything model. 
    The model generates a depth map visualized using the Inferno colormap, where warmer colors 
    (e.g., red, orange) indicate objects closer to the camera, and cooler colors (e.g., blue, purple) 
    represent objects farther away.

    This tool is useful for:
        - Understanding the spatial relationships between objects in a scene.
        - Analyzing 3D structures in static images.
        - Reasoning about relative depth to identify which objects are closer or farther from the camera.
        - Supporting motion analysis across frames or assisting in applications like augmented reality.

    Limitations:
        - The model is not perfect and may produce approximate or incorrect depth estimates in complex scenes.
        - The depth map should be used as a reference rather than a definitive answer, especially in critical tasks.

    Args:
        image_index (int): The index of the input image to process. For multiple images, 
                           the correct index should be specified as 0, 1, etc.

    Returns:
        annotated_depth_map (Image.Image): The depth map of the input image, represented as a colormapped image (Inferno colormap) on top of the original image.
        
    Visualization:
        - Warmer colors (e.g., red) indicate closer objects.
        - Cooler colors (e.g., purple) indicate farther objects.

    Note:
        Use this output for qualitative reasoning or as an input to downstream processing tasks. 
        Validate the depth results in critical applications.
    """
    image_index: int


class TerminateAction(BaseModel):
    """
    Use TerminateAction when you finish your resoning process and provide your final response to the user's query here.
    For example, if the user asks you to output an option letter or return an answer, you should do it here.
    """
    final_response: str
    

class VisualSearchAction(BaseModel):
    """
    Deal with the case when the user query is asking about objects that are not seen by the model.
    In that case, the most common reason is that the object is too small such that both the vision-language model and the object detection model fail to detect it.
    This function tries to detect the object by sliding window search.
    With the help of the detection model, it tries to detect the object in the zoomed-in patches.
    The function returns a list of annotated images that may contain at least one of the objects, annotated with bounding boxes.
    It also returns a list of a list of bounding boxes of the detected objects.

    Args:
        image_index (int): the index of the image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.
        
    Returns:
        possible_patches (List[PIL.Image.Image]): a list of annotated zoomed-in images that may contain the object, annotated with bounding boxes.
        possible_boxes (List[List[Float]]): For each image in possible_patches, a list of bounding boxes of the detected objects. 
            The coordinates are w.r.t. each zoomed-in image. The order of the boxes is the same as the order of the images in possible_patches.
    """
    image_index: int
    objects: List[str]


class TextAction(BaseModel):
    """
    TextAction allows you to respond user request with pure text. This action is useful when you are reflecting or thinking in pure text and do not need any extra tools.
    """
    text: str
    

class ProportionBoundingBox(BaseModel):
    x_min: float # ranging from 0 to 1
    y_min: float # ranging from 0 to 1
    x_max: float # ranging from 0 to 1
    y_max: float # ranging from 0 to 1


class OverlayAction(BaseModel):
    """
    Overlay two images together. 
    This tool is useful when you want to overlay two outputs into one. For example, overlay the results of depth estimation and segmentation.
    
    Args:
        background_image_index (int): the background image
        overlay_image_index (int): the overlay image
        overlay_proportion (List[float]): Four float numbers of bounding box (xmin, ymin, xmax, ymax) that shows how much should the overlaid image take up the background.  
                                          Usually, it should be [0, 0, 1, 1], meaning the overlay image will be resized to background image and overlay fully. 
                                          In other times, such as jigsaw, the overlaid image should fit the size of the missing part of the background image. For example, [0.5, 0.5, 1, 1]
    """
    """
    Overlay two images together with controllable transparency and positioning.
    This is particularly useful for:
        - Visualizing heatmaps/depth maps while preserving original image context
        - Combining segmentation results with original images
        - Highlighting specific regions with annotations
    
    Key Features:
        1. Transparency Control (alpha): Adjust how strongly the overlay appears
        2. Precise Positioning: Place overlays in specific regions using normalized coordinates
        3. Multi-modal Fusion: Combine different processing results (depth, segmentation, etc.)

    Args:
        background_image_index (int): Index of the background image
        overlay_image_index (int): Index of the overlay image
        overlay_proportion (ProportionBoundingBox): Normalized coordinates [x_min,y_min,x_max,y_max] specifying:
            - x_min: Left boundary (0=far left, 1=far right)
            - y_min: Top boundary (0=top, 1=bottom)
            - x_max: Right boundary
            - y_max: Bottom boundary
            Default covers full image ([0,0,1,1])

    Returns:
        List[Image.Image]: Contains the composited image with overlay applied

    Example Use Cases:
        1. Depth Visualization:
            overlay(depth_map, original_img, alpha=0.4, proportion=[0,0,1,1])
        
        2. Localized Segmentation:
            overlay(seg_mask, original_img, alpha=0.7, proportion=[0.5,0.5,1,1])

        3. Annotation Highlighting:
            overlay(text_annotations, diagram, alpha=0.9, proportion=[0.1,0.1,0.5,0.3])
    """
    background_image_index: int
    overlay_image_index: int
    overlay_proportion: ProportionBoundingBox
    


class CropAction(BaseModel):
    """
    CropAction crops a specific region from an image based on bounding box coordinates.
    
    Args:
        image_index (int): The index of the image to crop
        bounding_box (BoundingBox): The bounding box coordinates defining the crop region
        padding (float, optional): Additional padding around the bounding box. Defaults to 0.05.
        
    Returns:
        The cropped image
    """
    image_index: int
    bounding_box: BoundingBox
    padding: float = 0.05


class SegmentAction(BaseModel):
    """
    SegmentAction performs segmentation on an image using bounding box prompts.
    
    Args:
        image_index (int): The index of the image to segment
        bounding_boxes (List[BoundingBox]): List of bounding boxes to use as prompts
        
    Returns:
        The original image with segmentation masks overlaid
    """
    image_index: int
    bounding_boxes: List[BoundingBox]


class OCRAction(BaseModel):
    """
    Extract text from an image using OCR (Optical Character Recognition).
    Useful when the user asks about text content in the image.
    
    Args:
        image_index (int): Index of the image containing text
        engine (Literal['easyocr', 'openai']): OCR engine to use (default: 'easyocr')
        
    Returns:
        List of extracted text strings (may be empty if no text found)
    """
    image_index: int
    engine: Literal['easyocr', 'openai'] = 'easyocr'



class TextToImagesSimilarityAction(BaseModel):
    """
    Calculate similarity scores between a text query and multiple images using CLIP model.
    Useful for finding the most relevant image to a text description.
    
    Args:
        text (str): The reference text description
        image_indices (List[int]): List of image indices to compare with the text
        
    Returns:
        Dict containing:
            - similarity_scores: List[float] of similarity scores (0-1)
            - best_match_index: int index of most similar image
            - best_match_image: PIL.Image of most similar image
    """
    text: str
    image_indices: List[int]


class ImageToTextsSimilarityAction(BaseModel):
    """
    Calculate similarity scores between an image and multiple text descriptions using CLIP model.
    Useful for finding the best text caption for an image.
    
    Args:
        image_index (int): Index of reference image
        texts (List[str]): List of text descriptions to compare
        
    Returns:
        Dict containing:
            - similarity_scores: List[float] of similarity scores (0-1)
            - best_match_index: int index of most similar text
            - best_match_text: str of most similar text
    """
    image_index: int
    texts: List[str]


class ImageToImagesSimilarityAction(BaseModel):
    """
    Calculate similarity scores between a reference image and multiple other images using CLIP model.
    Useful for finding visually similar images.
    
    Args:
        reference_image_index (int): Index of reference image
        other_image_indices (List[int]): List of image indices to compare
        
    Returns:
        Dict containing:
            - similarity_scores: List[float] of similarity scores (0-1)
            - best_match_index: int index of most similar image
            - best_match_image: PIL.Image of most similar image
    """
    reference_image_index: int
    other_image_indices: List[int]




class AllStep(BaseModel):
    thought: str
    action_name: Literal['grounding_action', 'depth_action', 'zoomin_action', 'visual_search_action', 'crop_action', 'segment_action', 'ocr_action', 'overlay_action','image_to_images_similarity_action', 'terminate_action']
    action: Union[GroundingAction, DepthAction, ZoomInAction, VisualSearchAction, CropAction, SegmentAction, OCRAction, OverlayAction,ImageToImagesSimilarityAction, TerminateAction]

class ActionStep(BaseModel):
    thought: str
    action_name: Literal['grounding_action', 'depth_action', 'zoomin_action', 'visual_search_action', 'crop_action', 'segment_action', 'overlay_action','ocr_action', 'image_to_images_similarity_action']
    action: Union[GroundingAction, DepthAction, ZoomInAction, VisualSearchAction, CropAction, SegmentAction, OverlayAction, OCRAction, ImageToImagesSimilarityAction]


class TerminateStep(BaseModel):
    thought: str
    action_name: Literal['terminate_action']
    action: TerminateAction


################################################################
######################## vision expert models 
################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
detector_id = "IDEA-Research/grounding-dino-base"
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device='cuda')  
depth_model.eval()

sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(device='cuda')
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
sam_model.eval()

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

PIL_COLORS = [tuple(int(c * 255) for c in color) for color in COLORS]



def crop_image_action(
    image: Image.Image,
    bounding_box: Dict[str, int],
    padding: float = 0.05
) -> List[Image.Image]:
    """Perform cropping action with padding"""
    x_min = bounding_box["x_min"]
    y_min = bounding_box["y_min"]
    x_max = bounding_box["x_max"]
    y_max = bounding_box["y_max"]
    
    # Apply padding
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - padding * width)
    y_min = max(0, y_min - padding * height)
    x_max = min(image.width, x_max + padding * width)
    y_max = min(image.height, y_max + padding * height)
    
    cropped = image.crop((x_min, y_min, x_max, y_max))
    if cropped.width * cropped.height == 0:
        return []
    return [cropped]


def segment_image(
    image: Image.Image,
    bounding_boxes: List[Dict[str, int]]
) -> List[Image.Image]:
    """Perform segmentation using bounding box prompts"""
    if not bounding_boxes:
        return []
    
    # Convert to format expected by SAM
    input_boxes = [[
        box["x_min"], 
        box["y_min"], 
        box["x_max"], 
        box["y_max"]
    ] for box in bounding_boxes]
    
    # Process with SAM
    inputs = sam_processor(
        images=image, 
        input_boxes=[input_boxes], 
        return_tensors="pt"
    ).to(device)
    
    outputs = sam_model(**inputs)
    masks = sam_processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]
    
    masks = refine_masks(masks, polygon_refinement=True)
    # print(f"mask{masks}")
    # Create detection results for annotation
    detections = []
    for i, mask in enumerate(masks):
        detections.append(DetectionResult(
            score=1.0,
            label=f"object_{i}",
            box=DetBoundingBox(
                xmin=input_boxes[i][0],
                ymin=input_boxes[i][1],
                xmax=input_boxes[i][2],
                ymax=input_boxes[i][3]
            ),
            mask=mask
        ))
    
    # print("detection:", detections)
    # Return annotated image
    annotated_image = plot_detections(image, detections)
    return [annotated_image]






import easyocr
EASYOCR_READER = None
def init_easyocr(model_storage_dir: str = "./local_easyocr_models"):
    """Initialize EasyOCR and pre-load the model."""
    global EASYOCR_READER
    if EASYOCR_READER is None:
        if model_storage_dir:
            os.makedirs(model_storage_dir, exist_ok=True)
        EASYOCR_READER = easyocr.Reader(
            ['en'],
            model_storage_directory=model_storage_dir,  # Custom model path
            download_enabled=True  # Automatic download allowed (for the first time)
        )


def ocr_extract_text(
    image: Image.Image,
    engine: str = 'easyocr'
) -> Dict[str, str]:
    if engine == 'easyocr':
        global EASYOCR_READER
        if EASYOCR_READER is None:
            init_easyocr()
        
        results = EASYOCR_READER.readtext(np.array(image))
        texts = [text for (_, text, _) in results]
        return {"text": ", ".join(texts) if texts else "No text detected"}
    
    elif engine == 'openai':
        from openai import OpenAI
        import base64
        from io import BytesIO
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return {"text": response.choices[0].message.content}
    
    return {"text": "OCR engine not supported"}


def overlay(
    background_image: Image.Image,
    overlay_image: Image.Image,
    overlay_proportion: Union[Dict[str, float], ProportionBoundingBox,List[float]],
    alpha: float = 0.3
) -> List[Image.Image]:
    """
    Overlay two images together with transparency control.
    
    Args:
        background_image: The background/base image
        overlay_image: The image to overlay on top
        overlay_proportion: Either a ProportionBoundingBox object or dict with x_min,y_min,x_max,y_max
        alpha: Transparency level (0-1) of the overlay
        
    Returns:
        List containing the resulting overlayed image
    """
    # Convert dict to ProportionBoundingBox if needed
    if isinstance(overlay_proportion, dict):
        overlay_proportion = ProportionBoundingBox(**overlay_proportion)
    elif isinstance(overlay_proportion, list):
        if len(overlay_proportion) != 4:
            raise ValueError("List must contain exactly 4 elements [x_min, y_min, x_max, y_max]")
        overlay_proportion = ProportionBoundingBox(
            x_min=overlay_proportion[0],
            y_min=overlay_proportion[1],
            x_max=overlay_proportion[2],
            y_max=overlay_proportion[3]
        )
    # Validate proportions
    if not (0 <= overlay_proportion.x_min < overlay_proportion.x_max <= 1 and
            0 <= overlay_proportion.y_min < overlay_proportion.y_max <= 1):
        raise ValueError("Invalid proportion values - must be between 0 and 1 with min < max")
    
    # Calculate actual pixel coordinates
    bg_width, bg_height = background_image.size
    x_min = int(overlay_proportion.x_min * bg_width)
    y_min = int(overlay_proportion.y_min * bg_height)
    x_max = int(overlay_proportion.x_max * bg_width)
    y_max = int(overlay_proportion.y_max * bg_height)
    
    # Calculate width and height with validation
    width = max(1, x_max - x_min)  # Ensure at least 1 pixel
    height = max(1, y_max - y_min)  # Ensure at least 1 pixel
    
    # Resize overlay image to match target area
    try:
        overlay_resized = overlay_image.resize((width, height), Image.Resampling.LANCZOS)
    except Exception as e:
        raise ValueError(f"Failed to resize overlay image: {str(e)}")

    # Apply transparency
    overlay_with_alpha = overlay_resized.copy()
    overlay_with_alpha.putalpha(int(255 * alpha))

    # Create new image and paste background
    result_img = Image.new('RGBA', background_image.size)
    result_img.paste(background_image, (0, 0))

    # Paste the overlay with transparency
    result_img.paste(overlay_with_alpha, (x_min, y_min), overlay_with_alpha)

    return [result_img.convert('RGB')]


import open_clip
open_clip_model_version = "ViT-H-14-378-quickgelu"
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(open_clip_model_version, pretrained='dfn5b')
open_clip_model = open_clip_model.to(device).eval()
open_clip_tokenizer = open_clip.get_tokenizer(open_clip_model_version)


def calculate_text_to_images_similarity(
    text: str, 
    images: List[Image.Image]
) -> Dict[str, Any]:
    """CLIP-based similarity between text and images"""
    
    # Preprocess inputs
    if isinstance(text, list):
        text = text[0]
    text_input = open_clip_tokenizer([text]).to(device)
    image_inputs = torch.stack([open_clip_preprocess(img).to(device) for img in images])
    
    # Calculate features
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        text_features = open_clip_model.encode_text(text_input)
        image_features = open_clip_model.encode_image(image_inputs)
        
        # Normalize features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = text_features @ image_features.T
    
    scores = [round(score, 4) for score in similarity[0].tolist()]
    best_idx = torch.argmax(similarity).item()
    
    return {
        "similarity_scores": scores,
        "best_match_index": best_idx,
        "best_match_image": images[best_idx]
    }


def calculate_image_to_texts_similarity(
    image: Image.Image,
    texts: List[str]
) -> Dict[str, Any]:
    """CLIP-based similarity between image and texts"""
    
    # Preprocess inputs
    image_input = open_clip_preprocess(image).unsqueeze(0).to(device)
    text_inputs = open_clip_tokenizer(texts).to(device)
    
    # Calculate features
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        image_features = open_clip_model.encode_image(image_input)
        text_features = open_clip_model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = image_features @ text_features.T
    
    scores = [round(score, 4) for score in similarity[0].tolist()]
    best_idx = torch.argmax(similarity).item()
    
    return {
        "similarity_scores": scores,
        "best_match_index": best_idx,
        "best_match_text": texts[best_idx]
    }



def calculate_image_to_images_similarity(
    reference_image: Image.Image,
    other_images: List[Image.Image]
) -> Dict[str, Any]:
    """CLIP-based similarity between images"""

    old_other_images = other_images

    # Preprocess inputs
    ref_image = open_clip_preprocess(reference_image).unsqueeze(0).to(device)
    other_images = torch.stack([open_clip_preprocess(img).to(device) for img in other_images])
    
    # Calculate features
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):  
        ref_features = open_clip_model.encode_image(ref_image)
        other_features = open_clip_model.encode_image(other_images)
        
        # Normalize features
        ref_features /= ref_features.norm(dim=-1, keepdim=True)
        other_features /= other_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = ref_features @ other_features.T
    
    scores = [round(score, 4) for score in similarity[0].tolist()]
    best_idx = torch.argmax(similarity).item()
    
    return {
        "similarity_scores": scores,
        "best_match_index": best_idx,
        "best_match_image": old_other_images[best_idx]
    }


def crop_image(
    image: Image.Image, 
    x:float, 
    y:float, 
    width:float, 
    height:float
):
    """Crop the image based on the normalized coordinates.
    Return the cropped image.
    This has the effect of zooming in on the image crop.

    Args:
        image (PIL.Image.Image): the input image
        x (float): the horizontal coordinate of the upper-left corner of the box
        y (float): the vertical coordinate of that corner
        width (float): the box width
        height (float): the box height

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
    """
    w, h = image.size
    
    # limit the range of x and y
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    width = max(0, min(width, w))
    height = max(0, min(height, h))
    
    
    cropped_img = image.crop((x, y, width, height))
    
    if cropped_img.width * cropped_img.height == 0:
        return
    
    return cropped_img


def zoom_in_image_by_bbox(
    image: Image.Image,
    bounding_box: Dict[str, int],
    padding: float = 0.05,
    min_zoom_ratio: float = 0.1
) -> List[Image.Image]:
    """
    Zoom into a region of interest while maintaining image quality and aspect ratio.
    
    Args:
        image: Input PIL Image
        bounding_box: {
            'x_min': int, 'y_min': int, 
            'x_max': int, 'y_max': int
        }
        padding: Percentage padding around the box (default: 5%)
        min_zoom_ratio: Minimum size relative to original image (default: 10%)
    
    Returns:
        List containing the zoomed-in image (or empty list if failed)
    """
    # Obtain the original image dimensions and bounding box parameters.
    orig_width, orig_height = image.size
    x_min, y_min = bounding_box['x_min'], bounding_box['y_min']
    x_max, y_max = bounding_box['x_max'], bounding_box['y_max']
    
    # Calculate the original dimensions of the bounding box.
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # Apply padding (proportional to the bounding box dimensions).
    padding_x = padding * box_width
    padding_y = padding * box_height
    
    # Calculate the cropping area with padding (ensuring it does not exceed the image boundaries).
    crop_x_min = max(0, x_min - padding_x)
    crop_y_min = max(0, y_min - padding_y)
    crop_x_max = min(orig_width, x_max + padding_x)
    crop_y_max = min(orig_height, y_max + padding_y)
    
    # Ensure a minimum scaling ratio (at least 10% of the original image).
    min_width = orig_width * min_zoom_ratio
    min_height = orig_height * min_zoom_ratio
    
    if (crop_x_max - crop_x_min) < min_width:
        # Adjust the width to the minimum size (keeping the center point unchanged).
        center_x = (crop_x_min + crop_x_max) / 2
        crop_x_min = max(0, center_x - min_width/2)
        crop_x_max = min(orig_width, center_x + min_width/2)
    
    if (crop_y_max - crop_y_min) < min_height:
        # Adjust the height to the minimum size (keeping the center point unchanged).
        center_y = (crop_y_min + crop_y_max) / 2
        crop_y_min = max(0, center_y - min_height/2)
        crop_y_max = min(orig_height, center_y + min_height/2)
    
    # crop
    try:
        cropped = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    except Exception as e:
        print(f"Zoom in failed: {str(e)}")
        return []
    
    # Scale up to the original image dimensions while maintaining the original aspect ratio.
    zoomed = cropped.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
    
    return [zoomed]


def get_depth(
    image: Image.Image
):
    """
    Use Depth Anything V2 model to generate a depth map of the image
    """
    
    pixel_values = depth_processor(images=image, return_tensors="pt").to(device='cuda').pixel_values
    
    with torch.no_grad():
        outputs = depth_model(pixel_values)
        predicted_depth = outputs.predicted_depth
        
    h, w = image.size[::-1]

    depth = torch.nn.functional.interpolate(predicted_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    colored_depth = Image.fromarray(cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1])
    
    # overlay the colored depth with the original image
    colored_depth.putalpha(int(255 * 0.7))
    
    copied_image = deepcopy(image)
    copied_image.paste(colored_depth, (0, 0, w, h), colored_depth)
    
    return [copied_image.convert("RGB")]
    
    

def visual_search(
    image: Image.Image, 
    objects
):
    """Deal with the case when the user query is asking about objects that are not seen by the model.
    In that case, the most common reason is that the object is too small such that both the vision-language model and the object detection model fail to detect it.
    This function tries to detect the object by sliding window search.
    With the help of the detection model, it tries to detect the object in the zoomed-in patches.
    The function returns a list of annotated images that may contain at least one of the objects, annotated with bounding boxes.
    It also returns a list of a list of bounding boxes of the detected objects.

    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.
        
    Returns:
        possible_patches (List[PIL.Image.Image]): a list of annotated zoomed-in images that may contain the object, annotated with bounding boxes.
        possible_boxes (List[List[List[Float]]]): For each image in possible_patches, a list of bounding boxes of the detected objects. 
            The coordinates are w.r.t. each zoomed-in image. The order of the boxes is the same as the order of the images in possible_patches.
        possible_labels (List[List[labels]]): A list of list of possible labels returned.
    """
    image_width, image_height = image.size
    
    def check_if_box_margin(box, margin=0.005):
        x_margin = min(box[0], image_width - box[0] - box[2])
        y_margin = min(box[1], image_height - box[1] - box[3])
        return x_margin < margin * image_width or y_margin < margin * image_height
    
    
    box_width = 1/3
    box_height = 1/3

    possible_patches = []
    possible_boxes = []
    possible_labels = []
    
    for x in np.arange(0, 7/9, 2/9):
        for y in np.arange(0, 7/9, 2/9):
            cropped_img = crop_image(image, x * image_width, y * image_height, (box_width + x) * image_width, (box_height + y)* image_height)
            # detected_images, detection_boxes, labels = detect(cropped_img, objects)
            
            detection_results: List[DetectionResult] = detect(cropped_img, objects)
            detection_boxes = [detection.box.xyxy for detection in detection_results]
            
            labels = [detection.label for detection in detection_results]
            
            # if one of the boxes is not too close to the edge, save it
            margin_flag = True
            for box in detection_boxes:
                if not check_if_box_margin(box):
                    margin_flag = False
                    break
            
            # if the object is detected and the box is not too close to the edge
            if len(detection_boxes) != 0 and not margin_flag:
                detected_images = [plot_detections(cropped_img, detection_results)]
                possible_patches.extend(detected_images)
                possible_boxes.append(detection_boxes)
                possible_labels.append(labels)

    return possible_patches, possible_boxes, possible_labels



@dataclass
class DetBoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: DetBoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=DetBoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    
    boxes = np.array([detection.box.xyxy for detection in detection_results])
    labels = [f"{detection.label}: {detection.score:.2f}" for detection in detection_results]
    masks = np.array([detection.mask for detection in detection_results])
    
    if all([detection.mask is not None for detection in detection_results]):
        detections = sv.Detections(xyxy=boxes, mask=masks)
    else:
        detections = sv.Detections(xyxy=boxes)
    

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annoatotor = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    annotated_frame = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    annotated_frame = mask_annoatotor.annotate(scene=annotated_frame.copy(), detections=detections)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame.copy(), detections=detections, labels=labels)
    
    if all([detection.mask is not None for detection in detection_results]):
        colored_mask = annotated_frame.copy()
        for detection_idx in np.flip(np.argsort(detections.area)):
                color = resolve_color(
                    color=sv.ColorPalette.DEFAULT,
                    detections=detections,
                    detection_idx=detection_idx,
                    color_lookup=sv.ColorLookup.INDEX
                )
                mask = detections.mask[detection_idx]
                colored_mask[mask != 0] = color.as_bgr()
                
        cv2.addWeighted(
                colored_mask, 0.5, annotated_frame, 0.5, 0, dst=annotated_frame
        )
    
    
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)


def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    if isinstance(image, Image.Image):
        image = np.array(image)
    annotated_image = annotate(image, detections)
    
    if save_name:
        Image.fromarray(annotated_image).save(save_name)
    
    return Image.fromarray(annotated_image)
    
    

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Warning: in mask_to_polygon, contours is empty!")
        return None
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(1,))

    return mask

def load_image(image_str: str) -> Image.Image:
    image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes = get_boxes(detection_results)
    inputs = sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = sam_model(**inputs)
    masks = sam_processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
):
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold)
    
    boxes = get_boxes(detections)
    
    try:
        detections = segment(image, detections, polygon_refinement, segmenter_id)
        
        plot_result = plot_detections(
            image,
            detections
        )

        boxes = [detection.box.xyxy for detection in detections]
        labels = [f"{detection.label}" for detection in detections]
        
        return [plot_result, image], boxes, labels
    except Exception as e:
        print(f"Error: {e}\nboxes: {boxes}")
        return [], [], []
    

def encode_image(image: Image.Image) -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format="jpeg")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


