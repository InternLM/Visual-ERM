import base64
import io
from PIL import Image
import os
from openai import OpenAI
import httpx

# Singleton client instance
_rm_client = None
_rm_model_name = None

# Same to training prompt
PROMPT_CHART2CODE_JUDGEMENT = """\
You are an **Experienced Specialist for Data Visualization**. You will be provided with **two images**:

1. **Original Image**: a chart rendered using ground-truth Matplotlib code.
2. **Generated Image**: a chart rendered using AI-generated Matplotlib code for the **Original Image**.

Your task is to **compare the Generated Image against the Original Image** and identify **all visual discrepancies**, then summarize them in a strict JSON format.

Additionally, You should assign a **severity score** for each error:
- 1 (minor): small errors that barely affect readability or understanding.
- 2 (medium): errors that affect partial understanding and require manual correction for reliable use.
- 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead.

You MUST output a single JSON object **ONLY**.
- Do NOT include any extra text.
- The JSON schema MUST match exactly:
{
  "structure_error_count": int,
  "data_error_count": int,
  "text_error_count": int,
  "style_error_count": int,
  "errors": [
    {
      "category": "structure_error | data_error | text_error | style_error",
      "severity": 1 | 2 | 3,
      "location": "Specific location (e.g., 'Left subplot title', 'Red line data', 'Legend')",
      "description": "Concise description of the error."
    }
  ]
}
"""

# PROMPT_IMG2SVG_JUDGEMENT = """\
# You are an expert QA Specialist for Vector Graphics & Icon Generation. You will be provided with two images to compare visually:

# - Image 1 (Ground Truth): The original icon/graphic rendered from correct SVG code.
# - Image 2 (Prediction): An icon/graphic rendered from AI-generated SVG code attempting to reproduce Image 1.

# Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed image relative to the original, then summarize them in a strict JSON format.

# Assign a severity score for each error:
# - 1 (minor): small errors that barely affect readability or understanding
# - 2 (medium): errors that affect partial understanding and require manual correction for reliable use
# - 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead

# You MUST output a single JSON object ONLY.
# - Do NOT include any extra text.
# - Do NOT wrap the JSON in markdown code fences like ```json.
# - The JSON schema MUST match exactly:

# {
#   "structure_error_count": int,
#   "shape_error_count": int,
#   "style_error_count": int,
#   "text_symbol_error_count": int,
#   "other_error_count": int,
#   "errors": [
#     {
#       "category": "structure_error | shape_error | style_error | text_symbol_error | other_error",
#       "severity": 1 | 2 | 3,
#       "location": "Visual location (e.g., 'Train Wheel', 'Eye Contour', 'Canvas Background')",
#       "description": "Concise description of the visual mismatch."
#     }
#   ]
# }
# """

PROMPT_IMG2SVG_JUDGEMENT = """\
You are an expert QA Specialist for Vector Graphics & Icon Generation. You will be provided with two images to compare visually:

- Image 1 (Ground Truth): The original icon/graphic rendered from correct SVG code.
- Image 2 (Prediction): An icon/graphic rendered from AI-generated SVG code attempting to reproduce Image 1.

Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed image relative to the original, then summarize them in a strict JSON format.

Assign a severity score for each error:
- 1 (minor): small errors that barely affect readability or understanding
- 2 (medium): errors that affect partial understanding and require manual correction for reliable use
- 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead

You MUST output a single JSON object ONLY.
- Do NOT include any extra text.
- Do NOT wrap the JSON in markdown code fences like ```json.
- The JSON schema MUST match exactly:

{
  "structure_error_count": int,
  "shape_error_count": int,
  "style_error_count": int,
  "text_symbol_error_count": int,
  "errors": [
    {
      "category": "structure_error | shape_error | style_error | text_symbol_error",
      "severity": 1 | 2 | 3,
      "location": "Visual location (e.g., 'Train Wheel', 'Eye Contour', 'Canvas Background')",
      "description": "Concise description of the visual mismatch."
    }
  ]
}
"""

MAX_IMG_SIDE = os.getenv('MAX_IMG_SIDE', 1024)

def get_rm_client():
    """Get or create the OpenAI client singleton instance."""
    global _rm_client
    if _rm_client is None:
        RM_API_BASE = os.getenv('RM_API_BASE', None)
        if not RM_API_BASE:
            raise ValueError("RM_API_BASE is not set in the environment variables")
        RM_API_KEY = os.getenv('RM_API_KEY', None)
        if not RM_API_KEY:
            raise ValueError("RM_API_KEY is not set in the environment variables")
        _rm_client = OpenAI(
            api_key=RM_API_KEY,
            base_url=RM_API_BASE,
            http_client=httpx.Client(verify=False),
        )
    return _rm_client


def get_rm_model_name():
    """Get the model name from environment variables (singleton)."""
    global _rm_model_name
    if _rm_model_name is None:
        _rm_model_name = os.getenv('RM_NAME', None)
        if not _rm_model_name:
            raise ValueError("RM_NAME is not set in the environment variables")
    return _rm_model_name


def encode_image(img_path: str) -> str:
    with Image.open(img_path) as img:
        buffered = io.BytesIO()
        img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def message_format(user_prompt, image_files):
    """
    Args:
        user_prompt (str): User prompt for the model
        image_files (str | list | None): Image files, can be a single string, a list of strings, or None
    """
    
    # 1. Initialize content list
    content_parts = []

    # 2. Process images (if any)
    if image_files:
        # Compatibility handling: if the user only passes a single string path, automatically convert it to a list
        if isinstance(image_files, str):
            image_files = [image_files]
        
        # Iterate through all image paths
        for file_path in image_files:
            # Add a simple check to ensure the path is not empty
            if file_path: 
                try:
                    base64_image = encode_image(file_path)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            # "detail": "low"
                        },
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {file_path}: {e}")
                    # According to demand, here you can choose to skip or report an error

    # 3. Append text prompt (usually put text after images, or according to model poriginal)
    content_parts.append({
        "type": "text", 
        "text": user_prompt
    })

    # 4. Return the final structure
    return [
        {
            "role": "user",
            "content": content_parts,
        },
    ]

