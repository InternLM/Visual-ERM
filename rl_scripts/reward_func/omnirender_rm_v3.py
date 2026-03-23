import os
import re
import sys
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError
import logging
import traceback
from reward_utils import (
    get_rm_client,
    get_rm_model_name,
    message_format,
    PROMPT_CHART2CODE_JUDGEMENT,
    PROMPT_IMG2SVG_JUDGEMENT,
)
import json
import io
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)
# default level is WARNING, set in verl/__init__.py
logger.setLevel(logging.INFO)

TEMP_SAVE_DIR_ROOT = os.getenv("TEMP_SAVE_DIR_ROOT", None)
if not TEMP_SAVE_DIR_ROOT:
    raise ValueError("TEMP_SAVE_DIR_ROOT is not set in environment variables")
os.makedirs(TEMP_SAVE_DIR_ROOT, exist_ok=True)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", None)
if not EXPERIMENT_NAME:
    raise ValueError("EXPERIMENT_NAME is not set in environment variables")
TEMP_SAVE_DIR = os.path.join(TEMP_SAVE_DIR_ROOT, EXPERIMENT_NAME)
os.makedirs(TEMP_SAVE_DIR, exist_ok=True)

SEVERITY_AVG = -25.0
# WHITE_SEVERITY = -32.0
REWARD_MIN = 0.0
REWARD_MAX = 2.0
# (-10 / 11 + 1) * 2 = 0.1818
# severity > 18 or render failed
# REWARD_MIN = -18.0
# REWARD_SVG_MIN = -34
# REWARD_MIN = -1
# MAX_SEVERITY = 36.0  # empirical, from RM statistics
MAX_SEVERITY = 50.0 # empirical, from RM statistics
BASE_RENDER_SUCCESS_REWARD = 1.0    # actually use 1 + 1
RM_REWARD_SCALE = REWARD_MAX - BASE_RENDER_SUCCESS_REWARD

# reward = 1.0 + reward_score_from_rm / MAX_SEVERITY
# reward = np.clip(reward, 0.0, 1.0)
# reward *= 2.0
# return reward


DEFAULT_MODEL_KWARGS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    # normal
    "max_tokens": 4096,
    "n": 1,
}


def run_once_with_prompt_single_turn(
    model_client, model_name, messages, retry=2, **kwargs
):
    num_retries = 0

    while num_retries < retry:
        try:
            # these two keys are needed to be added in `extra_body` in vllm
            top_k = kwargs.pop("top_k", -1)
            repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
            request_body = {**kwargs}
            if top_k != -1:
                if "extra_body" not in request_body:
                    request_body["extra_body"] = {}
                request_body["extra_body"]["top_k"] = top_k
            if repetition_penalty != 1.0:
                if "extra_body" not in request_body:
                    request_body["extra_body"] = {}
                request_body["extra_body"]["repetition_penalty"] = repetition_penalty
            chat_response = model_client.chat.completions.create(
                model=model_name,
                messages=messages,
                **request_body,
            )
            return chat_response
        except Exception as e:
            logger.info(f"[Retry {num_retries+1}/{retry}] Exception type: {type(e).__name__}")
            logger.info(f"Error message: {e}")
            logger.info("Traceback:\n" + traceback.format_exc())
            num_retries += 1
    raise RuntimeError(f"Calling OpenAI API failed after {retry} retries.")

def extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object from text.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_error_response(resp) -> Tuple[Optional[dict], str]:
    if resp is None:
        return None, ""

    if isinstance(resp, list):
        resp = "".join(str(x) for x in resp)

    raw = str(resp).strip()
    original_raw = raw

    # remove <think>...</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # remove markdown code block
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", raw, count=1)
        raw = re.sub(r"\n?\s*```$", "", raw).strip()

    # directly try to parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, original_raw
    except Exception:
        pass

    # try to extract JSON from text
    json_str = extract_json_object(raw)
    if json_str:
        try:
            return json.loads(json_str), original_raw
        except Exception:
            pass

    return None, original_raw



def extract_score_from_rm_response(rm_response_content: str) -> Optional[float]:
    """
    Extract total severity score from reward model response.

    The response should contain a JSON object with an "errors" array,
    where each error has a "severity" field (1, 2, or 3).

    Args:
        rm_response_content: Raw response content from reward model

    Returns:
        Total severity score (sum of all error severities), or None if extraction fails.
    """
    if not rm_response_content:
        logger.warning("Empty response content for score extraction")
        return None

    obj, raw_text = parse_error_response(rm_response_content)
    if obj is None:
        logger.warning(
            f"Failed to parse JSON from response. Raw text:\n{raw_text}"
        )
        return None

    if not isinstance(obj, dict):
        logger.warning(f"Parsed object is not a dict, got type: {type(obj)}")
        return None

    try:
        errors = obj.get("errors", [])
        if not isinstance(errors, list):
            logger.warning(f"Expected 'errors' to be a list, got type: {type(errors)}")
            return None

        total_severity = 0.0
        for idx, error in enumerate(errors):
            if not isinstance(error, dict):
                logger.warning(f"Error item at index {idx} is not a dict, skipping")
                continue

            severity = error.get("severity")
            if severity is None:
                logger.debug(f"Error at index {idx} missing 'severity' field, skipping")
                continue

            try:
                severity = float(severity)
                if severity < 0:
                    logger.warning(
                        f"Negative severity {severity} at index {idx}, treating as 0"
                    )
                    severity = 0
                total_severity += severity
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid severity value at index {idx}: {severity}, error: {e}"
                )
                continue

        return -1.0 * total_severity

    except Exception as e:
        logger.error(f"Unexpected error while extracting score: {e}", exc_info=True)
        return None


# The retry here is for the case that cannot extract correctly from the response
def get_reward_from_rm(pred_image_path: str, gt_image_path: str, task_type: str, retry=2) -> float:
    """
    Get reward from RM using singleton client.
    """
    # Get singleton client and model name
    model_client = get_rm_client()
    model_name = get_rm_model_name()

    # user_prompt = PROMPT_PYTHON_CODE_RENDER_JUDGEMENT
    if "icon2svg" in task_type:
        user_prompt = PROMPT_IMG2SVG_JUDGEMENT
    elif "chart2code" in task_type:
        user_prompt = PROMPT_CHART2CODE_JUDGEMENT
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    messages = message_format(user_prompt, [gt_image_path, pred_image_path])

    num_retries = 0
    while num_retries < retry:
        rm_response = run_once_with_prompt_single_turn(
            model_client,
            model_name,
            messages,
            **DEFAULT_MODEL_KWARGS,
        )
        rm_score = extract_score_from_rm_response(
            rm_response.choices[0].message.content
        )
        if rm_score is not None:
            logger.info(f"rm_response:\n{rm_response.choices[0].message.content}")
            logger.info(f"rm_score:\n{rm_score}")
            return rm_score
        else:
            num_retries += 1
            continue
    # RM's failure, so use avg score instead of min score
    # return REWARD_AVG
    return SEVERITY_AVG


# def extract_python_code(content: str) -> Optional[str]:
#     """
#     Extract the LAST python code block from markdown content.
#     Only supports ```python ... ``` blocks.
#     """
#     if not content:
#         return None

#     pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
#     matches = pattern.findall(content)

#     if not matches:
#         return None

#     # IMPORTANT: use the LAST code block
#     return matches[-1].strip()
SUPPORTED_LANGS = ["python", "svg"]

def extract_last_code_block(predict_str: str) -> Tuple[Optional[str], Optional[str]]:
    if not predict_str:
        return None, None

    pattern = re.compile(r"```([a-zA-Z0-9_+-]+)\s*(.*?)\s*```", re.DOTALL)
    matches = list(pattern.finditer(predict_str))
    if not matches:
        return None, None

    for m in reversed(matches):
        lang = (m.group(1) or "").strip().lower()
        code = (m.group(2) or "").strip()
        if lang in SUPPORTED_LANGS and code:
            if lang == "tikz":
                lang = "latex"
            return lang, code

    return None, None

def fix_svg(svg_content):
    # fix the svg tag if not closed
    svg_content = re.sub(r'<[^>]*$', '', svg_content)
    if not svg_content.strip().endswith('</svg>'):
        svg_content += '</svg>'
    return svg_content

def robust_extract_code(text):
    """
    Robust extraction logic:
    1. try to extract python/svg code from markdown block
    2. if not found, fallback to extract svg tag
    """
    # 1. try to extract python/svg code from markdown block
    lang, code = extract_last_code_block(text)
    if code:
        return lang, code

    # 2. fallback to extract svg tag
    pattern = re.compile(r'<svg[^>]*>.*', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return "svg", fix_svg(match.group(0))
    
    return None, None

def is_valid_image(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) <= 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()  # only verify file structure, not decode
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        print(f"invalid image path: {path}")
        return False

def is_white_image(image_path):
    try:
        with open(image_path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        return np.all(img_array == 255)
    except Exception as e:
        print(f"Error checking white image: {e}")
        return False

def render_python_code(
    code: str, save_path: str, timeout: int = 15
) -> Tuple[bool, str]:
    """
    Modifies the code to save image to `save_path` and executes it.
    Returns (success: bool, message: str)
    """
    # Create a temporary Python file
    # uuid is used to avoid conflicts between steps, may be sampled repeatedly
    image_temp_file_path = save_path
    python_temp_file_path = image_temp_file_path.replace(".png", ".py")

    # --- Code Modification Logic ---
    modified_code = code

    # Ensure matplotlib is imported if we are going to use plt
    if "matplotlib.pyplot" not in modified_code and "plt." in modified_code:
        modified_code = "import matplotlib.pyplot as plt\n" + modified_code

    # ---- remove all matplotlib save/show ----
    modified_code = re.sub(r'plt\.savefig\s*\(.*?\)', '# removed by render hack', modified_code, flags=re.DOTALL)
    modified_code = re.sub(r'fig\.savefig\s*\(.*?\)', '# removed by render hack', modified_code, flags=re.DOTALL)
    modified_code = re.sub(r'plt\.show\s*\(\s*\)', '# removed by render hack', modified_code)
    modified_code = re.sub(r'fig\.show\s*\(\s*\)', '# removed by render hack', modified_code)

    # ---- special libraries (KEEP) ----
    if "rdkit" in code:
        pattern = r"(drawer\.WriteDrawingText\s*\()(.*?)(\))"
        replacement = rf'\1"{image_temp_file_path}"\3'
        modified_code = re.sub(pattern, replacement, code)

    elif "indigo" in code:
        pattern = r"(renderer\.renderToFile\s*\(\s*[^,]+,\s*)(.*?)(\))"
        replacement = rf'\1"{image_temp_file_path}"\3'
        modified_code = re.sub(pattern, replacement, code)

    elif "fig.write_image" in code:
        pattern = r'(fig\.write_image\s*\()([\'"].*?[\'"]|[a-zA-Z0-9_]+)'
        modified_code = re.sub(pattern, f'\\1r"{image_temp_file_path}"', code)

    # ---- final unified save ----
    modified_code += f"""
# ===== render hack final save =====
import matplotlib.pyplot as plt
plt.savefig(r"{image_temp_file_path}")"""

    # --- Write modified code to temporary file ---
    with open(python_temp_file_path, "w", encoding="utf-8") as f:
        f.write(modified_code)

    # --- Execute modified code ---
    try:
        # Execute in a subprocess
        result = subprocess.run(
            [sys.executable, python_temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        # Check if image file exists as a success criteria
        if is_valid_image(image_temp_file_path):
            return True, "Success"
        else:
            error_msg = (
                result.stderr if result.stderr else "No stderr, but file not created."
            )
            return False, f"Execution failed. Stderr: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, f"Exception: {str(e)}"

# def render_svg_code(code: str, save_path: str) -> Tuple[bool, str]:
#     """
#     Aligned with Code 2:
#     # - Fixed output size 336x336
#     - Fixed output size 672x672
#     - White background
#     """
#     if not save_path.lower().endswith(".png"):
#         save_path += ".png"
#     try:
#         # cairosvg.svg2png(
#         #     bytestring=code.encode("utf-8"), 
#         #     write_to=save_path,
#         #     output_height=336,
#         #     output_width=336,
#         #     background_color="white",
#         # )
#         logger.info(f"Start rendering SVG code to {save_path}")
#         cairosvg.svg2png(
#             bytestring=code.encode("utf-8"), 
#             write_to=save_path,
#             output_height=672,
#             output_width=672,
#             background_color="white",
#         )
#         logger.info(f"Finished rendering SVG code to {save_path}")
#         # Code 2 logic: exists and size > 0
#         if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
#             return True, "Success"
#         else:
#             return False, "Render failed: Empty Output File"
#     except Exception as e:
#         logger.info(f"Failed to render SVG code: {e}")
#         return False, str(e)

def _svg_worker(code, save_path, result_dict):
    """Isolated worker for SVG rendering."""
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=code.encode("utf-8"),
            write_to=save_path,
            output_height=336,
            output_width=336,
            background_color="white",
        )
        result_dict['success'] = os.path.exists(save_path) and os.path.getsize(save_path) > 0
    except Exception as e:
        result_dict['success'] = False
        result_dict['error'] = str(e)

def render_svg_code(code: str, save_path: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Render SVG with a hard timeout using multiprocessing to prevent distributed training hangs.
    """
    if not save_path.lower().endswith(".png"):
        save_path += ".png"

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    
    # Use 'spawn' or default starting method
    process = multiprocessing.Process(
        target=_svg_worker, 
        args=(code, save_path, result_dict)
    )
    
    try:
        process.start()
        logger.info(f"Started SVG rendering process for {save_path}")
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            logger.error(f"SVG render timeout for {save_path}")
            return False, "Timeout"
        
        if result_dict.get('success', False):
            logger.info(f"SVG rendering process for {save_path} finished successfully")
            return True, "Success"
        else:
            err = result_dict.get('error', "Render failed: Empty Output File")
            logger.error(f"SVG rendering process for {save_path} failed: {err}")
            return False, err
            
    except Exception as e:
        if process.is_alive():
            process.terminate()
        logger.error(f"SVG rendering process for {save_path} encountered an error: {str(e)}")
        return False, f"Process error: {str(e)}"


def render_by_language(lang: str, code: str, idx: str) -> Tuple[bool, str, Optional[str]]:
    img_path = os.path.join(TEMP_SAVE_DIR, lang, f"{idx}_{uuid.uuid4().hex}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if lang == "python":
        ok, msg = render_python_code(code, img_path)
    elif lang == "svg":
        ok, msg = render_svg_code(code, img_path)
    # elif lang == "html":
    #     ok, msg = render_html_code(code, img_path)
    # elif lang == "latex":
    #     ok, msg = render_latex_code(code, img_path)
    else:
        return False, None, f"Unsupported lang: {lang}"

    return (True, img_path, msg) if ok else (False, None, msg)

def severity_to_reward(severity: float) -> float:
    reward = 1.0 + severity / MAX_SEVERITY
    reward = np.clip(reward, 0.0, 1.0)
    reward *= RM_REWARD_SCALE  
    return reward

# extra_info:
# - "gt_img_path": str, path to the ground truth image
# - "idx": str, index for the data item
# "ground_truth" [Temperarily not included in extra_info, not used for now], gt_code is currently not used for now
def compute_score(
    predict_str: str, ground_truth: str, extra_info: dict
) -> float:
    # logger.info(f"Start computing score for:\n{predict_str}")
    try:
        if extra_info is None or "idx" not in extra_info or "gt_img_path" not in extra_info or "task_type" not in extra_info:
            # logger.warning("extra_info is None or missing 'idx' or 'gt_img_path' key")
            raise ValueError("extra_info is None or missing 'idx' or 'gt_img_path' or 'task_type' key")

        # Extract Python code from predict_str
        # python_code = extract_python_code(predict_str)
        # lang, code = extract_last_code_block(predict_str)
        lang, code = robust_extract_code(predict_str)
        if not code:
            logger.info(f"No code found for {extra_info['idx']}, return min reward")
            return REWARD_MIN

        # Render the code
        # success, msg, pred_image_path = render_python_code(
        #     python_code, extra_info["idx"]
        # )
        success, pred_image_path, msg = render_by_language(lang, code, extra_info["idx"])
        logger.info(f"Render by language result: success: {success}, pred_image_path: {pred_image_path}, msg: {msg}")
        if not success:
            logger.info(f"Render failed for {extra_info['idx']}, return min reward")
            return REWARD_MIN
        
        # specific for svg, if the image is white, return min reward
        # if is_white_image(pred_image_path):
        #     logger.info(f"White image detected for {extra_info['idx']}, return min reward")
        #     return REWARD_SVG_MIN
        if is_white_image(pred_image_path):
            logger.info(f"White image detected for {extra_info['idx']}, skipping RM.")
            
            # white_reward = severity_to_reward(WHITE_SEVERITY)
            
            # logger.info(f"Returning simulated score for white image: {white_reward}")
            # return white_reward
            return REWARD_MIN
        else:
            final_reward = BASE_RENDER_SUCCESS_REWARD
            logger.info(f"Start getting reward from RM for {extra_info['idx']}")

            reward_score_from_rm = get_reward_from_rm(
                pred_image_path, extra_info["gt_img_path"], extra_info["task_type"]
            )
            logger.info(f"Reward from RM for {extra_info['idx']}: {reward_score_from_rm}")
            final_reward += severity_to_reward(reward_score_from_rm)
            logger.info(f"Final reward for {extra_info['idx']}: {final_reward}")
            return final_reward

    except Exception as e:
        logger.error(f"Error in compute_score: {e}")
        return REWARD_MIN


