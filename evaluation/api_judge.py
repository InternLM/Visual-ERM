import json
import os
import re
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import multiprocessing as mp
from openai import OpenAI
import base64

# ---------------- 配置部分 ----------------
api_key = ""
base_url = ""
client = OpenAI(api_key=api_key, base_url=base_url)
# ------------------------------------------

def call_api(query, image_paths):
    """调用 API 模型"""
    base64_images = []
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                img_data = f.read()
                base64_images.append(base64.b64encode(img_data).decode("utf-8"))
        except Exception as e:
            print(f"[警告] 无法读取图像: {img_path}, {e}")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                }
                for b64 in base64_images
            ],
        ],
    })

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            # model="gpt-4.1",
            # model="gpt-5-mini",
            # model="gemini-3-pro-preview",
            # model="gemini-3-flash-preview",
            # model="gpt-5.2-2025-12-11",
            # model="gpt-4o-2024-08-06",
            # model="qwen3-vl-235b-a22b-instruct",
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[错误] API调用失败: {e}")
        return ""

# ---------- 加载数据 ------------
path = "./VC-RewardBench.jsonl"
data = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data.append(json.loads(line))
print(len(data))
num_workers = 64
save_path = f"./results/results.json"


def build_prompt_for_table() -> str:
    prompt = """
You are a table parsing quality auditor. You will be given two table images:

- Image 1: the original table screenshot (original)
- Image 2: the parsed / rendered table image (parsed)

Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed table relative to the original, then summarize them in a strict JSON format.

For every discrepancy, assign exactly ONE of the following categories:
1) layout_error (structure/layout errors)
2) text_error (text recognition errors)
3) numeric_error (numeric/symbol/unit errors)

Assign a severity score for each error:
- 1 (minor): small errors that barely affect readability or understanding
- 2 (medium): errors that affect partial understanding and require manual correction for reliable use
- 3 (severe): structural or key-content errors that break reliable alignment or can significantly mislead

You MUST output a single JSON object ONLY.
- Do NOT include any extra text.
- Do NOT wrap the JSON in markdown code fences like ```json.
- The JSON schema MUST match exactly:

{
  "layout_error_count": int,
  "text_error_count": int,
  "numeric_error_count": int,
  "errors": [
    {
      "type": "layout_error | text_error | numeric_error",
      "description": "A short description of what is wrong and where it is wrong",
      "severity": 1|2|3
    }
  ]
}
"""
    return prompt


def build_prompt_for_chart() -> str:
    prompt = """\
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
    return prompt

def build_prompt_for_svg() -> str:
    prompt = """\
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
    return prompt

def parse_error_response(resp: str):
    """
    解析模型输出的错误分析 JSON。
    允许有 ```json ... ``` 包裹。
    返回 (obj, raw_text)。解析失败则 obj=None。
    """
    if resp is None:
        return None, ""

    if isinstance(resp, list):
        resp = "".join(str(x) for x in resp)

    raw = str(resp).strip()

    # 去掉 ```json ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    # 尝试直接 json.loads
    try:
        return json.loads(raw), raw
    except Exception:
        # 抽取 {...}
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None, raw
        try:
            return json.loads(m.group(0)), raw
        except Exception:
            return None, raw


# ---------- 定义多进程子任务函数 ------------
def process_item(item):
    new_item = dict(item)
    orig_img = new_item["gt_img_path"]
    parsed_img = new_item["pred_img_path"]
    task_type = new_item["category"]

    if task_type == "table":
        prompt = build_prompt_for_table()
    elif task_type == "chart":
        prompt = build_prompt_for_chart()
    elif task_type == "svg":
        prompt = build_prompt_for_svg()
    else:
        print("Task type undefined")

    # 添加了前缀  prompt = "Original table:<image>\nParsed table:<image>\n" + prompt
    prompt = "<image>\n<image>\n" + prompt

    try:
        resp = call_api(prompt, image_paths=[orig_img, parsed_img])
    except Exception as e:
        print("############ Error #################")
        new_item["pred_json"] = None
        new_item["error_parse_failed_reason"] = f"调用 模型API 失败: {e}"
        return new_item

    obj, raw = parse_error_response(resp)
    if obj is None:
        print("############ Error #################")
        new_item["pred_json"] = None
        new_item["error_parse_failed_reason"] = f"解析 JSON 失败，原始回复: {raw}"
        return new_item

    # 写入新字段
    new_item["pred_json"] = obj
    new_item["error_parse_failed_reason"] = None

    keys_to_remove = ["pred_json_gemini_2_5_pro", "pred_json_gemini_3_pro_preview", "pred_json_gpt_5_mini"]  # x 不存在也没关系
    for k in keys_to_remove:
        new_item.pop(k, None)

    return new_item

# ---------- 并行处理数据 ------------
with mp.Pool(processes=int(num_workers)) as pool:  # 可根据你电脑核数调整
    results = list(tqdm(pool.imap(process_item, data), total=len(data)))

# ---------- 保存结果 ------------
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)