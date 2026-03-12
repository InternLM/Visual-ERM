import os, re, json, uuid, base64
from datetime import datetime
from PIL import Image, ImageChops
from html2image import Html2Image
from openai import OpenAI

# -------------------------
# Per-process singletons
# -------------------------
_HTI = None
_HTI_PID = None
_OUT_DIR = None

_VLM_CLIENTS = None
_VLM_PID = None

openai_api_key = "EMPTY"

# # MODEL_PATH=Qwen3-VL-8B-Instruct-full-sft-e3-mix-table-125k-chart-104k-svg-111k-rm
# VLM_BASE_URLS = [
#     "http://100.100.60.142:8020/v1",
#     "http://100.100.60.143:8021/v1",
#     "http://100.97.19.161:8022/v1",
#     "http://100.97.19.162:8023/v1",
#     "http://100.96.204.132:8024/v1",
#     "http://100.96.204.133:8025/v1",
#     "http://100.96.99.205:8026/v1",
#     "http://100.96.99.206:8027/v1",
# ]

# MODEL_PATH=Qwen3-VL-8B-Instruct-full-sft-e3-mix-table-125k-chart-104k-svg-111k-rm-v2
VLM_BASE_URLS = [
    "http://100.103.181.131:8020/v1",
    "http://100.103.181.132:8021/v1",
    "http://100.99.23.103:8022/v1",
    "http://100.99.23.104:8023/v1",
]

def _get_hti():
    global _HTI, _HTI_PID, _OUT_DIR
    pid = os.getpid()
    if _HTI is None or _HTI_PID != pid:
        _OUT_DIR = f"/mnt/shared-storage-user/liuziyu/DocReward/verl/temp_imgs_v2/p{pid}"
        os.makedirs(_OUT_DIR, exist_ok=True)
        _HTI = Html2Image(
            output_path=_OUT_DIR,
            browser_executable="/usr/bin/google-chrome",
            custom_flags=[
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-features=TranslateUI",
                "--disable-extensions",
                "--mute-audio",
                "--log-level=3",
                "--disable-logging",
            ],
        )
        _HTI_PID = pid
    return _HTI, _OUT_DIR


def _get_vlm_client(index: int) -> OpenAI:
    """lazy init per-process VLM clients, avoid heavy init during module import"""
    global _VLM_CLIENTS, _VLM_PID
    pid = os.getpid()
    if _VLM_CLIENTS is None or _VLM_PID != pid:
        _VLM_CLIENTS = [
            OpenAI(api_key=openai_api_key, base_url=url) for url in VLM_BASE_URLS
        ]
        _VLM_PID = pid
    return _VLM_CLIENTS[index % len(_VLM_CLIENTS)]



# ---------- 函数实现 ------------
def vlm_client_qa(vlm_client, serve_name, prompt, image_path, max_new_tokens=512, temperature=0.0):
    if image_path != None:
        assert not type(prompt) is list
        if type(image_path) is not list:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read())
            encoded_image_text = encoded_image.decode("utf-8")
            base64_qwen = f"data:image;base64,{encoded_image_text}"

            chat_response = vlm_client.chat.completions.create(
                model=serve_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_qwen
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            # print("Chat response:", chat_response)
            output_text = chat_response.choices[0].message.content
            return output_text
        else:
            assert len(image_path)==2
            base64_qwen_list = []
            for image_path_item in image_path:
                with open(image_path_item, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                encoded_image_text = encoded_image.decode("utf-8")
                base64_qwen = f"data:image;base64,{encoded_image_text}"
                base64_qwen_list.append(base64_qwen)

            chat_response = vlm_client.chat.completions.create(
                model=serve_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_qwen_list[0]
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_qwen_list[1]
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            # print("Chat response:", chat_response)
            output_text = chat_response.choices[0].message.content
            return output_text

# # 第一版本的 PT
# def build_error_prompt_for_images() -> str:
#     prompt = """
# You are a table parsing quality auditor. You will be given two table images:

# - Image 1: the original table screenshot (original)
# - Image 2: the parsed / rendered table image (parsed)

# Your job is to compare the original vs. parsed images and identify all discrepancies in the parsed table relative to the original, then summarize them in a strict JSON format.

# You MUST output a single JSON object ONLY.
# - Do NOT include any extra text.
# - Do NOT wrap the JSON in markdown code fences like ```json.
# - The JSON schema MUST match exactly:

# {
#   "layout_error_count": int,
#   "text_error_count": int,
#   "numeric_error_count": int,
#   "errors": [
#     {
#       "type": "layout_error | text_error | numeric_error",
#       "description": "A short description of what is wrong and where it is wrong",
#       "severity": 1|2|3
#     }
#   ]
# }
# """
#     return prompt

def build_error_prompt_for_images() -> str:
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


def calculate_error_score(error_analysis):
    """计算单个条目的错误总分"""
    total_severity = 0
    
    try:
        # 如果 error_analysis 为空字典，则直接返回 None
        if not error_analysis:
            return None
        
        errors = error_analysis["errors"]
        for error in errors:
            severity = error.get("severity", 0)
            total_severity += severity
            
        return total_severity
    except Exception as e:
        return None


def crop_with_padding(image_path, output_path, padding=12):
    img = Image.open(image_path).convert("RGB")

    # 以左上角像素作为背景色
    bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox:
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(img.width, right + padding)
        lower = min(img.height, lower + padding)
        img = img.crop((left, upper, right, lower))
    img.thumbnail((768, 768))
    img.save(output_path)


def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    # ---------- 渲染解析之后的图片并暂存 ------------
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    body {{
        background: white;
        margin: 20px;
    }}
    table {{
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }}
    td {{
        border: 1px solid #333;
        padding: 6px 10px;
    }}
    </style>
    </head>
    <body>
    {predict_str}
    </body>
    </html>
    """

    try: 
        start_time = datetime.now()
        # ---------- 调用hti渲染图片 ------------
        hti, out_dir = _get_hti()
        # print(out_dir)
        uid = uuid.uuid4().hex
        raw_img = f"table_raw_{uid}.png"
        final_img = f"table_final_{uid}.png"
        hti.screenshot(
            html_str=html,
            save_as=raw_img,
            size=(3600, 2400),
        )
        raw_img = out_dir + '/' + raw_img
        final_img = out_dir + '/' + final_img
        crop_with_padding(raw_img, final_img, padding=12)

        render_time = datetime.now()

        # ---------- 调用service对输出给reward ------------
        # 根据 extra_info 里面的 index, 把数据均分到不同 vllm service 上
        index = int(extra_info["index"])
        vlm_client = _get_vlm_client(index)
        prompt = build_error_prompt_for_images()

        # prompt = "Original table:<image>\nParsed table:<image>\n" + prompt
        prompt = "<image>\n<image>\n" + prompt
        
        orig_img = extra_info["image_ori"]

        # ---- 新增：resize orig_img ----
        img = Image.open(orig_img).convert("RGB")
        img.thumbnail((768, 768))  # 等比例缩放，最长边不超过 768

        # 生成临时路径
        tmp_dir = out_dir  # 或你自己的 temp 目录
        resized_orig_img = os.path.join(
            tmp_dir, f"orig_resized_{uuid.uuid4().hex}.png"
        )
        img.save(resized_orig_img)

        resize_time = datetime.now()

        resp = vlm_client_qa(vlm_client, "reward_service", prompt, image_path=[resized_orig_img, final_img], max_new_tokens=8192)
        # print(resp)

        rm_infer_time = datetime.now()

        # 用完之后删除图片
        for img_path in [raw_img, final_img, resized_orig_img]:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Failed to delete {img_path}: {e}")

        delete_img_time = datetime.now()

        reward_score = None
        obj, raw = parse_error_response(resp)
        reward_score = calculate_error_score(obj)
        print(reward_score)

        parse_time = datetime.now()

        # print(
        #     f"""
        #     time breakdown (s):
        #     render        : {(render_time - start_time).total_seconds():.4f}
        #     resize        : {(resize_time - render_time).total_seconds():.4f}
        #     rm_infer      : {(rm_infer_time - resize_time).total_seconds():.4f}
        #     delete_img    : {(delete_img_time - rm_infer_time).total_seconds():.4f}
        #     parse         : {(parse_time - delete_img_time).total_seconds():.4f}
        #     total         : {(parse_time - start_time).total_seconds():.4f}
        #     """
        # )

        # # 如果解析失败，给一个适中的错误值
        if reward_score == None:
            print("########## Parse Error ##########")
            reward_score = 4
        
        return -reward_score

    except Exception as e:
        print("############ Reward Error ############")
        print(e)
        reward_score = 4
        return -reward_score