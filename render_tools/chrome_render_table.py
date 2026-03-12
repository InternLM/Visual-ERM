import os
import json
import uuid
import traceback
import multiprocessing as mp
from tqdm import tqdm

from html2image import Html2Image
from PIL import Image, ImageChops


JSON_IN = "./qwen3vl8b_253.json"
JSON_OUT = "./qwen3vl8b_253_with_table_imgs.json"
IMG_DIR = "./render_images/test_100_images_verl_grpo_table_40k_teds_32gpus"

# ======== global ========
_HTI = None

def init_worker():
    """每个进程初始化一次 Html2Image，避免重复启动 Chrome。"""
    global _HTI
    _HTI = Html2Image(
        output_path=IMG_DIR,  # 截图保存的文件夹位置
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

def wrap_table_html(table_html_or_md: str) -> str:
    """
    把 table html 包进完整 HTML 文档并加上样式。
    注意：这里假设你传进来的就是可渲染的 <table>...</table> 或者包含 table 的 html。
    """
    return f"""
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
td, th {{
    border: 1px solid #333;
    padding: 6px 10px;
}}
</style>
</head>
<body>
{table_html_or_md}
</body>
</html>
"""

def crop_with_padding(image_path, output_path, padding=12):
    img = Image.open(image_path).convert("RGB")

    # 以左上角颜色作为背景色
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

def render_table_html_to_image(table_html: str, final_img_path: str, size=(3600, 2400), padding=12) -> bool:
    """
    用 html2image 渲染 table -> raw 临时图 -> 裁剪白边并保存到 final_img_path
    成功返回 True，否则 False
    """
    global _HTI
    if _HTI is None:
        # 保险：万一没走 initializer，也能工作（只是慢）
        init_worker()

    html = wrap_table_html(table_html)

    # 生成一个同目录临时 raw 文件名，避免并发冲突
    tmp_raw = f"raw_{os.path.basename(final_img_path)}_{uuid.uuid4().hex}.png"
    tmp_raw_path = os.path.join(
        os.path.dirname(final_img_path),
        tmp_raw
    )

    try:
        _HTI.screenshot(
            html_str=html,
            save_as=tmp_raw,
            size=size,
        )
        crop_with_padding(tmp_raw_path, final_img_path, padding=padding)
        return True
    except Exception as e:
        # 你需要更详细日志就打开下面这一行
        # traceback.print_exc()
        print(f"Screenshot Error:{e}")
        return False
    finally:
        try:
            if os.path.exists(tmp_raw_path):
                os.remove(tmp_raw_path)
        except Exception as e:
            print(f"Remove Error:{e}")
            pass

def process_one(args):
    """
    worker：处理一个样本（orig + corrupted），返回 (idx, updated_item)
    """
    idx, item, img_dir = args

    table_md_list = item.get("table_md", [])
    table_md_corr_list = item.get("table_md_corrupted", [])

    # 兼容 list / str
    if isinstance(table_md_list, list) and table_md_list:
        html_orig = table_md_list[0]
    elif isinstance(table_md_list, str):
        html_orig = table_md_list
    else:
        html_orig = None

    if isinstance(table_md_corr_list, list) and table_md_corr_list:
        html_corr = table_md_corr_list[0]
    elif isinstance(table_md_corr_list, str):
        html_corr = table_md_corr_list
    else:
        html_corr = None

    base_name = f"sample_{idx:03d}"
    orig_img_path = os.path.join(img_dir, f"{base_name}_orig.png")
    corr_img_path = os.path.join(img_dir, f"{base_name}_corrupted.png")

    # orig
    if html_orig:
        ok_orig = render_table_html_to_image(html_orig, orig_img_path)
        item["table_md_img_path"] = orig_img_path if ok_orig else None
    else:
        item["table_md_img_path"] = None

    # corrupted
    if html_corr:
        ok_corr = render_table_html_to_image(html_corr, corr_img_path)
        item["table_md_corrupted_img_path"] = corr_img_path if ok_corr else None
    else:
        item["table_md_corrupted_img_path"] = None

    return idx, item

def main():
    assert os.path.exists(JSON_IN), f"输入 JSON 不存在: {JSON_IN}"

    with open(JSON_IN, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON 根节点应为 list，实际为: {type(data)}")

    os.makedirs(IMG_DIR, exist_ok=True)

    # 多进程参数
    ctx = mp.get_context("spawn")  # html2image/Chrome 场景下 spawn 更稳
    num_workers = 16
    chunksize = 5  # 每个任务渲染两张图，chunksize 不要太大

    tasks = [(i, data[i], IMG_DIR) for i in range(len(data))]

    results = {}
    with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
        for idx, item in tqdm(pool.imap_unordered(process_one, tasks, chunksize=chunksize), total=len(tasks)):
            results[idx] = item

    # 保持输出顺序
    new_data = [results[i] for i in range(len(data))]

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 所有表格已处理完毕，结果保存到: {JSON_OUT}")
    print(f"✅ 渲染图片保存目录: {IMG_DIR}")

if __name__ == "__main__":
    main()
