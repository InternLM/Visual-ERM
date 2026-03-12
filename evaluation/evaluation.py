import json
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm
import argparse
import base64
from openai import OpenAI
from multiprocessing import Pool, cpu_count
import traceback
from typing import Any, Dict, List, Tuple, Optional


# =========================
# 0) 你需要提供的 LLM 接口
# =========================

# ---------------- 配置部分 ----------------
api_key = ""
base_url = ""
client = OpenAI(api_key=api_key, base_url=base_url)
# ------------------------------------------

def call_api(query, image_paths=None):
    """调用 API 模型"""
    if image_paths!=None:
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
    else:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query}
            ],
        })


    try:
        response = client.chat.completions.create(
            # model="gemini-2.5-pro",
            # model="gpt-4.1",
            model="gpt-5-mini",
            # model="gemini-3-pro-preview",
            # model="gemini-3-flash-preview",
            # model="gpt-5.2-2025-12-11",
            # model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=8192,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[错误] API调用失败: {e}")
        return ""

# ==========================
# 配置：不同category允许的错误类型
# ==========================
CATEGORY_ERROR_TYPES = {
    "table": ["layout_error", "text_error", "numeric_error"],
    "chart": ["structure_error", "data_error", "text_error", "style_error"],
    "svg": ["structure_error", "shape_error", "style_error", "text_symbol_error"],
}

# ==========================
# 评估：数值工具
# ==========================
def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def f1_score(tp: int, fp: int, fn: int) -> float:
    # 空集合完美情况：gt无错且pred无错
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)

def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """
    Pearson correlation. 若全为常数或样本不足，返回nan。
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")

    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)

def get_err_type(e: Dict[str, Any]) -> Optional[str]:
    return e.get("type") or e.get("category")

# ==========================
# Prompt：一次性让LLM做全量匹配
# ==========================
def build_llm_prompt(
    category: str,
    allowed_types: List[str],
    pred_errors: List[Dict[str, Any]],
    gt_errors: List[Dict[str, Any]],
) -> str:
    """
    让LLM做一次性匹配：
    - 输入pred与gt全部错误
    - 输出匹配 pairs + 未匹配 pred/gt
    - 必须严格：只允许同type匹配，否则算不匹配
    """

    def normalize_err(e: Dict[str, Any]) -> Dict[str, Any]:
        err_type = get_err_type(e)
        return {
            "type": err_type if err_type is not None else "unknown",
            "description": e.get("description", ""),
            "severity": e.get("severity", None),
        }

    pred_list = [{"pred_id": i, **normalize_err(e)} for i, e in enumerate(pred_errors)]
    gt_list = [{"gt_id": i, **normalize_err(e)} for i, e in enumerate(gt_errors)]

    prompt = f"""
你是一个严格的“错误对齐评估器”。你的任务是：对比 PRED 的错误列表与 GT 的错误列表，判断它们哪些是同一个具体错误点，并输出结构化JSON。

【数据类别】
{category}

【允许的错误类型（只能在这些类型中匹配）】
{allowed_types}

【匹配规则（非常重要）】
1) 只有当 pred.type == gt.type 或者 pred.category == gt.category 时，才允许匹配，否则必须判断为不匹配。
2) “匹配”要求描述指向同一个具体错误点（同一个位置/对象/单元格/图表元素），并且错误现象一致或高度一致。
3) 如果 pred 描述比 gt 更泛化，但明显指向同一错误点，可以算 match_level="partial"，否则为 "no"。
4) 每条 pred 最多匹配一个 gt，每条 gt 最多匹配一个 pred（1-1匹配）。
5) 不能乱配：宁可不匹配，也不要错误匹配。

【你的输出必须是严格合法的JSON，不能包含多余文本】
输出JSON格式如下：
{{
  "matches": [
    {{
      "pred_id": 0,
      "gt_id": 3,
      "match_level": "yes" | "partial"
    }}
  ],
  "unmatched_pred": [1, 2],
  "unmatched_gt": [0, 4],
  "notes": "可选，最多一句话"
}}

【PRED错误列表】
{json.dumps(pred_list, ensure_ascii=False, indent=2)}

【GT错误列表】
{json.dumps(gt_list, ensure_ascii=False, indent=2)}
""".strip()

    return prompt

# ==========================
# 解析LLM输出
# ==========================
def parse_llm_output(raw: str) -> Dict[str, Any]:
    """
    要求LLM输出为JSON。这里做强健解析。
    """
    raw = raw.strip()
    # 有些模型可能会包 ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        # 再尝试找第一段JSON
        idx = raw.find("{")
        if idx >= 0:
            raw = raw[idx:]

    # 截取最外层JSON（防止混入多余文字）
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last >= 0 and last > first:
        raw = raw[first:last + 1]

    return json.loads(raw)

# ==========================
# 评估单条数据（一次LLM调用）
# ==========================
def evaluate_one_item_with_llm(
    item: Dict[str, Any],
    call_api,
) -> Dict[str, Any]:
    """
    对单条数据做：
    - 调用LLM进行pred vs gt匹配
    - 计算每个错误类型 + 总体 TP/FP/FN/F1
    - 统计 pred & gt severity sums（按类型 + 总）
    """
    category = item.get("category")
    allowed_types = CATEGORY_ERROR_TYPES.get(category, [])

    gt_errors = item.get("human_errors", []) or []
    pred_errors = (item.get("pred_json", {}) or {}).get("errors", []) or []

    # 过滤掉不在allowed_types的错误（防脏数据）
    gt_errors = [e for e in gt_errors if e.get("type") in allowed_types]
    ### lzy modify, 由于error中的错误类型的key不对齐，table用的是type，chart和svg用的是category
    # pred_errors = [e for e in pred_errors if e.get("type") in allowed_types]
    pred_errors = [
        e for e in pred_errors
        if (get_err_type(e)) in allowed_types
    ]

    prompt = build_llm_prompt(category, allowed_types, pred_errors, gt_errors)
    raw_resp = call_api(prompt)
    llm_out = parse_llm_output(raw_resp)

    matches = llm_out.get("matches", []) or []
    unmatched_pred = set(llm_out.get("unmatched_pred", []) or [])
    unmatched_gt = set(llm_out.get("unmatched_gt", []) or [])

    # 对 matches 做合法性与1-1约束的二次清洗
    used_pred = set()
    used_gt = set()
    clean_matches = []
    for m in matches:
        pid = m.get("pred_id")
        gid = m.get("gt_id")
        level = m.get("match_level", "no")

        if level not in ("yes", "partial"):
            continue
        if not isinstance(pid, int) or not isinstance(gid, int):
            continue
        if pid < 0 or pid >= len(pred_errors):
            continue
        if gid < 0 or gid >= len(gt_errors):
            continue

        ### 部分 error type (error category) 其实语义有重复，这里放宽一些，不需要type(category)归类一致
        # # 强制同type才能匹配
        # if pred_errors[pid].get("type") != gt_errors[gid].get("type"):
        #     continue

        # 强制1-1
        if pid in used_pred or gid in used_gt:
            continue

        used_pred.add(pid)
        used_gt.add(gid)
        clean_matches.append({"pred_id": pid, "gt_id": gid, "match_level": level})

    # 如果LLM没给unmatched，我们自己补全（更稳）
    all_pred_ids = set(range(len(pred_errors)))
    all_gt_ids = set(range(len(gt_errors)))
    matched_pred_ids = set(m["pred_id"] for m in clean_matches)
    matched_gt_ids = set(m["gt_id"] for m in clean_matches)

    if not llm_out.get("unmatched_pred"):
        unmatched_pred = all_pred_ids - matched_pred_ids
    if not llm_out.get("unmatched_gt"):
        unmatched_gt = all_gt_ids - matched_gt_ids

    # --- 计算TP/FP/FN（支持partial软TP） ---
    # 软匹配：partial算0.5TP（你也可以改成1或0）
    soft_tp = 0.0
    hard_tp = 0
    for m in clean_matches:
        if m["match_level"] == "yes":
            soft_tp += 1.0
            hard_tp += 1
        else:
            soft_tp += 0.5

    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    # 总体F1（软/硬两套）
    hard_f1 = f1_score(hard_tp, fp, fn)
    # soft F1：把soft_tp当tp使用（不是整数也OK）
    soft_precision = safe_div(soft_tp, soft_tp + fp)
    soft_recall = safe_div(soft_tp, soft_tp + fn)
    soft_f1 = safe_div(2 * soft_precision * soft_recall, soft_precision + soft_recall)

    # --- 子类型指标 ---
    # 按类型统计TP/FP/FN
    type_metrics = {}
    for t in allowed_types:
        # 匹配到的该类型tp
        t_soft_tp = 0.0
        t_hard_tp = 0
        for m in clean_matches:
            pid = m["pred_id"]
            pred_t = get_err_type(pred_errors[pid])
            if pred_t != t:
                continue
            if m["match_level"] == "yes":
                t_soft_tp += 1.0
                t_hard_tp += 1
            else:
                t_soft_tp += 0.5

        # 该类型FP：该类型pred里未匹配的数量
        t_fp = sum(
            1 for pid in unmatched_pred
            if get_err_type(pred_errors[pid]) == t
        )
        # 该类型FN：该类型gt里未匹配的数量
        t_fn = sum(
            1 for gid in unmatched_gt
            if get_err_type(gt_errors[gid]) == t
        )

        t_hard_f1 = f1_score(t_hard_tp, t_fp, t_fn)
        t_soft_precision = safe_div(t_soft_tp, t_soft_tp + t_fp)
        t_soft_recall = safe_div(t_soft_tp, t_soft_tp + t_fn)
        t_soft_f1 = safe_div(2 * t_soft_precision * t_soft_recall, t_soft_precision + t_soft_recall)

        type_metrics[t] = {
            "tp_hard": t_hard_tp,
            "tp_soft": t_soft_tp,
            "fp": t_fp,
            "fn": t_fn,
            "f1_hard": t_hard_f1,
            "f1_soft": t_soft_f1,
        }

    # --- severity sums ---
    def sum_severity_by_type(errs: List[Dict[str, Any]]) -> Dict[str, float]:
        sums = {t: 0.0 for t in allowed_types}
        for e in errs:
            t = get_err_type(e)
            if t not in allowed_types:
                continue
            sev = e.get("severity", 0)
            try:
                sev = float(sev)
            except Exception:
                sev = 0.0
            sums[t] += sev
        sums["__total__"] = sum(sums[t] for t in allowed_types)
        return sums

    pred_sev_sums = sum_severity_by_type(pred_errors)
    gt_sev_sums = sum_severity_by_type(gt_errors)

    return {
        "id": item.get("id"),
        "idx": item.get("idx"),
        "category": category,

        "counts": {
            "pred_error_num": len(pred_errors),
            "gt_error_num": len(gt_errors),
            "match_num": len(clean_matches),
        },

        "matches": clean_matches,  # 可审计
        "overall": {
            "tp_hard": hard_tp,
            "tp_soft": soft_tp,
            "fp": fp,
            "fn": fn,
            "f1_hard": hard_f1,
            "f1_soft": soft_f1,
        },
        "by_type": type_metrics,

        "severity_pred": pred_sev_sums,
        "severity_gt": gt_sev_sums,

        "raw_resp": raw_resp,
    }

# ==========================
# Step1: 按category拆分
# ==========================
def group_by_category(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for item in data:
        cat = item.get("category")
        if cat in CATEGORY_ERROR_TYPES:
            groups[cat].append(item)
        else:
            # 不认识的category直接忽略或收集到unknown
            groups["unknown"].append(item)
    return dict(groups)

# ==========================
# Step2 + Step3: 并行评估全量数据
# ==========================
def evaluate_dataset(
    json_path: str,
    call_api,
    max_workers: int = 16,
) -> Dict[str, Any]:
    # 读json
    print("Reading JSON Files...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("你的json文件顶层应该是一个 list，每个元素是一条样本数据。")

    groups = group_by_category(data)

    # 并行跑每条样本（LLM调用适合线程池）
    results_all = []
    errors = []

    def job(item):
        try:
            return evaluate_one_item_with_llm(item, call_api)
        except Exception as e:
            return {
                "id": item.get("id"),
                "idx": item.get("idx"),
                "category": item.get("category"),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    items_to_process = []
    for cat in ("table", "chart", "svg"):
        items_to_process.extend(groups.get(cat, []))

    print("Evaluation Start...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(job, item) for item in items_to_process]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating (LLM)",
            ncols=100,
        ):
            res = fut.result()
            if res.get("error"):
                errors.append(res)
            else:
                results_all.append(res)


    # ==========================
    # 汇总：F1（按category + 总体）
    # ==========================
    summary = {
        "overall": {},
        "by_category": {},
        "errors": errors,
    }

    # helper: 聚合TP/FP/FN（硬/软）
    def aggregate_metrics(rows: List[Dict[str, Any]], category: Optional[str] = None):
        if category:
            allowed_types = CATEGORY_ERROR_TYPES[category]
        else:
            # overall across all categories: union
            allowed_types = sorted(set(sum(CATEGORY_ERROR_TYPES.values(), [])))

        # 总体
        hard_tp = sum(r["overall"]["tp_hard"] for r in rows)
        soft_tp = sum(r["overall"]["tp_soft"] for r in rows)
        fp = sum(r["overall"]["fp"] for r in rows)
        fn = sum(r["overall"]["fn"] for r in rows)

        hard_f1 = f1_score(hard_tp, fp, fn)
        soft_precision = safe_div(soft_tp, soft_tp + fp)
        soft_recall = safe_div(soft_tp, soft_tp + fn)
        soft_f1 = safe_div(2 * soft_precision * soft_recall, soft_precision + soft_recall)

        # 子类型
        type_agg = {}
        for t in allowed_types:
            # 只有在当前category中才统计
            relevant = []
            for r in rows:
                if t in r["by_type"]:
                    relevant.append(r["by_type"][t])
            if not relevant:
                continue
            t_hard_tp = sum(x["tp_hard"] for x in relevant)
            t_soft_tp = sum(x["tp_soft"] for x in relevant)
            t_fp = sum(x["fp"] for x in relevant)
            t_fn = sum(x["fn"] for x in relevant)

            t_hard_f1 = f1_score(t_hard_tp, t_fp, t_fn)
            t_soft_precision = safe_div(t_soft_tp, t_soft_tp + t_fp)
            t_soft_recall = safe_div(t_soft_tp, t_soft_tp + t_fn)
            t_soft_f1 = safe_div(2 * t_soft_precision * t_soft_recall, t_soft_precision + t_soft_recall)

            type_agg[t] = {
                "tp_hard": t_hard_tp,
                "tp_soft": t_soft_tp,
                "fp": t_fp,
                "fn": t_fn,
                "f1_hard": t_hard_f1,
                "f1_soft": t_soft_f1,
            }

        return {
            "overall": {
                "tp_hard": hard_tp,
                "tp_soft": soft_tp,
                "fp": fp,
                "fn": fn,
                "f1_hard": hard_f1,
                "f1_soft": soft_f1,
            },
            "by_type": type_agg
        }

    # category级别汇总
    results_by_cat = defaultdict(list)
    for r in results_all:
        results_by_cat[r["category"]].append(r)

    for cat in ("table", "chart", "svg"):
        rows = results_by_cat.get(cat, [])
        summary["by_category"][cat] = aggregate_metrics(rows, category=cat)

    # 全体汇总
    summary["overall"] = aggregate_metrics(results_all, category=None)

    # ==========================
    # Step3: Severity相关度（Pearson）
    # 输出：每类数据每类错误相关度、每类数据总相关度、总体总相关度
    # ==========================
    severity_corr = {
        "by_category": {},
        "overall": {},
    }

    def calc_corr_for_rows(rows: List[Dict[str, Any]], category: str):
        allowed_types = CATEGORY_ERROR_TYPES[category]
        out = {"by_type": {}, "total": None}

        # 每个错误类型相关度
        for t in allowed_types:
            xs = [r["severity_pred"].get(t, 0.0) for r in rows]
            ys = [r["severity_gt"].get(t, 0.0) for r in rows]
            out["by_type"][t] = pearson_corr(xs, ys)

        # 总severity相关度
        xs_total = [r["severity_pred"].get("__total__", 0.0) for r in rows]
        ys_total = [r["severity_gt"].get("__total__", 0.0) for r in rows]
        out["total"] = pearson_corr(xs_total, ys_total)
        return out

    for cat in ("table", "chart", "svg"):
        rows = results_by_cat.get(cat, [])
        severity_corr["by_category"][cat] = calc_corr_for_rows(rows, cat)

    # overall total severity corr across all categories
    xs_all = [r["severity_pred"].get("__total__", 0.0) for r in results_all]
    ys_all = [r["severity_gt"].get("__total__", 0.0) for r in results_all]
    severity_corr["overall"]["total"] = pearson_corr(xs_all, ys_all)

    summary["severity_correlation"] = severity_corr

    # 返回最终结构化结果
    return {
        "summary": summary,
        "per_item_results": results_all,   # 每条样本结果（含匹配关系）
        "failed_items": errors,            # 失败样本（解析/LLM异常）
    }

# ==========================
# CLI用法示例
# ==========================
if __name__ == "__main__":
    # 你的json路径
    file_name = "qwen3vl235b_instruct_api"
    print(f"Evaluation for {file_name}")
    json_path = f"./results/{file_name}.json"

    # 跑评估
    results = evaluate_dataset(
        json_path=json_path,
        call_api=call_api,
        max_workers=64,  # 并行线程数（建议按你的LLM QPS 调整）
    )

    # 输出summary
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))

    # 可选：保存完整结果
    with open(f"./results/{file_name}_score.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved!")
