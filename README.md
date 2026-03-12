# Visual-ERM: Reward Modeling for Visual Equivalence

<p align="center">
  <a href="[TODO: arXiv link]">📄 Paper</a> |
  <a href="[TODO: project page link]">🌐 Project Page</a> |
  <a href="[TODO: Hugging Face model link]">🤗 Models</a> |
  <a href="[TODO: dataset link]">📦 Data</a> |
  <a href="[TODO: benchmark link]">📊 VC-RewardBench</a>
</p>

🌈 Official repository for **Visual-ERM**, a multimodal generative reward model for **vision-to-code** tasks.  
Visual-ERM evaluates outputs directly in the **rendered visual space** and provides **fine-grained**, **interpretable**, and **task-agnostic** feedback for structured visual reconstruction, including **chart-to-code**, **table-to-markdown**, and **SVG-to-code**.

## 📢 News

- 🚀 **[2026/03/15]** Release of Visual-ERM training data and VC-RewardBench.
- 🚀 **[2026/03/14]** Release of pretrained Visual-ERM checkpoints.
- 🚀 **[2026/03/13]** Initial release of Visual-ERM codebase.

## 💡 Highlights

- 🔥 **Visual-space reward modeling.** Instead of relying on text-only rules or coarse visual embedding similarity, Visual-ERM evaluates predictions in the rendered visual space.

- 🔥 **Fine-grained and interpretable feedback.** Visual-ERM predicts structured discrepancies with fields such as **category**, **severity**, **location**, and **description**, making reward signals actionable for both training and refinement.

- 🔥 **Task-agnostic reward supervision.** A single reward model generalizes across multiple vision-to-code tasks, including charts, tables, and SVGs.

- 🔥 **Effective for both RL and test-time scaling.** Visual-ERM can be used as a reward model in RL and as a visual critic for reflection-and-revision at inference time.

- 🔥 **A new benchmark for visual discrepancy judgment.** We introduce **VC-RewardBench**, a benchmark for fine-grained image-to-image discrepancy evaluation on structured visual data.

## Overview

Vision-to-code tasks require models to reconstruct structured visual inputs into executable or structured representations with high visual fidelity. However, existing reward designs have major limitations:

- **Text-based rewards** (e.g., edit distance, TEDS) ignore important visual cues such as layout, alignment, spacing, and style.
- **Vision embedding rewards** (e.g., DINO similarity) are often coarse-grained, semantically biased, and vulnerable to reward hacking.

To address this, we propose **Visual Equivalence Reward Model (Visual-ERM)**, a multimodal generative reward model that compares:

- the **ground-truth image**, and
- the **rendered image** from a model prediction,

and then outputs fine-grained discrepancy annotations that can be converted into reward signals or used for reflection-based refinement.

<p align="center">
  <img src="assets/teaser" width="95%">
</p>

## Framework

Visual-ERM consists of three major components:

1. **Reward data generation**  
   We construct image pairs by:
   - editing ground-truth structured outputs to inject controlled errors, and
   - sampling natural errors from weaker model predictions.

2. **Fine-grained discrepancy annotation**  
   Each image pair is annotated with structured visual discrepancies, including:
   - category
   - severity
   - location
   - description

3. **Integration into RL and test-time scaling**  
   Visual-ERM can be used:
   - as a **reward model** for GRPO-based RL, and
   - as a **visual critic** for iterative reflection and revision during inference.

<p align="center">
  <img src="assets/framework.jpg" width="95%">
</p>

## Main Results

Visual-ERM consistently improves vision-to-code performance across multiple tasks.

### Reinforcement Learning

- **Chart-to-Code**: improves **Qwen3-VL-8B-Instruct** by **+8.4** on average.
- **Table-to-Markdown**: yields **+2.7** average improvement.
- **SVG-to-Code**: yields **+4.1** average improvement.

### Visual Critic Benchmark

On **VC-RewardBench**, Visual-ERM substantially improves over the base model on fine-grained discrepancy judgment and outperforms **Qwen3-VL-235B-Instruct** as an open-source judge.

## VC-RewardBench

We introduce **VisualCritic-RewardBench (VC-RewardBench)**, a benchmark for evaluating fine-grained image-to-image discrepancy judgment on structured visual data.

<p align="center">
  <img src="assets/vc-bench.jpg" width="95%">
</p>

### Benchmark Features

- Covers **charts**, **tables**, and **SVGs**
- Contains **1,335** carefully curated instances
- Each instance includes:
  - a ground-truth image
  - a corrupted / rendered counterpart
  - fine-grained discrepancy annotations

## Quick Start

### 1. Reward Inference

Use Visual-ERM to compare a reference image and a rendered prediction:

```bash
python [TODO: path/to/infer_reward.py] \
    --ref_image [TODO: reference image path] \
    --pred_image [TODO: rendered image path] \
    --model_path [TODO: visual-erm checkpoint]
```

Expected output format:

```json
{
  "score": 0.84,
  "errors": [
    {
      "category": "text_error",
      "severity": 2,
      "location": "y-axis label",
      "description": "The label text differs from the reference image."
    }
  ]
}
```

### 2. Train Visual-ERM

```bash
python [TODO: path/to/train_reward_model.py] \
    --config [TODO: configs/train_visual_erm.yaml]
```

### 3. RL with Visual-ERM

```bash
python [TODO: path/to/train_rl.py] \
    --policy_model [TODO: policy checkpoint] \
    --reward_model [TODO: visual-erm checkpoint] \
    --task [chart|table|svg] \
    --config [TODO: configs/rl.yaml]
```

### 4. Test-Time Scaling with Reflection

```bash
python [TODO: path/to/reflect_and_revise.py] \
    --policy_model [TODO: policy checkpoint] \
    --critic_model [TODO: visual-erm checkpoint] \
    --input [TODO: input file or dataset] \
    --num_rounds 3
```

## Data

### Reward Modeling Data

Visual-ERM is trained on reward data spanning three vision-to-code domains:

- **Chart-to-Code**
- **Table-to-Markdown**
- **SVG-to-Code**

> **TODO:** Add download links, license notes, and data release policy.

### VC-RewardBench

> **TODO:** Add benchmark download link and evaluation instructions.

## Evaluation

### Supported Tasks

- Chart-to-Code
- Table-to-Markdown
- SVG-to-Code
- Fine-grained discrepancy judgment on VC-RewardBench

### Example Evaluation

```bash
python [TODO: path/to/eval.py] \
    --model_path [TODO: checkpoint] \
    --task [chart|table|svg|vc_rewardbench] \
    --config [TODO: configs/eval.yaml]
```

## Why Visual-ERM?

Visual-ERM is designed for settings where **visual equivalence matters more than textual similarity**.

It is particularly useful when:
- semantic similarity is not sufficient,
- reward hacking is common under proxy rewards,
- test-time self-correction requires interpretable visual feedback.

## ✒️ Citation

If you find this work useful, please consider citing:

```bibtex
@article{liu2026visualerm,
  title   = {Visual-ERM: Reward Modeling for Visual Equivalence},
  author  = {Ziyu Liu and Shengyuan Ding and Xinyu Fang and Xuanlang Dai and Penghui Yang and Jianze Liang and Jiaqi Wang and Kai Chen and Dahua Lin and Yuhang Zang},
  journal = {arXiv preprint arXiv:[TODO]},
  year    = {2026}
}
```

---

If you are interested in **visual reward modeling**, **vision-to-code**, or **reinforcement learning for multimodal models**, feel free to communicate with us.
