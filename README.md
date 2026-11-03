# `watch-the-loop-break` 🧠

Reproducing and extending Experiment 8 of **["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)** ([Takehi, Teoh, Zhang & Khattab, 2026](https://rikiyat.github.io/blog/weight-space-programming/)) on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

---

## Overview

Tiny reasoning models frequently fail on competition math not because they cannot perform the algebra, but because their `<think>` chain enters a **repetitive loop** until exhausting the token limit. In *Weight-Space Programming*, a frontier LLM ("Astra") inspected `Qwen/Qwen3.5-0.8B`'s weights without gradients or training, isolated the repetition loop to a single residual direction written by the MLPs of **Layer 15** and **Layer 19**, and projected that direction out of 64 columns in each layer's `down_proj.weight` (`131,072` entries = `0.016%` of the model).

This repository:
1. **Verifies the exact weight-space diff (`weights.py`)**: Confirms across all 320 tensors (`852,985,920` parameters) that **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ, verifies their `bfloat16` SHA-256 digests, and proves via SVD that both edits are exact **rank-1 projections** ($\sigma_1^2 / \sum \sigma_i^2 > 99.999\%$).
2. **Implements continuous weight-space interpolation (`blend(alpha)`)**:
   $$W(\alpha) = W_{\text{base}} + \alpha \cdot (W_{\text{edit}} - W_{\text{base}})$$
   Allowing continuous tuning from the unmodified base model ($\alpha = 0$), through partial loop suppression ($\alpha = 0.5$), the published edit ($\alpha = 1.0$), and over-projection ($\alpha \in \{1.25, 1.5\}$).
3. **Provides a Side-by-Side Live Streaming Gradio App (`app.py`)**: Streams $\alpha_{\text{left}}$ and $\alpha_{\text{right}}$ side by side on any chosen [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problem with a live **repeated 8-gram ratio meter** and a **token-cap truncation flag**.
4. **Runs an $\alpha$-Dose-Response Sweep (`sweep.py`)**: Evaluates $N=50$ `MATH-500` problems across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`, `max_new_tokens=4096`), recording accuracy, `% hitting the cap`, `mean tokens`, `repeated 8-gram ratio`, and `wall time`, and saving CSVs + charts to `assets/`.

---

## Visuals

### 1. Side-by-Side Gradio App (`app.py`)
![Side-by-Side Gradio App](assets/app_side_by_side.gif)
![Side-by-Side Gradio Screenshot](assets/app_screenshot.png)

### 2. $\alpha$-Sweep Results on MATH-500 ($N=50$)
![Alpha Sweep Dashboard](assets/sweep_charts.png)
![Token Length vs 8-Gram Repetition Distribution](assets/alpha_token_distributions.png)

### 3. Weight-Space Edit Structure (Layers 15 & 19 `mlp.down_proj.weight`)
![Weight Diff Analysis](assets/weight_diff_analysis.png)

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Verify exact tensor diff, SHA-256 hashes, and generate weight diff chart
.venv/bin/python weights.py

# 2. Run the N=50 MATH-500 sweep across alpha in [0, 0.5, 1.0, 1.25, 1.5]
.venv/bin/python sweep.py --n 50 --alphas 0,0.5,1.0,1.25,1.5

# 3. Launch the side-by-side streaming Gradio app (binds to 127.0.0.1:7860)
.venv/bin/python app.py
```

See [`NOTES.md`](NOTES.md) for the full build log, setup details, experimental measurements, surprises, and next steps.
