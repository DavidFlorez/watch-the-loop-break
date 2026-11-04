# `watch-the-loop-break` 🧠

Reproducing and extending Experiment 8 of **["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)** ([Takehi, Teoh, Zhang & Khattab, 2026](https://rikiyat.github.io/blog/weight-space-programming/)) on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

---

## Overview

Tiny reasoning models frequently fail on competition math not because they cannot perform the algebra, but because their `<think>` chain enters a **repetitive loop** until exhausting the token limit. In *Weight-Space Programming*, a frontier LLM ("Astra") inspected `Qwen/Qwen3.5-0.8B`'s weights without gradients or training, isolated the repetition loop to a single residual direction written by the MLPs of **Layer 15** and **Layer 19**, and projected that direction out of 64 columns in each layer's `down_proj.weight` (`131,072` entries = `0.0150%` of the model).

This repository:
1. **Verifies the exact weight-space diff (`weights.py`)**: Confirms across all 488 tensors (`873,438,784` parameters) that **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ, verifies their `bfloat16` SHA-256 digests, and proves via SVD that both edits are exact **rank-1 projections** ($\sigma_1^2 / \sum \sigma_i^2 > 99.999\%$).
2. **Implements continuous weight-space interpolation (`blend(alpha)`)**:
   $$W(\alpha) = W_{\text{base}} + \alpha \cdot (W_{\text{edit}} - W_{\text{base}})$$
   Allowing continuous tuning from the unmodified base model ($\alpha = 0$), through partial loop suppression ($\alpha = 0.5$), the published edit ($\alpha = 1.0$), and over-projection ($\alpha \in \{1.25, 1.5\}$).
3. **Provides a Side-by-Side Live Streaming Gradio App (`app.py`)**: Streams $\alpha_{\text{left}}$ and $\alpha_{\text{right}}$ side by side on any chosen [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problem with a live **repeated 8-gram ratio meter** and a **token-cap truncation flag**.
4. **Runs an $\alpha$-Dose-Response Sweep (`sweep.py`)**: Evaluates $N=50$ `MATH-500` problems across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`, `max_new_tokens=4096`), recording accuracy, `% hitting the cap`, `mean tokens`, `repeated 8-gram ratio`, and `wall time`, and saving CSVs + charts to `assets/`.

---

## Key Results ($N=50$ MATH-500 Sweep, `max_new_tokens=4096`)

Evaluated across `seed=0` (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) on Apple M3 (`bfloat16`):

| Blend $\alpha$ | Ablation Scale ($6\alpha$) | Correct ($N=50$) | Accuracy (%) | Hit 4096 Cap (%) | Mean Tokens | Full-Trace 8-Gram Rep (%) | Final-512w Tail Rep (%) | Wall Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** (Base) | $0.0\times$ | 2 / 50 | 4.0% | 98.0% (49/50) | 4055.9 | 8.44% | **14.71%** | 3157.01s |
| **0.50** | $3.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 3990.4 | 3.74% | 1.22% | 1918.79s |
| **1.00** (Published) | $6.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 4028.7 | 2.99% | 1.39% | 2609.49s |
| **1.25** (**Optimal @ 4k**) | $7.5\times$ | **6 / 50** | **12.0%** | **88.0% (44/50)** | **3982.9** | 2.69% | 1.35% | 1995.77s |
| **1.50** (Over-ablation) | $9.0\times$ | 4 / 50 | 8.0% | 92.0% (46/50) | 4019.4 | **2.37%** | **0.74%** | **1890.71s** |

- **Exact 2-Tensor Diff**: Across all `488` tensors (`873,438,784` parameters), **`486` are bit-identical** in `bfloat16`; only `layers.15.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1771.2$, `99.99945%` rank-1 energy) and `layers.19.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1552.0$, `99.99914%` rank-1 energy) differ across 64 columns each (`0.0150%` of parameters).
- **Inverted-U Dose Response under a 4k Budget**: Verbatim loops collapse **12×** by $\alpha = 0.5$–$1.0$ (tail 512w repetition `14.71% -> 1.22%–1.39%`), while extrapolating to **$\alpha = 1.25$** trims over-cautious `<think>` verification chains enough to double accuracy (`12.0%` vs `6.0%`) before over-projection at $\alpha = 1.5$ degrades reasoning (`8.0%`). Every edited ($\alpha > 0$) run that finished before the 4,096-token cap solved the problem (`100%` precision conditioned on completion: `3/3`, `3/3`, `6/6`, `4/4`).

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

## Quickstart (Apple Silicon MPS / Local)

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

---

## Run on Cloud GPU (Google Cloud GCE L4/T4, Kaggle T4 x2, or Google Colab)

To sweep larger thinking budgets (`8192`, `16384`, and optional `32768` tokens) with **vLLM (`0.33.0`)**:

### Option A: One-Command Google Cloud Ephemeral GPU (`gpu/gcp_run.py`)
Spins up a Spot **NVIDIA L4** (`g2-standard-4`, 24 GB VRAM, native **`bfloat16`**) or **NVIDIA T4** (`n1-standard-4`, automatic fallback if L4 quota is 0) in your GCP project, installs `vLLM` into an isolated `uv` virtualenv (`--torch-backend=auto`), streams logs live, syncs completed results into `assets/` and `NOTES.md`, and **auto-deletes the VM** immediately on exit (enforced both by `finally:` cleanup and GCP's `--max-run-duration` server-side timer):

```bash
# Preview VM zone, GPU selection, and remote commands without spending anything:
.venv/bin/python gpu/gcp_run.py --project YOUR_GCP_PROJECT_ID --dry-run

# Launch the full 8k + 16k sweep on Spot L4 (auto-fallback to T4) and auto-delete VM on completion:
.venv/bin/python gpu/gcp_run.py --project YOUR_GCP_PROJECT_ID
```

### Option B: Notebook Runner (`notebooks/gpu_sweep.ipynb` on Colab / Kaggle)
1. **Open [`notebooks/gpu_sweep.ipynb`](notebooks/gpu_sweep.ipynb)** in **Kaggle** (`GPU T4 x2`, Internet enabled) or **Google Colab** (`T4` or `L4` GPU). *(Requires NVIDIA compute capability `>= 7.5`; P100 `6.0` is blocked automatically.)*
2. **Isolated `uv` GPU Environment**: Step 1 installs [`requirements-gpu.txt`](requirements-gpu.txt) into an isolated virtual environment (`/tmp/wsp_venv` or `/kaggle/tmp/wsp_venv`) via `uv pip install --torch-backend=auto`, avoiding Colab's preloaded `torch`/`protobuf` conflicts without restarting the kernel. **Do NOT install `requirements.txt` on GPU.**
3. **Numerical Smoke Gate**: Step 3 runs [`gpu/smoke_test.py`](gpu/smoke_test.py) across 8 MATH-500 problems (`seed=0`, `temperature=0`, `max_tokens=1024`) — validating native **`bfloat16`** directly on `L4`/`A100` (`CC >= 8.0`) or gating **`float16` vs. `float32`** on `T4` (`CC 7.5`).
4. **Two-Pass Sweep & Aggregation**: Step 4 builds temporary blended safetensors checkpoints in scratch (`/tmp` or `/kaggle/tmp`) via [`gpu/make_ckpt.py`](gpu/make_ckpt.py), evaluates Pass 1 (`budget` tokens) + Pass 2 forced `</think>` answer continuation (`512` tokens) via [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py), and aggregates cross-budget charts into `assets/budget_curve_gpu.png` via [`gpu/aggregate.py`](gpu/aggregate.py).

See [`NOTES.md`](NOTES.md) for the full build log, setup details, experimental measurements, surprises, and next steps.
