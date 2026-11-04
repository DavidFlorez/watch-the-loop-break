# NOTES.md — `watch-the-loop-break` Build Log

## Goal

Reproduce the Qwen3.5-0.8B "Weight-Space Programming" result ([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)), confirm that only `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` differ between base and edited models, and extend the result by sweeping continuous blends $W(\alpha) = W_{\text{base}} + \alpha(W_{\text{edit}} - W_{\text{base}})$ across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ on MATH-500 alongside a live streaming Gradio app.

## Setup

- **Models**:
  - Base: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (`24` hidden layers, `1024` hidden size, `3584` MLP intermediate size, hybrid 18 GatedDeltaNet + 6 GQA attention architecture, `488` tensors / `873,438,784` total parameters in `bfloat16`)
  - Edited: [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) (`p50c` weight-space edit)
  - Reference logs & rebuild scripts: [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/runs/p50c`)
- **Dataset**:
  - [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (`test` split, 500 competition math problems across 7 subjects and 5 difficulty levels; deterministic $N=50$ evaluation subset sampled with `seed=0`)
- **Hardware**:
  - **Local (MPS)**: Apple MacBook Air (`Mac15,12`), Apple M3 chip (8-core CPU, 10-core GPU), 16 GB unified memory (`mps` backend auto-detected via `CUDA -> MPS -> CPU` fallback in [`detect_hardware`](weights.py))
  - **Cloud GPU (vLLM)**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB` VRAM, Compute Capability `12.0`, CUDA `13.0`) on Google Colab Pro (`G4`) via [`gpu/colab_run.py`](gpu/colab_run.py), alongside NVIDIA `T4` (`16 GB`) / `L4` (`24 GB`) runners in [`notebooks/gpu_sweep.ipynb`](notebooks/gpu_sweep.ipynb) and [`gpu/gcp_run.py`](gpu/gcp_run.py)
- **Library Versions**:
  - **Local / MPS ([`requirements.txt`](requirements.txt))**: Python `3.13.7`, `torch==2.14.1`, `transformers==5.21.0`, `safetensors==0.8.0`, `accelerate==1.15.0`, `datasets==5.1.0`, `huggingface_hub==2.3.0`, `gradio==6.30.0`, `math-verify==0.9.0`, `numpy==2.5.4`, `pandas==3.1.0`, `matplotlib==3.11.2`, `pillow==12.3.0`, `tqdm==4.70.1`
  - **Cloud GPU / vLLM ([`requirements-gpu.txt`](requirements-gpu.txt))**: `vllm==0.33.0`, `torch==2.14.0+cu132`, `transformers==5.20.2`, `triton==3.8.0`, `flashinfer-python==0.7.2.post1`, `ninja==1.13.2`, `datasets==5.1.0`, `math-verify==0.9.0`, `safetensors==0.8.0` *(installed into an isolated `uv` virtualenv because `requirements.txt`'s `transformers==5.21.0` and `torch==2.14.1` conflict with `vllm==0.33.0`)*

## Log

- **2026-11-03 10:06 PST — Project initialization & mathematical audit of the edit**
  - Inspected [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming), and [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs).
  - Verified the exact weight-space programming mechanism: without gradient training, Astra identified unit repetition directions $u_{15}, u_{19} \in \mathbb{R}^{1024}$ at the residual outputs of layers 15 and 19, selected the 64 MLP intermediate columns in `down_proj.weight` (`[1024, 3584]`) with strongest alignment to $u_\ell$, and applied $W_{\text{edit}} = W_{\text{base}} - 6 \cdot u_\ell (u_\ell^\top W_{\text{base}} \odot \mathbb{1}_{64})$.
  - Implemented [`blend`](weights.py), computing $W(\alpha) = W_{\text{base}} + \alpha (W_{\text{edit}} - W_{\text{base}})$ in `float32` and casting to `bfloat16` (continuously scaling the ablation dose as $6\alpha$: $0\times$ at $\alpha=0$, $3\times$ at $\alpha=0.5$, $6\times$ at $\alpha=1.0$, $7.5\times$ at $\alpha=1.25$, $9\times$ at $\alpha=1.5$).

- **2026-11-03 10:31 PST — Full 488-tensor SHA-256 verification (`0a73ef1`)**
  - Executed [`verify_weight_diff`](weights.py) across all `488` tensors (`873,438,784` parameters). Confirmed bit-for-bit (`uint16` view of `bfloat16`) that **486/488 tensors are identical** and **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ.
  - Verified exact SHA-256 hashes for `blend(0.0)` and `blend(1.0)` against both Hugging Face checkpoints and saved [`assets/weight_verification.json`](assets/weight_verification.json) and [`assets/weight_diff_analysis.png`](assets/weight_diff_analysis.png).

- **2026-11-03 11:17–12:11 PST — Resolving MPS GatedDeltaNet bottlenecks & unified-memory swap (`d802a6c`..`34fa5bd`)**
  - **What broke #1**: On Apple Silicon MPS (where `flash-linear-attention` and `causal_conv1d` CUDA kernels are absent), `transformers`'s pure-PyTorch fallback executed `F.conv1d(..., groups=6144)` and 4D broadcast state updates on every single decode token, taking >2s/step and holding finished sequences in the batch until step 4096.
  - **Fix #1**: Added [`install_fast_qwen35_kernels`](weights.py) to replace `seq_len=1` decode calls with in-place 4-tap `conv_state` shifts (`float32` dot accumulation, exact match with reference) and in-place `float32` `torch.matmul` recurrent state updates (`max diff = 4.7e-7`), alongside dynamic EOS batch compaction (`past.reorder_cache(keep_idx)`) and custom batched [`PresencePenaltyLogitsProcessor`](metrics.py) (since `transformers.GenerationConfig` lacked a native `presence_penalty` field).
  - **What broke #2**: Under Chrome + macOS Desktop load on a 16 GB unified-memory machine, keeping the unused ViT vision encoder in VRAM (`~220 MB`) plus transient state copies pushed Python past `3.0 GB` (`627 MB` macOS memory compression), slowing 512 decode steps from `45s` to `149s`.
  - **Fix #2**: Dropped unused `model.model.visual` weights (`drop_visual=True`, saving `~220 MB`), sorted pending prompts by length to minimize left-padding waste, updated GatedDeltaNet `recurrent_state` in-place, flushed `torch.mps.empty_cache()` every 16 steps, and benchmarked batch scaling (`B=1: 8.9 tok/s`, `B=4: 17.3 tok/s`, `B=8: 29.7 tok/s`, `B=12: 31.6 tok/s` at `2.29 GB` driver memory).

- **2026-11-03 12:11–15:19 PST — Full $N=50$ MATH-500 $\alpha$-sweep & chart generation**
  - Ran all `250` evaluations (`N=50` MATH-500 problems × 5 $\alpha$ values at `max_new_tokens=4096`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) with JSONL checkpointing ([`assets/sweep_per_problem.jsonl`](assets/sweep_per_problem.jsonl)).
  - Enhanced Panel C of [`assets/sweep_charts.png`](assets/sweep_charts.png) to plot both full-trace repeated 8-gram ratio and final-512-token tail loop ratio with auto-scaled axes.

- **2026-11-03 15:20–15:30 PST — Side-by-side Gradio app polish, headless Chrome capture, and unit tests**
  - **What broke #3**: Initial headless Chrome capture showed Gradio dark-mode `.prose` CSS overriding text color inside white meter cards (`#334155` rendered light gray on `#ffffff`) and unescaped `\alpha` / `\text` in Markdown.
  - **Fix #3**: Enforced explicit high-contrast inline `#0f172a` / `#1e293b` styles in [`render_repetition_meter_html`](app.py), replaced raw TeX escapes with clean Unicode math, paced [`replay_recorded_comparison`](app.py) by token progress so shorter solved runs finish visually before capped base runs, and verified all 7 unit tests in [`tests/test_core.py`](tests/test_core.py).

- **2026-11-03 16:15 PST — Added vLLM GPU Execution Path (`gpu/`, `requirements-gpu.txt`, `notebooks/gpu_sweep.ipynb`)**
  - Added [`requirements-gpu.txt`](requirements-gpu.txt) pinning `vllm==0.33.0` (`torch==2.14.0`, `transformers>=5.16.1,<5.21.0`) separately from MPS `requirements.txt`.
  - Implemented [`gpu/make_ckpt.py`](gpu/make_ckpt.py) (builds temporary `bfloat16` safetensors checkpoints at revision `2fc06364715b967f1860aea9cf38778875588b17` in scratch directories outside git with SHA-256 & `config.json` assertions), [`gpu/smoke_test.py`](gpu/smoke_test.py) (8-problem greedy `fp16` vs `fp32` check with NaN/corruption + 128-token prefix divergence gating), [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py) (single-alpha process isolation, `mamba_ssm_cache_dtype="float32"`, compatibility fallbacks for `language_model_only` and `enforce_eager`, Pass 1 + Pass 2 forced `</think>` answer continuation on capped traces, strict vs forced accuracy, Wilson 95% CIs, and `fcntl.flock` atomic multi-GPU merging), [`gpu/aggregate.py`](gpu/aggregate.py), and [`notebooks/gpu_sweep.ipynb`](notebooks/gpu_sweep.ipynb) (supporting 2-GPU parallel execution on Kaggle `T4 x2` and 1-GPU on Colab `T4`).
- **2026-11-04 11:35–13:10 PST — Ephemeral Colab Pro G4 (`RTX PRO 6000 Blackwell`) Runner (`gpu/colab_run.py`) & Full 4k–32k Multi-Budget Sweep**
  - **What broke #1 (Colab system `site-packages` & `MPLBACKEND` inheritance)**: Installing `vllm==0.33.0` into Colab's system `/usr/local/lib/python3.x/dist-packages` conflicted with preloaded packages, while Colab's IPython kernel exported `MPLBACKEND=module://matplotlib_inline.backend_inline` into subprocesses (crashing isolated venvs without `matplotlib_inline`) and left the isolated venv's `bin/` directory (`ninja` / CUDA toolchain) off `PATH` during FlashInfer `SM120` JIT compilation.
  - **Fix #1**: Built [`gpu/colab_run.py`](gpu/colab_run.py) (`colab --auth=oauth2` automated session provisioning, isolated `uv` virtualenv at `/content/wsp/venv`, explicit `MPLBACKEND=Agg`, prepended venv + CUDA `bin/` directories onto `PATH` in both [`gpu/colab_run.py`](gpu/colab_run.py) and [`create_vllm_engine`](gpu/common.py), chunked result tar downloads, and guaranteed session teardown in `finally:`), alongside [`gpu/gcp_run.py`](gpu/gcp_run.py) and isolated `uv` bootstrap in [`notebooks/gpu_sweep.ipynb`](notebooks/gpu_sweep.ipynb).
  - **Live GPU Execution on NVIDIA RTX PRO 6000 Blackwell (`96 GB` VRAM, `bfloat16`)**:
    - [`gpu/smoke_test.py`](gpu/smoke_test.py) passed with **`0/8` greedy prefix disagreements** against `float32` (`recommended_dtype = "bfloat16"`).
    - Executed `N=50` MATH-500 evaluations across **4 token budgets** (`4096`, `8192`, `16384`, `32768`) × **3 blend alphas** (`0.0`, `1.0`, `1.25`) = `600` rollouts (+ forced Pass-2 continuations) at `4,500–9,700 tok/s`. At `32,768` tokens, **Base (`α=0.0`)** stayed trapped in repetition loops (`94.0%` cap-hit, `73.4%` 8-gram repetition, **`6.0%` strict accuracy**), whereas **Published Edit (`α=1.0`)** and **$1.25\times$ Extrapolation (`α=1.25`)** broke the loops (`40.0%` and `32.0%` cap-hit, `20.0%` and `12.5%` 8-gram repetition) and surged **8× to `40.0%` and `48.0%` strict accuracy** — matching the paper's `7.9%` / `48.1%` 32k benchmarks.

## Key results

### 1. Exact Weight-Space Diff & SVD Analysis ([`assets/weight_verification.json`](assets/weight_verification.json))

- **Tensors checked**: `488` total (`873,438,784` parameters); **`486` bit-identical** (`bfloat16` `uint16` view); **`2` modified** (`only_target_tensors_differ = true`).
- **Parameter footprint**: `64` columns per `[1024, 3584]` matrix (`65,536 × 2 = 131,072` column entries; `130,188` non-zero `bfloat16` entries changed) = **`0.0150%`** of model parameters.
- **Rank-1 structure**:
  - `model.language_model.layers.15.mlp.down_proj.weight`: $\sigma_1 = 1.7489$, $\sigma_2 = 0.00099$ ($\sigma_1 / \sigma_2 = \mathbf{1771.2}$, **`99.99945%`** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 10.64\%$, `base_sha256` = `ef60d2c6...`, `edit_sha256` = `82e2d733...`.
  - `model.language_model.layers.19.mlp.down_proj.weight`: $\sigma_1 = 1.3988$, $\sigma_2 = 0.00090$ ($\sigma_1 / \sigma_2 = \mathbf{1552.0}$, **`99.99914%`** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 8.11\%$, `base_sha256` = `638eedeb...`, `edit_sha256` = `8ef3c61b...`.
  - Principal left singular vectors align across layers 15 and 19 with $\cos(u_{15}, u_{19}) = \mathbf{0.8490}$.
  - `blend(0.0)` and `blend(1.0)` match the exact `bfloat16` SHA-256 digests of the base and edited checkpoints.

### 2. MATH-500 $\alpha$-Sweep Results ($N=50$, `max_new_tokens=4096` on MPS) ([`assets/sweep_results.csv`](assets/sweep_results.csv))

Measured on 50 deterministic `HuggingFaceH4/MATH-500` problems (`seed=0`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) using `math-verify` + Hendrycks LaTeX grading:

| Blend $\alpha$ | Ablation Scale ($6\alpha$) | Correct ($N=50$) | Accuracy (%) | Hit 4096 Cap (%) | Mean Tokens | Full-Trace 8-Gram Rep (%) | Final-512w Tail Rep (%) | Wall Time (s) | Speed (s/prob) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** (Base) | $0.0\times$ | 2 / 50 | 4.0% | 98.0% (49/50) | 4055.9 | 8.44% | **14.71%** | 3157.01s | 63.14s |
| **0.50** | $3.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 3990.4 | 3.74% | 1.22% | 1918.79s | 38.38s |
| **1.00** (Published) | $6.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 4028.7 | 2.99% | 1.39% | 2609.49s | 52.19s |
| **1.25** (**Optimal @ 4k**) | $7.5\times$ | **6 / 50** | **12.0%** | **88.0% (44/50)** | **3982.9** | 2.69% | 1.35% | 1995.77s | 39.92s |
| **1.50** (Over-ablation) | $9.0\times$ | 4 / 50 | 8.0% | 92.0% (46/50) | 4019.4 | **2.37%** | **0.74%** | **1890.71s** | **37.81s** |

- **Official 32k Budget Comparison (`RikiyaT/lm_understand-logs` `exp8_math/runs/p50c` `merged.json`)**:
  - **Paper 32k Base (`alpha=0.0`, `max_new_tokens=32768`)**: **`7.9%`** strict accuracy / **`93.0%`** cut share.
  - **Paper 32k Edited (`alpha=1.0`, `p50c`, `max_new_tokens=32768`)**: **`48.1%`** strict accuracy / **`33.05%`** cut share.
- **GPU Two-Pass Evaluation Protocol (`gpu/vllm_sweep.py`)**:
  - **Strict Accuracy (`pred_strict` / `correct_strict`)**: Evaluates [`extract_last_boxed_strict`](metrics.py) strictly after the final `</think>` tag (no fallback into unclosed thinking blocks), averaged across all $N \times S$ samples (`avg@n`) with 95% Wilson confidence intervals ([`wilson_ci`](metrics.py)).
  - **Forced-Answer Accuracy (`pred_forced` / `correct_forced`)**: For every sample hitting the Pass-1 `budget` cap, appends `\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n` and generates up to `512` additional tokens on the same loaded model to measure how many capped traces already solved the problem internally before running out of thinking budget.
- **MPS Decode Kernel Speedup**:
  - Replacing `transformers`'s pure-PyTorch `causal_conv1d_update` and 4D-broadcast `torch_recurrent_gated_delta_rule` + dropping unused visual weights reduced 512 decode steps (`B=10`) from `148.9s` (under swap) down to **`40–45s`** (~`111–125 tok/s` across 10 sequences; `3.34x` throughput scaling from `B=1` `8.9 tok/s` to `B=8` `29.7 tok/s` at `2.29 GB` MPS driver memory).

<!-- GPU_RESULTS_START -->
### 3. GPU vLLM Multi-Budget Sweep Results (RTX-PRO-6000 (bfloat16), updated 2026-11-04)

- **Smoke Test (`gpu/smoke_test.py`)**: GPU=`rtx-pro-6000` | fp16_passed=`True` | prompt_disagreements=`0/8` | fp16_nan_failures=`0` -> selected **`bfloat16`**.
- **GPU Runtime Environment**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` | Driver `580.82.07` | CUDA `13.0` | Python `3.12.3` | Packages: `vllm==0.33.0, torch==2.14.0+cu132, transformers==5.20.2, triton==3.8.0, datasets==5.1.0, math_verify==installed`.
- **Budgets & Blend Coefficients**: Budgets `[4096, 8192, 16384, 32768]` across `alpha` in `[0.0, 1.0, 1.25]` (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`).

| Budget | Backend / GPU (`dtype`) | Blend $\alpha$ | Samples (`N×S`) | Strict Acc % (95% CI) | Forced Acc % (95% CI) | Cap-Hit % (95% CI) | Mean Tokens | 8-Gram Rep % | Speed (`tok/s`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | `mps` / `apple-m3` (`bfloat16`) | **0.00** | 50×1=50 | **4.0%** (1.1–13.5%) | — | 98.0% (89.5–99.7%) | 4055.9 | 8.44% | — |
| 4096 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **0.00** | 50×1=50 | **0.0%** (0.0–7.1%) | 14.0% (7.0–26.2%) | 100.0% (92.9–100.0%) | 4096.0 | 10.49% | 8240.6 |
| 4096 | `mps` / `apple-m3` (`bfloat16`) | **0.50** | 50×1=50 | **6.0%** (2.1–16.2%) | — | 94.0% (83.8–97.9%) | 3990.4 | 3.74% | — |
| 4096 | `mps` / `apple-m3` (`bfloat16`) | **1.00** | 50×1=50 | **6.0%** (2.1–16.2%) | — | 94.0% (83.8–97.9%) | 4028.7 | 2.99% | — |
| 4096 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.00** | 50×1=50 | **6.0%** (2.1–16.2%) | 12.0% (5.6–23.8%) | 94.0% (83.8–97.9%) | 4000.9 | 2.68% | 9657.9 |
| 4096 | `mps` / `apple-m3` (`bfloat16`) | **1.25** | 50×1=50 | **12.0%** (5.6–23.8%) | — | 88.0% (76.2–94.4%) | 3982.9 | 2.69% | — |
| 4096 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.25** | 50×1=50 | **6.0%** (2.1–16.2%) | 10.0% (4.3–21.4%) | 94.0% (83.8–97.9%) | 3986.7 | 2.45% | 9706.8 |
| 4096 | `mps` / `apple-m3` (`bfloat16`) | **1.50** | 50×1=50 | **8.0%** (3.1–18.8%) | — | 92.0% (81.2–96.8%) | 4019.4 | 2.37% | — |
| 8192 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **0.00** | 50×1=50 | **8.0%** (3.1–18.8%) | 22.0% (12.8–35.2%) | 94.0% (83.8–97.9%) | 8022.7 | 27.42% | 7734.2 |
| 8192 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.00** | 50×1=50 | **10.0%** (4.3–21.4%) | 16.0% (8.3–28.5%) | 90.0% (78.6–95.7%) | 7779.7 | 5.76% | 8279.9 |
| 8192 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.25** | 50×1=50 | **12.0%** (5.6–23.8%) | 18.0% (9.8–30.8%) | 86.0% (73.8–93.0%) | 7733.2 | 4.54% | 8241.9 |
| 16384 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **0.00** | 50×1=50 | **6.0%** (2.1–16.2%) | 24.0% (14.3–37.4%) | 94.0% (83.8–97.9%) | 15723.2 | 55.52% | 6321.4 |
| 16384 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.00** | 50×1=50 | **18.0%** (9.8–30.8%) | 22.0% (12.8–35.2%) | 80.0% (67.0–88.8%) | 14727.7 | 11.79% | 6390.6 |
| 16384 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.25** | 50×1=50 | **28.0%** (17.5–41.7%) | 30.0% (19.1–43.8%) | 64.0% (50.1–75.9%) | 13855.9 | 8.30% | 6496.6 |
| 32768 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **0.00** | 50×1=50 | **6.0%** (2.1–16.2%) | 20.0% (11.2–33.0%) | 94.0% (83.8–97.9%) | 31124.1 | 73.43% | 4495.9 |
| 32768 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.00** | 50×1=50 | **40.0%** (27.6–53.8%) | 40.0% (27.6–53.8%) | 40.0% (27.6–53.8%) | 23432.5 | 20.01% | 4691.2 |
| 32768 | `vllm` / `rtx-pro-6000` (`bfloat16`) | **1.25** | 50×1=50 | **48.0%** (34.8–61.5%) | 48.0% (34.8–61.5%) | 32.0% (20.8–45.8%) | 21502.9 | 12.46% | 4664.6 |

- **Published 32k Reference (`RikiyaT/lm_understand-logs` `exp8_math/runs/p50c`)**: Base (`α = 0.0`): **`7.9%`** strict accuracy / **`93.0%`** cut share; Edited `p50c` (`α = 1.0`): **`48.1%`** strict accuracy / **`33.05%`** cut share.
- **Visual**: See [`assets/budget_curve_gpu.png`](assets/budget_curve_gpu.png) and per-budget dashboards under `assets/budget_<budget>/gpu-*/sweep_charts.png`.
<!-- GPU_RESULTS_END -->

## Visuals

- `assets/weight_diff_analysis.png` — Column-wise L2 edit norms ($\|\Delta W[:, j]\|_2$) and log-scale singular value spectra for Layers 15 and 19 `mlp.down_proj.weight`, proving only 64 columns per tensor change via a rank-1 projection.
- `assets/app_screenshot.png` — Side-by-side Gradio interface on `test/algebra/1936.json` comparing Base (`α = 0.00`, hitting the `4096 / 4096` token cap) vs. Edited (`α = 1.00`, finishing cleanly in `2689` tokens with exact `\boxed{-1}`).
- `assets/app_side_by_side.gif` — Animated capture of `app.py` streaming Base (`α = 0.00`) vs. Edited (`α = 1.00`), showing the edited model boxing `-1` while the base model keeps circling until hitting the red `TOKEN CAP HIT` alert.
- `assets/sweep_charts.png` — Four-panel dose-response dashboard across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ showing MATH-500 accuracy peaking at $\alpha = 1.25$ (`12.0%`), 4096-token cap truncation dropping to `88.0%`, full-trace and tail 8-gram repetition collapsing monotonically (`14.71% -> 0.74%`), and evaluation wall times.
- `assets/alpha_token_distributions.png` — Per-problem token lengths (solved vs. capped/wrong) and token count vs. repeated 8-gram ratio scatter across all 250 runs.
- `assets/budget_curve_gpu.png` — Cross-budget scaling curve (`4096` MPS + `8192` / `16384` / `32768` GPU vLLM) comparing Strict Accuracy (with 95% Wilson CIs), Pass-2 Forced-Answer Accuracy, and Token-Cap Truncation Rate alongside the paper's 32k benchmarks (`7.9% / 93.0%` Base, `48.1% / 33.05%` Edited).

## Surprises and limitations

1. **Loop breaking happens immediately by $\alpha = 0.5$–$1.0$, but coherent over-thinking still saturates a 4,096-token budget**:
   - At $\alpha = 0.0$, worst-case traces lock into severe verbatim repetition loops (`test/prealgebra/2066.json` reached `52.1%` overall / `70.8%` tail repeated 8-grams; `test/number_theory/631.json` reached `48.2%` overall / `94.8%` tail repetition).
   - Raising $\alpha$ from `0.0 -> 0.5 -> 1.0` slashes final-512-token repetition **12×** (`14.71% -> 1.22% -> 1.39%`), completely eliminating verbatim loops. However, because `Qwen3.5-0.8B` was trained for `32,768`-token `<think>` budgets, a strict `4,096`-token cap still truncates `94.0%` (`47/50`) of $\alpha=1.0$ trajectories mid-verification.
2. **Inverted-U dose-response curve ($\alpha = 1.25$ doubles 4k accuracy over $\alpha = 1.0$)**:
   - Extrapolating the directional ablation slightly past the published edit to **$\alpha = 1.25$** ($7.5\times$ subtraction) shortens over-cautious `<think>` verification chains just enough to raise completion before 4,096 tokens (`88.0%` hit cap vs. `94.0%`), **doubling accuracy from `6.0% (3/50)` at $\alpha = 1.0$ to `12.0% (6/50)` at $\alpha = 1.25$** (`3×` base accuracy), while **every single run across $\alpha \in \{0.5, 1.0, 1.25, 1.5\}$ that finished before the 4,096-token cap produced the exact right answer (`100%` precision conditioned on termination: `3/3`, `3/3`, `6/6`, `4/4`)**.
   - Pushing further to **$\alpha = 1.5$** ($9\times$ subtraction) reduces repetition even further (`0.74%` tail), but over-subtracts the residual direction on harder problems, dropping accuracy back to `8.0% (4/50)`.
3. **`fp16` vs. `bf16` numerics on Turing (`T4`) GatedDeltaNet layers**:
   - `Qwen/Qwen3.5-0.8B` is trained and distributed in `bfloat16` (8-bit exponent, max `~3.4e38`), whereas NVIDIA **T4** (`sm_75`) lacks hardware `bf16` tensor cores and executes native half precision in `float16` (5-bit exponent, max `65,504`). Across 18 GatedDeltaNet recurrent layers, `float16` can occasionally overflow or diverge on long reasoning prefixes; [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py) keeps SSM state caches in `float32` (`mamba_ssm_cache_dtype="float32"`) and [`gpu/smoke_test.py`](gpu/smoke_test.py) gates `float16` against `float32` on 8 greedy prompts (`temperature=0`, `max_tokens=1024`) before any sweep runs.
4. **Exact confirmation of loop-onset vs. full-trace repetition across `4k -> 8k -> 16k -> 32k` (and why Base fails even with Pass-2 forced answers)**:
   - Loops in `Qwen3.5-0.8B` typically onset around tokens `2,500–4,500` and then repeat for the remaining `28,000+` tokens under a `32,768` cap. On RTX PRO 6000 (`bfloat16`), Base (`α = 0.0`) full-trace repeated 8-gram ratio grew `10.49% (4k) -> 27.42% (8k) -> 55.52% (16k) -> 73.43% (32k)` (`85.4%` tail), with `94.0%` of trajectories hitting the cap at every budget from `8k` to `32k`. Because the late KV cache is saturated by verbatim loops, even forced `</think>` Pass-2 accuracy on Base plateaued at `14% -> 22% -> 24% -> 20%`, while strict accuracy stayed stuck at **`6.0%`** at both `16k` and `32k`.
   - In contrast, loop breaking unlocks long-horizon test-time scaling: from `4k -> 8k -> 16k -> 32k`, strict accuracy climbed **`6.0% -> 10.0% -> 18.0% -> 40.0%`** for **Published Edit (`α = 1.0`)** and **`6.0% -> 12.0% -> 28.0% -> 48.0%`** (**8× Base**) for **$1.25\times$ Extrapolation (`α = 1.25`)**, cutting cap-hits from `94.0%` down to `40.0%` and `32.0%` and saving **~9,621 tokens/problem** at `32k` (`31,124.1 -> 21,502.9`).
5. **Statistical power ($N=50$), `avg@n`, and forced-pass `presence_penalty` scope**:
   - At $N=50$ single-sample evaluation, accuracy changes move in `2.0%` (`1/50`) discrete steps (`±7–13%` Wilson 95% confidence intervals); `gpu/vllm_sweep.py --samples K` supports multi-sample `avg@n` evaluation with Wilson CIs to reduce sampling variance.
   - In Pass 2 (forced answer after hitting the token cap), `presence_penalty=1.5` applies across the `512` newly sampled forced-answer tokens rather than carrying over a penalty table from Pass 1's `<think>` prefix—avoiding spurious penalties on mathematical constants/symbols already written during reasoning while still suppressing loops in the final answer block.

## Open questions / next steps

1. **Multi-sample `avg@n` evaluation at `32k` and finer $\alpha \in [1.0, 1.35]$ grid**: Run `gpu/vllm_sweep.py --samples 4` across $\alpha \in \{1.0, 1.15, 1.25, 1.35\}$ at `budget=32768` via [`gpu/colab_run.py`](gpu/colab_run.py) to narrow the ±12% Wilson CIs at $N=50$ and pinpoint the exact optimum above $\alpha = 1.0$.
2. **Decoupled 2D layer interpolation $(\alpha_{15}, \alpha_{19})$**: Since $\cos(u_{15}, u_{19}) = 0.849$, sweeping $\alpha_{15} \neq \alpha_{19}$ (e.g., testing whether Layer 15 controls topic circling while Layer 19 controls token-level repetition) could separate loop breaking from premature stopping.
3. **Token-budget-aware dynamic $\alpha(t)$ scheduling**: Annealing $\alpha(t)$ upward as $t \to T_{\max}$ (`0.75 -> 1.35`) could preserve full exploration early in `<think>` while nudging the model to conclude and `\boxed{}` cleanly before any deployment token ceiling (`2048` or `4096`).
4. **Custom Metal / MLX GatedDeltaNet kernel**: Porting the 18 GatedDeltaNet recurrent layers to custom Metal / `mlx-lm` kernels on Apple Silicon to reach `300+ tok/s` for interactive 16k–32k math evaluation on consumer MacBooks.
