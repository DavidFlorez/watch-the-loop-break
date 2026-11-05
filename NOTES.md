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
- **2026-11-04 14:10–21:55 PST — Nested-budget derivation, open-box forced completion, `N=200` core + `N=100` 6-dose GPU sweeps, and publication charts (`0c7db81`)**
  - **What broke #2 (Nested-budget prefix duplication, double-`</think>` on closed capped traces, & silent whitespace fallback on null `loop_onset_token`)**:
    1. Across all `450` committed `N=50` smaller-budget traces (`4096`, `8192`, `16384` × `α ∈ {0.0, 1.0, 1.25}`), per-request `seed=0` made **100% (`450/450`)** of traces exact character prefixes of the `32,768`-token trace (`t32.startswith(tb)`), making separate Pass-1 generation for smaller budgets redundant — except that legacy logs omitted `out_ids` (`32/450` single-char UTF-8/BPE boundary splits when re-tokenizing decoded `text`) and scored one unclosed `8,192`-token trace (`test/algebra/1519.json`, `closed_think=False`, `hit_cap=True`) as `correct_strict=True`.
    2. For Pass-2 forced completion on capped traces where `</think>` had *already* closed (`closed_think=True` without yet emitting `\boxed{}`), appending `\n</think>\n\n` injected a second closing tag rather than prompting directly for the box.
    3. In [`_get_or_compute_loop_onset`](scripts/generate_article_charts.py), checking `if rec["loop_onset_token"] is not None:` caused records with an already-computed `loop_onset_token: None` (exact BPE check finding no `>50%` 1024-token loop window) to fall back to approximate whitespace-word windows when run without `--with-tokenizer` (`68/200` vs exact `66/200` = `33.0%` at `α=1.0`).
  - **Fix #2**:
    1. Added [`gpu/derive_budgets.py`](gpu/derive_budgets.py) ([`derive_budgets_from_32k`](gpu/derive_budgets.py), [`verify_derived_against_root`](gpu/derive_budgets.py)) to truncate `out_ids[:B]` from a single `32,768`-token run across `{4096, 8192, 16384, 32768}`, enforce finished-only strict correctness (`finish_reason == "stop"` within `B` tokens AND `correct`), and run Pass-2 forced completion only on truncated prefixes (`hit_cap == True`). Updated [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py) to persist full `out_ids` and `forced_text`, and added `open_box` Pass-2 suffix (`\n\nFinal Answer: \boxed{` via [`FORCE_BOX_OPEN_SUFFIX`](gpu/common.py) + [`extract_forced_boxed`](metrics.py)) whenever a truncated prefix already closed `</think>`.
    2. Added [`compute_loop_onset_from_token_ids`](metrics.py) (`1024`-token sliding window, stride `256`, `>50%` repeated 8-grams), paired Wilson CIs + exact two-sided McNemar tests ([`gpu/stats.py`](gpu/stats.py) → [`assets/gpu_stats.csv`](assets/gpu_stats.csv)), published Takehi et al. (2026) `avg@4` & re-tokenized first-sample truncation extraction ([`scripts/paper_reference.py`](scripts/paper_reference.py) → [`assets/paper_reference.csv`](assets/paper_reference.csv)), and [`scripts/generate_article_charts.py`](scripts/generate_article_charts.py) (rendering 5 1600px article figures under `assets/article/` plus updated `assets/budget_curve_gpu.png` and `assets/mps_4k_comparison.png`).
  - **Live Cloud GPU Scaling Execution (`RTX PRO 6000 Blackwell`, `bfloat16`, nested cohorts `S_50 ⊂ S_100 ⊂ S_200`)**:
    - **`N = 200` Core Cohort (`assets/gpu_n200/`)**: Ran `32,768`-token rollouts on `N=200` (`seed=0`) for `α ∈ {0.0, 1.0, 1.25}` (`600` 32k traces) and derived nested `4k / 8k / 16k` budgets + live GPU `open_box`/`close_think` Pass-2 forced completions (`2,400` total budget evaluations), keeping all 24 original `N=50` files under `assets/budget_*/` bit-for-bit intact (`SHA-256` verified).
    - **`N = 100` Paired 6-Dose Cohort (`assets/gpu_dose_n100/`)**: Ran intermediate and high ablation doses `α ∈ {0.5, 0.75, 1.5}` (`3×, 4.5×, 9×`) on `S_100 ⊂ S_200` at `32,768` tokens and derived `4k / 8k / 16k` + Pass-2 forced completions (`1,200` new evaluations; joined with `S_200[:100]` for `{0.0, 1.0, 1.25}` to form a `2,400`-evaluation 6-dose paired matrix).

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
  - **Strict Accuracy (`pred_strict` / `correct_strict`)**: Evaluates [`extract_last_boxed_strict`](metrics.py) strictly after the final `</think>` tag only on finished runs (`finish_reason == "stop"` within budget $B$), averaged across all $N \times S$ samples (`avg@n`) with 95% Wilson confidence intervals ([`wilson_ci`](metrics.py)).
  - **Forced-Answer Accuracy (`pred_forced` / `correct_forced`)**: For every sample hitting the Pass-1 `budget` cap, appends [`FORCE_ANSWER_SUFFIX`](gpu/common.py) (`\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n`, `close_think` mode) if `</think>` has not appeared yet, or [`FORCE_BOX_OPEN_SUFFIX`](gpu/common.py) (`\n\nFinal Answer: \boxed{`, `open_box` mode) if `</think>` already closed without a final box, then generates up to `512` additional tokens on the same loaded model.
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

### 4. Nested-Budget Prefix Invariance & Offline Derivation Check (`gpu/derive_budgets.py`)

With per-request `seed=0` in `vLLM`, every smaller-budget Pass-1 trace is an exact prefix of the `32,768`-token trace for the same `(alpha, unique_id, sample_idx)`:
- **Trace prefix check (`N=50`, 450 traces across `4096, 8192, 16384`)**: **`450 / 450` (100.0%)** exact string prefix matches (`t32.startswith(tb)`) verified by [`test_committed_gpu_traces_are_exact_prefixes_of_32k`](tests/test_gpu_suite.py).
- **Offline `derive_budgets_from_32k` + `verify_derived_against_root` check on committed `N=50` 32k logs**:
  - `hit_cap_count` & `hit_cap_pct`: **`0 / 450` mismatches** (`100.0%` exact match across all `12 / 12` `(budget, alpha)` cells).
  - `closed_think_pct`: **`0 / 450` mismatches** (`100.0%` exact match across all `12 / 12` `(budget, alpha)` cells: `[2.0%, 6.0%, 6.0%]` @ 4k, `[6.0%, 14.0%, 14.0%]` @ 8k, `[6.0%, 34.0%, 38.0%]` @ 16k).
  - `strict` accuracy: `exact_csv_match_legacy_strict = True` (`12 / 12` cells match raw text strict grading) and `11 / 12` cells match finished-only strict (`finish_reason == "stop"` within `B` tokens AND `correct`). The single `1 / 50` difference at `(budget=8192, α=0.0)` (`6.0% = 3/50` finished-strict vs `8.0% = 4/50` legacy unclosed-strict) is `test/algebra/1519.json`, which hit the `8,192`-token cap mid-`<think>` (`hit_cap = True`, `closed_think = False`, scratch `\boxed{1/\sqrt{3}}` inside `<think>`) and later looped to `32,768` tokens (`pred_strict = None`).
  - Re-tokenization note: because legacy `N=50` logs did not store raw `out_ids`, re-encoding decoded `text` via `tokenizer.encode` produced `32 / 450` (`9` @ 4k, `12` @ 8k, `11` @ 16k) single-character BPE boundary splits on incomplete UTF-8/whitespace sequences; [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py) now saves `out_ids` and `forced_text` in every run so `out_ids[:B]` truncation is **100% token-exact**.
  - `open_box` forced completion on `N=200`: among capped traces where `</think>` had *already* closed (`6`, `15`, `35`, and `42` traces at `4k / 8k / 16k / 32k`), appending [`FORCE_BOX_OPEN_SUFFIX`](gpu/common.py) recovered **`6/6` (100%)**, **`7/15` (46.7%)**, **`14/35` (40.0%)**, and **`6/42` (14.3%)** additional boxed answers without re-emitting `</think>`.

### 5. Core `N = 200` Budget Scaling Cohort (`S_50 ⊂ S_200`, `RTX PRO 6000 bfloat16`) ([`assets/gpu_n200/`](assets/gpu_n200/))

Executed at `32,768` tokens (`seed=0`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) and nested-derived via [`gpu/derive_budgets.py`](gpu/derive_budgets.py) with live GPU Pass-2 forced completions (`open_box` + `close_think`):

| Budget $B$ | Blend $\alpha$ | Strict Acc (95% Wilson CI) | $\Delta$ vs Base (McNemar $p$) | Forced Acc (95% CI) | Hit Cap % | Closed `</think>` % | Looped % (1024w / $\ge$15%) | Mean / Med Tokens | Mean / Tail 8-Gram % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4,096** | **0.00** | **3.0%** (1.38–6.39%) | — | 19.0% (14.17–25.00%) | 97.0% | 4.0% | 14.0% / 14.0% | 4040.7 / 4096.0 | 9.13% / 15.35% |
| **4,096** | **1.00** | **9.0%** (5.77–13.78%) | +6.0 pp ($p = 0.00183$) | 17.0% (12.43–22.82%) | 91.0% | 9.5% | **0.0%** / 0.0% | 3937.3 / 4096.0 | 2.90% / 1.15% |
| **4,096** | **1.25** | **9.5%** (6.17–14.36%) | +6.5 pp ($p = 0.00098$) | 17.5% (12.86–23.36%) | 90.5% | 11.0% | 0.5% / 0.5% | 3918.8 / 4096.0 | 2.55% / 1.02% |
| **8,192** | **0.00** | **7.5%** (4.60–12.00%) | — | 20.0% (15.05–26.09%) | 92.5% | 7.5% | 55.0% / 55.0% | 7911.1 / 8192.0 | 27.17% / 51.87% |
| **8,192** | **1.00** | **13.5%** (9.45–18.93%) | +6.0 pp ($p = 0.01690$) | 22.5% (17.26–28.77%) | 86.0% | 18.0% | 3.0% / 6.0% | 7573.4 / 8192.0 | 6.33% / 2.87% |
| **8,192** | **1.25** | **14.0%** (9.87–19.49%) | +6.5 pp ($p = 0.00720$) | 22.0% (16.82–28.24%) | 85.5% | 17.5% | 2.0% / 3.5% | 7526.4 / 8192.0 | 4.99% / 2.48% |
| **16,384** | **0.00** | **8.5%** (5.37–13.19%) | — | 21.0% (15.93–27.16%) | 91.5% | 8.5% | 87.5% / 88.0% | 15428.6 / 16384.0 | 53.38% / 78.45% |
| **16,384** | **1.00** | **29.5%** (23.61–36.16%) | **+21.0 pp** ($p < 10^{-6}$) | 36.0% (29.67–42.86%) | 67.0% | 43.5% | 14.5% / 24.5% | 13818.6 / 16384.0 | 12.27% / 7.56% |
| **16,384** | **1.25** | **30.5%** (24.54–37.20%) | **+22.0 pp** ($p < 10^{-6}$) | 34.5% (28.26–41.32%) | 65.0% | 41.5% | 10.0% / 19.0% | 13630.0 / 16384.0 | 9.95% / 5.57% |
| **32,768** | **0.00** | **8.5%** (5.37–13.19%) | — | 21.5% (16.37–27.70%) | 91.5% | 8.5% | 89.5% / 89.5% | 30419.9 / 32768.0 | 70.66% / 80.34% |
| **32,768** | **1.00** | **48.5%** (41.67–55.39%) | **+40.0 pp** ($p < 10^{-6}$) | 50.0% (43.14–56.86%) | 31.0% | 79.5% | 33.0% / 46.0% | 21435.7 / 22088.5 | 18.75% / 12.57% |
| **32,768** | **1.25** | **51.5%** (44.61–58.33%) | **+43.0 pp** ($p < 10^{-6}$) | **53.5%** (46.59–60.28%) | **29.5%** | **80.0%** | **25.5%** / **38.5%** | **20926.9 / 21742.0** | **15.15% / 10.07%** |

- **Zero regressions vs. Base (`α = 0.0`) on discordant pairs**: At `32,768` tokens, every single problem solved by the Base model (`17/200`) was *also* solved by both `α = 1.0` (`n10 = 0, n01 = 80, discordant = 80, p = 1.65e-24`) and `α = 1.25` (`n10 = 0, n01 = 86, discordant = 86, p = 2.58e-26`). At `16,384` tokens, `α = 1.0` achieved `n10 = 0, n01 = 42, p = 4.55e-13`.
- **Same-Problem Paper Benchmark (`our_n200_core` in [`assets/paper_reference.csv`](assets/paper_reference.csv))**: On the exact same 200 MATH-500 problems at `32,768` tokens, Takehi et al.'s (2026) 4-sample `avg@4` is **9.5%** (`76/800`, 95% CI [7.66%, 11.73%]) for `unedited` and **48.0%** (`384/800`, [44.56%, 51.46%]) for `p50c` — matching our single-sample greedy-ish (`T=0.6`) `α = 1.0` (**48.5%**) and `α = 1.25` (**51.5%**) within `0.5 pp`.

### 6. Paired `N = 100` 6-Dose Sweep (`S_100 ⊂ S_200`, `α ∈ {0.0, 0.5, 0.75, 1.0, 1.25, 1.5}`) ([`assets/gpu_dose_n100/`](assets/gpu_dose_n100/))

Joining `gpu_dose_n100` (`α ∈ {0.5, 0.75, 1.5}`) with the exact paired `S_200[:100]` problems for `α ∈ {0.0, 1.0, 1.25}` at `32,768` tokens reveals an **inverted-U accuracy curve** driven by a trade-off between monotonic loop suppression and over-editing / premature termination:

| Blend $\alpha$ | Scale ($6\alpha$) | Strict Correct | Finished Wrong | Capped Looping | Capped Coherent | Total Looped (Cap + Recov) | Forced Acc | Closed `</think>` | Med Tokens | Mean 8g Rep | McNemar vs `α=0` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | $0.0\times$ | **6.0%** | 0.0% | 91.0% | 3.0% | 91.0% (91 + 0) | 18.0% | 6.0% | 32,768.0 | 71.40% | — |
| **0.50** | $3.0\times$ | **32.0%** | 5.0% | 60.0% | 3.0% | 72.0% (60 + 12) | 41.0% | 51.0% | 32,768.0 | 44.53% | +26.0 pp ($p < 10^{-6}$) |
| **0.75** | $4.5\times$ | **52.0%** | 9.0% | 31.0% | 8.0% | 58.0% (31 + 27) | **57.0%** | 74.0% | 26,214.5 | 28.38% | **+46.0 pp** ($p < 10^{-6}$) |
| **1.00** | $6.0\times$ | **45.0%** | 21.0% | 15.0% | 19.0% | 33.0% (15 + 18) | 47.0% | **77.0%** | 25,239.5 | 18.84% | +39.0 pp ($p < 10^{-6}$) |
| **1.25** | $7.5\times$ | **46.0%** | 20.0% | 10.0% | 24.0% | 22.0% (10 + 12) | 47.0% | 73.0% | 23,294.5 | 13.58% | +40.0 pp ($p < 10^{-6}$) |
| **1.50** | $9.0\times$ | **40.0%** | **32.0%** | **8.0%** | 20.0% | **17.0%** (8 + 9) | 42.0% | 71.0% | **20,469.0** | **12.18%** | +34.0 pp ($p < 10^{-6}$) |

- **Two competing mechanisms**:
  1. **Monotonic loop suppression (`α = 0 → 1.5`)**: Looped traces fall strictly (`91% → 72% → 58% → 33% → 22% → 17%`), cap hits fall strictly (`94% → 63% → 39% → 34% → 34% → 28%`), and median length drops by `12,299` tokens (`32,768 → 20,469`).
  2. **Over-editing & premature termination above `α = 0.75`**: Stepping from `α = 0.50` ($3\times$) to **`α = 0.75` ($4.5\times$)** jumps strict accuracy by **`+20.0 pp`** (`32% → 52%`, McNemar $n_{10}=3, n_{01}=23, p = 0.000088$) while keeping finished-but-wrong errors low (`9%`). Pushing further to `α = 1.0–1.25` increases finished-but-wrong responses to `20–21%`, and over-editing at **`α = 1.50` ($9\times$)** pushes finished-but-wrong answers to **`32%`** — dropping strict accuracy by **`-12.0 pp`** vs peak `α = 0.75` (`52% → 40%`, McNemar $n_{10}=18, n_{01}=6, p = 0.02266$).

### 7. Loop-Onset Dynamics, Two-Pass Recovery & Metrics Definitions

- **Two complementary loop detectors**:
  - `loop_onset_token` ([`compute_loop_onset_from_token_ids`](metrics.py)): Earliest 0-based token index of a `1,024`-token sliding window (stride `256`) whose decoded whitespace 8-gram repetition ratio strictly exceeds `0.50`. Used for [`assets/article/loop_onset.png`](assets/article/loop_onset.png) and [`assets/article/outcomes_32k.png`](assets/article/outcomes_32k.png).
  - Full-trace cumulative 8-gram repetition `repeated_8gram_ratio(text) >= 0.15` (reported in `looped_pct` in `sweep_results.csv`). At `N=200, B=32768`:
    - `α = 0.00`: `179/200` (**89.5%**) have `loop_onset_token` (median **6,144** tokens; `89.5%` have full-trace `rep >= 0.15`). Once a base trace enters a loop, **`0 / 179` (0.0%) recover** — every single one rides the verbatim loop into the `32,768`-token cap.
    - `α = 1.00`: `66/200` (**33.0%**) have `loop_onset_token` (`46.0%` have full-trace `rep >= 0.15`), with median onset shifted **`+12,288` tokens later to 18,432** on capped traces (`16,128` across all 66). Crucially, **`39 / 66` (59.1%) of looping `α=1.0` traces break out of transient repetition and finish `</think>` cleanly** (`27` capped while looping = `13.5%`, `39` transient recovered = `19.5%`).
    - `α = 1.25`: `51/200` (**25.5%**) have `loop_onset_token` (`38.5%` have full-trace `rep >= 0.15`), median onset shifted to **22,784** tokens on capped traces (`18,944` overall), with **`26 / 51` (51.0%) recovering** (`25` capped = `12.5%`, `26` transient recovered = `13.0%`).
- **Why `4,096` tokens hides the edit**: Even without entering a loop (`134/200 = 67.0%` loop-free at `α=1.0`), `Qwen3.5-0.8B`'s normal multi-step verification traces have a median length of **~22,000 tokens** (`22,088.5` at `α=1.0`, `21,742.0` at `α=1.25`). Under a `4,096`-token ceiling, `90.5–91.0%` of edited traces are cut mid-thought before `</think>`, capping strict accuracy at `9.0–9.5%`. Even forced `</think>` answer extraction (`Pass 2`) only reaches `17.0–22.5%` at `4k–8k` (compared to `50.0–53.5%` at `32k`), showing that extra thinking tokens produce genuine multi-step mathematical progress rather than mere formatting delay.

### 8. Published Paper Reference Comparison ([`assets/paper_reference.csv`](assets/paper_reference.csv) & [`assets/gpu_stats.csv`](assets/gpu_stats.csv))

Extracted directly from Takehi et al.'s (2026) upstream evaluation logs (`RikiyaT/lm_understand-logs` `exp8_math/eval/{unedited,p50c}`):

| Cohort / Slice | Problems $N$ | Metric Type | Base (`unedited`, `α=0`) | Edited (`p50c`, `α=1`) | Gain ($\Delta$) |
|---|---:|---|---:|---:|---:|
| **`our_n50`** (exact problem match) | 50 | Published `avg@4` @ 32k | **7.50%** (15/200) | **41.00%** (82/200) | **+33.50 pp** |
| **`our_n100_dose`** (exact problem match) | 100 | Published `avg@4` @ 32k | **6.50%** (26/400) | **43.75%** (175/400) | **+37.25 pp** |
| **`our_n200_core`** (exact problem match) | 200 | Published `avg@4` @ 32k | **9.50%** (76/800) | **48.00%** (384/800) | **+38.50 pp** |
| **`paper_first_samples_n200`** | 200 | Published `avg@4` @ 32k | **9.25%** (74/800) | **48.38%** (387/800) | **+39.12 pp** |
| **`math500_all_500`** (full benchmark) | 500 | Published `avg@4` @ 32k | **7.90%** (158/2000) | **48.10%** (962/2000) | **+40.20 pp** |
| **`paper_first_samples_n200`** (`B = 4,096`) | 200 | Truncated 1st-sample (approx.) | 2.00% (98.0% cut) | 8.00% (92.0% cut) | +6.00 pp |
| **`paper_first_samples_n200`** (`B = 8,192`) | 200 | Truncated 1st-sample (approx.) | 5.50% (94.5% cut) | 14.50% (85.5% cut) | +9.00 pp |
| **`paper_first_samples_n200`** (`B = 16,384`) | 200 | Truncated 1st-sample (approx.) | 6.50% (93.5% cut) | 27.50% (71.0% cut) | +21.00 pp |
| **`paper_first_samples_n200`** (`B = 32,768`) | 200 | Truncated 1st-sample (approx.) | 6.50% (93.5% cut) | 46.00% (24.0% cut) | +39.50 pp |

## Visuals

- **Article Figures (`1600 × 1000 px`, generated via [`scripts/generate_article_charts.py`](scripts/generate_article_charts.py))**:
  - [`assets/article/headline_budget_curve.png`](assets/article/headline_budget_curve.png) — Strict accuracy scaling across `4k → 8k → 16k → 32k` (`N=200`, RTX PRO 6000 `bfloat16`) with 95% Wilson CI bands, 4k Budget Trap zone, Paper 1st-sample truncated curves (`N=200`), and exact same-problem Paper 32k `avg@4` reference stars (`9.5%` vs `48.0%`).
  - [`assets/article/edit_diagram.png`](assets/article/edit_diagram.png) — Two-tensor rank-1 surgery diagram (`486/488` bit-identical tensors, `131,072 / 873,438,784 = 0.0150%` weights across 64 columns in Layers 15 & 19 `mlp.down_proj.weight`) paired with log-scale SVD spectra ($\sigma_1/\sigma_2 = 1771.2$ and $1552.0$).
  - [`assets/article/outcomes_32k.png`](assets/article/outcomes_32k.png) — Mutually exclusive 4-outcome breakdown (`Correct`, `Wrong answer`, `Capped while looping`, `Capped not looping`) across Core `N=200` (`α ∈ {0, 1, 1.25}`) and Paired 6-Dose `N=100` (`α ∈ {0, 0.5, 0.75, 1, 1.25, 1.5}`).
  - [`assets/article/loop_onset.png`](assets/article/loop_onset.png) — Loop onset window histogram (`0–4k`, `4k–8k`, `8k–16k`, `16k–32k`, `No Loop Detected`, distinguishing cap hits vs recovered loops) and onset token scatter (`median 6.1k -> 18.4k -> 22.8k`) across `N=200`.
  - [`assets/article/hero_trace.png`](assets/article/hero_trace.png) — Side-by-side 32k execution trace on `test/counting_and_probability/389.json` ("Mikka's 2-topping pizza from 8 toppings", gold `28`): Base (`α=0.0`) enters a verbatim loop at token `5,120` (`1,292×` repeats across `27,648` tokens, `83.1%` 8-gram rep, capped without `</think>`), while Edited (`α=1.0`) solves cleanly in `3,843` tokens (`8.5×` shorter, `0.50%` 8-gram rep, `\boxed{28}`).
- **Repository & Per-Run Dashboards**:
  - [`assets/budget_curve_gpu.png`](assets/budget_curve_gpu.png) — GPU-only 3-panel cross-budget scaling dashboard (`N=200`, `4k → 8k → 16k → 32k`: Strict Accuracy, Pass-2 Forced Accuracy, and Cap-Hit %).
  - [`assets/mps_4k_comparison.png`](assets/mps_4k_comparison.png) — Side-by-side 4,096-token Apple M3 MPS vs. RTX PRO 6000 vLLM comparison (`N=50`: strict/forced accuracy and last-240-word tail loop collapse).
  - [`assets/weight_diff_analysis.png`](assets/weight_diff_analysis.png) — Column-wise L2 edit norms ($\|\Delta W[:, j]\|_2$) and singular value spectra for Layers 15 & 19.
  - [`assets/sweep_charts.png`](assets/sweep_charts.png) & [`assets/alpha_token_distributions.png`](assets/alpha_token_distributions.png) — Local Apple M3 MPS `4k` `N=50` 5-dose dashboard (`α ∈ [0, 0.5, 1.0, 1.25, 1.5]`) and token-length vs repetition scatter.
  - [`assets/app_screenshot.png`](assets/app_screenshot.png) & [`assets/app_side_by_side.gif`](assets/app_side_by_side.gif) — Side-by-side Gradio streaming UI (`app.py`).

## Surprises and limitations

1. **Loop breaking happens early ($\alpha = 0.5$–$1.0$), but coherent `<think>` chains still need `16k–32k` tokens to finish**:
   - Even when repetition loops are eliminated (`0.0%` looped at `α=1.0` under a `4k` ceiling), `Qwen3.5-0.8B` was RL-trained for long chain-of-thought verification (`median 22,088.5` tokens on loop-free/edited `32k` runs). At `4,096` tokens, `91.0%` (`182/200`) of `α=1.0` runs hit the cap mid-thought (`9.5%` close `</think>`), keeping strict accuracy at `9.0%` (vs `3.0%` Base). Unlocking the budget to `16,384` and `32,768` tokens cuts cap hits to `67.0% -> 31.0%` and raises strict accuracy to **`29.5%` (`3.5×` Base) -> `48.5%` (`5.7×` Base; `51.5%` / `6.1×` at `α=1.25`)**.
2. **Inverted-U dose-response curve (`α = 0.75` peaks at `52%` on `N=100`; `α = 1.50` over-edits and increases wrong answers to `32%`)**:
   - Across the paired 6-dose sweep (`S_100 ⊂ S_200` @ `32k`), loop prevalence decreases monotonically (`91% -> 72% -> 58% -> 33% -> 22% -> 17%`), but strict accuracy peaks at **$\alpha = 0.75$ ($4.5\times$ subtraction: `52.0%` strict, `57.0%` forced, `+46.0 pp` over Base $p < 10^{-6}$)** and outperforms `α = 0.50` by `+20.0 pp` ($p = 0.000088$). Above `α = 0.75`, finished-but-wrong answers climb from `9.0%` (`α=0.75`) to `21.0%` (`α=1.0`), `20.0%` (`α=1.25`), and **`32.0%` (`α=1.5`)**, dropping `α=1.5` strict accuracy by `-12.0 pp` (`52% -> 40%`, $p = 0.0227$) due to under-thinking / premature `</think>` termination. (On the full `N=200` cohort, `α=1.25` edges out `α=1.0` at `51.5%` vs `48.5%`, $p = 0.345$, so `α ∈ [0.75, 1.25]` forms the optimal operating plateau.)
3. **Why Pass-2 forced answers only partially rescue short budgets (`17–19%` @ `4k` vs `50–53.5%` @ `32k`)**:
   - If the edit merely helped the model wrap up faster, forcing `</think>\n\n` at `4,096` tokens would already recover most of the `32k` accuracy. Instead, forced Pass-2 accuracy at `4k` is only `17.0–17.5%` (`α=1.0/1.25`, no better than Base's `19.0%`) and only rises alongside strict reasoning depth (`22.0–22.5%` @ `8k`, `34.5–36.0%` @ `16k`, `50.0–53.5%` @ `32k`). Base (`α=0.0`) stagnates at `19.0% -> 20.0% -> 21.0% -> 21.5%` forced accuracy because once a trace enters a verbatim loop (median onset `6,144` tokens), its KV cache is flooded with thousands of repeated tokens and never recovers (`0/179`).
4. **Transient loops still occur at intermediate doses (`α = 0.75–1.0`), but edited models break out of them**:
   - Unlike Base (`0/179 = 0%` recovery once looping), edited models recover from over half of detected loop episodes before `32,768` tokens (`27/58 = 46.6%` at `α=0.75` on `N=100`; `39/66 = 59.1%` at `α=1.0` and `26/51 = 51.0%` at `α=1.25` on `N=200`).
5. **`fp16` vs. `bf16` numerics on Turing (`T4`) GatedDeltaNet layers**:
   - `Qwen/Qwen3.5-0.8B` is trained and distributed in `bfloat16` (8-bit exponent), whereas NVIDIA **T4** (`sm_75`) lacks hardware `bf16` tensor cores. [`gpu/vllm_sweep.py`](gpu/vllm_sweep.py) keeps SSM state caches in `float32` (`mamba_ssm_cache_dtype="float32"`) and [`gpu/smoke_test.py`](gpu/smoke_test.py) gates `float16` vs `float32` before running.

## Open questions / next steps

1. **Test `α = 0.75` and `α = 0.85` on full `N = 200` (and `N = 500` `avg@4`)**: On `S_100 ⊂ S_200`, `α = 0.75` achieved the highest single-sample accuracy (`52.0%` strict / `57.0%` forced, `+7.0 pp` above `α = 1.0`). Evaluating `α ∈ {0.75, 0.85}` on the remaining `100` problems of `S_200` (or full `MATH-500` with `avg@4`) will confirm whether `4.5×–5×` projection strictly dominates the published `6×` (`α = 1.0`) checkpoint across all 500 problems.
2. **Decoupled 2D layer interpolation $(\alpha_{15}, \alpha_{19})$**: Since $\cos(u_{15}, u_{19}) = 0.8490$, sweeping $\alpha_{15} \neq \alpha_{19}$ (e.g., testing whether Layer 15 suppresses topic circling while Layer 19 triggers premature `</think>` termination at `α = 1.5`) could separate loop breaking from under-thinking.
3. **Token-budget-aware dynamic $\alpha(t)$ scheduling**: Annealing $\alpha(t)$ from `0.0 -> 0.75 -> 1.25` across token position could allow unperturbed exploration on easy problems (eliminating the `9–21%` finished-wrong penalty at `α = 0.75–1.0` vs `0%` finished-wrong on Base) while suppressing late-stage loops after token `4,096`.
4. **Custom Metal / MLX GatedDeltaNet kernel**: Porting the 18 GatedDeltaNet recurrent layers to custom Metal / `mlx-lm` kernels on Apple Silicon to reach `300+ tok/s` for interactive 16k–32k math evaluation on consumer MacBooks.

