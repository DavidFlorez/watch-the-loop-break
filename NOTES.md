# NOTES.md — `watch-the-loop-break` Build Log

## Goal

Reproduce the Qwen3.5-0.8B "Weight-Space Programming" edit ([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)), verify that only `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` differ between base and edited models, and extend the result by interpolating $W(\alpha) = W_{\text{base}} + \alpha(W_{\text{edit}} - W_{\text{base}})$ across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ on MATH-500 and in a side-by-side live Gradio streaming app.

## Setup

- **Models**:
  - Base: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (`24` hidden layers, `1024` hidden size, `3584` MLP intermediate size, hybrid GatedDeltaNet + full-attention architecture, `488` tensors / `873,438,784` total parameters in `bfloat16`)
  - Edited: [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) (`p50c` weight-space edit)
  - Reference logs & rebuild scripts: [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/runs/p50c`)
- **Dataset**:
  - [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (`test` split, 500 competition math problems across 7 subjects and 5 difficulty levels; deterministic $N=50$ evaluation subset sampled with `seed=0`)
- **Hardware**:
  - Apple MacBook Air (`Mac15,12`), Apple M3 chip (8-core CPU, 10-core GPU), 16 GB unified memory (`mps` backend auto-detected via `CUDA -> MPS -> CPU` fallback in [`detect_hardware`](weights.py))
- **Library Versions** ([`requirements.txt`](requirements.txt)):
  - Python `3.13.7`
  - `torch==2.14.1`, `transformers==5.21.0`, `safetensors==0.8.0`, `accelerate==1.15.0`
  - `datasets==5.1.0`, `huggingface_hub==2.3.0`, `gradio==6.30.0`, `math-verify==0.9.0`
  - `numpy==2.5.4`, `pandas==3.1.0`, `matplotlib==3.11.2`, `pillow==12.3.0`, `tqdm==4.70.1`

## Log

- **2026-11-03 10:06 PST — Project initialization & literature/code audit**
  - Read ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/), the [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) model card, and the official evaluation/rebuild scripts in [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/code/apply_edit.py`, `exp8_math/code/evaluate.py`, `exp8_math/code/mathqa.py`, `exp8_math/runs/p50c/work_notes/final_report.md`).
  - Confirmed the mathematical form of Astra's `p50c` edit: in `Qwen3.5-0.8B`, Astra identified unit repetition directions $u_{15}, u_{19} \in \mathbb{R}^{1024}$ at the residual outputs of layers 15 and 19, selected the 64 MLP output columns (`down_proj.weight` shape `[1024, 3584]`) that wrote most strongly along those directions, and applied $W_{\text{edit}} = W_{\text{base}} - 6 \cdot u (u^\top W_{\text{base}} \odot \mathbb{1}_{64})$.
  - Consequently, linear interpolation $W(\alpha) = W_{\text{base}} + \alpha (W_{\text{edit}} - W_{\text{base}})$ directly varies the directional ablation dose by $6\alpha$ along those 64 columns in layers 15 and 19 ($\alpha = 0$: base model; $\alpha = 0.5$: $3\times$ projection subtraction; $\alpha = 1.0$: Astra's published $6\times$ edit; $\alpha = 1.25$: $7.5\times$ over-subtraction; $\alpha = 1.5$: $9\times$ over-subtraction).
  - Initialized git repository on `main`, created `.venv`, `.gitignore`, [`pyproject.toml`](pyproject.toml), and [`requirements.txt`](requirements.txt).

- **2026-11-03 10:31 PST — Full tensor verification & `blend(alpha)` implementation (`0a73ef1`)**
  - Executed [`verify_weight_diff`](weights.py) across all **488 tensors (`873,438,784` parameters)** in [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
  - Confirmed bit-for-bit in `bfloat16` (`uint16` view) that **486 of 488 tensors are identical** and **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ (`[1024, 3584]`, `bfloat16`).
  - Verified that [`blend`](weights.py) reproduces exact `bfloat16` SHA-256 digests at $\alpha = 0.0$ and $\alpha = 1.0$, and generated [`assets/weight_verification.json`](assets/weight_verification.json) and [`assets/weight_diff_analysis.png`](assets/weight_diff_analysis.png).
  - Downloaded official HF evaluation reference files (`eval/scores_base_math500.json`, `eval/scores_edited_math500.json`, `eval/samples_base_math500.jsonl`, `eval/samples_edited_math500.jsonl`) for cross-checking grading logic (`math-verify` + Hendrycks LaTeX normalization) and 32k-token baseline metrics.

- **2026-11-03 11:17 PST — Apple Silicon MPS decode kernel optimization & dynamic EOS batching (`d802a6c`)**
  - **What broke**: Initial batched evaluation on MPS fell back to `transformers`'s pure-PyTorch `causal_conv1d_update` (`F.conv1d(..., groups=6144)`) and 4D-broadcast `torch_recurrent_gated_delta_rule`, causing multi-minute stalled prefill/decode steps, and kept finished sequences active in the batch all the way to step 4096.
  - **Fix**: Added [`install_fast_qwen35_kernels`](weights.py) (replacing single-token `causal_conv1d_update` with in-place 4-tap state shift + elementwise multiply and replacing `torch_recurrent_gated_delta_rule` with `torch.matmul` state updates), incremental `_penalty_buf` in [`PresencePenaltyLogitsProcessor`](metrics.py), and dynamic EOS batch compaction (`past.reorder_cache(keep_idx)`) in [`generate_batch`](sweep.py).

- **2026-11-03 11:46–11:50 PST — Solving MPS unified-memory swap thrashing (`4e98088`, `0fb9182`)**
  - **What broke**: On a 16 GB Apple M3 unified-memory machine, large batches (`batch_size=50` or `batch_size=10` without cache flushing) pushed `MPSHeapAllocator` + attention KV cache (`DynamicLayer` growing `+1` token per step via `torch.cat`) past the ~3.5 GB unthrottled RAM headroom, triggering macOS memory compression (`CMPRS` > 5 GB) and SSD swap thrashing.
  - **Fix**: Sorted pending prompts by character length before batching (minimizing left-padding waste), set default `batch_size=8` (keeping `mps.driver_allocated_memory` steady at `~2.44 GB` with `0` swap pressure), pre-allocated `full_mask`, top-k-filtered before top-p (`topk(20)`), added `torch.mps.empty_cache()` every 64 steps, and sorted `summary_df` ascending by `alpha`. Also mirrored the MPS sampling/cache optimizations in [`app.py`](app.py) (`4da771e`).

- **2026-11-03 11:50–12:15 PST — Running $N=50$ MATH-500 $\alpha$-sweep, unit tests, and Gradio capture**
  - Added automated unit test suite [`tests/test_core.py`](tests/test_core.py) verifying hardware detection, SHA-256 endpoint fidelity and linearity of `blend(alpha)`, 8-gram repetition metrics, presence penalty, boxed math grading, and XSS-safe HTML rendering.
  - Executed `sweep.py --n 50 --alphas 1.0,1.25,1.5,0.5,0 --batch-size 8 --max-new-tokens 4096` and captured live side-by-side Gradio streaming screenshots/GIFs.

## Key results

### 1. Weight Verification Summary ([`assets/weight_verification.json`](assets/weight_verification.json))
- **Total tensors checked**: `488` across both checkpoints; **identical tensors**: `486`; **differing tensors**: `2` (`only_target_tensors_differ = true`).
- **Differing parameter footprint**: `64` columns per `[1024, 3584]` matrix (`65,536` column entries × `2` = `131,072` touched entries; `130,188` non-zero `bfloat16` bit changes) out of `873,438,784` total parameters (**`0.0150%`** of the model).
- **Exact rank-1 structure & shared loop direction**:
  - `model.language_model.layers.15.mlp.down_proj.weight`: $\sigma_1 = 1.7489$, $\sigma_2 = 0.00099$ ($\sigma_1/\sigma_2 = 1771.2$, **99.99945%** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 10.64\%$, `base_sha256` = `ef60d2c6...`, `edit_sha256` = `82e2d733...`.
  - `model.language_model.layers.19.mlp.down_proj.weight`: $\sigma_1 = 1.3988$, $\sigma_2 = 0.00090$ ($\sigma_1/\sigma_2 = 1552.0$, **99.99914%** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 8.11\%$, `base_sha256` = `638eedeb...`, `edit_sha256` = `8ef3c61b...`.
  - Principal left singular vectors across layers 15 and 19 have cosine similarity $\cos(u_{15}, u_{19}) = \mathbf{0.8490}$, confirming both edits subtract a shared residual repetition subspace.

### 2. MATH-500 $\alpha$-Sweep Results ($N=50$, `max_new_tokens=4096`) vs. Official 32k Reference
*(Updating automatically with final $N=50$ sweep measurements from [`assets/sweep_results.csv`](assets/sweep_results.csv)...)*

## Visuals

- `assets/weight_diff_analysis.png` — Column-wise L2 edit norms ($\|\Delta W[:, j]\|_2$) and log-scale singular value spectra for Layers 15 and 19 `mlp.down_proj.weight`, demonstrating that only 64 columns per tensor are modified via a rank-1 projection.
- `assets/app_screenshot.png` — Side-by-side Gradio interface comparing Base (`alpha=0.0`) vs. Weight-Space Edited (`alpha=1.0`) on a MATH-500 problem with live repeated 8-gram meters and token-cap badges.
- `assets/app_side_by_side.gif` — Animated live capture of `app.py` streaming Base (`alpha=0.0`) vs. Edited (`alpha=1.0`), showing the base model entering a repetitive thinking loop while the edited model terminates cleanly with `\boxed{}`.
- `assets/sweep_charts.png` — Four-panel dose-response dashboard across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ showing MATH-500 accuracy, 4096-token cap truncation rate, mean generated tokens, repeated 8-gram ratio, and wall time.
- `assets/alpha_token_distributions.png` — Per-problem token length distributions (solved vs. capped/wrong) and token count vs. repeated 8-gram ratio scatter across $\alpha$.

## Surprises and limitations

- *(To be finalized after sweep completion.)*

## Open questions / next steps

- *(To be finalized after sweep completion.)*
