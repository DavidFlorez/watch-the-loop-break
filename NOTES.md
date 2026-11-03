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
  - Apple MacBook Air (`Mac15,12`), Apple M3 chip (8-core CPU, 10-core GPU), 16 GB unified memory (`mps` backend auto-detected via `CUDA -> MPS -> CPU` fallback in [`detect_hardware`](weights.py))
- **Library Versions** ([`requirements.txt`](requirements.txt)):
  - Python `3.13.7`
  - `torch==2.14.1`, `transformers==5.21.0`, `safetensors==0.8.0`, `accelerate==1.15.0`
  - `datasets==5.1.0`, `huggingface_hub==2.3.0`, `gradio==6.30.0`, `math-verify==0.9.0`
  - `numpy==2.5.4`, `pandas==3.1.0`, `matplotlib==3.11.2`, `pillow==12.3.0`, `tqdm==4.70.1`

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
  - **Fix #3**: Enforced explicit high-contrast inline `#0f172a` / `#1e293b` styles in [`render_repetition_meter_html`](app.py), replaced raw TeX escapes with clean Unicode math, paced [`replay_recorded_comparison`](app.py) by token progress so shorter solved runs finish visually before capped base runs, and verified all 7 unit tests in [`tests/test_core.py`](tests/test_core.py) (`0.510s`, `OK`).

## Key results

### 1. Exact Weight-Space Diff & SVD Analysis ([`assets/weight_verification.json`](assets/weight_verification.json))

- **Tensors checked**: `488` total (`873,438,784` parameters); **`486` bit-identical** (`bfloat16` `uint16` view); **`2` modified** (`only_target_tensors_differ = true`).
- **Parameter footprint**: `64` columns per `[1024, 3584]` matrix (`65,536 × 2 = 131,072` column entries; `130,188` non-zero `bfloat16` entries changed) = **`0.0150%`** of model parameters.
- **Rank-1 structure**:
  - `model.language_model.layers.15.mlp.down_proj.weight`: $\sigma_1 = 1.7489$, $\sigma_2 = 0.00099$ ($\sigma_1 / \sigma_2 = \mathbf{1771.2}$, **`99.99945%`** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 10.64\%$, `base_sha256` = `ef60d2c6...`, `edit_sha256` = `82e2d733...`.
  - `model.language_model.layers.19.mlp.down_proj.weight`: $\sigma_1 = 1.3988$, $\sigma_2 = 0.00090$ ($\sigma_1 / \sigma_2 = \mathbf{1552.0}$, **`99.99914%`** rank-1 energy), $\|\Delta W\|_F / \|W_{\text{base}}\|_F = 8.11\%$, `base_sha256` = `638eedeb...`, `edit_sha256` = `8ef3c61b...`.
  - Principal left singular vectors align across layers 15 and 19 with $\cos(u_{15}, u_{19}) = \mathbf{0.8490}$.
  - `blend(0.0)` and `blend(1.0)` match the exact `bfloat16` SHA-256 digests of the base and edited checkpoints.

### 2. MATH-500 $\alpha$-Sweep Results ($N=50$, `max_new_tokens=4096`) ([`assets/sweep_results.csv`](assets/sweep_results.csv))

Measured on 50 deterministic `HuggingFaceH4/MATH-500` problems (`seed=0`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) using `math-verify` + Hendrycks LaTeX grading:

| Blend $\alpha$ | Ablation Scale ($6\alpha$) | Correct ($N=50$) | Accuracy (%) | Hit 4096 Cap (%) | Mean Tokens | Full-Trace 8-Gram Rep (%) | Final-512w Tail Rep (%) | Wall Time (s) | Speed (s/prob) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** (Base) | $0.0\times$ | 2 / 50 | 4.0% | 98.0% (49/50) | 4055.9 | 8.44% | **14.71%** | 3157.01s | 63.14s |
| **0.50** | $3.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 3990.4 | 3.74% | 1.22% | 1918.79s | 38.38s |
| **1.00** (Published) | $6.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 4028.7 | 2.99% | 1.39% | 2609.49s | 52.19s |
| **1.25** (**Optimal @ 4k**) | $7.5\times$ | **6 / 50** | **12.0%** | **88.0% (44/50)** | **3982.9** | 2.69% | 1.35% | 1995.77s | 39.92s |
| **1.50** (Over-ablation) | $9.0\times$ | 4 / 50 | 8.0% | 92.0% (46/50) | 4019.4 | **2.37%** | **0.74%** | **1890.71s** | **37.81s** |

- **Official 32k Budget Comparison (`RikiyaT/lm_understand-logs` `exp8_math/runs/p50c`)**:
  - Base (`alpha=0.0`, `max_new_tokens=32768`): `5.0%` accuracy, `95.0%` cut share, `31,690.0` mean tokens, `73.79%` mean repeated 8-gram ratio (`79.84%` median).
  - Edited (`alpha=1.0`, `max_new_tokens=32768`): `51.0%` accuracy, `31.6%` cut share, `21,703.8` mean tokens, `23.09%` mean repeated 8-gram ratio (`16.50%` median).
- **MPS Decode Kernel Speedup**:
  - Replacing `transformers`'s pure-PyTorch `causal_conv1d_update` and 4D-broadcast `torch_recurrent_gated_delta_rule` + dropping unused visual weights reduced 512 decode steps (`B=10`) from `148.9s` (under swap) down to **`40–45s`** (~`111–125 tok/s` across 10 sequences; `3.34x` throughput scaling from `B=1` `8.9 tok/s` to `B=8` `29.7 tok/s` at `2.29 GB` MPS driver memory).

## Visuals

- `assets/weight_diff_analysis.png` — Column-wise L2 edit norms ($\|\Delta W[:, j]\|_2$) and log-scale singular value spectra for Layers 15 and 19 `mlp.down_proj.weight`, proving only 64 columns per tensor change via a rank-1 projection.
- `assets/app_screenshot.png` — Side-by-side Gradio interface on `test/algebra/1936.json` comparing Base (`α = 0.00`, hitting the `4096 / 4096` token cap) vs. Edited (`α = 1.00`, finishing cleanly in `2689` tokens with exact `\boxed{-1}`).
- `assets/app_side_by_side.gif` — Animated capture of `app.py` streaming Base (`α = 0.00`) vs. Edited (`α = 1.00`), showing the edited model boxing `-1` while the base model keeps circling until hitting the red `TOKEN CAP HIT` alert.
- `assets/sweep_charts.png` — Four-panel dose-response dashboard across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ showing MATH-500 accuracy peaking at $\alpha = 1.25$ (`12.0%`), 4096-token cap truncation dropping to `88.0%`, full-trace and tail 8-gram repetition collapsing monotonically (`14.71% -> 0.74%`), and evaluation wall times.
- `assets/alpha_token_distributions.png` — Per-problem token lengths (solved vs. capped/wrong) and token count vs. repeated 8-gram ratio scatter across all 250 runs.

## Surprises and limitations

1. **Loop breaking happens immediately by $\alpha = 0.5$–$1.0$, but coherent over-thinking still saturates a 4,096-token budget**:
   - At $\alpha = 0.0$, worst-case traces lock into severe verbatim repetition loops (`test/prealgebra/2066.json` reached `52.1%` overall / `70.8%` tail repeated 8-grams; `test/number_theory/631.json` reached `48.2%` overall / `94.8%` tail repetition).
   - Raising $\alpha$ from `0.0 -> 0.5 -> 1.0` slashes final-512-token repetition **12×** (`14.71% -> 1.22% -> 1.39%`), completely eliminating verbatim loops. However, because `Qwen3.5-0.8B` was trained for `32,768`-token `<think>` budgets (where $\alpha=1.0$ averages `21,704` tokens and `51.0%` accuracy), a strict `4,096`-token cap still truncates `94.0%` (`47/50`) of $\alpha=1.0$ trajectories mid-verification!
2. **Inverted-U dose-response curve ($\alpha = 1.25$ doubles 4k accuracy over $\alpha = 1.0$)**:
   - Extrapolating the directional ablation slightly past the published edit to **$\alpha = 1.25$** ($7.5\times$ subtraction) shortens over-cautious `<think>` verification chains just enough to raise completion before 4,096 tokens (`88.0%` hit cap vs. `94.0%`), **doubling accuracy from `6.0% (3/50)` at $\alpha = 1.0$ to `12.0% (6/50)` at $\alpha = 1.25$** (`3×` base accuracy), while **every single run across $\alpha \in \{0.5, 1.0, 1.25, 1.5\}$ that finished before the 4,096-token cap produced the exact right answer (`100%` precision conditioned on termination: `3/3`, `3/3`, `6/6`, `4/4`)**.
   - Pushing further to **$\alpha = 1.5$** ($9\times$ subtraction) reduces repetition even further (`0.74%` tail), but over-subtracts the residual direction on harder problems, dropping accuracy back to `8.0% (4/50)`.
3. **Why $\alpha=0$ full-trace 8-gram ratio is `8.44%` at 4k vs `73.8%` at 32k**:
   - Loops in `Qwen3.5-0.8B` typically onset around tokens `2,500–4,500` and then repeat for the remaining `28,000+` tokens under a `32,768` cap. At `4,096` tokens, those loops are caught right as they begin — which is why the final-512-token window (`mean_tail_rep_8gram_pct = 14.71%`, up to `94.8%` on individual problems) exposes the base-model loop onset much more sharply than the full-4096 average.
4. **Statistical power ($N=50$) & single-machine runtime**:
   - At $N=50$ under a strict `4,096`-token ceiling, accuracy shifts from `2/50` (`4%`) to `6/50` (`12%`) reflect finite-sample variance; full-dataset (`N=500`) runs at `8192` or `16384` tokens would reduce quantization from the `4096` cutoff.

## Open questions / next steps

1. **Decoupled 2D layer interpolation $(\alpha_{15}, \alpha_{19})$**: Since $\cos(u_{15}, u_{19}) = 0.849$, sweeping $\alpha_{15} \neq \alpha_{19}$ (e.g., testing whether Layer 15 controls topic circling while Layer 19 controls token-level repetition) could separate loop breaking from premature stopping.
2. **Token-budget-aware dynamic $\alpha(t)$ scheduling**: Annealing $\alpha(t)$ upward as $t \to T_{\max}$ (`0.75 -> 1.35`) could preserve full exploration early in `<think>` while nudging the model to conclude and `\boxed{}` cleanly before any deployment token ceiling (`2048` or `4096`).
3. **Context-length frontier curve (`max_new_tokens` $\in \{2048, 4096, 8192, 16384, 32768\}$)**: Mapping the exact Pareto frontier of accuracy vs. token budget across $\alpha \in [1.0, 1.25]$ to determine whether $\alpha = 1.25$ strictly dominates $\alpha = 1.0$ for low-latency deployment (`<= 8k` tokens).
4. **Custom Metal / MLX GatedDeltaNet kernel**: Porting the 18 GatedDeltaNet recurrent layers to custom Metal / `mlx-lm` kernels on Apple Silicon to reach `300+ tok/s` for interactive 16k–32k math evaluation on consumer MacBooks.
