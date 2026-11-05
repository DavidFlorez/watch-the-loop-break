# `watch-the-loop-break` 🧠

Reproducing, stress-testing, and extending Experiment 8 of **["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)** ([Takehi, Teoh, Zhang & Khattab, 2026](https://rikiyat.github.io/blog/weight-space-programming/)) on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) vs. [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

> **Headline Finding — Why a 4k Token Cap Hides a 6× Gain:**
> Subtracting a single rank-1 repetition direction from **2 of 488 tensors** (`layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight`, **0.0150%** of parameters) looks deceptively modest under a `4,096`-token ceiling (**3.0% → 9.0–9.5%** strict accuracy across $N=200$ `MATH-500` problems, $+6.0\text{–}6.5\text{ pp}$), yet yields a **5.7×–6.1× jump at `32,768` tokens** (**8.5% → 48.5–51.5%**, $+40.0\text{–}43.0\text{ pp}$, two-sided exact McNemar $p < 10^{-6}$ with **0 regressions** across 80–86 discordant pairs). Across a paired 6-dose sweep ($\alpha \in [0, 1.5]$ on $N=100$), loop prevalence falls monotonically (**91.0% → 17.0%**) while accuracy traces an **inverted-U** peaking at **$\alpha = 0.75$ ($4.5\times$ subtraction, `52.0%` strict / `57.0%` forced)** before over-editing degrades accuracy at $\alpha = 1.50$ (`40.0%`).

---

## Overview

Tiny reasoning models frequently fail on competition math not because they cannot perform the algebra, but because their `<think>` chain enters a **repetitive loop** until exhausting the token limit. In *Weight-Space Programming*, a frontier LLM ("Astra") inspected `Qwen/Qwen3.5-0.8B`'s weights without gradients or training, isolated the repetition loop to a single residual direction written by the MLPs of **Layer 15** and **Layer 19**, and projected that direction out of 64 columns in each layer's `down_proj.weight` (`131,072 / 873,438,784 = 0.0150%` of the model).

This repository:
1. **Verifies the exact weight-space diff ([`weights.py`](weights.py))**: Confirms across all `488` tensors (`873,438,784` parameters) that **486 are bit-identical** in `bfloat16` and **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ, verifies their SHA-256 digests, and proves via SVD that both edits are exact **rank-1 projections** ($\sigma_1 / \sigma_2 = 1771.2$ and $1552.0$, $> 99.999\%$ rank-1 energy).
2. **Implements continuous weight-space interpolation ([`blend`](weights.py))**:
   $$W(\alpha) = W_{\text{base}} + \alpha \cdot (W_{\text{edit}} - W_{\text{base}})$$
   Scaling the subtraction factor continuously across $6\alpha$: from unmodified Base ($\alpha = 0$), through partial suppression ($\alpha \in \{0.5, 0.75\}$), the published checkpoint ($\alpha = 1.0$, $6\times$), to over-projection ($\alpha \in \{1.25, 1.5\}$).
3. **Evaluates Nested Budget Scaling & 6-Dose Curves on GPU ([`gpu/vllm_sweep.py`](gpu/vllm_sweep.py), [`gpu/derive_budgets.py`](gpu/derive_budgets.py), [`gpu/stats.py`](gpu/stats.py))**: Runs `32,768`-token rollouts on **NVIDIA RTX PRO 6000 Blackwell** (`96 GB`, `bfloat16`, `vllm==0.33.0`), derives exact truncated prefixes at `B ∈ {4096, 8192, 16384, 32768}`, runs live Pass-2 forced completions (`close_think` + `open_box`), and computes exact two-sided McNemar tests and 95% Wilson confidence intervals across nested cohorts $S_{50} \subset S_{100} \subset S_{200}$ (`seed=0`).
4. **Benchmarks Against Published Upstream Logs & Generates Publication Figures ([`scripts/paper_reference.py`](scripts/paper_reference.py), [`scripts/generate_article_charts.py`](scripts/generate_article_charts.py))**: Cross-checks exact `avg@4` and 1st-sample truncated accuracy on the identical problem subsets from [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) and renders all figures under `assets/article/`.
5. **Provides a Side-by-Side Live Streaming Gradio App ([`app.py`](app.py)) & Local MPS Sweep ([`sweep.py`](sweep.py))**: Streams $\alpha_{\text{left}}$ and $\alpha_{\text{right}}$ side by side on any chosen [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problem with a live **repeated 8-gram ratio meter** and **token-cap truncation flag**, plus custom in-place Metal/MPS GatedDeltaNet decode kernels ([`install_fast_qwen35_kernels`](weights.py)).

---

## Key Results

### 1. Multi-Budget Scaling on $N = 200$ Core Cohort (`RTX PRO 6000 bfloat16`, `4k → 32k` Tokens)

Evaluated on $N = 200$ deterministic [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problems ($S_{50} \subset S_{100} \subset S_{200}$, `seed=0`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) across `2,400` budget evaluations ([`assets/gpu_n200/`](assets/gpu_n200/), [`assets/gpu_stats.csv`](assets/gpu_stats.csv)):

| Budget $B$ | Blend $\alpha$ | Strict Acc (95% Wilson CI) | $\Delta$ vs Base (McNemar $p$) | Forced Acc (95% CI) | Hit Cap % | Closed `</think>` % | Looped % (`1024w` / $\ge 15\%$) | Mean / Med Tokens | Mean / Tail 8-Gram % |
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

- **Zero Regressions Across Discordant Pairs**: At `32,768` tokens, every single problem solved by Base (`17/200`) is *also* solved by `α = 1.0` (`n10 = 0, n01 = 80, p = 1.65e-24`) and `α = 1.25` (`n10 = 0, n01 = 86, p = 2.58e-26`).
- **Why `4,096` Tokens Hides the Gain**: Even loop-free edited reasoning chains in `Qwen3.5-0.8B` have a median natural completion length of **~22,000 tokens** (`22,088.5` at `α=1.0`). At `4,096` tokens, `91.0%` (`182/200`) of edited traces are cut mid-`<think>` before finishing, capping strict accuracy at `9.0%` — and even Pass-2 forced `</think>` extraction only reaches `17.0–22.5%` at `4k–8k` vs `50.0–53.5%` at `32k`, proving the edit unlocks multi-step mathematical progress rather than formatting shortcuts.
- **Transient Loop Recovery (`59.1%` vs `0.0%`)**: Using the `1024`-token sliding window BPE detector ([`compute_loop_onset_from_token_ids`](metrics.py)), `89.5%` (`179/200`) of Base traces enter a loop (median onset **`6,144` tokens**) and **`0 / 179` (0.0%) recover**. At `α = 1.0`, loop onset shifts by `+12,288` tokens (`33.0% = 66/200`, median onset **`18,432` tokens** on capped traces), and **`39 / 66` (59.1%) break out of transient repetition** to finish `</think>` cleanly (`51.0% = 26/51` recovery at `α = 1.25`).

---

### 2. Paired 6-Dose Sweep ($N = 100$ Cohort, $\alpha \in [0, 1.5]$ @ `32,768` Tokens)

Joining [`assets/gpu_dose_n100/`](assets/gpu_dose_n100/) ($\alpha \in \{0.5, 0.75, 1.5\}$) with the identical $S_{100} \subset S_{200}$ problems from `gpu_n200` ($\alpha \in \{0.0, 1.0, 1.25\}$) across all 6 doses (`2,400` budget evaluations):

| Blend $\alpha$ | Scale ($6\alpha$) | Strict Correct | Finished Wrong | Capped Looping | Capped Coherent | Total Looped (Cap + Recov) | Forced Acc | Closed `</think>` | Med Tokens | Mean 8g Rep | McNemar vs `α=0` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | $0.0\times$ | **6.0%** | 0.0% | 91.0% | 3.0% | 91.0% (91 + 0) | 18.0% | 6.0% | 32,768.0 | 71.40% | — |
| **0.50** | $3.0\times$ | **32.0%** | 5.0% | 60.0% | 3.0% | 72.0% (60 + 12) | 41.0% | 51.0% | 32,768.0 | 44.53% | +26.0 pp ($p < 10^{-6}$) |
| **0.75** | $4.5\times$ | **52.0%** | 9.0% | 31.0% | 8.0% | 58.0% (31 + 27) | **57.0%** | 74.0% | 26,214.5 | 28.38% | **+46.0 pp** ($p < 10^{-6}$) |
| **1.00** | $6.0\times$ | **45.0%** | 21.0% | 15.0% | 19.0% | 33.0% (15 + 18) | 47.0% | **77.0%** | 25,239.5 | 18.84% | +39.0 pp ($p < 10^{-6}$) |
| **1.25** | $7.5\times$ | **46.0%** | 20.0% | 10.0% | 24.0% | 22.0% (10 + 12) | 47.0% | 73.0% | 23,294.5 | 13.58% | +40.0 pp ($p < 10^{-6}$) |
| **1.50** | $9.0\times$ | **40.0%** | **32.0%** | **8.0%** | 20.0% | **17.0%** (8 + 9) | 42.0% | 71.0% | **20,469.0** | **12.18%** | +34.0 pp ($p < 10^{-6}$) |

- **Two Competing Forces**: Loop prevalence drops monotonically across the entire dose axis (`91% → 72% → 58% → 33% → 22% → 17%`). Stepping from $\alpha = 0.50$ ($3\times$) to **$\alpha = 0.75$ ($4.5\times$)** adds **+20.0 pp** (`32.0% → 52.0%`, McNemar $p = 0.000088$) while keeping finished-but-wrong errors low (`9.0%`). Above $\alpha = 0.75$, finished-but-wrong answers rise to `20–21%` ($\alpha = 1.0\text{–}1.25$) and **`32.0%` at $\alpha = 1.50$ ($9\times$)**, cutting strict accuracy by **-12.0 pp** vs. peak $\alpha = 0.75$ (`52.0% → 40.0%`, McNemar $p = 0.02266$) due to premature `</think>` termination.

---

### 3. Published Paper Benchmark & Nested-Budget Verification ([`assets/paper_reference.csv`](assets/paper_reference.csv))

Exact problem-matched comparison against Takehi et al.'s (2026) upstream logs (`RikiyaT/lm_understand-logs` `exp8_math/eval/{unedited,p50c}`):

| Cohort / Slice | Problems $N$ | Metric Type | Base (`unedited`, $\alpha=0$) | Edited (`p50c`, $\alpha=1$) | Gain ($\Delta$) |
|---|---:|---|---:|---:|---:|
| **`our_n50`** (exact problem match) | 50 | Published `avg@4` @ 32k | **7.50%** (15/200) | **41.00%** (82/200) | **+33.50 pp** |
| **`our_n100_dose`** (exact problem match) | 100 | Published `avg@4` @ 32k | **6.50%** (26/400) | **43.75%** (175/400) | **+37.25 pp** |
| **`our_n200_core`** (exact problem match) | 200 | Published `avg@4` @ 32k | **9.50%** (76/800) | **48.00%** (384/800) | **+38.50 pp** |
| **`math500_all_500`** (full benchmark) | 500 | Published `avg@4` @ 32k | **7.90%** (158/2000) | **48.10%** (962/2000) | **+40.20 pp** |
| **`paper_first_samples_n200`** (`B = 4,096`) | 200 | Truncated 1st-sample (approx.) | 2.00% (98.0% cut) | 8.00% (92.0% cut) | +6.00 pp |
| **`paper_first_samples_n200`** (`B = 16,384`) | 200 | Truncated 1st-sample (approx.) | 6.50% (93.5% cut) | 27.50% (71.0% cut) | +21.00 pp |
| **`paper_first_samples_n200`** (`B = 32,768`) | 200 | Truncated 1st-sample (approx.) | 6.50% (93.5% cut) | 46.00% (24.0% cut) | +39.50 pp |

- **Nested-Budget Prefix Invariance**: Across all `450` committed `N=50` traces at smaller budgets (`4096, 8192, 16384` × $\alpha \in \{0.0, 1.0, 1.25\}$), **450 / 450 (100.0%)** are exact prefixes of the `32,768`-token trace (`t32.startswith(tb)`), and offline `out_ids[:B]` derivation via [`derive_budgets_from_32k`](gpu/derive_budgets.py) matches `hit_cap_pct` and `closed_think_pct` with **0 / 450 mismatches** (`12 / 12` cells exact).

---

### 4. Local Apple Silicon MPS Sweep ($N = 50$, `max_new_tokens=4096`, [`assets/sweep_results.csv`](assets/sweep_results.csv))

Original local Apple M3 (`bfloat16`) 5-dose sweep under a `4,096`-token budget (`seed=0`):

| Blend $\alpha$ | Ablation Scale ($6\alpha$) | Correct ($N=50$) | Accuracy (%) | Hit 4096 Cap (%) | Mean Tokens | Full-Trace 8-Gram Rep (%) | Final-512w Tail Rep (%) | Wall Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** (Base) | $0.0\times$ | 2 / 50 | 4.0% | 98.0% (49/50) | 4055.9 | 8.44% | **14.71%** | 3157.01s |
| **0.50** | $3.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 3990.4 | 3.74% | 1.22% | 1918.79s |
| **1.00** (Published) | $6.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 4028.7 | 2.99% | 1.39% | 2609.49s |
| **1.25** (**Optimal @ 4k**) | $7.5\times$ | **6 / 50** | **12.0%** | **88.0% (44/50)** | **3982.9** | 2.69% | 1.35% | 1995.77s |
| **1.50** (Over-ablation) | $9.0\times$ | 4 / 50 | 8.0% | 92.0% (46/50) | 4019.4 | **2.37%** | **0.74%** | **1890.71s** |

- **Exact 2-Tensor Diff**: Across all `488` tensors (`873,438,784` parameters), **`486` are bit-identical** in `bfloat16`; only `layers.15.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1771.2$, `99.99945%` rank-1 energy, `10.64%` Frobenius norm diff) and `layers.19.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1552.0$, `99.99914%` rank-1 energy, `8.11%` Frobenius norm diff) differ across 64 columns each (`0.0150%` of parameters, $\cos(u_{15}, u_{19}) = 0.8490$).

---

## Visuals

### 1. Why a 4k Token Cap Hides the Gain ($N = 200$ Core Cohort, `4k → 32k`)
![Headline Budget Scaling Curve](assets/article/headline_budget_curve.png)

### 2. Outcome Breakdown (`N=200` Core & `N=100` 6-Dose Inverted-U) & Loop-Onset Dynamics
![32k Mutually Exclusive Outcomes](assets/article/outcomes_32k.png)
![Loop Onset Histogram and Scatter](assets/article/loop_onset.png)

### 3. Two-Tensor Rank-1 Surgery (`0.0150%` of Weights) & Side-by-Side 32k Trace
![Two-Tensor Rank-1 Surgery Diagram](assets/article/edit_diagram.png)
![Side-by-Side 32k Trace Comparison](assets/article/hero_trace.png)

### 4. GPU 3-Panel Scaling & 4k MPS vs. vLLM Cross-Backend Comparison
![GPU Budget Scaling Dashboard](assets/budget_curve_gpu.png)
![4k MPS vs RTX PRO 6000 Comparison](assets/mps_4k_comparison.png)

### 5. Local Apple Silicon `4k` Sweep, SVD Weight Analysis & Gradio App (`app.py`)
![Weight Diff Analysis](assets/weight_diff_analysis.png)
![Local 4k Alpha Sweep Dashboard](assets/sweep_charts.png)
![Token Length vs 8-Gram Repetition Distribution](assets/alpha_token_distributions.png)
![Side-by-Side Gradio App](assets/app_side_by_side.gif)
![Side-by-Side Gradio Screenshot](assets/app_screenshot.png)

---

## Reproducing Results

### Local Setup, Weight Verification, 4k MPS Sweep & Live Gradio App

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Verify 488-tensor SHA-256 digests, SVD rank-1 structure, and weight diff chart
.venv/bin/python weights.py

# 2. Run the local Apple Silicon N=50 MATH-500 4k sweep across alpha in [0, 0.5, 1.0, 1.25, 1.5]
.venv/bin/python sweep.py --n 50 --alphas 0,0.5,1.0,1.25,1.5

# 3. Launch the side-by-side streaming Gradio app (binds to 127.0.0.1:7860)
.venv/bin/python app.py
```

### Cloud GPU (`vLLM 0.33.0`) Multi-Budget & Dose-Response Sweeps

```bash
# Option A: Ephemeral Google Colab Pro GPU (e.g., G4 RTX PRO 6000 96GB / L4 / A100)
.venv/bin/python gpu/colab_run.py --n 200 --budgets 32768 --alphas 0.0,1.0,1.25

# Derive nested {4096, 8192, 16384, 32768} budgets from 32k traces + live Pass-2 forced completions
.venv/bin/python gpu/colab_run.py --derive-from assets/gpu_n200 --out-root assets/gpu_n200 --budgets 4096,8192,16384,32768

# Option B: Ephemeral Google Cloud Spot L4 / T4 VM (auto-deletes VM on completion)
.venv/bin/python gpu/gcp_run.py --project YOUR_GCP_PROJECT_ID --dry-run
.venv/bin/python gpu/gcp_run.py --project YOUR_GCP_PROJECT_ID
```

### Offline Statistical Tables, Paper Reference, Article Figures & Tests

```bash
# 1. Compute Wilson 95% CIs and exact two-sided McNemar tests -> assets/gpu_stats.csv
.venv/bin/python gpu/stats.py

# 2. Extract published Takehi et al. (2026) avg@4 & truncated reference curves -> assets/paper_reference.csv
.venv/bin/python scripts/paper_reference.py

# 3. Render all 5 article charts (assets/article/*.png) + GPU/MPS comparison charts
.venv/bin/python scripts/generate_article_charts.py

# 4. Run full unit & integration test suite
.venv/bin/python -m unittest discover -s tests -v
```

See [`NOTES.md`](NOTES.md) for the complete chronological engineering log, statistical tables, loop-recovery breakdown, and open research questions.
