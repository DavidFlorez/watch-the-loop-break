# `watch-the-loop-break` 🧠

Reproducing, stress-testing, and extending Experiment 8 of **["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)** ([Takehi, Teoh, Zhang & Khattab, 2026](https://rikiyat.github.io/blog/weight-space-programming/)) on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) vs. [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

> **Headline Finding — Why a 4k Token Cap Hides a 6× Gain:**
> Subtracting a single rank-1 repetition direction from **2 of 488 tensors** (`layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight`, **0.0150%** of parameters) looks deceptively modest under a `4,096`-token ceiling (**3.0% → 9.0–9.5%** strict accuracy across $N=200$ `MATH-500` problems, $+6.0\text{–}6.5\text{ pp}$), yet yields a **5.7×–6.1× jump at `32,768` tokens** (**8.5% → 48.5–51.5%**, $+40.0\text{–}43.0\text{ pp}$, two-sided exact McNemar $p < 10^{-6}$ with **0 regressions** across 80–86 discordant pairs). Across a paired 6-dose sweep ($\alpha \in [0, 1.5]$ on $N=100$), stronger projection cuts loops monotonically (**91.0% → 17.0%**) and keeps cap hits non-increasing (**94.0% → 28.0%**, flat at `34.0%` across $\alpha = 1.0\text{–}1.25$), while finished-but-wrong answers rise (**0.0% → 9.0% at $\alpha=0.75$ → 21.0% at $\alpha=1.0$ → 32.0% at $\alpha=1.50$**); differences among $\alpha = 0.75 / 1.0 / 1.25$ (**52.0% / 45.0% / 46.0%**, McNemar $p = 0.14$ for $0.75$ vs. $1.0$) and $1.5$ vs. $1.0$ ($p = 0.38$) lie within sampling noise, while only $0.50 \to 0.75$ ($32.0\% \to 52.0\%$, $+20.0\text{ pp}$, $p \approx 0.0001$) is statistically significant.

---

## Overview

Tiny reasoning models frequently fail on competition math not because they cannot perform the algebra, but because their `<think>` chain enters a **repetitive loop** until exhausting the token limit. In *Weight-Space Programming*, a frontier LLM ("Astra") inspected `Qwen/Qwen3.5-0.8B`'s weights without gradients or training, isolated the repetition loop to a single residual direction written by the MLPs of **Layer 15** and **Layer 19**, and projected that direction out of 64 columns in each layer's `down_proj.weight` (`131,072` edited column weights with `130,188` non-zero `bfloat16` entries changed out of `873,438,784` checkpoint parameters = **`0.0150%`**, rounded by Takehi et al., 2026 to **`0.016%`** of nominal `"0.8 billion"`).

This repository:
1. **Verifies the exact weight-space diff ([`weights.py`](weights.py))**: Confirms across all `488` tensors (`873,438,784` parameters) that **486 are bit-identical** in `bfloat16` and **only** `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ, verifies their SHA-256 digests, and proves via SVD that both edits are exact **rank-1 projections** ($\sigma_1 / \sigma_2 = 1771.2$ and $1552.0$, $> 99.999\%$ rank-1 energy).
2. **Implements continuous weight-space interpolation ([`blend`](weights.py))**:
   $$W(\alpha) = W_{\text{base}} + \alpha \cdot (W_{\text{edit}} - W_{\text{base}})$$
   Scaling the subtraction factor continuously across $6\alpha$: from unmodified Base ($\alpha = 0$), through partial suppression ($\alpha \in \{0.5, 0.75\}$), the published checkpoint ($\alpha = 1.0$, $6\times$), to over-projection ($\alpha \in \{1.25, 1.5\}$).
3. **Evaluates Nested Budget Scaling & 6-Dose Curves on GPU ([`gpu/vllm_sweep.py`](gpu/vllm_sweep.py), [`gpu/derive_budgets.py`](gpu/derive_budgets.py), [`gpu/stats.py`](gpu/stats.py))**: Runs `32,768`-token rollouts on **NVIDIA RTX PRO 6000 Blackwell** (`96 GB`, `bfloat16`, `vllm==0.33.0`), derives exact truncated prefixes at `B ∈ {4096, 8192, 16384, 32768}`, runs live Pass-2 forced completions (`close_think` + `open_box`), and computes exact two-sided McNemar tests and 95% Wilson confidence intervals across nested cohorts $S_{50} \subset S_{100} \subset S_{200}$ (`seed=0`).
4. **Benchmarks Against Published Upstream Logs & Generates Publication Figures ([`scripts/paper_reference.py`](scripts/paper_reference.py), [`scripts/generate_article_charts.py`](scripts/generate_article_charts.py))**: Cross-checks exact `avg@4` and 1st-sample truncated accuracy on the identical problem subsets from [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) and renders all 10 article figures under `assets/article/`.
5. **Provides a Side-by-Side Live Streaming Gradio App ([`app.py`](app.py)) & Local MPS Sweep ([`sweep.py`](sweep.py))**: Streams $\alpha_{\text{left}}$ and $\alpha_{\text{right}}$ side by side on any chosen [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problem with a live **repeated 8-gram ratio meter** and **token-cap truncation flag**, plus custom in-place Metal/MPS GatedDeltaNet decode kernels ([`install_fast_qwen35_kernels`](weights.py)).

---

## Key Results

### 1. Multi-Budget Scaling on $N = 200$ Core Cohort (`RTX PRO 6000 bfloat16`, `4k → 32k` Tokens)

Evaluated on $N = 200$ deterministic [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) problems ($S_{50} \subset S_{100} \subset S_{200}$, `seed=0`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`) across `2,400` budget evaluations ([`assets/gpu_n200/`](assets/gpu_n200/), [`assets/gpu_stats.csv`](assets/gpu_stats.csv)):

| Budget $B$ | Blend $\alpha$ | Strict Acc (95% Wilson CI) | $\Delta$ vs Base (McNemar $p$) | Forced Acc (95% CI) | Hit Cap % | Closed `</think>` % | Looped % (`1024w` / $\ge 15\%$) | Mean / Med Tokens | Mean / Tail 8-Gram % | Wall Time (s) | Speed (`tok/s`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **4,096** | **0.00** | **3.0%** (1.38–6.39%) | — | 19.0% (14.17–25.00%) | 97.0% | 4.0% | 14.0% / 14.0% | 4040.7 / 4096.0 | 9.13% / 15.35% | 60.24s* | 14,978.7* |
| **4,096** | **1.00** | **9.0%** (5.77–13.78%) | +6.0 pp ($p = 0.00183$) | 17.0% (12.43–22.82%) | 91.0% | 9.5% | **0.0%** / 0.0% | 3937.3 / 4096.0 | 2.90% / 1.15% | 97.34s* | 9,021.1* |
| **4,096** | **1.25** | **9.5%** (6.17–14.36%) | +6.5 pp ($p = 0.00098$) | 17.5% (12.86–23.36%) | 90.5% | 11.0% | 0.5% / 0.5% | 3918.8 / 4096.0 | 2.55% / 1.02% | 98.43s* | 8,875.4* |
| **8,192** | **0.00** | **7.5%** (4.60–12.00%) | — | 20.0% (15.05–26.09%) | 92.5% | 7.5% | 55.0% / 55.0% | 7911.1 / 8192.0 | 27.17% / 51.87% | 117.95s* | 14,185.8* |
| **8,192** | **1.00** | **13.5%** (9.45–18.93%) | +6.0 pp ($p = 0.01690$) | 22.5% (17.26–28.77%) | 86.0% | 18.0% | 3.0% / 6.0% | 7573.4 / 8192.0 | 6.33% / 2.87% | 140.21s* | 11,406.4* |
| **8,192** | **1.25** | **14.0%** (9.87–19.49%) | +6.5 pp ($p = 0.00720$) | 22.0% (16.82–28.24%) | 85.5% | 17.5% | 2.0% / 3.5% | 7526.4 / 8192.0 | 4.99% / 2.48% | 141.78s* | 11,211.6* |
| **16,384** | **0.00** | **8.5%** (5.37–13.19%) | — | 21.0% (15.93–27.16%) | 91.5% | 8.5% | 87.5% / 88.0% | 15428.6 / 16384.0 | 53.38% / 78.45% | 130.16s* | 24,397.1* |
| **16,384** | **1.00** | **29.5%** (23.61–36.16%) | **+21.0 pp** ($p < 10^{-6}$) | 36.0% (29.67–42.86%) | 67.0% | 43.5% | 14.5% / 24.5% | 13818.6 / 16384.0 | 12.27% / 7.56% | 303.58s* | 9,316.6* |
| **16,384** | **1.25** | **30.5%** (24.54–37.20%) | **+22.0 pp** ($p < 10^{-6}$) | 34.5% (28.26–41.32%) | 65.0% | 41.5% | 10.0% / 19.0% | 13630.0 / 16384.0 | 9.95% / 5.57% | 313.00s* | 8,915.6* |
| **32,768** | **0.00** | **8.5%** (5.37–13.19%) | — | 21.5% (16.37–27.70%) | 91.5% | 8.5% | 89.5% / 89.5% | 30419.9 / 32768.0 | 70.66% / 80.34% | **1,378.30s** | 4,479.3 |
| **32,768** | **1.00** | **48.5%** (41.67–55.39%) | **+40.0 pp** ($p < 10^{-6}$) | 50.0% (43.14–56.86%) | 31.0% | 79.5% | 33.0% / 46.0% | 21435.7 / 22088.5 | 18.75% / 12.57% | **935.64s** | 4,614.0 |
| **32,768** | **1.25** | **51.5%** (44.61–58.33%) | **+43.0 pp** ($p < 10^{-6}$) | **53.5%** (46.59–60.28%) | **29.5%** | **80.0%** | **25.5%** / **38.5%** | **20926.9 / 21742.0** | **15.15% / 10.07%** | **914.68s** | 4,606.5 |

*\*Timing note*: Summed across all resumed JSONL chunks at `32,768` tokens, full `N=200` Pass-1 + Pass-2 generation totaled **`3,228.62s`** (`53.81 min`, `14,704,248` total output tokens = **`4,554.3 tok/s`**: `1,378.30s` / `6,173,816` tok @ `α=0.0`, `935.64s` / `4,316,989` tok @ `α=1.0`, `914.68s` / `4,213,443` tok @ `α=1.25`). For nested derived budgets (`4k / 8k / 16k`), Pass-1 prefixes were already timed in the `32k` run (`0s` attributed to prefix slicing) and `Wall Time (s)` / `Speed (tok/s)` record live Pass-2 forced completion time + throughput on capped traces.

- **Zero Regressions Across Discordant Pairs**: At `32,768` tokens, every single problem solved by Base (`17/200`) is *also* solved by `α = 1.0` (`n10 = 0, n01 = 80, discordant = 80, p = 1.65e-24`) and `α = 1.25` (`n10 = 0, n01 = 86, discordant = 86, p = 2.58e-26`). Between `α = 1.0` (`48.5%`) and `α = 1.25` (`51.5%`), the `+3.0 pp` difference (`n10 = 11, n01 = 17`) is within noise (two-sided exact McNemar $p = 0.3449$).
- **Same-Problem Paper Benchmark (`our_n200_core` in [`assets/paper_reference.csv`](assets/paper_reference.csv))**: On the exact same `200` `MATH-500` problems at `32,768` tokens, Takehi et al.'s (2026) 4-sample `avg@4` is **`9.5%`** (`76/800`) for `unedited` and **`48.0%`** (`384/800`) for `p50c` — matching our single-sample (`T=0.6`) `α = 1.0` (**`48.5%`**) within **`0.5 pp`**, while `α = 1.25` (**`51.5%`**) is **`+3.5 pp` higher**.
- **Why `4,096` Tokens Hides the Gain**: Across all `200` runs at `α = 1.0`, `Qwen3.5-0.8B`'s median generation length is **`22,088.5` tokens** (`21,742.0` across all `200` runs at `α = 1.25`). At `4,096` tokens, `91.0%` (`182/200`) of `α=1.0` runs hit the cap mid-thought before finishing, capping strict accuracy at `9.0%` — and even Pass-2 forced `</think>` extraction only reaches `17.0–22.5%` at `4k–8k` vs `50.0–53.5%` at `32k`, proving the edit unlocks multi-step mathematical progress rather than formatting shortcuts.
- **Transient Loop Recovery (`59.1%` vs `0.0%`)**: Using the `1024`-token sliding window BPE detector ([`compute_loop_onset_from_token_ids`](metrics.py)), `89.5%` (`179/200`) of Base traces enter a loop (median onset **`6,144` tokens**) and **`0 / 179` (0.0%) recover**. At `α = 1.0`, loop onset shifts by `+12,288` tokens (`33.0% = 66/200`, median onset **`18,432` tokens** on capped traces; `16,128` across all 66), and **`39 / 66` (59.1%) break out of transient repetition** to finish `</think>` cleanly (`51.0% = 26/51` recovery at `α = 1.25`).

---

### 2. Paired 6-Dose Sweep ($N = 100$ Cohort, $\alpha \in [0, 1.5]$ @ `32,768` Tokens)

Joining [`assets/gpu_dose_n100/`](assets/gpu_dose_n100/) ($\alpha \in \{0.5, 0.75, 1.5\}$; `1,532.82s` / `25.55 min` total 32k generation wall time across `7,099,398` output tokens = `4,631.6 tok/s`) with the identical $S_{100} \subset S_{200}$ problems from `gpu_n200` ($\alpha \in \{0.0, 1.0, 1.25\}$) at `32,768` tokens (`2,400` total budget evaluations across `4k/8k/16k/32k`):

| Blend $\alpha$ | Scale ($6\alpha$) | Strict Correct | Finished Wrong | Capped Looping | Capped Coherent | Total Looped (Cap + Recov) | Forced Acc | Closed `</think>` | Med Tokens | Mean 8g Rep | Wall Time (s) | Speed (`tok/s`) | McNemar vs `α=0` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | $0.0\times$ | **6.0%** | **0.0%** | 91.0% | 3.0% | 91.0% (91 + 0) | 18.0% | 6.0% | 32,768.0 | 71.40% | 703.19s | 4,497.1 | — |
| **0.50** | $3.0\times$ | **32.0%** | 5.0% | 60.0% | 3.0% | 72.0% (60 + 12) | 41.0% | 51.0% | 32,768.0 | 44.53% | 571.29s | 4,601.3 | +26.0 pp ($p < 10^{-6}$) |
| **0.75** | $4.5\times$ | **52.0%** | **9.0%** | 31.0% | 8.0% | 58.0% (31 + 27) | **57.0%** | 74.0% | 26,214.5 | 28.38% | 511.48s | 4,676.6 | **+46.0 pp** ($p < 10^{-6}$) |
| **1.00** | $6.0\times$ | **45.0%** | **21.0%** | 15.0% | 19.0% | 33.0% (15 + 18) | 47.0% | **77.0%** | 25,239.5 | 18.84% | 508.64s | 4,654.6 | +39.0 pp ($p < 10^{-6}$) |
| **1.25** | $7.5\times$ | **46.0%** | 20.0% | 10.0% | 24.0% | 22.0% (10 + 12) | 47.0% | 73.0% | 23,294.5 | 13.58% | 489.41s | 4,625.6 | +40.0 pp ($p < 10^{-6}$) |
| **1.50** | $9.0\times$ | **40.0%** | **32.0%** | **8.0%** | 20.0% | **17.0%** (8 + 9) | 42.0% | 71.0% | **20,469.0** | **12.18%** | **450.05s** | 4,618.9 | +34.0 pp ($p < 10^{-6}$) |

- **Monotonic Loop Reduction vs. Rising Wrong-Answer Completion**: Stronger projection cuts loops monotonically (`91.0% → 72.0% → 58.0% → 33.0% → 22.0% → 17.0%`), keeps cap hits **non-increasing** (`94.0% → 63.0% → 39.0% → 34.0% → 34.0% → 28.0%` — flat at `34.0%` for `α = 1.0` and `1.25`), and shortens median generation length by `12,299` tokens (`32,768.0 → 20,469.0`), while finished-but-wrong responses increase from `0.0%` (`α = 0.0`) and `5.0%` (`α = 0.50`) to **`9.0%` (`α = 0.75`)**, **`21.0%` (`α = 1.0`)**, **`20.0%` (`α = 1.25`)**, and **`32.0%` (`α = 1.50`)**. Pairwise two-sided exact McNemar tests ([`assets/gpu_stats.csv`](assets/gpu_stats.csv)) show that **only `α = 0.50 → 0.75` is statistically significant (`32.0% → 52.0%`, `+20.0 pp`, $n_{11}=29, n_{10}=3, n_{01}=23, p = 0.000088 \approx 0.0001$)**; differences between **`α = 0.75` and `α = 1.0` (`52.0% vs 45.0%`, $n_{11}=40, n_{10}=12, n_{01}=5, p = 0.1435$)** and between **`α = 1.50` and `α = 1.0` (`40.0% vs 45.0%`, $n_{11}=32, n_{10}=8, n_{01}=13, p = 0.3833$)** are **not statistically significant** (`1.0 vs 1.25`: $n_{11}=39, n_{10}=6, n_{01}=7, p = 1.00$; `0.75 vs 1.25`: $n_{11}=40, n_{10}=12, n_{01}=6, p = 0.2379$; `0.75 vs 1.5`: $n_{11}=34, n_{10}=18, n_{01}=6, p = 0.0227$).

---

### 3. Published Paper Benchmark & Nested-Budget Verification ([`assets/paper_reference.csv`](assets/paper_reference.csv))

Exact problem-matched comparison against Takehi et al.'s (2026) upstream logs (`RikiyaT/lm_understand-logs` `exp8_math/eval/{unedited,p50c}`):

| Cohort / Slice | Problems $N$ | Metric Type | Base (`unedited`, $\alpha=0$) | Edited (`p50c`, $\alpha=1$) | Gain ($\Delta$) |
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

- **Nested-Budget Prefix Invariance & Token/Forced Accounting**: Across all `450` committed `N=50` traces at smaller budgets (`4096, 8192, 16384` × $\alpha \in \{0.0, 1.0, 1.25\}$), **450 / 450 (100.0%)** are exact prefixes of the `32,768`-token trace (`t32.startswith(tb)`), and offline `out_ids[:B]` derivation via [`derive_budgets_from_32k`](gpu/derive_budgets.py) matches `hit_cap_pct` and `closed_think_pct` with **0 / 450 mismatches** (`12 / 12` cells exact). Because legacy `N=50` logs saved decoded `text` without `<|im_end|>` (`ID 248046`), derived `n_tokens` is `1–2` tokens lower on legacy finished `S_50` runs (`56/67` by `1` token, `11/67` by `2–3` tokens; 100% token-exact on fresh `S_200 \ S_50` runs), and the `71` unclosed-`</think>` capped `S_50` traces in `gpu_n200` at `32,768` tokens retain `pred_forced`/`correct_forced`/`forced_tokens` with `forced_text = None` (`0` missing across all derived `4k/8k/16k` budgets and `gpu_dose_n100`).

---

### 4. Local Apple Silicon MPS Sweep ($N = 50$, `max_new_tokens=4096`, [`assets/sweep_results.csv`](assets/sweep_results.csv))

Original local Apple M3 (`bfloat16`) 5-dose sweep under a `4,096`-token budget (`seed=0`):

| Blend $\alpha$ | Ablation Scale ($6\alpha$) | Correct ($N=50$) | Accuracy (%) | Hit 4096 Cap (%) | Mean Tokens | Full-Trace 8-Gram Rep (%) | Tail Loop (last 240 words) | Wall Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** (Base) | $0.0\times$ | 2 / 50 | 4.0% | 98.0% (49/50) | 4055.9 | 8.44% | **14.71%** | 3157.01s |
| **0.50** | $3.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 3990.4 | 3.74% | 1.22% | 1918.79s |
| **1.00** (Published) | $6.0\times$ | 3 / 50 | 6.0% | 94.0% (47/50) | 4028.7 | 2.99% | 1.39% | 2609.49s |
| **1.25** ($1.25\times$ Extrap.) | $7.5\times$ | **6 / 50** | **12.0%** | **88.0% (44/50)** | **3982.9** | 2.69% | 1.35% | 1995.77s |
| **1.50** (Over-ablation) | $9.0\times$ | 4 / 50 | 8.0% | 92.0% (46/50) | 4019.4 | **2.37%** | **0.74%** | **1890.71s** |

- **Exact 2-Tensor Diff**: Across all `488` tensors (`873,438,784` parameters), **`486` are bit-identical** in `bfloat16`; only `layers.15.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1771.18\times$, `99.99945%` rank-1 energy, `10.64%` Frobenius norm diff) and `layers.19.mlp.down_proj.weight` ($\sigma_1/\sigma_2 = 1552.02\times$, `99.99914%` rank-1 energy, `8.11%` Frobenius norm diff) differ across 64 columns each (`65,260` + `64,928` = `130,188` non-zero `bfloat16` entries changed across `131,072` column entries = **`0.015006%`** of `873,438,784` checkpoint parameters, rounded to **`0.016%`** of nominal `"0.8 billion"` in Takehi et al., 2026; $\cos(u_{15}, u_{19}) = 0.849005$).

---

## Visuals

### 1. Cover & Why a 4k Token Cap Hides the Gain ($N = 200$ Core Cohort, `4k → 32k`)
![5:2 Phone Thumbnail Cover](assets/article/cover_5x2.png)
![Headline Budget Scaling Curve (Phone)](assets/article/headline_budget_curve.png)
![Full Budget Scaling Curve (Desktop, with alpha=1.25 & Paper Truncated Reference)](assets/article/headline_budget_curve_full.png)

### 2. Outcome Breakdown (`N=200` Core & `N=100` 6-Dose Sweep) & Loop-Onset Dynamics
![32k Mutually Exclusive Outcomes (2-Panel)](assets/article/outcomes_32k.png)
![32k Mutually Exclusive Outcomes (Core N=200 Standalone)](assets/article/outcomes_32k_core.png)
![Paired 6-Dose Sweep Outcomes (N=100 Standalone)](assets/article/outcomes_dose.png)
![Loop Onset Position Scatter (Phone)](assets/article/loop_onset.png)
![Loop Onset Histogram and Scatter (Desktop 2-Panel)](assets/article/loop_onset_full.png)

### 3. Two-Tensor Rank-1 Surgery (`0.0150%` of Weights) & Verbatim Side-by-Side 32k Trace
![Two-Tensor Rank-1 Surgery Diagram](assets/article/edit_diagram.png)
![Side-by-Side 32k Trace Comparison](assets/article/hero_trace.png)

### 4. GPU 3-Panel Scaling & 4k MPS vs. vLLM Cross-Backend Comparison
![GPU Budget Scaling Dashboard](assets/budget_curve_gpu.png)
![4k MPS vs RTX PRO 6000 Comparison](assets/mps_4k_comparison.png)

### 5. Local Apple Silicon `4k` Sweep, SVD Weight Analysis & Gradio App (`app.py`)
![Weight Diff Analysis](assets/weight_diff_analysis.png)
![Local 4k Alpha Sweep Dashboard](assets/sweep_charts.png)
![Token Length vs 8-Gram Repetition Distribution](assets/alpha_token_distributions.png)
![Side-by-Side Gradio App (1280px Animated Replay GIF)](assets/app_side_by_side.gif)
![Side-by-Side KaTeX Gradio UI Still (1600px Phone-Readable)](assets/article/mac_app_still.png)
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

# 3. Render all 10 article charts (assets/article/*.png) + GPU/MPS comparison charts
.venv/bin/python scripts/generate_article_charts.py

# 4. Run full unit & integration test suite
.venv/bin/python -m unittest discover -s tests -v
```

See [`NOTES.md`](NOTES.md) for the complete chronological engineering log, statistical tables, loop-recovery breakdown, and open research questions.
