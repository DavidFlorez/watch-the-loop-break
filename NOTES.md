# NOTES.md — `watch-the-loop-break` Build Log

## Goal

Reproduce the Qwen3.5-0.8B "Weight-Space Programming" edit ([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)), verify that only `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` differ between base and edited models, and extend the result by interpolating $W(\alpha) = W_{\text{base}} + \alpha(W_{\text{edit}} - W_{\text{base}})$ across $\alpha \in [0, 0.5, 1.0, 1.25, 1.5]$ on MATH-500 and in a side-by-side live Gradio streaming app.

## Setup

- **Models**:
  - Base: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (revision `2fc06364715b967f1860aea9cf38778875588b17`)
  - Edited: [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
  - Reference logs & rebuild scripts: [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/runs/p50c`)
- **Dataset**:
  - [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (`test` split, 500 competition math problems across 7 subjects and 5 difficulty levels)
- **Hardware**:
  - Apple MacBook Air (`Mac15,12`), Apple M3 chip, 16 GB unified memory (`mps` backend auto-detected via `CUDA -> MPS -> CPU` fallback)
- **Library Versions**:
  - Python `3.13.7` (populated with exact package versions after environment installation)

## Log

- **2026-11-03 10:06 PST — Project initialization & literature/code audit**
  - Read ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/), the [`RikiyaT/Qwen3.5-0.8B-weight-space-programming`](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) model card, and the official evaluation/rebuild scripts in [`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/code/apply_edit.py`, `exp8_math/code/evaluate.py`, `exp8_math/code/mathqa.py`, `exp8_math/runs/p50c/work_notes/final_report.md`).
  - Confirmed the mathematical form of Astra's `p50c` edit: in `Qwen3.5-0.8B`, Astra identified unit repetition directions $u_{15}, u_{19} \in \mathbb{R}^{1024}$ at the residual outputs of layers 15 and 19, selected the 64 MLP output columns (`down_proj.weight` shape `[1024, 3584]`) that wrote most strongly along those directions, and applied $W_{\text{edit}} = W_{\text{base}} - 6 \cdot u (u^\top W_{\text{base}} \odot \mathbb{1}_{64})$.
  - Consequently, linear interpolation $W(\alpha) = W_{\text{base}} + \alpha (W_{\text{edit}} - W_{\text{base}})$ directly varies the directional ablation dose by $6\alpha$ along those 64 columns in layers 15 and 19 ($\alpha = 0$: base model; $\alpha = 0.5$: $3\times$ projection subtraction; $\alpha = 1.0$: Astra's published $6\times$ edit; $\alpha = 1.25$: $7.5\times$ over-subtraction; $\alpha = 1.5$: $9\times$ over-subtraction).
  - Initialized git repository on `main`, created `.venv`, `.gitignore`, and `pyproject.toml`.

## Key results

*(Pending weight verification and $N=50$ MATH-500 $\alpha$-sweep execution.)*

## Visuals

*(Pending generation of sweep charts, weight-diff visualizations, and Gradio side-by-side screenshot/GIF in `assets/`.)*

## Surprises and limitations

*(To be updated as experiments run.)*

## Open questions / next steps

*(To be updated as experiments run.)*
