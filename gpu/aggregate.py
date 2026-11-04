"""Aggregate multi-budget GPU & MPS results, render `assets/budget_curve_gpu.png`, and update `NOTES.md`."""

from __future__ import annotations

import os

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpu.common import PAPER_32K_REFERENCE, atomic_write_text
from metrics import wilson_ci

GPU_NOTES_START = "<!-- GPU_RESULTS_START -->"
GPU_NOTES_END = "<!-- GPU_RESULTS_END -->"

ALPHA_STYLE: Dict[float, Dict[str, Any]] = {
    0.0: {"color": "#ef4444", "marker": "o", "label": "α = 0.0 (Base)"},
    0.5: {"color": "#f59e0b", "marker": "^", "label": "α = 0.5 (Half Edit)"},
    1.0: {"color": "#10b981", "marker": "s", "label": "α = 1.0 (Published Edit)"},
    1.25: {"color": "#2563eb", "marker": "D", "label": "α = 1.25 (1.25x Extrapolation)"},
    1.5: {"color": "#7c3aed", "marker": "v", "label": "α = 1.5 (1.5x Over-ablation)"},
}


def load_combined_budget_results(out_root: Union[str, Path] = "assets") -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the 4k MPS baseline (`assets/sweep_results.csv`) and all `assets/budget_*/gpu-*` results."""
    root = Path(out_root)
    rows: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []
    smoke_records: List[Dict[str, Any]] = []

    # 1. Include existing 4k MPS baseline if present
    mps_csv = root / "sweep_results.csv"
    if mps_csv.is_file():
        mps_df = pd.read_csv(mps_csv)
        for _, r in mps_df.iterrows():
            n_eval = int(r["n_problems"])
            n_corr = int(r["correct"])
            n_cap = int(r["hit_cap_count"])
            acc_lo, acc_hi = wilson_ci(n_corr, n_eval)
            cap_lo, cap_hi = wilson_ci(n_cap, n_eval)
            rows.append(
                {
                    "budget": 4096,
                    "backend": "mps",
                    "gpu": "apple-m3",
                    "dtype": "bfloat16",
                    "alpha": float(r["alpha"]),
                    "n_problems": n_eval,
                    "n_samples": 1,
                    "total_evaluations": n_eval,
                    "correct_strict": n_corr,
                    "accuracy_strict_pct": float(r["accuracy_pct"]),
                    "accuracy_ci95_low": round(acc_lo * 100.0, 2),
                    "accuracy_ci95_high": round(acc_hi * 100.0, 2),
                    "forced_correct": np.nan,
                    "forced_accuracy_pct": np.nan,
                    "forced_accuracy_ci95_low": np.nan,
                    "forced_accuracy_ci95_high": np.nan,
                    "hit_cap_pct": float(r["hit_cap_pct"]),
                    "hit_cap_ci95_low": round(cap_lo * 100.0, 2),
                    "hit_cap_ci95_high": round(cap_hi * 100.0, 2),
                    "mean_tokens": float(r["mean_tokens"]),
                    "median_tokens": float(r["median_tokens"]),
                    "mean_rep_8gram_pct": float(r["mean_rep_8gram_pct"]),
                    "mean_tail_rep_8gram_pct": float(r["mean_tail_rep_8gram_pct"]),
                    "wall_time_s": float(r["wall_time_s"]),
                    "output_tok_per_s": np.nan,
                }
            )

    # 2. Scan all `assets/budget_<budget>/gpu-*` directories
    budget_dirs: List[Tuple[int, Path]] = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        m = re.match(r"^budget_(\d+)$", child.name)
        if m and child.is_dir():
            budget_dirs.append((int(m.group(1)), child))
    budget_dirs.sort(key=lambda item: item[0])

    for b_val, b_dir in budget_dirs:
        for gpu_dir in sorted(b_dir.iterdir()):
            if not gpu_dir.is_dir() or not gpu_dir.name.startswith("gpu-"):
                continue
            meta_file = gpu_dir / "meta.json"
            if meta_file.is_file():
                try:
                    meta_records.append(json.loads(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    pass

            res_csv = gpu_dir / "sweep_results.csv"
            if not res_csv.is_file():
                continue
            gdf = pd.read_csv(res_csv)
            for _, r in gdf.iterrows():
                rows.append(
                    {
                        "budget": int(b_val),
                        "backend": str(r.get("backend", "vllm")),
                        "gpu": str(r.get("gpu", "t4")),
                        "dtype": str(r.get("dtype", "float16")),
                        "alpha": float(r["alpha"]),
                        "n_problems": int(r["n_problems"]),
                        "n_samples": int(r.get("n_samples", 1)),
                        "total_evaluations": int(r.get("total_evaluations", r["n_problems"])),
                        "correct_strict": int(r["correct"]),
                        "accuracy_strict_pct": float(r["accuracy_pct"]),
                        "accuracy_ci95_low": float(r.get("accuracy_ci95_low", np.nan)),
                        "accuracy_ci95_high": float(r.get("accuracy_ci95_high", np.nan)),
                        "forced_correct": float(r.get("forced_correct", np.nan)),
                        "forced_accuracy_pct": float(r.get("forced_accuracy_pct", np.nan)),
                        "forced_accuracy_ci95_low": float(r.get("forced_accuracy_ci95_low", np.nan)),
                        "forced_accuracy_ci95_high": float(r.get("forced_accuracy_ci95_high", np.nan)),
                        "hit_cap_pct": float(r["hit_cap_pct"]),
                        "hit_cap_ci95_low": float(r.get("hit_cap_ci95_low", np.nan)),
                        "hit_cap_ci95_high": float(r.get("hit_cap_ci95_high", np.nan)),
                        "mean_tokens": float(r["mean_tokens"]),
                        "median_tokens": float(r["median_tokens"]),
                        "mean_rep_8gram_pct": float(r["mean_rep_8gram_pct"]),
                        "mean_tail_rep_8gram_pct": float(r["mean_tail_rep_8gram_pct"]),
                        "wall_time_s": float(r["wall_time_s"]),
                        "output_tok_per_s": float(r.get("output_tok_per_s", np.nan)),
                    }
                )

    # 3. Scan any `assets/gpu_smoke_*.json` files
    for s_file in sorted(root.glob("gpu_smoke_*.json")) if root.is_dir() else []:
        try:
            smoke_records.append(json.loads(s_file.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(), meta_records, smoke_records

    combined_df = (
        pd.DataFrame(rows)
        .sort_values(["budget", "alpha", "gpu", "dtype"])
        .reset_index(drop=True)
    )
    return combined_df, meta_records, smoke_records


def save_budget_curve_chart(
    combined_df: pd.DataFrame,
    out_path: Union[str, Path] = "assets/budget_curve_gpu.png",
) -> str:
    """Render 3-panel budget curve (Strict Acc, Forced Acc, Cap-Hit % vs Token Budget) + Paper 32k stars."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    alphas = sorted(combined_df["alpha"].unique()) if not combined_df.empty else [0.0, 1.0, 1.25]
    all_budgets = sorted({int(b) for b in combined_df["budget"].unique()} | {32768}) if not combined_df.empty else [4096, 8192, 16384, 32768]

    def _style_x(ax: Any) -> None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_budgets)
        labels = []
        for b in all_budgets:
            if b == 4096 and not combined_df.empty and (combined_df["budget"] == 4096).any():
                backends = set(combined_df.loc[combined_df["budget"] == 4096, "backend"])
                labels.append("4096\n(MPS)" if backends == {"mps"} else f"{b}")
            else:
                labels.append(str(b))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Max New Token Budget", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)

    # Panel A: Strict Accuracy (%) vs Token Budget
    ax_a = axes[0]
    for a in alphas:
        sub = combined_df[combined_df["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        y = sub["accuracy_strict_pct"].to_numpy()
        ax_a.plot(x, y, color=st["color"], marker=st["marker"], linewidth=2.4, markersize=7.5, label=st["label"])
        if "accuracy_ci95_low" in sub.columns and sub["accuracy_ci95_low"].notna().all():
            yerr_low = np.clip(y - sub["accuracy_ci95_low"].to_numpy(), 0, None)
            yerr_high = np.clip(sub["accuracy_ci95_high"].to_numpy() - y, 0, None)
            ax_a.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", ecolor=st["color"], alpha=0.35, capsize=3)

    p_base = PAPER_32K_REFERENCE["base"]
    p_edit = PAPER_32K_REFERENCE["edit"]
    ax_a.scatter(
        [p_base["budget"]],
        [p_base["accuracy_pct"]],
        marker="*",
        s=190,
        color="#dc2626",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Base ({p_base['accuracy_pct']}%)",
    )
    ax_a.scatter(
        [p_edit["budget"]],
        [p_edit["accuracy_pct"]],
        marker="*",
        s=190,
        color="#059669",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Edit ({p_edit['accuracy_pct']}%)",
    )
    ax_a.set_title("A. Strict Accuracy vs. Token Budget", fontsize=11.5, fontweight="bold")
    ax_a.set_ylabel("Strict MATH-500 Accuracy (%)", fontsize=10)
    ax_a.set_ylim(-3, 85)
    _style_x(ax_a)
    ax_a.legend(loc="upper left", fontsize=8.0)

    # Panel B: Forced-Answer Accuracy (%) vs Token Budget
    ax_b = axes[1]
    forced_sub = combined_df[combined_df["forced_accuracy_pct"].notna()] if not combined_df.empty else combined_df
    for a in alphas:
        sub = forced_sub[forced_sub["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        yf = sub["forced_accuracy_pct"].to_numpy()
        ys = sub["accuracy_strict_pct"].to_numpy()
        ax_b.plot(x, yf, color=st["color"], marker=st["marker"], linewidth=2.4, markersize=7.5, label=f"{st['label']} (Forced)")
        ax_b.plot(x, ys, color=st["color"], linestyle=":", linewidth=1.3, alpha=0.55)
        if "forced_accuracy_ci95_low" in sub.columns and sub["forced_accuracy_ci95_low"].notna().all():
            yerr_low = np.clip(yf - sub["forced_accuracy_ci95_low"].to_numpy(), 0, None)
            yerr_high = np.clip(sub["forced_accuracy_ci95_high"].to_numpy() - yf, 0, None)
            ax_b.errorbar(x, yf, yerr=[yerr_low, yerr_high], fmt="none", ecolor=st["color"], alpha=0.35, capsize=3)

    ax_b.set_title("B. Forced-Answer Accuracy (Solid) vs Strict (Dotted)", fontsize=11.5, fontweight="bold")
    ax_b.set_ylabel("Accuracy with Pass-2 Forced </think> (%)", fontsize=10)
    ax_b.set_ylim(-3, 85)
    _style_x(ax_b)
    if not forced_sub.empty:
        ax_b.legend(loc="upper left", fontsize=8.0)
    else:
        ax_b.text(0.5, 0.5, "Awaiting GPU forced-pass runs", transform=ax_b.transAxes, ha="center", va="center", fontsize=10, color="#64748b")

    # Panel C: % Hitting Token Cap vs Token Budget
    ax_c = axes[2]
    for a in alphas:
        sub = combined_df[combined_df["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        yc = sub["hit_cap_pct"].to_numpy()
        ax_c.plot(x, yc, color=st["color"], marker=st["marker"], linewidth=2.4, markersize=7.5, label=st["label"])
        if "hit_cap_ci95_low" in sub.columns and sub["hit_cap_ci95_low"].notna().all():
            yerr_low = np.clip(yc - sub["hit_cap_ci95_low"].to_numpy(), 0, None)
            yerr_high = np.clip(sub["hit_cap_ci95_high"].to_numpy() - yc, 0, None)
            ax_c.errorbar(x, yc, yerr=[yerr_low, yerr_high], fmt="none", ecolor=st["color"], alpha=0.35, capsize=3)

    ax_c.scatter(
        [p_base["budget"]],
        [p_base["hit_cap_pct"]],
        marker="*",
        s=190,
        color="#dc2626",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Base ({p_base['hit_cap_pct']}%)",
    )
    ax_c.scatter(
        [p_edit["budget"]],
        [p_edit["hit_cap_pct"]],
        marker="*",
        s=190,
        color="#059669",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Edit ({p_edit['hit_cap_pct']}%)",
    )
    ax_c.set_title("C. Token-Cap Truncation Rate vs. Budget", fontsize=11.5, fontweight="bold")
    ax_c.set_ylabel("% of Samples Hitting Token Cap", fontsize=10)
    ax_c.set_ylim(-5, 108)
    _style_x(ax_c)
    ax_c.legend(loc="lower left", fontsize=8.0)

    fig.suptitle(
        "Scaling Weight-Space Programming Across Token Budgets (4k MPS -> 8k/16k/32k GPU vLLM)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(target, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return str(target)


def update_notes_with_gpu_results(
    combined_df: pd.DataFrame,
    meta_records: Sequence[Dict[str, Any]],
    smoke_records: Sequence[Dict[str, Any]],
    notes_path: Union[str, Path] = "NOTES.md",
) -> bool:
    """Idempotently update `NOTES.md` inside `<!-- GPU_RESULTS_START --> ... <!-- GPU_RESULTS_END -->`."""
    npath = Path(notes_path)
    if not npath.is_file():
        return False

    gpu_rows = combined_df[combined_df["backend"] == "vllm"] if not combined_df.empty else pd.DataFrame()
    if gpu_rows.empty:
        return False

    today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    gpu_labels = sorted({f"{r['gpu'].upper()} ({r['dtype']})" for _, r in gpu_rows.iterrows()})
    budgets_executed = sorted({int(r["budget"]) for _, r in gpu_rows.iterrows()})
    alphas_executed = sorted({float(r["alpha"]) for _, r in gpu_rows.iterrows()})

    # Build compact markdown summary block
    lines: List[str] = [
        GPU_NOTES_START,
        f"### 3. GPU vLLM Multi-Budget Sweep Results ({', '.join(gpu_labels)}, updated {today_str})",
        "",
    ]

    if smoke_records:
        sm = smoke_records[-1]
        lines.append(
            f"- **Smoke Test (`gpu/smoke_test.py`)**: GPU=`{sm.get('gpu_tag', 'unknown')}` | "
            f"fp16_passed=`{sm.get('fp16_passed')}` | prompt_disagreements=`{sm.get('prompt_disagreements')}/{sm.get('n_problems', 8)}` | "
            f"fp16_nan_failures=`{len(sm.get('fp16_nan_failures', []))}` -> selected **`{sm.get('recommended_dtype', 'float32')}`**."
        )

    if meta_records:
        m0 = meta_records[-1]
        env_meta = m0.get("environment", {})
        smi = env_meta.get("nvidia_smi", {})
        freezes = [x for x in env_meta.get("pip_freeze_filtered", []) if x]
        if not freezes:
            pkg_vers = env_meta.get("package_versions", {})
            freezes = [f"{k}=={v}" for k, v in pkg_vers.items() if v]
        lines.append(
            f"- **GPU Runtime Environment**: `{smi.get('gpu_name', gpu_labels[0])}` | "
            f"Driver `{smi.get('driver_version', 'n/a')}` | CUDA `{smi.get('cuda_version', 'n/a')}` | "
            f"Python `{env_meta.get('python_version', 'n/a')}` | "
            f"Packages: `{', '.join(freezes[:6])}`."
        )

    lines.extend(
        [
            f"- **Budgets & Blend Coefficients**: Budgets `{budgets_executed}` across `alpha` in `{alphas_executed}` (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`).",
            "",
            "| Budget | Backend / GPU (`dtype`) | Blend $\\alpha$ | Samples (`N×S`) | Strict Acc % (95% CI) | Forced Acc % (95% CI) | Cap-Hit % (95% CI) | Mean Tokens | 8-Gram Rep % | Speed (`tok/s`) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, r in combined_df.iterrows():
        b = int(r["budget"])
        backend_str = f"`{r['backend']}` / `{r['gpu']}` (`{r['dtype']}`)"
        a_str = f"**{r['alpha']:.2f}**"
        n_eval_str = f"{int(r['n_problems'])}×{int(r['n_samples'])}={int(r['total_evaluations'])}"
        strict_str = (
            f"**{r['accuracy_strict_pct']:.1f}%** ({r['accuracy_ci95_low']:.1f}–{r['accuracy_ci95_high']:.1f}%)"
            if pd.notna(r.get("accuracy_ci95_low"))
            else f"**{r['accuracy_strict_pct']:.1f}%**"
        )
        if pd.notna(r.get("forced_accuracy_pct")):
            forced_str = (
                f"{r['forced_accuracy_pct']:.1f}% ({r['forced_accuracy_ci95_low']:.1f}–{r['forced_accuracy_ci95_high']:.1f}%)"
                if pd.notna(r.get("forced_accuracy_ci95_low"))
                else f"{r['forced_accuracy_pct']:.1f}%"
            )
        else:
            forced_str = "—"
        cap_str = (
            f"{r['hit_cap_pct']:.1f}% ({r['hit_cap_ci95_low']:.1f}–{r['hit_cap_ci95_high']:.1f}%)"
            if pd.notna(r.get("hit_cap_ci95_low"))
            else f"{r['hit_cap_pct']:.1f}%"
        )
        tps_str = f"{r['output_tok_per_s']:.1f}" if pd.notna(r.get("output_tok_per_s")) else "—"
        lines.append(
            f"| {b} | {backend_str} | {a_str} | {n_eval_str} | {strict_str} | {forced_str} | {cap_str} | "
            f"{r['mean_tokens']:.1f} | {r['mean_rep_8gram_pct']:.2f}% | {tps_str} |"
        )

    lines.extend(
        [
            "",
            "- **Published 32k Reference (`RikiyaT/lm_understand-logs` `exp8_math/runs/p50c`)**: Base (`α = 0.0`): **`7.9%`** strict accuracy / **`93.0%`** cut share; Edited `p50c` (`α = 1.0`): **`48.1%`** strict accuracy / **`33.05%`** cut share.",
            "- **Visual**: See [`assets/budget_curve_gpu.png`](assets/budget_curve_gpu.png) and per-budget dashboards under `assets/budget_<budget>/gpu-*/sweep_charts.png`.",
            GPU_NOTES_END,
        ]
    )
    block_text = "\n".join(lines)

    raw_notes = npath.read_text(encoding="utf-8")
    if GPU_NOTES_START in raw_notes and GPU_NOTES_END in raw_notes:
        updated_notes = re.sub(
            re.escape(GPU_NOTES_START) + r".*?" + re.escape(GPU_NOTES_END),
            lambda _m: block_text,
            raw_notes,
            count=1,
            flags=re.DOTALL,
        )
    else:
        anchor = "\n## Visuals"
        if anchor in raw_notes:
            updated_notes = raw_notes.replace(anchor, f"\n{block_text}\n{anchor}", 1)
        else:
            updated_notes = raw_notes.rstrip() + "\n\n" + block_text + "\n"

    atomic_write_text(npath, updated_notes)
    return True


def run_aggregation(
    out_root: str = "assets",
    notes_path: Optional[str] = "NOTES.md",
) -> pd.DataFrame:
    """Aggregate all budget sweep results, save `assets/budget_curve_gpu.png`, and update `NOTES.md`."""
    combined_df, meta_records, smoke_records = load_combined_budget_results(out_root=out_root)
    chart_path = Path(out_root) / "budget_curve_gpu.png"
    save_budget_curve_chart(combined_df, out_path=chart_path)
    print(f"[Aggregate] Saved budget curve chart -> {chart_path}", flush=True)

    if notes_path:
        updated = update_notes_with_gpu_results(
            combined_df=combined_df,
            meta_records=meta_records,
            smoke_records=smoke_records,
            notes_path=notes_path,
        )
        if updated:
            print(f"[Aggregate] Updated GPU results section in {notes_path}", flush=True)

    return combined_df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-budget GPU & MPS results and render `assets/budget_curve_gpu.png`."
    )
    parser.add_argument("--out-root", type=str, default="assets", help="Assets root directory (default: assets).")
    parser.add_argument("--notes", type=str, default="NOTES.md", help="Path to NOTES.md (or empty to skip).")
    args = parser.parse_args(argv)

    df = run_aggregation(out_root=args.out_root, notes_path=args.notes or None)
    if not df.empty:
        cols = [
            "budget",
            "backend",
            "gpu",
            "dtype",
            "alpha",
            "accuracy_strict_pct",
            "forced_accuracy_pct",
            "hit_cap_pct",
            "mean_tokens",
        ]
        avail = [c for c in cols if c in df.columns]
        print(df[avail].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
