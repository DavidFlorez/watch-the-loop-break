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
    """Render GPU-only 3-panel budget curve (Strict Acc, Forced Acc, Cap-Hit % vs Token Budget) + Paper 32k stars."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    gpu_df = (
        combined_df[combined_df["backend"] != "mps"].copy()
        if not combined_df.empty and "backend" in combined_df.columns
        else combined_df.copy()
    )
    if gpu_df.empty:
        gpu_df = combined_df.copy()

    # If multiple GPU cohorts exist (e.g. legacy N=50 and new N=200), keep the largest N cohort so lines never zig-zag
    if not gpu_df.empty and "n_problems" in gpu_df.columns and "budget" in gpu_df.columns and "alpha" in gpu_df.columns:
        gpu_df = (
            gpu_df.sort_values(["n_problems", "total_evaluations"], ascending=[True, True])
            .drop_duplicates(subset=["budget", "alpha"], keep="last")
            .sort_values(["budget", "alpha"])
            .reset_index(drop=True)
        )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    alphas = sorted(gpu_df["alpha"].unique()) if not gpu_df.empty else [0.0, 1.0, 1.25]
    all_budgets = (
        sorted({int(b) for b in gpu_df["budget"].unique()} | {32768})
        if not gpu_df.empty
        else [4096, 8192, 16384, 32768]
    )
    n_label = (
        f"N={int(gpu_df['n_problems'].max())}"
        if not gpu_df.empty and "n_problems" in gpu_df.columns
        else "GPU"
    )
    gpu_tag_label = (
        str(gpu_df["gpu"].iloc[0]).upper()
        if not gpu_df.empty and "gpu" in gpu_df.columns
        else "GPU"
    )

    def _style_x(ax: Any) -> None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_budgets)
        ax.set_xticklabels([f"{b // 1024}k ({b})" if b % 1024 == 0 else str(b) for b in all_budgets], fontsize=9.0)
        ax.set_xlabel("Max New Token Budget (log₂ scale)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.35)

    # Panel A: Strict Accuracy (%) vs Token Budget
    ax_a = axes[0]
    for a in alphas:
        sub = gpu_df[gpu_df["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        y = sub["accuracy_strict_pct"].to_numpy()
        lw = 2.0 if float(a) in (0.5, 0.75, 1.25, 1.5) else 2.8
        ax_a.plot(x, y, color=st["color"], marker=st["marker"], linewidth=lw, markersize=7.5, label=st["label"])
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
        s=200,
        color="#dc2626",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Base avg@4 ({p_base['accuracy_pct']}%)",
    )
    ax_a.scatter(
        [p_edit["budget"]],
        [p_edit["accuracy_pct"]],
        marker="*",
        s=200,
        color="#059669",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Edit avg@4 ({p_edit['accuracy_pct']}%)",
    )
    ax_a.set_title("A. Strict Accuracy vs. Token Budget", fontsize=12.0, fontweight="bold")
    ax_a.set_ylabel("Strict MATH-500 Accuracy (%)", fontsize=10.5)
    ax_a.set_ylim(-3, 72)
    _style_x(ax_a)
    ax_a.legend(loc="upper left", fontsize=8.2, frameon=True)

    # Panel B: Forced-Answer Accuracy (%) vs Token Budget
    ax_b = axes[1]
    forced_sub = gpu_df[gpu_df["forced_accuracy_pct"].notna()] if not gpu_df.empty else gpu_df
    for a in alphas:
        sub = forced_sub[forced_sub["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        yf = sub["forced_accuracy_pct"].to_numpy()
        ys = sub["accuracy_strict_pct"].to_numpy()
        lw = 2.0 if float(a) in (0.5, 0.75, 1.25, 1.5) else 2.8
        ax_b.plot(x, yf, color=st["color"], marker=st["marker"], linewidth=lw, markersize=7.5, label=f"{st['label']} (Forced)")
        ax_b.plot(x, ys, color=st["color"], linestyle=":", linewidth=1.4, alpha=0.55)
        if "forced_accuracy_ci95_low" in sub.columns and sub["forced_accuracy_ci95_low"].notna().all():
            yerr_low = np.clip(yf - sub["forced_accuracy_ci95_low"].to_numpy(), 0, None)
            yerr_high = np.clip(sub["forced_accuracy_ci95_high"].to_numpy() - yf, 0, None)
            ax_b.errorbar(x, yf, yerr=[yerr_low, yerr_high], fmt="none", ecolor=st["color"], alpha=0.35, capsize=3)

    ax_b.set_title("B. Forced-Answer Accuracy (Solid) vs. Strict (Dotted)", fontsize=12.0, fontweight="bold")
    ax_b.set_ylabel("Accuracy with Pass-2 Forced Completion (%)", fontsize=10.5)
    ax_b.set_ylim(-3, 72)
    _style_x(ax_b)
    if not forced_sub.empty:
        ax_b.legend(loc="upper left", fontsize=8.2, frameon=True)
    else:
        ax_b.text(0.5, 0.5, "Awaiting GPU forced-pass runs", transform=ax_b.transAxes, ha="center", va="center", fontsize=10, color="#64748b")

    # Panel C: % Hitting Token Cap vs Token Budget
    ax_c = axes[2]
    for a in alphas:
        sub = gpu_df[gpu_df["alpha"] == a].sort_values("budget")
        if sub.empty:
            continue
        st = ALPHA_STYLE.get(float(a), {"color": "#475569", "marker": "o", "label": f"α = {a:g}"})
        x = sub["budget"].to_numpy()
        yc = sub["hit_cap_pct"].to_numpy()
        lw = 2.0 if float(a) in (0.5, 0.75, 1.25, 1.5) else 2.8
        ax_c.plot(x, yc, color=st["color"], marker=st["marker"], linewidth=lw, markersize=7.5, label=st["label"])
        if "hit_cap_ci95_low" in sub.columns and sub["hit_cap_ci95_low"].notna().all():
            yerr_low = np.clip(yc - sub["hit_cap_ci95_low"].to_numpy(), 0, None)
            yerr_high = np.clip(sub["hit_cap_ci95_high"].to_numpy() - yc, 0, None)
            ax_c.errorbar(x, yc, yerr=[yerr_low, yerr_high], fmt="none", ecolor=st["color"], alpha=0.35, capsize=3)

    ax_c.scatter(
        [p_base["budget"]],
        [p_base["hit_cap_pct"]],
        marker="*",
        s=200,
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
        s=200,
        color="#059669",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=6,
        label=f"Paper 32k Edit ({p_edit['hit_cap_pct']}%)",
    )
    ax_c.set_title("C. Token-Cap Truncation Rate vs. Budget", fontsize=12.0, fontweight="bold")
    ax_c.set_ylabel("% of Samples Hitting Token Cap", fontsize=10.5)
    ax_c.set_ylim(-5, 108)
    _style_x(ax_c)
    ax_c.legend(loc="lower left", fontsize=8.2, frameon=True)

    fig.suptitle(
        f"GPU-Only Token-Budget Scaling ({gpu_tag_label} vLLM, {n_label}, 4k → 8k → 16k → 32k)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return str(target)


def save_mps_4k_comparison_chart(
    *,
    mps_csv_path: Union[str, Path] = "assets/sweep_results.csv",
    gpu_4k_csv_path: Union[str, Path] = "assets/budget_4096/gpu-rtx-pro-6000-bfloat16/sweep_results.csv",
    out_path: Union[str, Path] = "assets/mps_4k_comparison.png",
) -> str:
    """Render side-by-side 4,096-token Apple M3 MPS vs. NVIDIA RTX PRO 6000 GPU comparison."""
    mps_p = Path(mps_csv_path)
    gpu_p = Path(gpu_4k_csv_path)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    mps_df = pd.read_csv(mps_p).sort_values("alpha") if mps_p.is_file() else pd.DataFrame()
    gpu_df = pd.read_csv(gpu_p).sort_values("alpha") if gpu_p.is_file() else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    all_alphas = sorted(
        set(mps_df["alpha"].tolist() if not mps_df.empty else [])
        | set(gpu_df["alpha"].tolist() if not gpu_df.empty else [])
    )
    x_pos = np.arange(len(all_alphas))
    width = 0.27

    mps_strict = []
    mps_lo, mps_hi = [], []
    gpu_strict, gpu_forced = [], []
    gpu_slo, gpu_shi = [], []
    mps_tail, gpu_tail = [], []

    for a in all_alphas:
        m_sub = mps_df[mps_df["alpha"] == a] if not mps_df.empty else pd.DataFrame()
        g_sub = gpu_df[gpu_df["alpha"] == a] if not gpu_df.empty else pd.DataFrame()
        if not m_sub.empty:
            r = m_sub.iloc[0]
            val = float(r["accuracy_pct"])
            lo, hi = wilson_ci(int(r["correct"]), int(r["n_problems"]))
            mps_strict.append(val)
            mps_lo.append(max(0.0, val - lo * 100.0))
            mps_hi.append(max(0.0, hi * 100.0 - val))
            mps_tail.append(float(r["mean_tail_rep_8gram_pct"]))
        else:
            mps_strict.append(np.nan)
            mps_lo.append(0.0)
            mps_hi.append(0.0)
            mps_tail.append(np.nan)

        if not g_sub.empty:
            r = g_sub.iloc[0]
            val = float(r["accuracy_pct"])
            fval = float(r.get("forced_accuracy_pct", np.nan))
            lo, hi = wilson_ci(int(r["correct"]), int(r["n_problems"]))
            gpu_strict.append(val)
            gpu_forced.append(fval)
            gpu_slo.append(max(0.0, val - lo * 100.0))
            gpu_shi.append(max(0.0, hi * 100.0 - val))
            gpu_tail.append(float(r["mean_tail_rep_8gram_pct"]))
        else:
            gpu_strict.append(np.nan)
            gpu_forced.append(np.nan)
            gpu_slo.append(0.0)
            gpu_shi.append(0.0)
            gpu_tail.append(np.nan)

    # Panel A: Accuracy @ 4096 Tokens
    b1 = ax1.bar(x_pos - width, np.nan_to_num(mps_strict, nan=0.0), width, label="Apple M3 MPS (Strict)", color="#6366f1", alpha=0.9)
    b2 = ax1.bar(x_pos, np.nan_to_num(gpu_strict, nan=0.0), width, label="RTX PRO 6000 vLLM (Strict)", color="#10b981", alpha=0.9)
    b3 = ax1.bar(x_pos + width, np.nan_to_num(gpu_forced, nan=0.0), width, label="RTX PRO 6000 vLLM (Forced Pass 2)", color="#0ea5e9", alpha=0.75)

    for idx, val in enumerate(mps_strict):
        if pd.notna(val):
            ax1.text(x_pos[idx] - width, val + 0.6, f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#312e81")
    for idx, val in enumerate(gpu_strict):
        if pd.notna(val):
            ax1.text(x_pos[idx], val + 0.6, f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#065f46")
        else:
            ax1.text(x_pos[idx] + width * 0.5, 0.8, "n/a", ha="center", va="bottom", fontsize=7.5, color="#64748b")
    for idx, val in enumerate(gpu_forced):
        if pd.notna(val):
            ax1.text(x_pos[idx] + width, val + 0.6, f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0369a1")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"α = {a:g}\n({6*a:g}×)" for a in all_alphas], fontsize=9.5)
    ax1.set_ylabel("MATH-500 Accuracy @ 4,096 Tokens (%)", fontsize=10.5)
    ax1.set_title("A. 4k Accuracy: Local MPS vs. Cloud GPU vLLM", fontsize=12.0, fontweight="bold")
    ax1.set_ylim(0, 22)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Panel B: Last 240 Words Tail Loop Ratio (%) & Throughput
    ax2.plot(x_pos, mps_tail, marker="o", linewidth=2.5, markersize=8, color="#6366f1", label="Apple M3 MPS — Last 240 Words Tail Loop (%)")
    valid_g = [i for i, v in enumerate(gpu_tail) if pd.notna(v)]
    if valid_g:
        ax2.plot(
            [x_pos[i] for i in valid_g],
            [gpu_tail[i] for i in valid_g],
            marker="s",
            linewidth=2.5,
            markersize=8,
            color="#10b981",
            label="RTX PRO 6000 vLLM — Last 240 Words Tail Loop (%)",
        )
    for i, val in enumerate(mps_tail):
        if pd.notna(val):
            g_v = gpu_tail[i] if i < len(gpu_tail) and pd.notna(gpu_tail[i]) else None
            above = True if g_v is None else (float(val) >= float(g_v))
            ax2.annotate(
                f"{val:.1f}%",
                (x_pos[i], val),
                textcoords="offset points",
                xytext=(12 if (i > 0 and above and g_v is not None) else 0, 8 if above else -14),
                ha="left" if (i > 0 and above and g_v is not None) else "center",
                fontsize=8.5,
                fontweight="bold",
                color="#312e81",
            )
    for i in valid_g:
        val = float(gpu_tail[i])
        m_v = mps_tail[i] if i < len(mps_tail) and pd.notna(mps_tail[i]) else None
        above = False if m_v is None else (val > float(m_v))
        ax2.annotate(
            f"{val:.1f}%",
            (x_pos[i], val),
            textcoords="offset points",
            xytext=(0, 8 if above else -14),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#065f46",
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"α = {a:g}\n({6*a:g}×)" for a in all_alphas], fontsize=9.5)
    ax2.set_ylabel("Last 240 Words Tail Loop Ratio (%)", fontsize=10.5)
    ax2.set_title("B. Tail Loop Collapse Across Backends @ 4,096 Tokens", fontsize=12.0, fontweight="bold")
    ax2.set_ylim(-2.2, 19.2)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "4,096-Token Budget Comparison: Local Apple M3 MPS vs. RTX PRO 6000 vLLM (N=50)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(target, dpi=160)
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
            f"- **Budgets & Blend Coefficients**: Initial `N=50` GPU cohort (`assets/budget_*/gpu-*`, separate from the `N=50` MPS table in §2 above; see §5 for the `N=200` scaling cohort) across budgets `{budgets_executed}` and `alpha` in `{alphas_executed}` (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=True`).",
            "",
            "| Budget | Backend / GPU (`dtype`) | Blend $\\alpha$ | Samples (`N×S`) | Strict Acc % (95% CI) | Forced Acc % (95% CI) | Cap-Hit % (95% CI) | Mean Tokens | 8-Gram Rep % | Speed (`tok/s`) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, r in gpu_rows.iterrows():
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
            "- **Published 32k Reference (`RikiyaT/lm_understand-logs` `exp8_math/runs/p50c`)**: Same-50 problem `avg@4`: **`7.5%`** Base (`unedited`) vs **`41.0%`** Edited (`p50c`); full-500 benchmark `avg@4`: **`7.9%`** Base / **`93.0%`** cut share vs **`48.1%`** Edited / **`33.05%`** cut share (see §5 for the `N=200` cohort where same-problem `avg@4` is **`9.5%`** vs **`48.0%`**).",
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
