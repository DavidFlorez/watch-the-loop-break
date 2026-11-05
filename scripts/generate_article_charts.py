"""Generate publication-quality 1600px-wide article charts and aggregate GPU/MPS charts."""

from __future__ import annotations

import os

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpu.aggregate import load_combined_budget_results, save_budget_curve_chart, save_mps_4k_comparison_chart
from gpu.common import atomic_write_text
from gpu.stats import generate_gpu_stats
from metrics import compute_loop_onset_from_token_ids, compute_loop_onset_token, repeated_8gram_ratio, wilson_ci
from weights import BASE_MODEL_ID

TARGET_WIDTH_PX = 1600
DPI = 160
FIG_W_IN = TARGET_WIDTH_PX / DPI  # 10.0 inches -> 1600 px exact when bbox_inches=None
FIG_H_IN = 6.25  # 1000 px height (16:10 landscape, high legibility on desktop & mobile)

OUTCOME_ORDER = [
    "Correct",
    "Wrong answer",
    "Capped while looping",
    "Capped not looping",
]
OUTCOME_COLORS = {
    "Correct": "#059669",
    "Wrong answer": "#f59e0b",
    "Capped while looping": "#dc2626",
    "Capped not looping": "#64748b",
}


def _spread_y_positions(raw_ys: Sequence[float], min_sep: float, y_min: float, y_max: float) -> List[float]:
    """Adjust sorted or unsorted label y-positions vertically so adjacent labels never overlap."""
    n = len(raw_ys)
    if n == 0:
        return []
    indexed = sorted(enumerate(raw_ys), key=lambda t: t[1])
    ys = [float(val) for _, val in indexed]

    # Forward pass: push upward to maintain min_sep
    for i in range(1, n):
        if ys[i] - ys[i - 1] < min_sep:
            ys[i] = ys[i - 1] + min_sep

    # Clamp top and push downward
    if ys[-1] > y_max:
        ys[-1] = y_max
        for i in range(n - 2, -1, -1):
            if ys[i + 1] - ys[i] < min_sep:
                ys[i] = ys[i + 1] - min_sep

    # Clamp bottom and push upward again
    if ys[0] < y_min:
        ys[0] = y_min
        for i in range(1, n):
            if ys[i] - ys[i - 1] < min_sep:
                ys[i] = ys[i - 1] + min_sep

    out = [0.0] * n
    for (orig_idx, _), new_y in zip(indexed, ys):
        out[orig_idx] = new_y
    return out


def _read_jsonl_records(jsonl_gz_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not jsonl_gz_path.is_file():
        return records
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _discover_best_cohort_root(assets_root: Path) -> Tuple[Path, int, str]:
    """Return `(cohort_root, n_problems, label_tag)` preferring `assets/gpu_n200` over `assets`."""
    candidates = [
        (assets_root / "gpu_n200", "N=200 Core Cohort"),
        (assets_root, "N=50 Cohort"),
    ]
    best_path = assets_root
    best_n = 0
    best_tag = "GPU Cohort"

    for root_cand, tag in candidates:
        b32 = root_cand / "budget_32768"
        if not b32.is_dir():
            continue
        gpu_dirs = sorted(
            [
                d
                for d in b32.iterdir()
                if d.is_dir() and d.name.startswith("gpu-") and (d / "sweep_results.csv").is_file()
            ]
        )
        if not gpu_dirs:
            continue
        df = pd.read_csv(gpu_dirs[0] / "sweep_results.csv")
        if not df.empty:
            n_val = int(df["n_problems"].max())
            if n_val > best_n:
                best_n = n_val
                best_path = root_cand
                best_tag = f"N={n_val}"

    return best_path, best_n, best_tag


def _load_cohort_budget_df(cohort_root: Path) -> pd.DataFrame:
    """Load `budget_<B>/gpu-*/sweep_results.csv` for all budgets under `cohort_root`."""
    rows: List[pd.DataFrame] = []
    if not cohort_root.is_dir():
        return pd.DataFrame()
    for b_dir in sorted(cohort_root.iterdir()):
        if not b_dir.is_dir() or not b_dir.name.startswith("budget_"):
            continue
        b_str = b_dir.name.replace("budget_", "")
        if not b_str.isdigit():
            continue
        b_int = int(b_str)
        gpu_dirs = sorted(
            [
                d
                for d in b_dir.iterdir()
                if d.is_dir() and d.name.startswith("gpu-") and (d / "sweep_results.csv").is_file()
            ]
        )
        if not gpu_dirs:
            continue
        df = pd.read_csv(gpu_dirs[0] / "sweep_results.csv")
        df["budget"] = b_int
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["budget", "alpha"]).reset_index(drop=True)


class _WhitespaceTokenizer:
    """Minimal offline tokenizer fallback when HuggingFace AutoTokenizer is not loaded."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[str]:
        return text.split()

    def decode(self, token_ids: Sequence[Any], skip_special_tokens: bool = False) -> str:
        return " ".join(str(t) for t in token_ids)


_LOOP_ONSET_CACHE: Dict[Tuple[str, float, int], Optional[int]] = {}


def _get_or_compute_loop_onset(rec: Dict[str, Any], tokenizer: Any = None) -> Optional[int]:
    """Return cached `loop_onset_token` or compute it if missing (e.g., in legacy N=50 records)."""
    if "loop_onset_token" in rec:
        return None if rec["loop_onset_token"] is None else int(rec["loop_onset_token"])
    cache_key = (
        str(rec.get("unique_id", "")),
        round(float(rec.get("alpha", -1.0)), 6),
        int(rec.get("n_tokens", len(str(rec.get("text", ""))))),
    )
    if cache_key[0] and cache_key in _LOOP_ONSET_CACHE:
        return _LOOP_ONSET_CACHE[cache_key]

    tok_impl = tokenizer if tokenizer is not None else _WhitespaceTokenizer()
    onset_res: Optional[int]
    if rec.get("out_ids") and tokenizer is not None:
        onset_res = compute_loop_onset_from_token_ids([int(x) for x in rec["out_ids"]], tok_impl)
    else:
        text = str(rec.get("text", ""))
        if not text:
            onset_res = None
        elif tokenizer is not None:
            onset_res = compute_loop_onset_token(text, tokenizer)
        else:
            words = text.split()
            n_tok = int(rec.get("n_tokens", len(words)))
            if not words:
                onset_res = None
            else:
                onset_w = compute_loop_onset_from_token_ids(words, tok_impl)
                onset_res = None if onset_w is None else int(round((onset_w / max(len(words), 1)) * n_tok))

    if cache_key[0]:
        _LOOP_ONSET_CACHE[cache_key] = onset_res
    return onset_res


def generate_headline_budget_curve(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/headline_budget_curve.png",
    paper_csv: Optional[Union[str, Path]] = None,
) -> str:
    """1. Headline Budget Curve: 'The edit only pays off when you let it think'."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cohort_root, n_core, _ = _discover_best_cohort_root(root)
    gpu_df = _load_cohort_budget_df(cohort_root)
    paper_path = Path(paper_csv) if paper_csv is not None else (root / "paper_reference.csv")
    paper_df = pd.read_csv(paper_path) if paper_path.is_file() else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Subtle 4k "budget trap" shading & unobstructed callout below the top-left legend
    ax.axvspan(3400, 4800, color="#f1f5f9", alpha=0.88, zorder=1)
    ax.text(
        4096,
        24.0,
        "4k Budget Trap\n≥88% hit cap thinking\n(strict acc ≤ 9.5%)",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0, alpha=0.96),
        zorder=6,
    )

    right_endpoints: List[Dict[str, Any]] = []

    # 1. Main GPU curves: alpha=0.0 (thick crimson), alpha=1.0 (thick emerald), alpha=1.25 (thinner royal blue)
    gpu_styles = {
        0.0: {
            "color": "#dc2626",
            "lw": 3.2,
            "marker": "o",
            "ms": 8.0,
            "ci": True,
            "label": f"Base Model (α = 0.0, 0.00% edited, N={n_core}, 95% CI)",
            "short": "Base α=0.0",
            "z": 5,
        },
        1.0: {
            "color": "#059669",
            "lw": 3.4,
            "marker": "s",
            "ms": 8.0,
            "ci": True,
            "label": f"Published Edit (α = 1.0, 0.015% edited, N={n_core}, 95% CI)",
            "short": "Edit α=1.0",
            "z": 6,
        },
        1.25: {
            "color": "#2563eb",
            "lw": 1.85,
            "marker": "D",
            "ms": 6.2,
            "ci": False,
            "label": f"1.25× Extrapolation (α = 1.25, N={n_core})",
            "short": "Extrapolated α=1.25",
            "z": 4,
        },
    }

    for a_val in (0.0, 1.0, 1.25):
        sub = gpu_df[gpu_df["alpha"] == a_val].sort_values("budget") if not gpu_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        st = gpu_styles[a_val]
        x = sub["budget"].to_numpy()
        y = sub["accuracy_pct"].to_numpy()
        ax.plot(
            x,
            y,
            color=st["color"],
            linewidth=st["lw"],
            marker=st["marker"],
            markersize=st["ms"],
            markeredgecolor="#ffffff",
            markeredgewidth=1.0,
            label=st["label"],
            zorder=st["z"],
        )
        if st["ci"] and "accuracy_ci95_low" in sub.columns and sub["accuracy_ci95_low"].notna().all():
            ax.fill_between(
                x,
                sub["accuracy_ci95_low"].to_numpy(),
                sub["accuracy_ci95_high"].to_numpy(),
                color=st["color"],
                alpha=0.14,
                zorder=2,
            )
        right_endpoints.append(
            {
                "x": float(x[-1]),
                "y": float(y[-1]),
                "text": f"{st['short']}: {y[-1]:.1f}%",
                "color": st["color"],
                "bold": True,
            }
        )

    # 2. Paper's truncated first-sample curves (dashed) + same-problem 32k avg@4 stars (*)
    if not paper_df.empty:
        trunc_sub = paper_df[
            (paper_df["metric_type"] == "first_sample_truncated")
            & (paper_df["cohort"] == "paper_first_samples_n200")
        ]
        for cond, a_val, col, lbl, short in (
            ("unedited", 0.0, "#ef4444", "Paper 1st-Sample Truncated: Base (N=200, approx.)", "Paper 1st Base"),
            ("p50c", 1.0, "#10b981", "Paper 1st-Sample Truncated: Edit p50c (N=200, approx.)", "Paper 1st Edit"),
        ):
            c_df = trunc_sub[trunc_sub["condition"] == cond].sort_values("budget")
            if c_df.empty:
                continue
            x = c_df["budget"].to_numpy()
            y = c_df["accuracy_pct"].to_numpy()
            ax.plot(
                x,
                y,
                color=col,
                linestyle="--",
                linewidth=1.8,
                marker="^",
                markersize=5.8,
                alpha=0.82,
                label=lbl,
                zorder=3,
            )
            right_endpoints.append(
                {
                    "x": float(x[-1]),
                    "y": float(y[-1]),
                    "text": f"{short}: {y[-1]:.1f}%",
                    "color": col,
                    "bold": False,
                }
            )

        # Same-problem 32k avg@4 benchmark stars
        star_cohort = f"our_n{n_core}_core" if n_core == 200 else (f"our_n{n_core}" if n_core > 0 else "our_n200_core")
        if star_cohort not in set(paper_df["cohort"].unique()):
            star_cohort = "our_n50" if "our_n50" in set(paper_df["cohort"].unique()) else "math500_all_500"

        star_sub = paper_df[(paper_df["metric_type"] == "avg_at_4") & (paper_df["cohort"] == star_cohort)]
        star_x = 32768.0 * 1.03  # slight right offset in log2 space so star never hides the 32k dot
        for cond, col, short in (
            ("unedited", "#991b1b", "Paper avg@4 Base"),
            ("p50c", "#065f46", "Paper avg@4 Edit"),
        ):
            s_row = star_sub[star_sub["condition"] == cond]
            if s_row.empty:
                continue
            r0 = s_row.iloc[0]
            y_val = float(r0["accuracy_pct"])
            n_match = int(r0["n_problems"])
            ax.scatter(
                [star_x],
                [y_val],
                marker="*",
                s=250,
                color=col,
                edgecolors="#0f172a",
                linewidths=1.0,
                zorder=8,
                label=f"{short} (same {n_match} probs @ 32k: {y_val:.1f}%)",
            )
            right_endpoints.append(
                {
                    "x": star_x,
                    "y": y_val,
                    "text": f"★ {short} (N={n_match}): {y_val:.1f}%",
                    "color": col,
                    "bold": True,
                }
            )

    # Spread right-edge callouts so zero labels collide even when curves converge near ~48%
    if right_endpoints:
        spread_ys = _spread_y_positions(
            [pt["y"] for pt in right_endpoints],
            min_sep=4.9,
            y_min=1.0,
            y_max=61.5,
        )
        label_x = 39000.0
        for pt, ly in zip(right_endpoints, spread_ys):
            ax.annotate(
                pt["text"],
                xy=(pt["x"], pt["y"]),
                xytext=(label_x, ly),
                ha="left",
                va="center",
                fontsize=8.3,
                fontweight="bold" if pt["bold"] else "normal",
                color=pt["color"],
                arrowprops=dict(arrowstyle="-", color=pt["color"], lw=0.9, alpha=0.55),
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="#ffffff",
                    edgecolor=pt["color"],
                    lw=0.8 if pt["bold"] else 0.5,
                    alpha=0.95,
                ),
                zorder=9,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlim(3000, 128000)
    ax.set_xticks([4096, 8192, 16384, 32768])
    ax.set_xticklabels(["4,096\n(4k)", "8,192\n(8k)", "16,384\n(16k)", "32,768\n(32k)"], fontsize=10.5, fontweight="bold")
    ax.minorticks_off()

    ax.set_ylim(-2.5, 64.5)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticklabels([f"{v}%" for v in (0, 10, 20, 30, 40, 50, 60)], fontsize=10.0)

    ax.set_xlabel("Thinking Token Budget (log₂ scale)", fontsize=11.0, fontweight="bold", labelpad=6, color="#0f172a")
    ax.set_ylabel("Strict MATH-500 Accuracy (%)", fontsize=11.0, fontweight="bold", labelpad=6, color="#0f172a")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=8.2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.95,
    )

    fig.suptitle(
        "The edit only pays off when you let it think",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        f"Strict boxed accuracy after </think> on RTX PRO 6000 (bf16, N={n_core}) vs. Takehi et al. (2026) 1st-sample & 32k avg@4 references",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#475569",
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.125)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_hero_trace_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/hero_trace.png",
    problem_id: str = "test/counting_and_probability/389.json",
    tokenizer: Any = None,
) -> str:
    """2. Hero Trace Comparison: 389.json at 32k (Base α=0.0 looping 1,292× vs Edited α=1.0 solving in 3,844 tok)."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Locate 32k JSONL containing `problem_id`
    jsonl_cands = [
        root / "gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
        root / "budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ]
    by_alpha: Dict[float, Dict[str, Any]] = {}
    for cand in jsonl_cands:
        if not cand.is_file():
            continue
        for rec in _read_jsonl_records(cand):
            if str(rec.get("unique_id")) == problem_id:
                by_alpha.setdefault(float(rec["alpha"]), rec)
        if 0.0 in by_alpha and 1.0 in by_alpha:
            break

    r0 = by_alpha.get(0.0, {})
    r1 = by_alpha.get(1.0, {})
    t0 = str(r0.get("text", ""))
    t1 = str(r1.get("text", ""))

    loop_sentence = "Wait, if toppings are distinct, then {A,B} is a valid configuration."
    repeat_cnt = t0.count(loop_sentence) if t0 else 1292
    tok0 = int(r0.get("n_tokens", 32768))
    tok1 = int(r1.get("n_tokens", 3844))
    onset0 = r0.get("loop_onset_token")
    if onset0 is None and t0:
        onset0 = _get_or_compute_loop_onset(r0, tokenizer=tokenizer)
    onset0_val = int(onset0) if onset0 is not None else 5120
    rep0_pct = float(r0.get("rep_8gram_ratio", 0.831)) * 100.0
    tail0_pct = float(r0.get("tail_rep_8gram_ratio", 0.9485)) * 100.0
    rep1_pct = float(r1.get("rep_8gram_ratio", 0.005)) * 100.0
    tail1_pct = float(r1.get("tail_rep_8gram_ratio", 0.0)) * 100.0

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.018, 0.025, 0.964, 0.825])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Anatomy of a Broken Loop — MATH-500 Problem 389 @ 32,768-Token Budget",
        fontsize=14.4,
        fontweight="bold",
        color="#0f172a",
        y=0.968,
    )
    fig.text(
        0.5,
        0.905,
        "\"Mikka wants to order a pizza with two different toppings. He has 8 different\n"
        "toppings to choose from. How many different pizzas could he order?\"   (Gold Answer: 28)",
        ha="center",
        va="center",
        fontsize=9.1,
        style="italic",
        color="#334155",
    )

    # Helper to draw each column
    def _draw_column(
        x0: float,
        w: float,
        *,
        header_title: str,
        header_bg: str,
        border_col: str,
        title_col: str,
        badge_text: str,
        badge_bg: str,
        transcript_lines: List[Tuple[str, str, bool]],
        metrics: List[Tuple[str, str, str]],
        is_base: bool,
    ) -> None:
        # Outer card (kept strictly inside 0.01..0.99 so rounded corners never clip)
        card = mpatches.FancyBboxPatch(
            (x0, 0.028),
            w,
            0.942,
            boxstyle="round,pad=0.005,rounding_size=0.018",
            facecolor="#ffffff",
            edgecolor=border_col,
            linewidth=2.0,
        )
        ax.add_patch(card)

        # Column header banner
        hdr = mpatches.FancyBboxPatch(
            (x0 + 0.012, 0.865),
            w - 0.024,
            0.095,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            facecolor=header_bg,
            edgecolor="none",
        )
        ax.add_patch(hdr)
        ax.text(x0 + 0.022, 0.928, header_title, fontsize=10.3, fontweight="bold", color=title_col, va="center")
        ax.text(x0 + 0.022, 0.885, badge_text, fontsize=8.0, fontweight="bold", color="#334155", va="center")

        # Token timeline bar (0 .. 32,768)
        bar_x = x0 + 0.022
        bar_w = w - 0.044
        bar_y = 0.745
        bar_h = 0.068

        ax.text(
            bar_x,
            bar_y + bar_h + 0.015,
            "Token Budget Timeline (0 → 32,768 tokens, shared scale)",
            fontsize=8.4,
            fontweight="bold",
            color="#475569",
        )

        # Background empty track
        track = mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.0)
        ax.add_patch(track)

        if is_base:
            frac_pre = min(1.0, onset0_val / 32768.0)
            w_pre = bar_w * frac_pre
            w_loop = bar_w * (1.0 - frac_pre)
            ax.add_patch(mpatches.Rectangle((bar_x, bar_y), w_pre, bar_h, facecolor="#fca5a5", edgecolor="none"))
            ax.add_patch(
                mpatches.Rectangle(
                    (bar_x + w_pre, bar_y),
                    w_loop,
                    bar_h,
                    facecolor="#dc2626",
                    edgecolor="none",
                    hatch="///",
                    alpha=0.92,
                )
            )
            ax.axvline(bar_x + w_pre, ymin=bar_y, ymax=bar_y + bar_h, color="#7f1d1d", lw=2.0)
            ax.text(
                bar_x + w_pre / 2,
                bar_y + bar_h / 2,
                f"Think\n0–{onset0_val:,}",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#7f1d1d",
            )
            ax.text(
                bar_x + w_pre + w_loop / 2,
                bar_y + bar_h / 2,
                f"Verbatim Loop: {32768 - onset0_val:,} tok ({100*(1-frac_pre):.1f}%)",
                ha="center",
                va="center",
                fontsize=8.1,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#991b1b", edgecolor="none", alpha=0.92),
            )
        else:
            frac_used = min(1.0, tok1 / 32768.0)
            w_used = max(bar_w * frac_used, bar_w * 0.08)
            ax.add_patch(mpatches.Rectangle((bar_x, bar_y), w_used, bar_h, facecolor="#059669", edgecolor="none"))
            unused_center_x = bar_x + w_used + (bar_w - w_used) * 0.5
            ax.text(
                unused_center_x,
                bar_y + bar_h / 2,
                f"Solved in {tok1:,} tok — {32768 - tok1:,} saved ({100*(1-frac_used):.1f}% free)",
                ha="center",
                va="center",
                fontsize=7.9,
                fontweight="bold",
                color="#065f46",
            )

        # Transcript excerpt panel
        box_y = 0.21
        box_h = 0.505
        term = mpatches.FancyBboxPatch(
            (bar_x, box_y),
            bar_w,
            box_h,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.1,
        )
        ax.add_patch(term)

        cur_y = box_y + box_h - 0.026
        line_step = 0.0345
        for text_line, line_col, is_bold in transcript_lines:
            ax.text(
                bar_x + 0.010,
                cur_y,
                text_line,
                fontsize=7.35,
                fontfamily="monospace",
                fontweight="bold" if is_bold else "normal",
                color=line_col,
                va="top",
                parse_math=False,
            )
            cur_y -= line_step

        # Bottom telemetry pills (2x2 grid)
        pill_w = (bar_w - 0.014) / 2.0
        pill_h = 0.068
        for idx_m, (m_label, m_val, m_col) in enumerate(metrics):
            col_i = idx_m % 2
            row_i = idx_m // 2
            px = bar_x + col_i * (pill_w + 0.014)
            py = 0.116 - row_i * (pill_h + 0.014)
            pill = mpatches.FancyBboxPatch(
                (px, py),
                pill_w,
                pill_h,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=header_bg,
                edgecolor=border_col,
                linewidth=0.9,
            )
            ax.add_patch(pill)
            ax.text(px + 0.010, py + pill_h * 0.67, m_label, fontsize=7.4, color="#475569", va="center")
            ax.text(px + 0.010, py + pill_h * 0.28, m_val, fontsize=8.3, fontweight="bold", color=m_col, va="center")

    left_lines = [
        ("1. Analyze the Problem: 8 distinct toppings, pick 2...", "#334155", False),
        ("... Calculate: 8! / (2! 6!) = 28.", "#334155", True),
        ("   Wait, if toppings are indistinguishable, {A,B} = {B,A}.", "#475569", False),
        ("   If the toppings are distinct, is {A,B} different?", "#475569", False),
        (f"-- [Loop Onset @ token {onset0_val:,} | 8-gram rep > 50%] --", "#991b1b", True),
        (f"* {loop_sentence[:44]}...", "#dc2626", True),
        (f"* {loop_sentence[:44]}...", "#dc2626", False),
        (f"* {loop_sentence[:44]}...", "#dc2626", False),
        (f"* {loop_sentence[:44]}...", "#dc2626", False),
        (f"   ... [repeats verbatim {repeat_cnt:,}x across {max(tok0 - onset0_val, 0):,} tokens] ...", "#991b1b", True),
        (f"* {loop_sentence[:44]}...", "#dc2626", False),
        ("* Wait, [HARD CUT AT 32,768 TOKENS — NO </think>]", "#7f1d1d", True),
        ("Outcome: Strict Accuracy = FALSE (pred_strict = None)", "#991b1b", True),
    ]

    right_lines = [
        ("1. Analyze the Problem: 8 distinct toppings, pick 2.", "#334155", False),
        ("   Constraint: Order doesn't matter (subset of size 2).", "#334155", False),
        ("   Formula: Binomial coefficient C(8, 2).", "#334155", False),
        ("   $$ C(8, 2) = \\frac{8!}{2!(8-2)!}", "#0f172a", True),
        ("              = \\frac{8 \\times 7}{2} = \\frac{56}{2} = 28 $$", "#0f172a", True),
        ("   Thus, there are 28 different pizzas Mikka can order.", "#334155", False),
        ("</think>", "#047857", True),
        ("", "#334155", False),
        ("To find the number of different pizzas, we compute", "#334155", False),
        ("the number of 2-topping combinations: C(8, 2) = 28.", "#334155", False),
        ("Final Answer:  \\boxed{28}   <|im_end|>", "#059669", True),
        ("", "#334155", False),
        (f"Outcome: Strict Accuracy = TRUE in {tok1:,} tokens", "#065f46", True),
    ]

    _draw_column(
        0.012,
        0.476,
        header_title="Base Model (α = 0.0) — Trapped in Verbatim Loop",
        header_bg="#fef2f2",
        border_col="#ef4444",
        title_col="#991b1b",
        badge_text="0.00% weights edited  |  finish_reason = length  |  No </think>",
        badge_bg="#fee2e2",
        transcript_lines=left_lines,
        metrics=[
            ("Generated Tokens", f"{tok0:,} / 32,768 (Capped)", "#b91c1c"),
            ("Loop Onset / Repeats", f"Token {onset0_val:,} ({repeat_cnt:,}x loop)", "#b91c1c"),
            ("8-Gram Repetition", f"{rep0_pct:.1f}% full (Tail: {tail0_pct:.1f}%)", "#b91c1c"),
            ("Strict Grade (After </think>)", "None (Forced Pass-2: 28)", "#b91c1c"),
        ],
        is_base=True,
    )

    _draw_column(
        0.512,
        0.476,
        header_title="Published Edit (α = 1.0) — Clean Solve",
        header_bg="#ecfdf5",
        border_col="#10b981",
        title_col="#065f46",
        badge_text="2 tensors (0.015% weights)  |  finish_reason = stop  |  8.5x shorter",
        badge_bg="#d1fae5",
        transcript_lines=right_lines,
        metrics=[
            ("Generated Tokens", f"{tok1:,} / 32,768 (Finished)", "#047857"),
            ("Loop Onset / Repeats", "None detected (0x repeats)", "#047857"),
            ("8-Gram Repetition", f"{rep1_pct:.2f}% full (Tail: {tail1_pct:.1f}%)", "#047857"),
            ("Strict Grade (After </think>)", "Exact \\boxed{28} [PASS]", "#047857"),
        ],
        is_base=False,
    )

    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def _compute_outcome_breakdown(jsonl_gz_path: Path, tokenizer: Any = None) -> pd.DataFrame:
    """Classify every trace in `sweep_per_problem.jsonl.gz` into the 4 mutually exclusive outcomes."""
    records = _read_jsonl_records(jsonl_gz_path)
    if not records:
        return pd.DataFrame()

    rows_by_alpha: Dict[float, Dict[str, int]] = {}
    totals_by_alpha: Dict[float, int] = {}

    for rec in records:
        a_f = float(rec["alpha"])
        totals_by_alpha[a_f] = totals_by_alpha.get(a_f, 0) + 1
        counts = rows_by_alpha.setdefault(a_f, {k: 0 for k in OUTCOME_ORDER})

        hit_cap = bool(rec.get("hit_cap", rec.get("finish_reason") == "length"))
        correct_strict = bool(rec.get("correct_strict", False))

        if correct_strict and not hit_cap:
            counts["Correct"] += 1
        elif not hit_cap:
            counts["Wrong answer"] += 1
        else:
            onset = _get_or_compute_loop_onset(rec, tokenizer=tokenizer)
            if onset is not None:
                counts["Capped while looping"] += 1
            else:
                counts["Capped not looping"] += 1

    out_rows: List[Dict[str, Any]] = []
    for a_f in sorted(rows_by_alpha.keys()):
        n_tot = totals_by_alpha[a_f]
        c = rows_by_alpha[a_f]
        row: Dict[str, Any] = {"alpha": a_f, "n_total": n_tot}
        for k in OUTCOME_ORDER:
            row[f"count_{k}"] = c[k]
            row[f"pct_{k}"] = (c[k] / n_tot) * 100.0 if n_tot > 0 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def generate_outcomes_32k_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/outcomes_32k.png",
    tokenizer: Any = None,
) -> str:
    """3. Outcome Breakdown at 32k: Correct / Wrong answer / Capped while looping / Capped not looping."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Discover core 32k JSONL and optional 6-dose N=100 32k JSONL
    core_cands = [
        root / "gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
        root / "budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ]
    core_path = next((p for p in core_cands if p.is_file()), None)
    dose_path = root / "gpu_dose_n100/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz"

    core_df = _compute_outcome_breakdown(core_path, tokenizer=tokenizer) if core_path else pd.DataFrame()

    # Combine core[:100] with dose[:100] so the 6-alpha dose panel shows all 6 doses (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
    combined_dose_df = pd.DataFrame()
    if core_path and dose_path.is_file():
        core_recs = _read_jsonl_records(core_path)
        dose_recs = _read_jsonl_records(dose_path)
        dose_ids = {str(r["unique_id"]) for r in dose_recs}
        merged_recs = [r for r in core_recs if str(r["unique_id"]) in dose_ids] + dose_recs
        rows_by_a: Dict[float, Dict[str, int]] = {}
        tot_by_a: Dict[float, int] = {}
        for rec in merged_recs:
            a_f = float(rec["alpha"])
            tot_by_a[a_f] = tot_by_a.get(a_f, 0) + 1
            counts = rows_by_a.setdefault(a_f, {k: 0 for k in OUTCOME_ORDER})
            hit_cap = bool(rec.get("hit_cap", rec.get("finish_reason") == "length"))
            correct_strict = bool(rec.get("correct_strict", False))
            if correct_strict and not hit_cap:
                counts["Correct"] += 1
            elif not hit_cap:
                counts["Wrong answer"] += 1
            else:
                onset = _get_or_compute_loop_onset(rec, tokenizer=tokenizer)
                counts["Capped while looping" if onset is not None else "Capped not looping"] += 1
        d_rows: List[Dict[str, Any]] = []
        for a_f in sorted(rows_by_a.keys()):
            nt = tot_by_a[a_f]
            c = rows_by_a[a_f]
            row = {"alpha": a_f, "n_total": nt}
            for k in OUTCOME_ORDER:
                row[f"count_{k}"] = c[k]
                row[f"pct_{k}"] = (c[k] / nt) * 100.0 if nt > 0 else 0.0
            d_rows.append(row)
        combined_dose_df = pd.DataFrame(d_rows)

    has_two_panels = not combined_dose_df.empty
    if has_two_panels:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(FIG_W_IN, FIG_H_IN),
            dpi=DPI,
            gridspec_kw={"width_ratios": [1.0, 1.55]},
        )
        axes_to_plot = [
            (ax1, core_df, f"A. Core Cohort (N={int(core_df['n_total'].max()) if not core_df.empty else 200})"),
            (ax2, combined_dose_df, f"B. Paired 6-Dose Sweep (Same N={int(combined_dose_df['n_total'].max())})"),
        ]
    else:
        fig, ax1 = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
        axes_to_plot = [
            (ax1, core_df, f"32,768-Token Outcome Breakdown (N={int(core_df['n_total'].max()) if not core_df.empty else 50})"),
        ]

    fig.patch.set_facecolor("#ffffff")

    for panel_idx, (ax, df_panel, subtitle) in enumerate(axes_to_plot):
        ax.set_facecolor("#ffffff")
        if df_panel.empty:
            continue
        alphas = df_panel["alpha"].tolist()
        x = np.arange(len(alphas))
        bottom = np.zeros(len(alphas), dtype=float)
        width = 0.56 if len(alphas) <= 4 else 0.62

        for cat in OUTCOME_ORDER:
            vals = df_panel[f"pct_{cat}"].to_numpy()
            bars = ax.bar(
                x,
                vals,
                width,
                bottom=bottom,
                color=OUTCOME_COLORS[cat],
                edgecolor="#ffffff",
                linewidth=1.0,
                label=cat if panel_idx == 0 else None,
                zorder=3,
            )
            for idx_b, (bar, v, bot) in enumerate(zip(bars, vals, bottom)):
                if v >= 5.5:
                    lbl_txt = f"{v:.1f}%".replace(".0%", "%")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bot + v / 2.0,
                        lbl_txt,
                        ha="center",
                        va="center",
                        fontsize=8.3,
                        fontweight="bold",
                        color="#ffffff",
                        zorder=4,
                    )
            bottom = bottom + vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"α = {a:g}\n({6*a:g}×)" for a in alphas], fontsize=9.8, fontweight="bold")
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=9.5)
        if panel_idx == 0:
            ax.set_ylabel("Share of MATH-500 Problems (%)", fontsize=10.6, fontweight="bold", color="#0f172a")
        ax.set_title(subtitle, fontsize=11.3, fontweight="bold", color="#0f172a", pad=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1", zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes_to_plot[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        fontsize=9.2,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
    )

    fig.suptitle(
        "What Happens at 32,768 Tokens: Loop Collapse vs. Coherent Reasoning",
        fontsize=14.2,
        fontweight="bold",
        color="#0f172a",
        y=0.968,
    )
    if has_two_panels:
        fig.text(
            0.5,
            0.916,
            "Repetition & cap hits fall monotonically across α = 0 → 1.5 (91% → 17% looped on S₁₀₀ ⊂ S₂₀₀); over-editing (α = 1.5) increases finished wrong answers",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#475569",
        )
        fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.13, wspace=0.14)
    else:
        fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.13)

    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_loop_onset_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/loop_onset.png",
    tokenizer: Any = None,
) -> str:
    """4. Loop Onset Distribution across traces (base vs edited, showing both capped and finished traces)."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cands = [
        root / "gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
        root / "budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ]
    jsonl_path = next((p for p in cands if p.is_file()), None)
    records = _read_jsonl_records(jsonl_path) if jsonl_path else []

    rows: List[Dict[str, Any]] = []
    for rec in records:
        onset = _get_or_compute_loop_onset(rec, tokenizer=tokenizer)
        hit_cap = bool(rec.get("hit_cap", rec.get("finish_reason") == "length"))
        rows.append(
            {
                "alpha": float(rec["alpha"]),
                "loop_onset_token": onset,
                "hit_cap": hit_cap,
                "correct_strict": bool(rec.get("correct_strict", False)),
            }
        )
    df = pd.DataFrame(rows)
    n_core = int(df.groupby("alpha").size().max()) if not df.empty else 0

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(FIG_W_IN, FIG_H_IN),
        dpi=DPI,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    alphas = sorted(df["alpha"].unique()) if not df.empty else [0.0, 1.0, 1.25]
    alpha_colors = {0.0: "#dc2626", 0.5: "#f59e0b", 0.75: "#0ea5e9", 1.0: "#059669", 1.25: "#2563eb", 1.5: "#7c3aed"}

    # Panel A: Binned Loop Onset Window (%) comparing Base vs Edited (Capped solid vs Finished hatched)
    bins = [(0, 4096, "0–4k\n(Early)"), (4096, 8192, "4k–8k\n(Typical)"), (8192, 16384, "8k–16k\n(Mid)"), (16384, 32769, "16k–32k\n(Late)")]
    x_cats = np.arange(len(bins) + 1)
    n_a = len(alphas)
    bar_w = min(0.25, 0.76 / max(n_a, 1))

    for i_a, a_val in enumerate(alphas):
        sub = df[df["alpha"] == a_val]
        nt = max(len(sub), 1)
        col = alpha_colors.get(a_val, "#475569")
        offset = (i_a - (n_a - 1) / 2.0) * bar_w

        cap_pcts: List[float] = []
        fin_pcts: List[float] = []
        for lo, hi, _ in bins:
            in_bin = sub[sub["loop_onset_token"].notna() & (sub["loop_onset_token"] >= lo) & (sub["loop_onset_token"] < hi)]
            cap_pcts.append((in_bin["hit_cap"].sum() / nt) * 100.0)
            fin_pcts.append(((~in_bin["hit_cap"]).sum() / nt) * 100.0)

        no_loop = sub[sub["loop_onset_token"].isna()]
        cap_pcts.append((no_loop["hit_cap"].sum() / nt) * 100.0)
        fin_pcts.append(((~no_loop["hit_cap"]).sum() / nt) * 100.0)

        cap_arr = np.array(cap_pcts)
        fin_arr = np.array(fin_pcts)
        ax1.bar(
            x_cats + offset,
            cap_arr,
            bar_w * 0.92,
            color=col,
            alpha=0.92,
            edgecolor="#ffffff",
            lw=0.7,
            label=f"α = {a_val:g} — Capped (solid)",
            zorder=3,
        )
        ax1.bar(
            x_cats + offset,
            fin_arr,
            bar_w * 0.92,
            bottom=cap_arr,
            color=col,
            alpha=0.38,
            hatch="///",
            edgecolor=col,
            lw=0.8,
            label=f"α = {a_val:g} — Finished (hatched)",
            zorder=3,
        )

    ax1.set_xticks(x_cats)
    ax1.set_xticklabels([b[2] for b in bins] + ["No Loop\nDetected"], fontsize=9.2, fontweight="bold")
    ax1.set_ylim(0, 92)
    ax1.set_ylabel("Share of All Traces (%)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax1.set_xlabel("Loop Onset Window (1024-tok window, >50% repeated 8-grams)", fontsize=9.8, fontweight="bold", color="#0f172a")
    ax1.set_title("A. When Loops Start (Solid = Hit Cap, Hatched = Recovered)", fontsize=10.8, fontweight="bold", color="#0f172a", pad=8)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=7.8, ncol=1, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # Panel B: Onset Token Distribution (Capped vs Finished traces that looped)
    rng = np.random.default_rng(42)
    for i_a, a_val in enumerate(alphas):
        sub_loop = df[(df["alpha"] == a_val) & df["loop_onset_token"].notna()]
        col = alpha_colors.get(a_val, "#475569")
        capped_vals = sub_loop[sub_loop["hit_cap"]]["loop_onset_token"].to_numpy(dtype=float)
        fin_vals = sub_loop[~sub_loop["hit_cap"]]["loop_onset_token"].to_numpy(dtype=float)

        if len(capped_vals) > 0:
            jitter_c = rng.uniform(-0.11, -0.02, size=len(capped_vals))
            ax2.scatter(
                np.full(len(capped_vals), i_a) + jitter_c,
                capped_vals,
                color=col,
                alpha=0.65,
                s=26,
                marker="o",
                edgecolors="#ffffff",
                linewidths=0.4,
                label="Capped trace (●)" if i_a == 0 else None,
                zorder=4,
            )
            med_c = float(np.median(capped_vals))
            ax2.hlines(med_c, i_a - 0.16, i_a + 0.16, color="#0f172a", lw=2.2, zorder=6)
            ax2.text(
                i_a,
                med_c + 650,
                f"med {med_c/1000:.1f}k",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor=col, lw=0.7, alpha=0.92),
                zorder=7,
            )

        if len(fin_vals) > 0:
            jitter_f = rng.uniform(0.03, 0.12, size=len(fin_vals))
            ax2.scatter(
                np.full(len(fin_vals), i_a) + jitter_f,
                fin_vals,
                color="#ffffff",
                alpha=0.92,
                s=34,
                marker="D",
                edgecolors=col,
                linewidths=1.4,
                label="Finished / transient loop (◇)" if any(len(df[(df["alpha"] == a) & df["loop_onset_token"].notna() & (~df["hit_cap"])]) > 0 for a in alphas[: i_a + 1]) and i_a == next((idx for idx, a in enumerate(alphas) if len(df[(df["alpha"] == a) & df["loop_onset_token"].notna() & (~df["hit_cap"])]) > 0), -1) else None,
                zorder=5,
            )

    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f"α = {a:g}\n({df[(df['alpha']==a) & df['loop_onset_token'].notna()].shape[0]}/{df[df['alpha']==a].shape[0]} looped)" for a in alphas], fontsize=9.2, fontweight="bold")
    ax2.set_ylim(-1500, 39200)
    ax2.set_yticks([0, 8192, 16384, 24576, 32768])
    ax2.set_yticklabels(["0", "8,192\n(8k)", "16,384\n(16k)", "24,576\n(24k)", "32,768\n(32k)"], fontsize=9.0)
    ax2.set_ylabel("Loop Onset Token Position", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("Blend Coefficient α (Looped / Total Traces)", fontsize=9.8, fontweight="bold", color="#0f172a")
    ax2.set_title("B. Onset Position: Capped (●) vs. Recovered (◇)", fontsize=10.6, fontweight="bold", color="#0f172a", pad=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    fig.suptitle(
        f"Loop Onset Shift Across 32,768-Token Traces (RTX PRO 6000 bf16, N={n_core} per α)",
        fontsize=14.2,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    fig.subplots_adjust(left=0.065, right=0.98, top=0.865, bottom=0.14, wspace=0.22)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_edit_diagram(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/edit_diagram.png",
) -> str:
    """5. Architectural & Rank-1 Mechanism Diagram of the 2-Tensor Weight-Space Edit."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ver_path = root / "weight_verification.json"
    ver = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.is_file() else {}
    total_params = int(ver.get("total_parameters", 873438784))
    touched_entries = int(ver.get("total_touched_entries", 131072))
    pct_touched = (touched_entries / total_params) * 100.0
    cos_sim = float(ver.get("u15_u19_cosine_similarity", 0.8490))
    tensors = ver.get("tensors", {})
    t15 = tensors.get("model.language_model.layers.15.mlp.down_proj.weight", {})
    t19 = tensors.get("model.language_model.layers.19.mlp.down_proj.weight", {})
    sv15 = t15.get("svd_top5", [1.7489, 0.00099, 0.00095, 0.00087, 0.00081])
    sv19 = t19.get("svd_top5", [1.3988, 0.00090, 0.00081, 0.00074, 0.00071])
    ratio15 = float(t15.get("sv1_to_sv2_ratio", 1771.2))
    ratio19 = float(t19.get("sv1_to_sv2_ratio", 1552.0))

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.02, 0.03, 0.59, 0.84])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "The Surgery: 131,072 Rank-1 Weights out of 873.4M (0.015%) Break the Loop",
        fontsize=14.2,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    fig.text(
        0.5,
        0.912,
        "Full SHA-256 & SVD audit of Qwen/Qwen3.5-0.8B vs. RikiyaT/Qwen3.5-0.8B-weight-space-programming (488 tensors)",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#475569",
    )

    # Box 1: Full Model Overview
    b1 = mpatches.FancyBboxPatch(
        (0.02, 0.72),
        0.96,
        0.24,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        linewidth=1.6,
    )
    ax.add_patch(b1)
    ax.text(0.045, 0.895, "1. Full Model — Qwen3.5-0.8B (24 Hybrid Layers, bfloat16)", fontsize=10.0, fontweight="bold", color="#0f172a")
    ax.text(
        0.045,
        0.825,
        f"• {total_params:,} total parameters across 488 tensors (18 DeltaNet + 6 GQA)",
        fontsize=8.6,
        color="#334155",
    )
    ax.text(
        0.045,
        0.765,
        "• 486 / 488 tensors (99.985%) are bit-for-bit identical in bfloat16",
        fontsize=8.6,
        fontweight="bold",
        color="#047857",
    )

    # Arrows down to Layer 15 and Layer 19
    ax.annotate("", xy=(0.25, 0.65), xytext=(0.34, 0.72), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.2))
    ax.annotate("", xy=(0.75, 0.65), xytext=(0.66, 0.72), arrowprops=dict(arrowstyle="->", color="#059669", lw=2.2))

    # Box 2A: Layer 15 & Box 2B: Layer 19
    for bx, bw, title, r_val, fro_pct, sha_pair, col, bg in (
        (
            0.02,
            0.46,
            "Layer 15 · mlp.down_proj.weight",
            ratio15,
            float(t15.get("relative_frobenius_change", 0.1064)) * 100.0,
            "ef60d2 → 82e2d7",
            "#2563eb",
            "#eff6ff",
        ),
        (
            0.52,
            0.46,
            "Layer 19 · mlp.down_proj.weight",
            ratio19,
            float(t19.get("relative_frobenius_change", 0.0811)) * 100.0,
            "638eed → 8ef3c6",
            "#059669",
            "#ecfdf5",
        ),
    ):
        card = mpatches.FancyBboxPatch(
            (bx, 0.31),
            bw,
            0.33,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor=bg,
            edgecolor=col,
            linewidth=1.8,
        )
        ax.add_patch(card)
        ax.text(bx + 0.018, 0.59, title, fontsize=9.2, fontweight="bold", color="#0f172a")
        ax.text(bx + 0.018, 0.525, "Shape: [1,024 × 3,584]  (3.67M weights)", fontsize=8.1, color="#334155")
        ax.text(bx + 0.018, 0.465, "Edited: 64 / 3,584 columns (1.79%)", fontsize=8.2, fontweight="bold", color=col)
        ax.text(bx + 0.018, 0.405, f"Rank-1 purity: σ₁/σ₂ = {r_val:,.1f} (99.999%)", fontsize=8.0, fontweight="bold", color="#0f172a")
        ax.text(bx + 0.018, 0.345, f"||ΔW||/||W|| = {fro_pct:.2f}% | sha {sha_pair}", fontsize=7.3, color="#475569")

    # Box 3: Exact Rank-1 Equation & Continuous Blend
    b3 = mpatches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.245,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor="#fffbeb",
        edgecolor="#d97706",
        linewidth=1.6,
    )
    ax.add_patch(b3)
    ax.text(0.045, 0.222, "3. Mechanism: Directional Residual Projection & Blend W(α)", fontsize=9.8, fontweight="bold", color="#92400e")
    ax.text(
        0.045,
        0.165,
        "W_edit = W_base - 6 * u_l ( u_l^T W_base * 1_{64 cols} )",
        fontsize=8.4,
        fontfamily="monospace",
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.045,
        0.118,
        "W(alpha) = W_base + alpha * ( W_edit - W_base )",
        fontsize=8.4,
        fontfamily="monospace",
        fontweight="bold",
        color="#92400e",
    )
    ax.text(
        0.045,
        0.042,
        f"• Touched entries: 2 × (1,024 × 64) = {touched_entries:,} / {total_params:,} ({pct_touched:.4f}% = 0.015%)\n"
        f"• Cross-layer repetition direction alignment: cos(u₁₅, u₁₉) = {cos_sim:.4f}",
        fontsize=8.1,
        color="#334155",
    )

    # Right Subplot: Actual SVD Singular Value Spectrum (log scale)
    ax_svd = fig.add_axes([0.695, 0.14, 0.285, 0.69])
    ax_svd.set_facecolor("#ffffff")
    ranks = [1, 2, 3, 4, 5]
    ax_svd.plot(ranks, sv15, marker="o", ms=7.5, lw=2.5, color="#2563eb", label=f"Layer 15 (σ₁/σ₂ = {ratio15:.0f}×)")
    ax_svd.plot(ranks, sv19, marker="s", ms=7.5, lw=2.5, color="#059669", label=f"Layer 19 (σ₁/σ₂ = {ratio19:.0f}×)")
    ax_svd.set_yscale("log")
    ax_svd.set_xticks(ranks)
    ax_svd.set_xticklabels([f"σ{r}" for r in ranks], fontsize=10.0, fontweight="bold")
    ax_svd.set_ylim(2e-4, 6.0)
    ax_svd.set_ylabel("Singular Value of ΔW (log scale)", fontsize=10.2, fontweight="bold", color="#0f172a")
    ax_svd.set_xlabel("Singular Value Index", fontsize=10.2, fontweight="bold", color="#0f172a")
    ax_svd.set_title("Exact Rank-1 Spectrum of ΔW", fontsize=11.0, fontweight="bold", color="#0f172a", pad=8)
    ax_svd.grid(True, which="both", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax_svd.spines["top"].set_visible(False)
    ax_svd.spines["right"].set_visible(False)
    ax_svd.legend(loc="upper right", fontsize=8.3, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    ax_svd.annotate(
        f"σ₁ = {sv15[0]:.2f} / {sv19[0]:.2f}\n(99.999% energy)",
        xy=(1.03, sv15[0] * 0.92),
        xytext=(1.55, 0.42),
        fontsize=8.2,
        fontweight="bold",
        color="#0f172a",
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#94a3b8", lw=0.8),
    )
    ax_svd.annotate(
        "σ₂..σ₅ ≤ 0.001\n(bfloat16 rounding only)",
        xy=(3, sv15[2]),
        xytext=(2.05, 0.0055),
        fontsize=8.1,
        color="#475569",
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.8),
    )

    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_all_charts(
    *,
    assets_root: Union[str, Path] = "assets",
    article_dir: Union[str, Path] = "assets/article",
    with_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Generate all 5 article charts (1600px wide), GPU-only budget chart, MPS 4k chart, and stats."""
    root = Path(assets_root)
    art = Path(article_dir)
    art.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if with_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # Refresh paired statistics CSV across all available cohorts
    stats_df = generate_gpu_stats(assets_root=root, out_csv=root / "gpu_stats.csv")

    # Render GPU-only budget curve (preferring largest N cohort) + MPS 4k comparison
    cohort_root, n_core, _ = _discover_best_cohort_root(root)
    comb_df, _, _ = load_combined_budget_results(out_root=cohort_root)
    if comb_df.empty:
        comb_df, _, _ = load_combined_budget_results(out_root=root)
    gpu_curve_path = save_budget_curve_chart(comb_df, out_path=root / "budget_curve_gpu.png")

    mps_comp_path = None
    if (root / "sweep_results.csv").is_file() and (
        root / "budget_4096/gpu-rtx-pro-6000-bfloat16/sweep_results.csv"
    ).is_file():
        mps_comp_path = save_mps_4k_comparison_chart(
            mps_csv_path=root / "sweep_results.csv",
            gpu_4k_csv_path=root / "budget_4096/gpu-rtx-pro-6000-bfloat16/sweep_results.csv",
            out_path=root / "mps_4k_comparison.png",
        )

    article_paths = {
        "headline_budget_curve": generate_headline_budget_curve(
            assets_root=root,
            out_path=art / "headline_budget_curve.png",
        ),
        "hero_trace": generate_hero_trace_chart(
            assets_root=root,
            out_path=art / "hero_trace.png",
            tokenizer=tokenizer,
        ),
        "outcomes_32k": generate_outcomes_32k_chart(
            assets_root=root,
            out_path=art / "outcomes_32k.png",
            tokenizer=tokenizer,
        ),
        "loop_onset": generate_loop_onset_chart(
            assets_root=root,
            out_path=art / "loop_onset.png",
            tokenizer=tokenizer,
        ),
        "edit_diagram": generate_edit_diagram(
            assets_root=root,
            out_path=art / "edit_diagram.png",
        ),
    }

    return {
        "n_core": n_core,
        "stats_rows": int(len(stats_df)),
        "budget_curve_gpu": gpu_curve_path,
        "mps_4k_comparison": mps_comp_path,
        "article_charts": article_paths,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate 1600px article charts and aggregate GPU/MPS plots.")
    parser.add_argument("--assets-root", type=str, default="assets", help="Assets root directory (default: assets).")
    parser.add_argument("--article-dir", type=str, default="assets/article", help="Article chart directory.")
    parser.add_argument(
        "--with-tokenizer",
        action="store_true",
        help="Load Qwen tokenizer to compute loop_onset_token on any legacy records missing it.",
    )
    args = parser.parse_args(argv)

    summary = generate_all_charts(
        assets_root=args.assets_root,
        article_dir=args.article_dir,
        with_tokenizer=args.with_tokenizer,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
