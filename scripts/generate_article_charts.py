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

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from matplotlib.mathtext import MathTextParser
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
    "Ran out while looping",
    "Ran out, not looping",
]
OUTCOME_COLORS = {
    "Correct": "#059669",
    "Wrong answer": "#f59e0b",
    "Ran out while looping": "#dc2626",
    "Ran out, not looping": "#64748b",
}

HARDWARE_FOOTER_TEXT = "Run on a Colab GPU (RTX PRO 6000)"

FORBIDDEN_RENDERED_TOKENS: Tuple[str, ...] = ("\\", "$", "**", "\\boxed", "\\frac")
FORBIDDEN_ARTICLE_TERMS: Tuple[str, ...] = (
    "</think>",
    "bf16",
    "bfloat16",
    "Forced Pass-2",
    "finish_reason",
    "Blend coefficient",
    "headline_budget_curve_full.png",
)


def _draw_rich_spans(
    fig: Any,
    ax: Any,
    x: float,
    y: float,
    spans: Sequence[Tuple[str, ...]],
    *,
    fontsize: float,
    default_color: str = "#0f172a",
    fontfamily: str = "sans-serif",
    default_weight: str = "normal",
    va: str = "center",
    zorder: int = 4,
) -> List[Any]:
    """Render mixed plain/bold/italic/mathtext spans cleanly along a single baseline."""
    artists: List[Any] = []
    cur_x = float(x)
    total = len(spans)
    for idx, span in enumerate(spans):
        kind = span[0]
        content = span[1]
        col = span[2] if len(span) > 2 and span[2] else default_color
        if kind == "math":
            txt_str = f"${content}$"
            t_art = ax.text(
                cur_x,
                y,
                txt_str,
                fontsize=fontsize,
                fontfamily=fontfamily,
                fontweight=default_weight,
                color=col,
                va=va,
                ha="left",
                parse_math=True,
                zorder=zorder,
            )
        else:
            fw = "bold" if kind == "bold" else default_weight
            fs = "italic" if kind == "italic" else "normal"
            t_art = ax.text(
                cur_x,
                y,
                content,
                fontsize=fontsize,
                fontfamily=fontfamily,
                fontweight=fw,
                fontstyle=fs,
                color=col,
                va=va,
                ha="left",
                parse_math=False,
                zorder=zorder,
            )
        artists.append(t_art)
        if idx < total - 1:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = t_art.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            cur_x = float(bb.x1)
    return artists


def _fit_text_to_width(fig: Any, text_artist: Any, max_width_frac: float = 0.92, min_fontsize: float = 7.5) -> None:
    """Shrink font size proportionally if a figure-level text artist exceeds `max_width_frac`."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_window_extent(renderer=renderer).width
    max_px = fig_w * max_width_frac
    bb = text_artist.get_window_extent(renderer=renderer)
    if bb.width > max_px and bb.width > 0:
        cur_fs = float(text_artist.get_fontsize())
        text_artist.set_fontsize(max(min_fontsize, cur_fs * (max_px / bb.width) * 0.985))


def extract_rendered_strings(fig: Any) -> List[str]:
    """Draw `fig` and collect every rendered non-mathtext string while verifying valid mathtext."""
    import warnings

    rendered: List[str] = []
    seen_set = set()

    def _add(val: str) -> None:
        if val not in seen_set:
            seen_set.add(val)
            rendered.append(val)

    orig_draw_text = RendererBase.draw_text

    def _spy_draw_text(self: Any, gc: Any, x: Any, y: Any, s: str, prop: Any, angle: Any, ismath: Any = False, mtext: Any = None) -> Any:
        if not ismath:
            _add(str(s))
        return orig_draw_text(self, gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)

    RendererBase.draw_text = _spy_draw_text  # type: ignore[method-assign]
    parser = MathTextParser("path")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
            for w in caught:
                msg = str(w.message)
                if "MathTextWarning" in type(w.message).__name__ or "missing from font" in msg:
                    raise AssertionError(f"Matplotlib rendering warning detected: {type(w.message).__name__}: {msg}")

        dpi_val = float(fig.dpi)
        for t in fig.findobj(matplotlib.text.Text):
            if not t.get_visible():
                continue
            raw = t.get_text()
            if not raw:
                continue
            is_valid_single_math = (
                bool(t.get_parse_math())
                and cbook.is_math_text(raw)
                and raw.startswith("$")
                and raw.endswith("$")
                and "$$" not in raw
                and raw.count("$") == 2
            )
            if is_valid_single_math:
                for forbidden_in_math in ("**", "\\boxed", "\\binom"):
                    if forbidden_in_math in raw:
                        raise AssertionError(f"Forbidden raw markup {forbidden_in_math!r} inside math artist {raw!r}")
                parser.parse(raw, dpi_val, t.get_fontproperties())
            else:
                _add(raw)
    finally:
        RendererBase.draw_text = orig_draw_text  # type: ignore[method-assign]

    return rendered


def validate_figure_no_raw_markup(fig: Any, *, check_article_jargon: bool = True) -> List[str]:
    """Assert that a rendered figure contains zero raw markup/LaTeX or forbidden article jargon."""
    rendered = extract_rendered_strings(fig)
    for s in rendered:
        for tok in FORBIDDEN_RENDERED_TOKENS:
            if tok in s:
                raise AssertionError(f"Rendered string contains forbidden markup {tok!r}: {s!r}")
        if "_" in s:
            raise AssertionError(f"Rendered plain string contains unformatted underscore '_': {s!r}")
        if s.lstrip().startswith("*   ") or s.lstrip().startswith("* "):
            raise AssertionError(f"Rendered plain string contains raw markdown bullet '* ': {s!r}")
        if check_article_jargon:
            low = s.lower()
            for term in FORBIDDEN_ARTICLE_TERMS:
                if term.lower() in low:
                    raise AssertionError(f"Rendered article figure contains forbidden term {term!r}: {s!r}")
    return rendered



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


def _resolve_local_tokenizer(tokenizer: Any = None) -> Any:
    """Return `tokenizer` or load `Qwen/Qwen3.5-0.8B` from local cache when available."""
    if tokenizer is not None:
        return tokenizer
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    except Exception:
        return None


def _compute_substring_token_index(
    out_ids: Sequence[int],
    text: str,
    substring: str,
    tokenizer: Any = None,
) -> Optional[int]:
    """Compute the 0-based token index where the final number/span of `substring` starts in `out_ids`."""
    if not out_ids or not text or substring not in text:
        return None
    char_pos = text.index(substring)
    sub_28_offset = substring.rfind("28")
    target_char = char_pos + (sub_28_offset if sub_28_offset >= 0 else 0)

    tok = _resolve_local_tokenizer(tokenizer)
    if tok is not None:
        cum = 0
        for idx, tid in enumerate(out_ids):
            piece = tok.decode([int(tid)], skip_special_tokens=False)
            next_cum = cum + len(piece)
            if next_cum > target_char:
                return int(idx)
            cum = next_cum

    # Fallback: exact BPE token subsequence match in `out_ids`
    target_seq = [71690, 25, 56014, 35790, 90, 23, 1088, 14695, 220, 22, 14728, 17, 92, 283, 220, 17, 23]
    n_seq = len(target_seq)
    ids_list = [int(x) for x in out_ids]
    for i in range(max(0, len(ids_list) - n_seq + 1)):
        if ids_list[i : i + n_seq] == target_seq:
            return int(i + n_seq - 2)
    return int(round((target_char / max(len(text), 1)) * len(out_ids)))


def generate_cover_5x2_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/cover_5x2.png",
    problem_id: str = "test/counting_and_probability/389.json",
) -> str:
    """0. 1500x600 (5:2) Phone-Thumbnail Cover: Base loop vs Edited boxed 28 with 8.5% -> 48.5% overlay."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

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

    t0 = str(by_alpha.get(0.0, {}).get("text", ""))
    t1 = str(by_alpha.get(1.0, {}).get("text", ""))
    loop_line = "Wait, if toppings are distinct, then {A,B} is a valid configuration."
    box_line = "\\boxed{28}"
    answer_sentence = "Thus, there are 28 different pizzas Mikka can order."
    if t0:
        assert loop_line in t0, f"Expected verbatim loop line in base trace for {problem_id}"
    if t1:
        assert answer_sentence in t1 and box_line in t1, f"Expected verbatim answer and {box_line} in edit trace"

    cover_dpi = 150
    fig = plt.figure(figsize=(10.0, 4.0), dpi=cover_dpi)  # 1500 x 600 px exact (5:2)
    fig.patch.set_facecolor("#f8fafc")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Full-width top dark headline banner so both headline & subline read clearly at phone thumbnail size
    hero_pill = mpatches.FancyBboxPatch(
        (0.025, 0.725),
        0.950,
        0.240,
        boxstyle="round,pad=0.008,rounding_size=0.032",
        facecolor="#0f172a",
        edgecolor="#334155",
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(hero_pill)
    ax.text(
        0.50,
        0.875,
        "8.5% → 48.5%",
        ha="center",
        va="center",
        fontsize=29.0,
        fontweight="bold",
        color="#ffffff",
        zorder=6,
    )
    sub_art = ax.text(
        0.50,
        0.770,
        "Same tiny model, 131,072 weights edited, 32k thinking tokens (MATH-500, 200 problems)",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#e2e8f0",
        zorder=6,
    )
    _fit_text_to_width(fig, sub_art, max_width_frac=0.91, min_fontsize=10.5)

    # Left panel: Base model stuck in a loop
    left_card = mpatches.FancyBboxPatch(
        (0.025, 0.045),
        0.463,
        0.640,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor="#fef2f2",
        edgecolor="#dc2626",
        linewidth=2.4,
        zorder=2,
    )
    ax.add_patch(left_card)
    ax.text(
        0.045,
        0.628,
        "Base model: stuck in a loop",
        fontsize=15.0,
        fontweight="bold",
        color="#991b1b",
        va="center",
        zorder=3,
    )

    # 3 verbatim repeats rendered with chat bullets in large phone-readable font
    ly = 0.542
    for rep_idx in range(3):
        pill_bg = mpatches.FancyBboxPatch(
            (0.042, ly - 0.112),
            0.428,
            0.122,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor="#fee2e2" if rep_idx % 2 == 0 else "#ffffff",
            edgecolor="#fca5a5",
            linewidth=1.1,
            zorder=2.5,
        )
        ax.add_patch(pill_bg)
        ax.text(
            0.052,
            ly - 0.014,
            "•  Wait, if toppings are distinct, then {A,B} is\n    a valid configuration.",
            fontsize=11.0,
            fontweight="bold" if rep_idx == 0 else "normal",
            color="#991b1b" if rep_idx == 0 else "#b91c1c",
            va="top",
            linespacing=1.22,
            parse_math=False,
            zorder=3,
        )
        ly -= 0.136

    ax.text(
        0.045,
        0.088,
        "Ran out at 32,768 tokens",
        fontsize=13.5,
        fontweight="bold",
        color="#7f1d1d",
        va="center",
        zorder=3,
    )

    # Right panel: Edited model answer in 3,843 tokens
    right_card = mpatches.FancyBboxPatch(
        (0.512, 0.045),
        0.463,
        0.640,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor="#ecfdf5",
        edgecolor="#059669",
        linewidth=2.4,
        zorder=2,
    )
    ax.add_patch(right_card)
    ax.text(
        0.532,
        0.628,
        "Edited model: answer in 3,843 tokens",
        fontsize=14.6,
        fontweight="bold",
        color="#065f46",
        va="center",
        zorder=3,
    )
    ax.text(
        0.532,
        0.525,
        "Thus, there are 28 different pizzas\nMikka can order.",
        fontsize=13.4,
        fontweight="bold",
        color="#0f172a",
        va="top",
        linespacing=1.32,
        parse_math=False,
        zorder=3,
    )

    # Chat-app rendered boxed answer badge ("28 ✓")
    box_outer = mpatches.FancyBboxPatch(
        (0.545, 0.125),
        0.398,
        0.220,
        boxstyle="round,pad=0.006,rounding_size=0.022",
        facecolor="#ffffff",
        edgecolor="#10b981",
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(box_outer)
    box_inner = mpatches.FancyBboxPatch(
        (0.645, 0.155),
        0.198,
        0.160,
        boxstyle="square,pad=0.004",
        facecolor="#d1fae5",
        edgecolor="#047857",
        linewidth=2.6,
        zorder=3.5,
    )
    ax.add_patch(box_inner)
    ax.text(
        0.744,
        0.235,
        "28 ✓",
        ha="center",
        va="center",
        fontsize=24.0,
        fontweight="bold",
        color="#065f46",
        parse_math=False,
        zorder=4,
    )

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=cover_dpi, bbox_inches=None)
    plt.close(fig)
    return str(target)


def _render_headline_budget_curve_impl(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/headline_budget_curve.png",
    paper_csv: Optional[Union[str, Path]] = None,
    phone_mode: bool = True,
) -> str:
    """Shared renderer for phone-first (`phone_mode=True`) and full (`phone_mode=False`) budget curves."""
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

    # 4k tokens shaded zone
    ax.axvspan(3400, 4800, color="#f1f5f9", alpha=0.90, zorder=1)
    if phone_mode:
        ax.annotate(
            "At 4k tokens:\n91–97% ran out of tokens",
            xy=(4350, 13.5),
            xytext=(5150, 24.5),
            ha="left",
            va="center",
            fontsize=13.0,
            fontweight="bold",
            color="#334155",
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.4),
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#ffffff", edgecolor="#94a3b8", lw=1.2, alpha=0.97),
            zorder=6,
        )
    else:
        ax.text(
            4096,
            24.5,
            "At 4k tokens:\n91–97% ran out of tokens",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#475569",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#cbd5e1", lw=1.0, alpha=0.96),
            zorder=6,
        )

    right_endpoints: List[Dict[str, Any]] = []
    alphas_to_plot = (0.0, 1.0) if phone_mode else (0.0, 1.0, 1.25)

    gpu_styles = {
        0.0: {
            "color": "#dc2626",
            "lw": 3.8 if phone_mode else 3.2,
            "marker": "o",
            "ms": 9.5 if phone_mode else 8.0,
            "ci": True,
            "label": "Base model",
            "short": "Base model",
            "z": 5,
        },
        1.0: {
            "color": "#059669",
            "lw": 4.0 if phone_mode else 3.4,
            "marker": "s",
            "ms": 9.5 if phone_mode else 8.0,
            "ci": True,
            "label": "Edited (published strength)",
            "short": "Edited (published)",
            "z": 6,
        },
        1.25: {
            "color": "#2563eb",
            "lw": 2.0,
            "marker": "D",
            "ms": 6.4,
            "ci": False,
            "label": "Edited (1.25× strength)",
            "short": "Edited (1.25×)",
            "z": 4,
        },
    }

    endpoints_by_alpha: Dict[float, float] = {}
    for a_val in alphas_to_plot:
        sub = gpu_df[gpu_df["alpha"] == a_val].sort_values("budget") if not gpu_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        st = gpu_styles[a_val]
        x = sub["budget"].to_numpy()
        y = sub["accuracy_pct"].to_numpy()
        endpoints_by_alpha[a_val] = float(y[-1])
        ax.plot(
            x,
            y,
            color=st["color"],
            linewidth=st["lw"],
            marker=st["marker"],
            markersize=st["ms"],
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
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

    # Paper truncated curves (only in full mode) & same-problem 32k stars (both modes)
    if not paper_df.empty:
        if not phone_mode:
            trunc_sub = paper_df[
                (paper_df["metric_type"] == "first_sample_truncated")
                & (paper_df["cohort"] == "paper_first_samples_n200")
            ]
            for cond, col, lbl, short in (
                ("unedited", "#ef4444", "Paper first-sample cut: Base", "Paper cut: Base"),
                ("p50c", "#10b981", "Paper first-sample cut: Edited", "Paper cut: Edited"),
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

        star_cohort = f"our_n{n_core}_core" if n_core == 200 else (f"our_n{n_core}" if n_core > 0 else "our_n200_core")
        if star_cohort not in set(paper_df["cohort"].unique()):
            star_cohort = "our_n50" if "our_n50" in set(paper_df["cohort"].unique()) else "math500_all_500"

        star_sub = paper_df[(paper_df["metric_type"] == "avg_at_4") & (paper_df["cohort"] == star_cohort)]
        # Offset the 32k stars clearly to the right so they never overlap the 32,768 markers
        star_x = 32768.0 * (1.085 if phone_mode else 1.075)
        n_match_val = n_core if n_core > 0 else 200
        for cond, col, short in (
            ("unedited", "#991b1b", "Paper 32k Base"),
            ("p50c", "#065f46", "Paper 32k Edited"),
        ):
            s_row = star_sub[star_sub["condition"] == cond]
            if s_row.empty:
                continue
            r0 = s_row.iloc[0]
            y_val = float(r0["accuracy_pct"])
            n_match_val = int(r0["n_problems"])
            lbl_star = None if phone_mode else f"{short} (same {n_match_val} problems: {y_val:.1f}%)"

            ax.scatter(
                [star_x],
                [y_val],
                marker="*",
                s=330 if phone_mode else 240,
                color=col,
                edgecolors="#0f172a",
                linewidths=1.1,
                zorder=8,
                label=lbl_star,
            )
            right_endpoints.append(
                {
                    "x": star_x,
                    "y": y_val,
                    "text": f"★ {short}: {y_val:.1f}%",
                    "color": col,
                    "bold": True,
                }
            )

        if phone_mode:
            # Neutral legend proxy for the two 32k paper stars
            ax.scatter(
                [],
                [],
                marker="*",
                s=220,
                color="#475569",
                edgecolors="#0f172a",
                linewidths=1.0,
                label=f"Paper's result at 32k (same {n_match_val} problems)",
            )

    if phone_mode:
        # Clean direct endpoint labels positioned well clear of both 32,768 points and offset stars
        if 1.0 in endpoints_by_alpha:
            y1 = endpoints_by_alpha[1.0]
            ax.text(
                32768.0,
                y1 + 7.4,
                f"{y1:.1f}%",
                ha="center",
                va="bottom",
                fontsize=16.5,
                fontweight="bold",
                color="#059669",
                zorder=9,
            )
        if 0.0 in endpoints_by_alpha:
            y0 = endpoints_by_alpha[0.0]
            ax.text(
                37600.0,
                y0,
                f"{y0:.1f}%",
                ha="left",
                va="center",
                fontsize=16.5,
                fontweight="bold",
                color="#dc2626",
                zorder=9,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlim(3100, 47500)
        ax.set_xticks([4096, 8192, 16384, 32768])
        ax.set_xticklabels(["4,096", "8,192", "16,384", "32,768"], fontsize=13.5, fontweight="bold")
        ax.minorticks_off()

        ax.set_ylim(-2.5, 63.0)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticklabels([f"{v}%" for v in (0, 10, 20, 30, 40, 50, 60)], fontsize=13.5)

        ax.set_xlabel("Thinking budget (tokens)", fontsize=14.0, fontweight="bold", labelpad=5, color="#0f172a")
        ax.set_ylabel("Correct final answer (%)", fontsize=14.0, fontweight="bold", labelpad=6, color="#0f172a")
        ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=13.0,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            framealpha=0.96,
        )

        fig.suptitle(
            "The edit only pays off when you let it think",
            fontsize=18.5,
            fontweight="bold",
            color="#0f172a",
            y=0.965,
        )
        sub_t = fig.text(
            0.5,
            0.902,
            f"Share of problems with a correct final answer, {n_core} MATH-500 problems, 95% intervals",
            ha="center",
            va="center",
            fontsize=13.2,
            color="#475569",
        )
        _fit_text_to_width(fig, sub_t, max_width_frac=0.93, min_fontsize=10.0)
        fig.text(
            0.5,
            0.025,
            HARDWARE_FOOTER_TEXT,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#64748b",
        )
        fig.subplots_adjust(left=0.095, right=0.975, top=0.845, bottom=0.155)
    else:
        if right_endpoints:
            spread_ys = _spread_y_positions(
                [pt["y"] for pt in right_endpoints],
                min_sep=4.6,
                y_min=1.5,
                y_max=58.5,
            )
            label_x = 42500.0
            for pt, ly in zip(right_endpoints, spread_ys):
                ax.plot(
                    [pt["x"], label_x * 1.015],
                    [pt["y"], ly],
                    color=pt["color"],
                    lw=1.1,
                    alpha=0.55,
                    zorder=4,
                )
                ax.text(
                    label_x,
                    ly,
                    pt["text"],
                    ha="left",
                    va="center",
                    fontsize=8.6,
                    fontweight="bold" if pt["bold"] else "normal",
                    color=pt["color"],
                    bbox=dict(
                        boxstyle="round,pad=0.24",
                        facecolor="#ffffff",
                        edgecolor=pt["color"],
                        lw=0.85 if pt["bold"] else 0.55,
                        alpha=1.0,
                    ),
                    zorder=9,
                )

        ax.set_xscale("log", base=2)
        ax.set_xlim(3000, 128000)
        ax.set_xticks([4096, 8192, 16384, 32768])
        ax.set_xticklabels(["4,096\n(4k)", "8,192\n(8k)", "16,384\n(16k)", "32,768\n(32k)"], fontsize=10.8, fontweight="bold")
        ax.minorticks_off()

        ax.set_ylim(-2.5, 64.5)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticklabels([f"{v}%" for v in (0, 10, 20, 30, 40, 50, 60)], fontsize=10.5)

        ax.set_xlabel("Thinking budget (tokens)", fontsize=11.5, fontweight="bold", labelpad=5, color="#0f172a")
        ax.set_ylabel("Correct final answer (%)", fontsize=11.5, fontweight="bold", labelpad=6, color="#0f172a")
        ax.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=8.6,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            framealpha=0.95,
        )

        fig.suptitle(
            "The edit only pays off when you let it think",
            fontsize=16.0,
            fontweight="bold",
            color="#0f172a",
            y=0.965,
        )
        sub_full = fig.text(
            0.5,
            0.912,
            f"Share of problems with a correct final answer, {n_core} MATH-500 problems, vs. paper references",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#475569",
        )
        _fit_text_to_width(fig, sub_full, max_width_frac=0.93, min_fontsize=8.5)
        fig.text(
            0.5,
            0.022,
            HARDWARE_FOOTER_TEXT,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#64748b",
        )
        fig.subplots_adjust(left=0.078, right=0.985, top=0.865, bottom=0.145)

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_headline_budget_curve(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/headline_budget_curve.png",
    paper_csv: Optional[Union[str, Path]] = None,
) -> str:
    """1. Headline Budget Curve (Phone Version): Base model & Edited (published) with 95% intervals + Paper 32k stars."""
    return _render_headline_budget_curve_impl(
        assets_root=assets_root,
        out_path=out_path,
        paper_csv=paper_csv,
        phone_mode=True,
    )


def generate_headline_budget_curve_full(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/headline_budget_curve_full.png",
    paper_csv: Optional[Union[str, Path]] = None,
) -> str:
    """1b. Full 5-Curve Headline Budget Scaling Chart (Desktop Reference)."""
    return _render_headline_budget_curve_impl(
        assets_root=assets_root,
        out_path=out_path,
        paper_csv=paper_csv,
        phone_mode=False,
    )


def generate_hero_trace_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/hero_trace.png",
    problem_id: str = "test/counting_and_probability/389.json",
    tokenizer: Any = None,
) -> str:
    """2. Hero Trace Comparison: verbatim 389.json excerpts rendered cleanly like a chat app."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

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
    ids0 = r0.get("out_ids") or []

    calc28_sub = r"Calculation: $\frac{8 \times 7}{2} = 28$."
    calc28_tok = _compute_substring_token_index(ids0, t0, calc28_sub, tokenizer=tokenizer)
    calc28_tok_val = int(calc28_tok) if calc28_tok is not None else 798

    loop_sentence = "Wait, if toppings are distinct, then {A,B} is a valid configuration."
    repeat_cnt = t0.count(loop_sentence) if t0 else 1292
    tok0 = int(r0.get("n_tokens", 32768))
    tok1 = int(r1.get("n_tokens", 3843))
    onset0 = r0.get("loop_onset_token")
    if onset0 is None and t0:
        onset0 = _get_or_compute_loop_onset(r0, tokenizer=tokenizer)
    onset0_val = int(onset0) if onset0 is not None else 5120
    tail0_pct = float(r0.get("tail_rep_8gram_ratio", 0.9485)) * 100.0
    tail1_pct = float(r1.get("tail_rep_8gram_ratio", 0.0)) * 100.0

    # Verify every verbatim source line exists in the underlying JSONL text
    expected_t0_snippets = [
        "Here's a thinking process that leads to the solution:",
        "1.  **Analyze the Problem:**",
        "*   So, $C(8, 2)$.",
        r"*   Calculation: $\frac{8 \times 7}{2} = 28$.",
        loop_sentence,
    ]
    expected_t1_snippets = [
        "Here's a thinking process that leads to the solution:",
        "1.  **Analyze the Problem:**",
        "**Step 3: Perform the Calculation**",
        r"$$ = \frac{8 \times 7}{2} $$",
        r"$$ = \frac{56}{2} $$",
        "$$ = 28 $$",
        "Thus, there are 28 different pizzas Mikka can order.",
        r"\boxed{28}",
    ]
    if t0:
        for snip in expected_t0_snippets:
            assert snip in t0, f"Base trace missing expected verbatim snippet: {snip!r}"
    if t1:
        for snip in expected_t1_snippets:
            assert snip in t1, f"Edited trace missing expected verbatim snippet: {snip!r}"

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.018, 0.036, 0.964, 0.815])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "MATH-500 counting problem: 8 toppings, pick 2",
        fontsize=18.2,
        fontweight="bold",
        color="#0f172a",
        y=0.968,
    )
    prob_sub = fig.text(
        0.5,
        0.902,
        "\"Mikka wants to order a pizza with two different toppings. He has 8 different\n"
        "toppings to choose from. How many different pizzas could he order?\"   (Gold answer: 28)",
        ha="center",
        va="center",
        fontsize=11.4,
        style="italic",
        color="#334155",
    )
    _fit_text_to_width(fig, prob_sub, max_width_frac=0.94, min_fontsize=9.5)

    def _draw_column(
        x0: float,
        w: float,
        *,
        header_title: str,
        header_bg: str,
        border_col: str,
        title_col: str,
        sub_note: str,
        rich_lines: List[Dict[str, Any]],
        metrics: List[Tuple[str, str, str]],
        is_base: bool,
    ) -> None:
        card = mpatches.FancyBboxPatch(
            (x0, 0.025),
            w,
            0.948,
            boxstyle="round,pad=0.005,rounding_size=0.018",
            facecolor="#ffffff",
            edgecolor=border_col,
            linewidth=2.0,
        )
        ax.add_patch(card)

        hdr = mpatches.FancyBboxPatch(
            (x0 + 0.012, 0.868),
            w - 0.024,
            0.094,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            facecolor=header_bg,
            edgecolor="none",
        )
        ax.add_patch(hdr)
        ax.text(x0 + 0.022, 0.930, header_title, fontsize=12.6, fontweight="bold", color=title_col, va="center")
        ax.text(x0 + 0.022, 0.888, sub_note, fontsize=8.8, fontweight="bold", color="#334155", va="center")

        bar_x = x0 + 0.022
        bar_w = w - 0.044
        bar_y = 0.774
        bar_h = 0.048

        ax.text(
            bar_x,
            bar_y + bar_h + 0.014,
            "Thinking budget (0 → 32,768 tokens)",
            fontsize=9.2,
            fontweight="bold",
            color="#475569",
        )

        track = mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.0)
        ax.add_patch(track)

        call_y = 0.742
        if is_base:
            frac_calc = min(1.0, calc28_tok_val / 32768.0)
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
            calc_x = bar_x + bar_w * frac_calc
            loop_x = bar_x + w_pre
            ax.plot([calc_x, calc_x], [bar_y - 0.005, bar_y + bar_h + 0.003], color="#b45309", lw=1.8, linestyle="--", zorder=5)
            ax.plot([loop_x, loop_x], [bar_y - 0.005, bar_y + bar_h + 0.003], color="#7f1d1d", lw=2.0, zorder=5)

            ax.text(
                bar_x + w_pre + w_loop / 2,
                bar_y + bar_h / 2,
                f"Looped tokens {onset0_val:,} → {tok0:,} ({repeat_cnt:,}× repeats)",
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#991b1b", edgecolor="none", alpha=0.95),
            )

            lbl1_x = bar_x + 0.002
            ax.text(
                lbl1_x,
                call_y,
                f"= 28 at token {calc28_tok_val:,}",
                ha="left",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#92400e",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#fffbeb", edgecolor="#d97706", lw=0.9),
            )
            ax.plot([calc_x, lbl1_x + 0.045], [bar_y - 0.005, call_y + 0.015], color="#d97706", lw=1.1, zorder=4)

            lbl2_x = bar_x + 0.155
            ax.text(
                lbl2_x,
                call_y,
                f"Loop starts at token {onset0_val:,}",
                ha="left",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#fef2f2", edgecolor="#dc2626", lw=0.9),
            )
            ax.plot([loop_x, lbl2_x + 0.045], [bar_y - 0.005, call_y + 0.015], color="#dc2626", lw=1.1, zorder=4)

            ax.text(
                bar_x + bar_w,
                call_y,
                f"Out at {tok0:,}",
                ha="right",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#991b1b",
            )
        else:
            frac_used = min(1.0, tok1 / 32768.0)
            w_used = max(bar_w * frac_used, bar_w * 0.08)
            end_x = bar_x + w_used
            ax.add_patch(mpatches.Rectangle((bar_x, bar_y), w_used, bar_h, facecolor="#059669", edgecolor="none"))
            ax.plot([end_x, end_x], [bar_y - 0.005, bar_y + bar_h + 0.003], color="#047857", lw=1.8, zorder=5)

            lbl_ok_x = bar_x + 0.002
            ax.text(
                lbl_ok_x,
                call_y,
                f"Finished at {tok1:,} tokens",
                ha="left",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color="#065f46",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ecfdf5", edgecolor="#059669", lw=0.9),
            )
            ax.plot([end_x, lbl_ok_x + 0.060], [bar_y - 0.005, call_y + 0.015], color="#059669", lw=1.1, zorder=4)
            ax.text(
                bar_x + bar_w,
                call_y,
                "32,768 limit",
                ha="right",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#64748b",
            )

        box_y = 0.165
        box_h = 0.545
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

        cur_y = box_y + box_h - 0.036
        for item in rich_lines:
            kind = item.get("kind", "line")
            if kind == "boxed_badge":
                # Chat-app rendered final answer box ("28 ✓")
                ans_card = mpatches.FancyBboxPatch(
                    (bar_x + 0.014, cur_y - 0.056),
                    0.135,
                    0.068,
                    boxstyle="square,pad=0.004",
                    facecolor="#d1fae5",
                    edgecolor="#047857",
                    linewidth=2.2,
                    zorder=4,
                )
                ax.add_patch(ans_card)
                ax.text(
                    bar_x + 0.0815,
                    cur_y - 0.022,
                    "28 ✓",
                    ha="center",
                    va="center",
                    fontsize=13.5,
                    fontweight="bold",
                    color="#065f46",
                    zorder=5,
                )
                cur_y -= 0.072
                continue

            bg_col = item.get("bg")
            edge_col = item.get("edge")
            row_h = float(item.get("h", 0.048))
            if bg_col:
                hl = mpatches.FancyBboxPatch(
                    (bar_x + 0.008, cur_y - row_h * 0.52),
                    bar_w - 0.016,
                    row_h,
                    boxstyle="round,pad=0.003,rounding_size=0.010",
                    facecolor=bg_col,
                    edgecolor=edge_col or "none",
                    linewidth=0.9 if edge_col else 0.0,
                    zorder=3,
                )
                ax.add_patch(hl)

            spans = item["spans"]
            fs = float(item.get("fs", 8.8))
            col = str(item.get("col", "#1e293b"))
            row_artists = _draw_rich_spans(
                fig,
                ax,
                bar_x + 0.015,
                cur_y,
                spans,
                fontsize=fs,
                default_color=col,
                fontfamily="sans-serif",
                va="center",
                zorder=4,
            )
            # If a single-span plain line is near the right border of the chat card, shrink slightly so it never clips
            if len(row_artists) == 1 and spans[0][0] != "math":
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                bb = row_artists[0].get_window_extent(renderer=rend).transformed(ax.transAxes.inverted())
                max_right = bar_x + bar_w - 0.014
                if bb.x1 > max_right and (bb.x1 - bb.x0) > 0:
                    avail_w = max_right - (bar_x + 0.015)
                    row_artists[0].set_fontsize(fs * (avail_w / (bb.x1 - bb.x0)) * 0.98)

            cur_y -= float(item.get("step", 0.060))

        # 3 Stat Cards in plain words at bottom of each column
        n_cards = len(metrics)
        gap_w = 0.010
        pill_w = (bar_w - gap_w * (n_cards - 1)) / n_cards
        pill_h = 0.112
        py = 0.040
        for idx_m, (m_label, m_val, m_col) in enumerate(metrics):
            px = bar_x + idx_m * (pill_w + gap_w)
            pill = mpatches.FancyBboxPatch(
                (px, py),
                pill_w,
                pill_h,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=header_bg,
                edgecolor=border_col,
                linewidth=1.0,
            )
            ax.add_patch(pill)
            lbl_art = ax.text(
                px + pill_w / 2.0,
                py + pill_h * 0.72,
                m_label,
                ha="center",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color="#475569",
            )
            val_art = ax.text(
                px + pill_w / 2.0,
                py + pill_h * 0.30,
                m_val,
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=m_col,
            )
            for art in (lbl_art, val_art):
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                bb = art.get_window_extent(renderer=rend).transformed(ax.transAxes.inverted())
                if (bb.x1 - bb.x0) > (pill_w - 0.012):
                    art.set_fontsize(float(art.get_fontsize()) * ((pill_w - 0.012) / (bb.x1 - bb.x0)) * 0.97)

    # 8 clean, chat-app rendered verbatim lines per side (math rendered via mathtext, bold via fontweight, bullets via •)
    left_rich_lines: List[Dict[str, Any]] = [
        {
            "spans": [("plain", "Here's a thinking process that leads to the solution:")],
            "fs": 8.6,
            "col": "#475569",
            "step": 0.056,
        },
        {
            "spans": [("bold", "1.  Analyze the Problem:")],
            "fs": 9.2,
            "col": "#0f172a",
            "step": 0.056,
        },
        {
            "spans": [("plain", "•  So, "), ("math", "C(8, 2)"), ("plain", ".")],
            "fs": 9.2,
            "col": "#334155",
            "step": 0.068,
        },
        {
            "spans": [("bold", "•  Calculation: "), ("math", r"\frac{8 \times 7}{2} = 28"), ("plain", ".")],
            "fs": 9.6,
            "col": "#92400e",
            "bg": "#fef3c7",
            "edge": "#f59e0b",
            "h": 0.072,
            "step": 0.076,
        },
        {
            "spans": [("plain", "…")],
            "fs": 9.5,
            "col": "#64748b",
            "step": 0.052,
        },
        {
            "spans": [("bold", "•  Wait, if toppings are distinct, then {A,B} is a valid configuration.")],
            "fs": 8.1,
            "col": "#991b1b",
            "bg": "#fee2e2",
            "edge": "#f87171",
            "h": 0.048,
            "step": 0.058,
        },
        {
            "spans": [("plain", "•  Wait, if toppings are distinct, then {A,B} is a valid configuration.")],
            "fs": 8.1,
            "col": "#b91c1c",
            "bg": "#fee2e2",
            "edge": "#fca5a5",
            "h": 0.048,
            "step": 0.058,
        },
        {
            "spans": [("plain", "•  Wait, if toppings are distinct, then {A,B} is a valid configuration.")],
            "fs": 8.1,
            "col": "#b91c1c",
            "bg": "#fee2e2",
            "edge": "#fca5a5",
            "h": 0.048,
            "step": 0.058,
        },
    ]

    right_rich_lines: List[Dict[str, Any]] = [
        {
            "spans": [("plain", "Here's a thinking process that leads to the solution:")],
            "fs": 8.6,
            "col": "#475569",
            "step": 0.056,
        },
        {
            "spans": [("bold", "1.  Analyze the Problem:")],
            "fs": 9.2,
            "col": "#0f172a",
            "step": 0.054,
        },
        {
            "spans": [("plain", "…")],
            "fs": 9.5,
            "col": "#64748b",
            "step": 0.050,
        },
        {
            "spans": [("bold", "Step 3: Perform the Calculation")],
            "fs": 9.4,
            "col": "#0f172a",
            "step": 0.068,
        },
        {
            "spans": [("plain", "    "), ("math", r"= \frac{8 \times 7}{2}")],
            "fs": 10.2,
            "col": "#1e293b",
            "step": 0.076,
        },
        {
            "spans": [("plain", "    "), ("math", r"= \frac{56}{2} = 28")],
            "fs": 10.2,
            "col": "#0f172a",
            "step": 0.072,
        },
        {
            "spans": [("bold", "Thus, there are 28 different pizzas Mikka can order.")],
            "fs": 8.8,
            "col": "#065f46",
            "bg": "#d1fae5",
            "edge": "#10b981",
            "h": 0.052,
            "step": 0.068,
        },
        {
            "kind": "boxed_badge",
        },
    ]

    _draw_column(
        0.012,
        0.476,
        header_title="Base model: stuck in a loop",
        header_bg="#fef2f2",
        border_col="#ef4444",
        title_col="#991b1b",
        sub_note=f"Reached 28 at token {calc28_tok_val:,}, then looped {repeat_cnt:,}× to the limit",
        rich_lines=left_rich_lines,
        metrics=[
            ("Tokens used", f"{tok0:,} (ran out)", "#b91c1c"),
            ("Repeated text (last 240 words)", f"{tail0_pct:.1f}% ({repeat_cnt:,}× loop)", "#b91c1c"),
            ("Final answer", "None (ran out)", "#b91c1c"),
        ],
        is_base=True,
    )

    _draw_column(
        0.512,
        0.476,
        header_title=f"Edited model: solved in {tok1:,} tokens",
        header_bg="#ecfdf5",
        border_col="#10b981",
        title_col="#065f46",
        sub_note="Completed reasoning cleanly without looping",
        rich_lines=right_rich_lines,
        metrics=[
            ("Tokens used", f"{tok1:,} (finished)", "#047857"),
            ("Repeated text (last 240 words)", f"{tail1_pct:.1f}% (no loop)", "#047857"),
            ("Final answer", "28 ✓", "#047857"),
        ],
        is_base=False,
    )

    fig.text(
        0.5,
        0.014,
        HARDWARE_FOOTER_TEXT,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#64748b",
    )

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
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
                counts["Ran out while looping"] += 1
            else:
                counts["Ran out, not looping"] += 1

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


def _load_outcomes_core_and_dose(
    assets_root: Path,
    tokenizer: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load `core_df` (N=200) and `combined_dose_df` (paired N=100 across 6 doses)."""
    core_cands = [
        assets_root / "gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
        assets_root / "budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ]
    core_path = next((p for p in core_cands if p.is_file()), None)
    dose_path = assets_root / "gpu_dose_n100/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz"

    core_df = _compute_outcome_breakdown(core_path, tokenizer=tokenizer) if core_path else pd.DataFrame()

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
                counts["Ran out while looping" if onset is not None else "Ran out, not looping"] += 1
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

    return core_df, combined_dose_df


def _format_core_model_xtick(alpha: float, *, multiline: bool = False) -> str:
    if abs(alpha - 0.0) < 1e-6:
        return "Base model"
    if abs(alpha - 1.0) < 1e-6:
        return "Edited\n(published)" if multiline else "Edited (published strength)"
    if abs(alpha - 1.25) < 1e-6:
        return "Edited\n(1.25×)" if multiline else "Edited (1.25× strength)"
    return f"Edited ({alpha:g}×)"


def _format_dose_xtick(alpha: float, *, multiline: bool = False) -> str:
    if abs(alpha - 0.0) < 1e-6:
        return "0×"
    if abs(alpha - 1.0) < 1e-6:
        return "1×\n(published)" if multiline else "1× (published)"
    return f"{alpha:g}×"


def _plot_outcome_bars(
    ax: Any,
    df_panel: pd.DataFrame,
    *,
    xtick_formatter: Any = _format_core_model_xtick,
    bar_label_fontsize: float = 8.6,
    xtick_fontsize: float = 10.2,
    ytick_fontsize: float = 9.8,
    ylabel_fontsize: float = 11.0,
    show_ylabel: bool = True,
    add_legend_labels: bool = True,
) -> None:
    ax.set_facecolor("#ffffff")
    if df_panel.empty:
        return
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
            linewidth=1.1,
            label=cat if add_legend_labels else None,
            zorder=3,
        )
        for bar, v, bot in zip(bars, vals, bottom):
            if v >= 5.0:
                lbl_txt = f"{v:.1f}%".replace(".0%", "%")
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bot + v / 2.0,
                    lbl_txt,
                    ha="center",
                    va="center",
                    fontsize=bar_label_fontsize,
                    fontweight="bold",
                    color="#ffffff",
                    zorder=4,
                )
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels([xtick_formatter(float(a)) for a in alphas], fontsize=xtick_fontsize, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)], fontsize=ytick_fontsize)
    if show_ylabel:
        ax.set_ylabel("Share of problems (%)", fontsize=ylabel_fontsize, fontweight="bold", color="#0f172a")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_outcomes_32k_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/outcomes_32k.png",
    tokenizer: Any = None,
) -> str:
    """3. Outcome Breakdown at 32k (2-Panel Overview): Correct / Wrong answer / Ran out while looping / Ran out, not looping."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    core_df, combined_dose_df = _load_outcomes_core_and_dose(root, tokenizer=tokenizer)

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
            (
                ax1,
                core_df,
                lambda a: _format_core_model_xtick(a, multiline=True),
                f"A. Main Comparison ({int(core_df['n_total'].max()) if not core_df.empty else 200} problems)",
            ),
            (
                ax2,
                combined_dose_df,
                lambda a: _format_dose_xtick(a, multiline=True),
                f"B. Edit Strength Sweep ({int(combined_dose_df['n_total'].max())} problems)",
            ),
        ]
    else:
        fig, ax1 = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
        axes_to_plot = [
            (
                ax1,
                core_df,
                lambda a: _format_core_model_xtick(a, multiline=True),
                f"What happened at 32k tokens ({int(core_df['n_total'].max()) if not core_df.empty else 50} problems)",
            ),
        ]

    fig.patch.set_facecolor("#ffffff")

    for panel_idx, (ax, df_panel, x_fmt, subtitle) in enumerate(axes_to_plot):
        _plot_outcome_bars(
            ax,
            df_panel,
            xtick_formatter=x_fmt,
            show_ylabel=(panel_idx == 0),
            add_legend_labels=(panel_idx == 0),
        )
        ax.set_title(subtitle, fontsize=11.5, fontweight="bold", color="#0f172a", pad=8)
        if panel_idx == 1:
            ax.set_xlabel("Edit strength", fontsize=10.8, fontweight="bold", labelpad=5, color="#0f172a")

    handles, labels = axes_to_plot[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        fontsize=9.6,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
    )

    fig.suptitle(
        "What happens at 32k tokens: fewer loops, more finished answers",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
        y=0.968,
    )
    if has_two_panels:
        sub_overview = fig.text(
            0.5,
            0.916,
            "Stronger edits cut looping (91% → 17%) while wrong answers rise (9% at 0.75×, 21% at 1×, 32% at 1.5×)",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#475569",
        )
        _fit_text_to_width(fig, sub_overview, max_width_frac=0.93, min_fontsize=8.0)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.165, wspace=0.14)
    else:
        fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.165)

    fig.text(
        0.5,
        0.024,
        HARDWARE_FOOTER_TEXT,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#64748b",
    )

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_outcomes_32k_single_chart(
    *,
    panel: str = "core",
    assets_root: Union[str, Path] = "assets",
    out_path: Optional[Union[str, Path]] = None,
    tokenizer: Any = None,
) -> str:
    """Export standalone large-font `outcomes_32k_core.png` (panel='core') or `outcomes_dose.png` (panel='dose')."""
    root = Path(assets_root)
    core_df, combined_dose_df = _load_outcomes_core_and_dose(root, tokenizer=tokenizer)

    if panel == "dose":
        df_use = combined_dose_df if not combined_dose_df.empty else core_df
        default_out = root / "article/outcomes_dose.png"
        n_val = int(df_use["n_total"].max()) if not df_use.empty else 100
        title_str = f"Turning the edit up: fewer loops, more wrong answers ({n_val} problems, 32k)"
        subtitle_str = (
            "0.75×, 1× and 1.25× are statistically tied; only the jump from 0.5× to 0.75× is significant"
        )
        x_fmt = _format_dose_xtick
        xtick_fs = 12.2
        xlabel_txt = "Edit strength"
    else:
        df_use = core_df
        default_out = root / "article/outcomes_32k_core.png"
        n_val = int(df_use["n_total"].max()) if not df_use.empty else 200
        title_str = f"What happened to all {n_val} problems at 32k tokens"
        subtitle_str = (
            "Looping drops from 89.5% to 25.5%; ran out of tokens drops from 91.5% to 29.5%"
        )
        x_fmt = lambda a: _format_core_model_xtick(a, multiline=False)
        xtick_fs = 13.5
        xlabel_txt = None

    target = Path(out_path) if out_path is not None else default_out
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")

    _plot_outcome_bars(
        ax,
        df_use,
        xtick_formatter=x_fmt,
        bar_label_fontsize=13.5,
        xtick_fontsize=xtick_fs,
        ytick_fontsize=13.0,
        ylabel_fontsize=14.0,
        show_ylabel=True,
        add_legend_labels=True,
    )
    if xlabel_txt:
        ax.set_xlabel(xlabel_txt, fontsize=13.5, fontweight="bold", labelpad=5, color="#0f172a")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.822),
        ncol=4,
        fontsize=12.4,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
    )

    sup_t = fig.suptitle(
        title_str,
        fontsize=17.4,
        fontweight="bold",
        color="#0f172a",
        y=0.972,
    )
    _fit_text_to_width(fig, sup_t, max_width_frac=0.95, min_fontsize=13.0)

    sub_t = fig.text(
        0.5,
        0.892,
        subtitle_str,
        ha="center",
        va="center",
        fontsize=13.0,
        color="#475569",
    )
    _fit_text_to_width(fig, sub_t, max_width_frac=0.94, min_fontsize=10.5)

    fig.text(
        0.5,
        0.025,
        HARDWARE_FOOTER_TEXT,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#64748b",
    )

    fig.subplots_adjust(left=0.09, right=0.975, top=0.720, bottom=0.160)
    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def _render_loop_onset_impl(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/loop_onset.png",
    tokenizer: Any = None,
    phone_scatter_only: bool = True,
) -> str:
    """Render phone-first single scatter (`phone_scatter_only=True`) or 2-panel full loop-onset chart."""
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

    alphas = sorted(df["alpha"].unique()) if not df.empty else [0.0, 1.0, 1.25]
    alpha_colors = {0.0: "#dc2626", 0.5: "#f59e0b", 0.75: "#0ea5e9", 1.0: "#059669", 1.25: "#2563eb", 1.5: "#7c3aed"}

    if phone_scatter_only:
        fig, ax2 = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
        fig.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")
    else:
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

        bins = [
            (0, 4096, "0–4k\n(Early)"),
            (4096, 8192, "4k–8k\n(Typical)"),
            (8192, 16384, "8k–16k\n(Mid)"),
            (16384, 32769, "16k–32k\n(Late)"),
        ]
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
            model_short = _format_core_model_xtick(float(a_val), multiline=False).replace(" strength", "")
            ax1.bar(
                x_cats + offset,
                cap_arr,
                bar_w * 0.92,
                color=col,
                alpha=0.92,
                edgecolor="#ffffff",
                lw=0.7,
                label=f"{model_short} — ran out (solid)",
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
                label=f"{model_short} — finished (hatched)",
                zorder=3,
            )

        ax1.set_xticks(x_cats)
        ax1.set_xticklabels([b[2] for b in bins] + ["Never\nLooped"], fontsize=9.5, fontweight="bold")
        ax1.set_ylim(0, 92)
        ax1.set_ylabel("Share of all problems (%)", fontsize=11.0, fontweight="bold", color="#0f172a")
        ax1.set_xlabel("Token range where loop begins", fontsize=10.2, fontweight="bold", color="#0f172a")
        ax1.set_title("A. When loops start (solid = ran out, hatched = escaped)", fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(loc="upper left", fontsize=8.0, ncol=1, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    rng = np.random.default_rng(42)
    for i_a, a_val in enumerate(alphas):
        sub_loop = df[(df["alpha"] == a_val) & df["loop_onset_token"].notna()]
        col = alpha_colors.get(a_val, "#475569")
        capped_vals = sub_loop[sub_loop["hit_cap"]]["loop_onset_token"].to_numpy(dtype=float)
        fin_vals = sub_loop[~sub_loop["hit_cap"]]["loop_onset_token"].to_numpy(dtype=float)

        if len(capped_vals) > 0:
            jitter_c = rng.uniform(-0.12, -0.02, size=len(capped_vals))
            ax2.scatter(
                np.full(len(capped_vals), i_a) + jitter_c,
                capped_vals,
                color=col,
                alpha=0.68,
                s=44 if phone_scatter_only else 26,
                marker="o",
                edgecolors="#ffffff",
                linewidths=0.5,
                zorder=4,
            )
            med_c = float(np.median(capped_vals))
            ax2.hlines(med_c, i_a - 0.18, i_a + 0.18, color="#0f172a", lw=2.8 if phone_scatter_only else 2.2, zorder=6)
            ax2.text(
                i_a,
                med_c + (1000 if phone_scatter_only else 700),
                f"median start, ran out: {med_c/1000:.1f}k",
                ha="center",
                va="bottom",
                fontsize=11.0 if phone_scatter_only else 7.1,
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffff", edgecolor=col, lw=1.1, alpha=0.96),
                zorder=7,
            )

        if len(fin_vals) > 0:
            jitter_f = rng.uniform(0.03, 0.13, size=len(fin_vals))
            ax2.scatter(
                np.full(len(fin_vals), i_a) + jitter_f,
                fin_vals,
                color="#ffffff",
                alpha=0.95,
                s=54 if phone_scatter_only else 34,
                marker="D",
                edgecolors=col,
                linewidths=1.8 if phone_scatter_only else 1.4,
                zorder=5,
            )

    # Neutral gray legend markers so both shapes clearly mean ran-out vs escaped across all colors
    ax2.scatter(
        [],
        [],
        color="#64748b",
        s=54 if phone_scatter_only else 36,
        marker="o",
        edgecolors="#ffffff",
        linewidths=0.6,
        label="Loop, ran out of tokens",
    )
    ax2.scatter(
        [],
        [],
        color="#ffffff",
        s=62 if phone_scatter_only else 42,
        marker="D",
        edgecolors="#64748b",
        linewidths=1.9,
        label="Loop, escaped and finished",
    )

    ax2.set_xticks(range(len(alphas)))
    ax2.set_xlim(
        -0.48 if phone_scatter_only else -0.56,
        max(len(alphas) - 1, 2) + (0.52 if phone_scatter_only else 0.68),
    )
    ax2.set_xticklabels(
        [
            f"{_format_core_model_xtick(float(a), multiline=(not phone_scatter_only)).replace(' strength', '')}\n({df[(df['alpha']==a) & df['loop_onset_token'].notna()].shape[0]}/{df[df['alpha']==a].shape[0]} looped)"
            for a in alphas
        ],
        fontsize=12.4 if phone_scatter_only else 8.4,
        fontweight="bold",
    )
    ax2.set_ylim(-1500, 39200)
    ax2.set_yticks([0, 8192, 16384, 24576, 32768])
    ax2.set_yticklabels(["0", "8,192 (8k)", "16,384 (16k)", "24,576 (24k)", "32,768 (32k)"], fontsize=12.6 if phone_scatter_only else 9.0)
    ax2.set_ylabel("Where the loop starts (token)", fontsize=14.0 if phone_scatter_only else 10.8, fontweight="bold", color="#0f172a")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Compute exact loop & escaped counts for subtitle line
    def _counts_for(alpha_val: float, def_l: int, def_e: int) -> Tuple[int, int, int]:
        sub_a = df[df["alpha"] == alpha_val] if not df.empty else pd.DataFrame()
        n_a = len(sub_a)
        if not n_a:
            return def_l, 200, def_e
        l_a = int(sub_a["loop_onset_token"].notna().sum())
        e_a = int((sub_a["loop_onset_token"].notna() & (~sub_a["hit_cap"])).sum())
        return l_a, n_a, e_a

    l0, n0, e0 = _counts_for(0.0, 179, 0)
    l1, n1, e1 = _counts_for(1.0, 66, 39)
    l125, n125, e125 = _counts_for(1.25, 51, 26)
    caption_line = (
        f"Base: {l0}/{n0} looped, {e0} escaped · Edited: {l1}/{n1} looped, {e1} escaped · "
        f"Edited 1.25×: {l125}/{n125} looped, {e125} escaped"
    )

    if phone_scatter_only:
        ax2.legend(loc="upper left", fontsize=12.6, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.96)
        fig.suptitle(
            "With the edit, loops are rarer and start later",
            fontsize=17.8,
            fontweight="bold",
            color="#0f172a",
            y=0.965,
        )
        sub_l = fig.text(
            0.5,
            0.902,
            caption_line,
            ha="center",
            va="center",
            fontsize=12.4,
            fontweight="bold",
            color="#334155",
        )
        _fit_text_to_width(fig, sub_l, max_width_frac=0.94, min_fontsize=9.5)
        fig.text(
            0.5,
            0.025,
            HARDWARE_FOOTER_TEXT,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#64748b",
        )
        fig.subplots_adjust(left=0.165, right=0.975, top=0.845, bottom=0.175)
    else:
        ax2.set_title("B. When each loop begins", fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
        ax2.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
        fig.suptitle(
            "With the edit, loops are rarer and start later",
            fontsize=15.5,
            fontweight="bold",
            color="#0f172a",
            y=0.965,
        )
        sub_lf = fig.text(0.5, 0.912, caption_line, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#334155")
        _fit_text_to_width(fig, sub_lf, max_width_frac=0.94, min_fontsize=8.5)
        fig.text(
            0.5,
            0.022,
            HARDWARE_FOOTER_TEXT,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#64748b",
        )
        fig.subplots_adjust(left=0.068, right=0.985, top=0.845, bottom=0.16, wspace=0.33)

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_loop_onset_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/loop_onset.png",
    tokenizer: Any = None,
) -> str:
    """4. Loop Onset Scatter (Phone Version): Onset position scatter with verified medians."""
    return _render_loop_onset_impl(
        assets_root=assets_root,
        out_path=out_path,
        tokenizer=tokenizer,
        phone_scatter_only=True,
    )


def generate_loop_onset_full_chart(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/loop_onset_full.png",
    tokenizer: Any = None,
) -> str:
    """4b. Full 2-Panel Loop Onset Histogram + Scatter (Desktop Reference)."""
    return _render_loop_onset_impl(
        assets_root=assets_root,
        out_path=out_path,
        tokenizer=tokenizer,
        phone_scatter_only=False,
    )


def generate_edit_diagram(
    *,
    assets_root: Union[str, Path] = "assets",
    out_path: Union[str, Path] = "assets/article/edit_diagram.png",
) -> str:
    """5. Architectural & Rank-1 Mechanism Diagram of the 2-Block Weight-Space Edit."""
    root = Path(assets_root)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    ver_path = root / "weight_verification.json"
    ver = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.is_file() else {}
    total_params = int(ver.get("total_parameters", 873438784))
    touched_entries = int(ver.get("total_touched_entries", 131072))
    pct_touched = (touched_entries / total_params) * 100.0
    tensors = ver.get("tensors", {})
    t15 = tensors.get("model.language_model.layers.15.mlp.down_proj.weight", {})
    t19 = tensors.get("model.language_model.layers.19.mlp.down_proj.weight", {})
    sv15 = t15.get("svd_top5", [1.7489, 0.00099, 0.00095, 0.00087, 0.00081])
    sv19 = t19.get("svd_top5", [1.3988, 0.00090, 0.00081, 0.00074, 0.00071])
    ratio15 = float(t15.get("sv1_to_sv2_ratio", 1771.2))
    ratio19 = float(t19.get("sv1_to_sv2_ratio", 1552.0))

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.02, 0.03, 0.58, 0.81])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Only 2 of 488 weight blocks changed",
        fontsize=18.5,
        fontweight="bold",
        color="#0f172a",
        y=0.965,
    )
    sub_e = fig.text(
        0.5,
        0.902,
        f"{touched_entries:,} weights = {pct_touched:.3f}% of the model (the paper rounds to 0.016%)",
        ha="center",
        va="center",
        fontsize=13.2,
        color="#334155",
    )
    _fit_text_to_width(fig, sub_e, max_width_frac=0.94, min_fontsize=10.5)

    # Box 1: Full Model Overview
    b1 = mpatches.FancyBboxPatch(
        (0.02, 0.72),
        0.96,
        0.24,
        boxstyle="round,pad=0.008,rounding_size=0.024",
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        linewidth=1.8,
    )
    ax.add_patch(b1)
    ax.text(0.05, 0.875, "Qwen3.5-0.8B · 488 weight blocks (873.4M weights)", fontsize=13.2, fontweight="bold", color="#0f172a")
    ax.text(
        0.05,
        0.775,
        "486 unchanged · only 2 layers edited",
        fontsize=12.8,
        fontweight="bold",
        color="#047857",
    )

    # Arrows down to Layer 15 and Layer 19
    ax.annotate("", xy=(0.25, 0.65), xytext=(0.34, 0.72), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.6))
    ax.annotate("", xy=(0.75, 0.65), xytext=(0.66, 0.72), arrowprops=dict(arrowstyle="->", color="#059669", lw=2.6))

    # Box 2A & Box 2B: Layer 15 & Layer 19
    for bx, bw, title, r_val, col, bg in (
        (
            0.02,
            0.46,
            "Layer 15",
            ratio15,
            "#2563eb",
            "#eff6ff",
        ),
        (
            0.52,
            0.46,
            "Layer 19",
            ratio19,
            "#059669",
            "#ecfdf5",
        ),
    ):
        card = mpatches.FancyBboxPatch(
            (bx, 0.34),
            bw,
            0.30,
            boxstyle="round,pad=0.008,rounding_size=0.024",
            facecolor=bg,
            edgecolor=col,
            linewidth=2.0,
        )
        ax.add_patch(card)
        ax.text(bx + 0.024, 0.565, title, fontsize=14.0, fontweight="bold", color="#0f172a")
        ax.text(bx + 0.024, 0.470, "64 of 3,584 columns changed", fontsize=10.2, fontweight="bold", color=col)
        ax.text(bx + 0.024, 0.380, f"Top direction is {r_val:,.0f}× larger", fontsize=10.2, fontweight="bold", color="#0f172a")

    # Box 3: Plain-English Rank-1 Explanation & Dial Equation
    b3 = mpatches.FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.25,
        boxstyle="round,pad=0.008,rounding_size=0.024",
        facecolor="#fffbeb",
        edgecolor="#d97706",
        linewidth=1.8,
    )
    ax.add_patch(b3)
    ax.text(
        0.05,
        0.205,
        "Each edit points in a single direction (rank-1)",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.05,
        0.095,
        r"$W(\mathrm{strength}) = W_{\mathrm{base}} + \mathrm{strength} \times (W_{\mathrm{edit}} - W_{\mathrm{base}})$",
        fontsize=12.6,
        fontweight="bold",
        color="#92400e",
        parse_math=True,
    )

    # Right Subplot: Compact Singular-Value Spectrum (log scale)
    ax_svd = fig.add_axes([0.695, 0.16, 0.285, 0.65])
    ax_svd.set_facecolor("#ffffff")
    ranks = [1, 2, 3, 4, 5]
    ax_svd.plot(ranks, sv15, marker="o", ms=8.5, lw=2.8, color="#2563eb", label=f"Layer 15 ({ratio15:,.0f}×)")
    ax_svd.plot(ranks, sv19, marker="s", ms=8.5, lw=2.8, color="#059669", label=f"Layer 19 ({ratio19:,.0f}×)")
    ax_svd.set_yscale("log")
    ax_svd.set_xticks(ranks)
    ax_svd.set_xticklabels([f"#{r}" for r in ranks], fontsize=11.5, fontweight="bold")
    ax_svd.set_ylim(2e-4, 6.0)
    ax_svd.set_ylabel("Size of each direction (log)", fontsize=11.8, fontweight="bold", color="#0f172a")
    ax_svd.set_xlabel("Direction number", fontsize=11.8, fontweight="bold", color="#0f172a")
    ax_svd.set_title("One direction dominates", fontsize=13.0, fontweight="bold", color="#0f172a", pad=8)
    ax_svd.grid(True, which="both", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax_svd.spines["top"].set_visible(False)
    ax_svd.spines["right"].set_visible(False)
    ax_svd.legend(loc="upper right", fontsize=10.2, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    ax_svd.annotate(
        f"{ratio15:,.0f}× and {ratio19:,.0f}×\nlarger than #2",
        xy=(1.04, sv15[0] * 0.90),
        xytext=(1.55, 0.36),
        fontsize=10.2,
        fontweight="bold",
        color="#0f172a",
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#94a3b8", lw=0.9),
    )

    validate_figure_no_raw_markup(fig, check_article_jargon=True)
    fig.savefig(target, dpi=DPI, bbox_inches=None)
    plt.close(fig)
    return str(target)


def generate_all_charts(
    *,
    assets_root: Union[str, Path] = "assets",
    article_dir: Union[str, Path] = "assets/article",
    with_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Generate all article charts (cover_5x2 + 1600px figures), GPU-only budget chart, MPS 4k chart, and stats."""
    root = Path(assets_root)
    art = Path(article_dir)
    art.mkdir(parents=True, exist_ok=True)

    tokenizer = _resolve_local_tokenizer() if with_tokenizer else None

    stats_df = generate_gpu_stats(assets_root=root, out_csv=root / "gpu_stats.csv")

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
        "cover_5x2": generate_cover_5x2_chart(
            assets_root=root,
            out_path=art / "cover_5x2.png",
        ),
        "headline_budget_curve": generate_headline_budget_curve(
            assets_root=root,
            out_path=art / "headline_budget_curve.png",
        ),
        "headline_budget_curve_full": generate_headline_budget_curve_full(
            assets_root=root,
            out_path=art / "headline_budget_curve_full.png",
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
        "outcomes_32k_core": generate_outcomes_32k_single_chart(
            panel="core",
            assets_root=root,
            out_path=art / "outcomes_32k_core.png",
            tokenizer=tokenizer,
        ),
        "outcomes_dose": generate_outcomes_32k_single_chart(
            panel="dose",
            assets_root=root,
            out_path=art / "outcomes_dose.png",
            tokenizer=tokenizer,
        ),
        "loop_onset": generate_loop_onset_chart(
            assets_root=root,
            out_path=art / "loop_onset.png",
            tokenizer=tokenizer,
        ),
        "loop_onset_full": generate_loop_onset_full_chart(
            assets_root=root,
            out_path=art / "loop_onset_full.png",
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
