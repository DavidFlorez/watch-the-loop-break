"""Gradio side-by-side streaming app for `watch-the-loop-break`.

Streams base (`alpha=0.0`) and weight-edited (`alpha=1.0`, or any custom `alpha`) answers
side by side on a chosen `HuggingFaceH4/MATH-500` problem, with:
- Full KaTeX math and Markdown rendering across problem statements, gold answers, and output streams
- Live repetition meter based on the repeated 8-gram ratio (`repeated_8gram_ratio`)
- Prominent flag when an answer hits the token cap vs. finishes naturally with a boxed answer
- Instant replay from recorded 32,768-token (N=200) and 4,096-token (N=50) GPU sweeps, plus live local streaming
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import torch
from datasets import load_dataset

from metrics import (
    build_math_prompt,
    extract_last_boxed,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
)
from sweep import load_model_and_tokenizer
from weights import (
    BASE_MODEL_ID,
    EDIT_MODEL_ID,
    TARGET_TENSORS,
    _resolve_parameter_by_name,
    blend,
    detect_hardware,
    load_target_tensors,
    register_active_model,
)

# Global state initialized on startup or first request
_STATE: Dict[str, Any] = {}
_MATH500_CACHE: List[Dict[str, Any]] = []
_RECORDED_SWEEP_CACHE: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}

KATEX_DELIMITERS: List[Dict[str, Any]] = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

BUDGET_CHOICES: Dict[str, int] = {
    "32,768 tokens (Paper Protocol — 200 problems)": 32768,
    "4,096 tokens (Fast Budget — 50 problems)": 4096,
}

RECORDED_JSONL_BY_BUDGET: Dict[int, List[str]] = {
    32768: [
        "assets/gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
        "assets/budget_32768/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ],
    4096: [
        "assets/sweep_per_problem.jsonl",
        "assets/gpu_n200/budget_4096/gpu-rtx-pro-6000-bfloat16/sweep_per_problem.jsonl.gz",
    ],
}

APP_CSS = """
:root, html, body, .gradio-container, .dark, .dark .gradio-container {
    color-scheme: light !important;
    --body-background-fill: #f8fafc !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f1f5f9 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #cbd5e1 !important;
    --block-label-background-fill: #f1f5f9 !important;
    --block-label-text-color: #1e293b !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --border-color-primary: #cbd5e1 !important;
    --color-accent-soft: #ecfdf5 !important;
}
body, .gradio-container {
    background: #f8fafc !important;
    color: #0f172a !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.gradio-container .generating,
.gradio-container .pending {
    opacity: 1 !important;
}
.gradio-container label,
.gradio-container label span,
.gradio-container [data-testid="block-label"] {
    color: #334155 !important;
    opacity: 1 !important;
}
.gradio-container .wrap-inner input,
.gradio-container [role="combobox"] input,
.gradio-container .secondary-wrap input {
    padding-right: 36px !important;
    text-overflow: ellipsis !important;
    overflow: hidden !important;
    white-space: nowrap !important;
}
.hero-header {
    background: #0f172a !important;
    color: #f8fafc !important;
    border-radius: 12px;
    padding: 11px 18px;
    margin-bottom: 4px;
}
.hero-header h1, .hero-header p, .hero-header strong, .hero-header .katex, .hero-header .katex * {
    color: #f8fafc !important;
}
.hero-header h1 {
    font-size: 1.34rem !important;
    margin: 0 0 3px 0 !important;
}
.problem-card {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 11px !important;
    padding: 9px 14px !important;
    font-size: 1.04rem !important;
    line-height: 1.48 !important;
    color: #0f172a !important;
}
.problem-card *, .problem-card .prose, .problem-card .prose *, .problem-card .katex, .problem-card .katex * {
    color: #0f172a !important;
    opacity: 1 !important;
}
.gold-card {
    background: #f0fdf4 !important;
    border: 1.5px solid #86efac !important;
    border-radius: 11px !important;
    padding: 8px 14px !important;
    font-size: 1.04rem !important;
    line-height: 1.42 !important;
    color: #064e3b !important;
}
.gold-card *, .gold-card .prose, .gold-card .prose *, .gold-card .katex, .gold-card .katex * {
    color: #064e3b !important;
    opacity: 1 !important;
}
.gold-card .katex-display {
    margin: 0.18em 0 0.04em 0 !important;
    font-size: 1.18rem !important;
}
.stream-panel {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    padding: 12px 16px 14px 16px !important;
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
    color: #0f172a !important;
}
.stream-panel *, .stream-panel .prose, .stream-panel .prose *, .stream-panel .katex, .stream-panel .katex * {
    color: #0f172a !important;
    opacity: 1 !important;
}
.stream-panel .prose {
    padding-bottom: 10px !important;
}
.katex .frac-line, .katex .overline-line, .katex .underline-line {
    border-bottom-width: max(1.3px, 0.055em) !important;
    border-color: currentColor !important;
}
.katex .fbox {
    border-width: max(1.5px, 0.06em) !important;
    border-style: solid !important;
    border-color: currentColor !important;
    padding: 0.14em 0.24em !important;
}
.stream-panel-left {
    border-top: 4px solid #dc2626 !important;
}
.stream-panel-right {
    border-top: 4px solid #059669 !important;
}
.stream-panel blockquote {
    border-left: 3px solid #94a3b8;
    background: #f8fafc !important;
    padding: 6px 12px;
    border-radius: 6px;
    color: #334155 !important;
    margin: 8px 0;
}
.katex-display {
    margin: 0.45em 0 !important;
    padding: 3px 0 !important;
    font-size: 1.12rem !important;
    overflow-x: auto;
    overflow-y: visible !important;
}
"""

APP_HEAD_JS = """
<script>
(function() {
  function scrollStreamsToBottom() {
    ['left-stream-md', 'right-stream-md'].forEach(function(id) {
      var el = document.getElementById(id);
      if (!el) return;
      if (el.scrollHeight > el.clientHeight + 2) {
        el.scrollTop = el.scrollHeight;
      }
      var descendants = el.querySelectorAll('*');
      for (var i = 0; i < descendants.length; i++) {
        var child = descendants[i];
        if (child.scrollHeight > child.clientHeight + 2) {
          child.scrollTop = child.scrollHeight;
        }
      }
    });
  }
  var timer = null;
  var obs = new MutationObserver(function() {
    window.requestAnimationFrame(scrollStreamsToBottom);
    if (timer) clearTimeout(timer);
    timer = setTimeout(scrollStreamsToBottom, 90);
  });
  window.addEventListener('DOMContentLoaded', function() {
    obs.observe(document.body, { childList: true, subtree: true, characterData: true });
  });
})();
</script>
"""


def _wrap_bare_boxed(text: str) -> str:
    """Wrap standalone or un-delimited `\\boxed{...}` in `$...$` / `$$...$$` so KaTeX renders every box."""
    out: List[str] = []
    i = 0
    n = len(text)
    in_inline = False
    in_display = False

    while i < n:
        if text[i] == "\\":
            if i + 1 < n and text[i + 1] == "$":
                out.append("\\$")
                i += 2
                continue
            if not in_inline and not in_display and text.startswith("\\boxed{", i):
                depth = 1
                j = i + len("\\boxed{")
                while j < n and depth > 0:
                    if text[j] == "\\" and j + 1 < n:
                        j += 2
                        continue
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    boxed_expr = text[i:j]
                    if boxed_expr == "\\boxed{}":
                        boxed_expr = "\\boxed{\\;}"
                    line_start = text.rfind("\n", 0, i) + 1
                    line_end = text.find("\n", j)
                    if line_end == -1:
                        line_end = n
                    before_on_line = text[line_start:i].strip()
                    after_on_line = text[j:line_end].strip()
                    if not before_on_line and not after_on_line:
                        out.append(f"\n$$\n{boxed_expr}\n$$\n")
                    else:
                        out.append(f"${boxed_expr}$")
                    i = j
                    continue
        if text.startswith("$$", i) and (i == 0 or text[i - 1] != "\\"):
            if not in_inline:
                in_display = not in_display
            out.append("$$")
            i += 2
            continue
        if text[i] == "$" and (i == 0 or text[i - 1] != "\\"):
            if not in_display:
                in_inline = not in_inline
            out.append("$")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _trim_incomplete_trailing_math(text: str) -> str:
    """Drop an unclosed trailing math delimiter (`$$`, `$`, `\\(`, `\\[`) on an in-flight stream frame."""
    in_inline = False
    in_display = False
    in_paren = False
    in_bracket = False
    last_open_pos = -1

    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "$":
                i += 2
                continue
            if nxt == "(" and not (in_display or in_bracket or in_inline or in_paren):
                in_paren = True
                last_open_pos = i
                i += 2
                continue
            if nxt == ")" and in_paren:
                in_paren = False
                last_open_pos = -1
                i += 2
                continue
            if nxt == "[" and not (in_display or in_bracket or in_inline or in_paren):
                in_bracket = True
                last_open_pos = i
                i += 2
                continue
            if nxt == "]" and in_bracket:
                in_bracket = False
                last_open_pos = -1
                i += 2
                continue
        if text.startswith("$$", i):
            if not (in_inline or in_paren or in_bracket):
                if not in_display:
                    in_display = True
                    last_open_pos = i
                else:
                    in_display = False
                    last_open_pos = -1
            i += 2
            continue
        if text[i] == "$":
            if not (in_display or in_paren or in_bracket):
                if not in_inline:
                    in_inline = True
                    last_open_pos = i
                else:
                    in_inline = False
                    last_open_pos = -1
            i += 1
            continue
        i += 1

    if (in_inline or in_display or in_paren or in_bracket) and last_open_pos != -1:
        return text[:last_open_pos].rstrip()
    return text


def format_problem_for_display(problem_text: str) -> str:
    """Format a MATH-500 problem statement with normalized math delimiters for clean KaTeX rendering."""
    cleaned = (problem_text or "").strip()
    if not cleaned:
        return "*Enter or select a math problem above.*"
    cleaned = cleaned.replace("\\[", "$$").replace("\\]", "$$")
    cleaned = cleaned.replace("\\(", "$").replace("\\)", "$")
    cleaned = _wrap_bare_boxed(cleaned)
    return f"**Problem Statement:**\n\n{cleaned}"


def format_gold_for_display(gold_answer: str) -> str:
    """Format a gold reference answer so KaTeX renders fractions, radicals, and tuples cleanly."""
    cleaned = (gold_answer or "").strip()
    if not cleaned:
        return "**Gold Answer:** —"
    if cleaned.startswith("$$") and cleaned.endswith("$$") and len(cleaned) > 4:
        inner = cleaned[2:-2].strip()
    elif cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) > 2:
        inner = cleaned[1:-1].strip()
    else:
        inner = cleaned
    if inner.startswith("\\boxed{") and inner.endswith("}"):
        inner = inner[len("\\boxed{") : -1].strip()
    return f"**Gold Answer:**\n\n$$\n\\boxed{{{inner}}}\n$$"


def format_stream_for_display(
    text: str,
    *,
    is_streaming: bool = False,
    max_head_chars: int = 1900,
    max_tail_chars: int = 3800,
) -> str:
    """Clean raw model output for KaTeX + Markdown rendering with zero raw `<think>` or `\boxed{}` leaks."""
    if not text:
        return "*Waiting to start…*"

    cleaned = text.replace("<|im_start|>assistant", "").replace("<|im_end|>", "")
    cleaned = re.sub(r"^\s*<think>\s*", "", cleaned)
    cleaned = re.sub(
        r"\s*</think>\s*",
        "\n\n---\n\n**✓ Finished Thinking — Final Response**\n\n",
        cleaned,
    )
    # Prevent CommonMark from turning 4-space-indented list items into raw <pre><code> blocks
    cleaned = re.sub(r"(?m)^[ \t]{4,}(\*|-|\d+\.)[ \t]+", r"  \1 ", cleaned)
    # Unwrap inline code backticks wrapping LaTeX expressions so KaTeX renders them instead of <code>
    cleaned = re.sub(r"`([^`\n]*\\[a-zA-Z]+[^`\n]*)`", r"\1", cleaned)

    if is_streaming:
        cleaned = _trim_incomplete_trailing_math(cleaned)

    cleaned = cleaned.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n")
    cleaned = cleaned.replace("\\(", "$").replace("\\)", "$")
    cleaned = _wrap_bare_boxed(cleaned).strip()

    if len(cleaned) > (max_head_chars + max_tail_chars):
        head_cut = cleaned.rfind("\n", 0, max_head_chars)
        if head_cut < max_head_chars // 2:
            head_cut = max_head_chars
        tail_start = cleaned.find("\n", len(cleaned) - max_tail_chars)
        if tail_start == -1 or tail_start >= len(cleaned) - 80:
            tail_start = len(cleaned) - max_tail_chars
        head_part = _trim_incomplete_trailing_math(cleaned[:head_cut].rstrip())
        tail_part = cleaned[tail_start:].lstrip()
        # Avoid starting the tail slice inside an unclosed $$ block
        if tail_part.count("$$") % 2 == 1:
            first_close = tail_part.find("$$")
            if first_close != -1:
                tail_part = tail_part[first_close + 2 :].lstrip()
        cleaned = (
            f"{head_part}\n\n"
            "> *… [middle repeating tokens condensed for display; full token count tracked above] …*\n\n"
            f"{tail_part}"
        )

    return cleaned


def get_math500_problems() -> List[Dict[str, Any]]:
    """Load and cache all 500 problems from `HuggingFaceH4/MATH-500` (test split)."""
    global _MATH500_CACHE
    if _MATH500_CACHE:
        return _MATH500_CACHE
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for i, row in enumerate(ds):
            prob_preview = row["problem"].replace("\n", " ")
            if len(prob_preview) > 78:
                prob_preview = prob_preview[:75] + "..."
            label = (
                f"[{i:03d}] {row.get('subject', 'Math')} (L{row.get('level', '?')}) | "
                f"{row.get('unique_id', f'p{i}')} — {prob_preview}"
            )
            _MATH500_CACHE.append(
                {
                    "idx": i,
                    "label": label,
                    "unique_id": str(row.get("unique_id", f"math500_{i}")),
                    "problem": str(row["problem"]),
                    "answer": str(row["answer"]),
                    "subject": str(row.get("subject", "Unknown")),
                    "level": str(row.get("level", "?")),
                }
            )
    except Exception:
        _MATH500_CACHE = [
            {
                "idx": 0,
                "label": "[000] Precalculus (L2) | test/precalculus/807.json — Convert the point (0,3) in rectangular coordinates to polar...",
                "unique_id": "test/precalculus/807.json",
                "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
                "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
                "subject": "Precalculus",
                "level": "2",
            },
            {
                "idx": 1,
                "label": "[001] Algebra (L2) | custom/algebra/1 — Let x and y be positive integers such that x+y=20 and xy=91...",
                "unique_id": "custom/algebra/1",
                "problem": "Let $x$ and $y$ be positive integers such that $x + y = 20$ and $xy = 91$. What is $x^2 + y^2$?",
                "answer": "218",
                "subject": "Algebra",
                "level": "2",
            },
        ]
    return _MATH500_CACHE


def ensure_model_loaded() -> Dict[str, Any]:
    """Lazy-load `Qwen/Qwen3.5-0.8B` and the target weight tensors from both checkpoints."""
    if _STATE.get("ready"):
        return _STATE
    tok, model, device, dtype, hw_desc = load_model_and_tokenizer(BASE_MODEL_ID)
    base_tensors, edit_tensors = load_target_tensors(BASE_MODEL_ID, EDIT_MODEL_ID)
    register_active_model(model, base_tensors, edit_tensors)
    _STATE.update(
        {
            "ready": True,
            "tokenizer": tok,
            "model": model,
            "device": device,
            "dtype": dtype,
            "hw_desc": hw_desc,
            "base_tensors": base_tensors,
            "edit_tensors": edit_tensors,
        }
    )
    return _STATE


def render_repetition_meter_html(
    alpha: float,
    rep_ratio: float,
    tail_rep_ratio: float,
    n_tokens: int,
    max_tokens: int,
    tok_per_s: float,
    status: str,
    hit_cap: bool,
    pred_box: Optional[str],
    gold_answer: Optional[str],
    is_correct: Optional[bool],
) -> str:
    """Render a secure, XSS-safe HTML dashboard card for live 8-gram repetition and token-cap status."""
    rep_pct = max(0.0, min(100.0, rep_ratio * 100.0))
    tail_pct = max(0.0, min(100.0, tail_rep_ratio * 100.0))
    tok_pct = max(0.0, min(100.0, (n_tokens / max(max_tokens, 1)) * 100.0))

    if rep_pct < 15.0:
        bar_color = "#16a34a"
        loop_label = "LOW REPETITION · Productive Reasoning"
        badge_bg = "#dcfce7"
        badge_fg = "#166534"
    elif rep_pct < 38.0:
        bar_color = "#d97706"
        loop_label = "MODERATE REPETITION · Re-checking"
        badge_bg = "#fef3c7"
        badge_fg = "#92400e"
    else:
        bar_color = "#dc2626"
        loop_label = "HIGH REPETITION · Reasoning Loop Detected!"
        badge_bg = "#fee2e2"
        badge_fg = "#991b1b"

    if abs(float(alpha)) < 1e-6:
        model_title = "Base Model (0.00× strength)"
        tok_bar_color = "#dc2626"
    elif abs(float(alpha) - 1.0) < 1e-6:
        model_title = "Edited Model (1.00× published strength)"
        tok_bar_color = "#059669"
    else:
        model_title = f"Edited Model ({float(alpha):.2f}× strength)"
        tok_bar_color = "#2563eb"

    if hit_cap:
        cap_banner = (
            '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
            'background:#fef2f2;border:2px solid #dc2626;color:#991b1b !important;font-weight:700;font-size:13.5px;">'
            f'🚨 RAN OUT OF TOKENS ({int(n_tokens):,} / {int(max_tokens):,} tokens) — Stopped mid-loop before answering!'
            "</div>"
        )
    elif status == "done":
        if is_correct is True:
            cap_banner = (
                '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
                'background:#f0fdf4;border:1.5px solid #16a34a;color:#166534 !important;font-weight:700;font-size:13.5px;">'
                f"✅ FINISHED CLEANLY ({int(n_tokens):,} / {int(max_tokens):,} tokens) — Correct Final Answer!"
                "</div>"
            )
        elif pred_box is not None:
            cap_banner = (
                '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
                'background:#fffbeb;border:1.5px solid #d97706;color:#92400e !important;font-weight:600;font-size:13.5px;">'
                f"⚠️ FINISHED ({int(n_tokens):,} / {int(max_tokens):,} tokens) — Final answer extracted"
                "</div>"
            )
        else:
            cap_banner = (
                '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
                'background:#f8fafc;border:1.5px solid #94a3b8;color:#334155 !important;font-weight:600;font-size:13.5px;">'
                f"⏹️ FINISHED ({int(n_tokens):,} / {int(max_tokens):,} tokens) — No boxed answer found"
                "</div>"
            )
    elif status == "streaming":
        cap_banner = (
            '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
            'background:#eff6ff;border:1px solid #93c5fd;color:#1e40af !important;font-weight:600;font-size:13px;">'
            f"⚡ Thinking live… ({int(n_tokens):,} / {int(max_tokens):,} tokens)"
            "</div>"
        )
    else:
        cap_banner = (
            '<div style="margin-top:6px;padding:7px 11px;border-radius:7px;'
            'background:#f8fafc;border:1px solid #e2e8f0;color:#475569 !important;font-size:13px;">'
            "Ready — select a MATH-500 problem and click Replay or Stream Live"
            "</div>"
        )

    safe_pred = html.escape(str(pred_box)) if pred_box is not None else "—"
    safe_gold = html.escape(str(gold_answer)) if gold_answer else "—"

    return f"""
<div style="border:1.5px solid #cbd5e1;border-radius:11px;padding:11px 14px;background:#ffffff !important;color:#0f172a !important;box-shadow:0 1px 3px rgba(0,0,0,0.05);font-family:system-ui,-apple-system,sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px;flex-wrap:wrap;">
    <span style="font-weight:800;font-size:15.5px;color:#0f172a !important;">{html.escape(model_title)}</span>
    <span style="padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;background:{badge_bg};color:{badge_fg} !important;">
      {html.escape(loop_label)}
    </span>
  </div>

  <div style="margin:5px 0;">
    <div style="display:flex;justify-content:space-between;font-size:13.5px;color:#0f172a !important;margin-bottom:3px;">
      <span style="color:#0f172a !important;"><strong style="color:#0f172a !important;">Repeated text:</strong> {rep_pct:.1f}%</span>
      <span style="color:#0f172a !important;"><strong style="color:#0f172a !important;">Last 240 words repeated:</strong> {tail_pct:.1f}%</span>
    </div>
    <div style="width:100%;height:11px;background:#e2e8f0;border-radius:999px;overflow:hidden;border:1px solid #cbd5e1;">
      <div style="width:{rep_pct:.1f}%;height:100%;background:{bar_color};transition:width 0.15s ease;"></div>
    </div>
  </div>

  <div style="margin:5px 0;">
    <div style="display:flex;justify-content:space-between;font-size:13.5px;color:#0f172a !important;margin-bottom:3px;">
      <span style="color:#0f172a !important;"><strong style="color:#0f172a !important;">Tokens used:</strong> {int(n_tokens):,} / {int(max_tokens):,} ({tok_pct:.0f}%)</span>
      <span style="color:#0f172a !important;"><strong style="color:#0f172a !important;">Extracted answer:</strong> <strong style="color:#0f172a !important;background:#f1f5f9 !important;padding:1px 6px;border-radius:4px;">{safe_pred}</strong> (Gold: <strong style="color:#0f172a !important;">{safe_gold}</strong>)</span>
    </div>
    <div style="width:100%;height:7px;background:#e2e8f0;border-radius:999px;overflow:hidden;">
      <div style="width:{tok_pct:.1f}%;height:100%;background:{tok_bar_color};transition:width 0.15s ease;"></div>
    </div>
  </div>

  {cap_banner}
</div>
"""


def _sample_next_token(
    logits: torch.Tensor,
    generated_ids: List[int],
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
) -> int:
    """Sample a single token from `logits` (`[1, vocab_size]`) with presence penalty, top-k, and top-p."""
    scores = logits[0].to(torch.float32).clone()

    if presence_penalty > 0.0 and generated_ids:
        unique_seen = torch.tensor(list(set(generated_ids)), device=scores.device, dtype=torch.long)
        scores[unique_seen] -= float(presence_penalty)

    if temperature <= 1e-5:
        return int(torch.argmax(scores).item())

    if top_k > 0 and top_k < scores.shape[-1]:
        cand_logits, cand_indices = torch.topk(scores, int(top_k))
    else:
        cand_logits, cand_indices = torch.sort(scores, descending=True)

    cand_logits = cand_logits / float(temperature)
    probs = torch.softmax(cand_logits, dim=-1)

    if 0.0 < top_p < 1.0:
        cum_probs = torch.cumsum(probs, dim=-1)
        remove_mask = (cum_probs - probs) > float(top_p)
        probs = probs.masked_fill(remove_mask, 0.0)

    sampled_rel = torch.multinomial(probs, num_samples=1)
    return int(cand_indices[sampled_rel].item())


@torch.no_grad()
def stream_side_by_side(
    problem_text: str,
    gold_answer: str,
    alpha_left: float,
    alpha_right: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    enable_thinking: bool,
    seed: int = 0,
) -> Generator[Tuple[str, str, str, str], None, None]:
    """Stream answers for `alpha_left` and `alpha_right` side by side in lockstep."""
    problem_clean = (problem_text or "").strip()
    if not problem_clean:
        err_html = render_repetition_meter_html(0.0, 0.0, 0.0, 0, max_new_tokens, 0.0, "idle", False, None, gold_answer, None)
        yield err_html, "Please enter a non-empty math problem.", err_html, "Please enter a non-empty math problem."
        return
    if len(problem_clean) > 8000:
        problem_clean = problem_clean[:8000]

    alpha_l = float(max(-2.0, min(3.0, alpha_left)))
    alpha_r = float(max(-2.0, min(3.0, alpha_right)))
    max_toks = int(max(32, min(32768, max_new_tokens)))
    temp = float(max(0.0, min(2.0, temperature)))
    tp = float(max(0.05, min(1.0, top_p)))
    tk = int(max(1, min(200, top_k)))
    pp = float(max(0.0, min(5.0, presence_penalty)))

    state = ensure_model_loaded()
    tok = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    base_tensors = state["base_tensors"]
    edit_tensors = state["edit_tensors"]

    param_refs = {name: _resolve_parameter_by_name(model, name) for name in TARGET_TENSORS}
    left_blended = {
        name: t.to(device=param_refs[name].device, dtype=param_refs[name].dtype)
        for name, t in blend(alpha_l, model=None, base_tensors=base_tensors, edit_tensors=edit_tensors).items()
    }
    right_blended = {
        name: t.to(device=param_refs[name].device, dtype=param_refs[name].dtype)
        for name, t in blend(alpha_r, model=None, base_tensors=base_tensors, edit_tensors=edit_tensors).items()
    }

    def apply_weights_fast(weight_dict: Dict[str, torch.Tensor]) -> None:
        for name, target_tensor in weight_dict.items():
            param_refs[name].data = target_tensor

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": build_math_prompt(problem_clean)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    enc = tok(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(device)

    eos_ids = set()
    if isinstance(tok.eos_token_id, int):
        eos_ids.add(tok.eos_token_id)
    elif isinstance(tok.eos_token_id, (list, tuple)):
        eos_ids.update(tok.eos_token_id)

    torch.manual_seed(int(seed))
    t0 = time.perf_counter()

    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None

    apply_weights_fast(left_blended)
    out_l = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    past_l = out_l.past_key_values
    gen_l: List[int] = []
    next_tok_l = _sample_next_token(out_l.logits[:, -1, :], gen_l, temp, tp, tk, pp)
    gen_l.append(next_tok_l)
    done_l = next_tok_l in eos_ids

    torch.manual_seed(int(seed))
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    apply_weights_fast(right_blended)
    out_r = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    past_r = out_r.past_key_values
    gen_r: List[int] = []
    next_tok_r = _sample_next_token(out_r.logits[:, -1, :], gen_r, temp, tp, tk, pp)
    gen_r.append(next_tok_r)
    done_r = next_tok_r in eos_ids

    emit_every = 8

    for step in range(1, max_toks):
        if done_l and done_r:
            break

        if not done_l:
            apply_weights_fast(left_blended)
            inp_l = torch.tensor([[gen_l[-1]]], device=device, dtype=torch.long)
            out_l = model(input_ids=inp_l, past_key_values=past_l, use_cache=True, logits_to_keep=1)
            past_l = out_l.past_key_values
            nt_l = _sample_next_token(out_l.logits[:, -1, :], gen_l, temp, tp, tk, pp)
            del out_l
            gen_l.append(nt_l)
            if nt_l in eos_ids:
                done_l = True

        if not done_r:
            apply_weights_fast(right_blended)
            inp_r = torch.tensor([[gen_r[-1]]], device=device, dtype=torch.long)
            out_r = model(input_ids=inp_r, past_key_values=past_r, use_cache=True, logits_to_keep=1)
            past_r = out_r.past_key_values
            nt_r = _sample_next_token(out_r.logits[:, -1, :], gen_r, temp, tp, tk, pp)
            del out_r
            gen_r.append(nt_r)
            if nt_r in eos_ids:
                done_r = True

        if device == "mps" and (step + 1) % 16 == 0 and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        if (step + 1) % emit_every == 0 or (done_l and done_r) or (step + 1 == max_toks):
            elapsed = max(time.perf_counter() - t0, 1e-3)
            text_l = tok.decode(gen_l, skip_special_tokens=False)
            text_r = tok.decode(gen_r, skip_special_tokens=False)

            rep_l = repeated_8gram_ratio(text_l)
            tail_l = tail_repeated_8gram_ratio(text_l)
            rep_r = repeated_8gram_ratio(text_r)
            tail_r = tail_repeated_8gram_ratio(text_r)

            hit_cap_l = (len(gen_l) >= max_toks) and not done_l
            hit_cap_r = (len(gen_r) >= max_toks) and not done_r
            status_l = "done" if (done_l or hit_cap_l) else "streaming"
            status_r = "done" if (done_r or hit_cap_r) else "streaming"

            pred_l = extract_last_boxed(text_l)
            pred_r = extract_last_boxed(text_r)
            ok_l = grade_math_answer(pred_l, gold_answer) if (status_l == "done" and gold_answer) else None
            ok_r = grade_math_answer(pred_r, gold_answer) if (status_r == "done" and gold_answer) else None

            html_l = render_repetition_meter_html(
                alpha_l, rep_l, tail_l, len(gen_l), max_toks, len(gen_l) / elapsed, status_l, hit_cap_l, pred_l, gold_answer, ok_l
            )
            html_r = render_repetition_meter_html(
                alpha_r, rep_r, tail_r, len(gen_r), max_toks, len(gen_r) / elapsed, status_r, hit_cap_r, pred_r, gold_answer, ok_r
            )
            yield (
                html_l,
                format_stream_for_display(text_l, is_streaming=(status_l == "streaming")),
                html_r,
                format_stream_for_display(text_r, is_streaming=(status_r == "streaming")),
            )

    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def load_recorded_sweep_pairs(
    jsonl_path: Optional[str] = None,
    *,
    budget: int = 32768,
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Load recorded sweep outputs indexed by `unique_id -> {alpha -> record}`."""
    candidates = [jsonl_path] if jsonl_path else RECORDED_JSONL_BY_BUDGET.get(int(budget), RECORDED_JSONL_BY_BUDGET[32768])
    chosen = next((p for p in candidates if p and os.path.exists(p)), None)
    if not chosen:
        return {}
    cache_key = os.path.abspath(chosen)
    if cache_key in _RECORDED_SWEEP_CACHE:
        return _RECORDED_SWEEP_CACHE[cache_key]

    by_problem: Dict[str, Dict[float, Dict[str, Any]]] = {}
    try:
        opener = gzip.open if chosen.endswith(".gz") else open
        with opener(chosen, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                uid = str(r["unique_id"])
                a = float(r["alpha"])
                by_problem.setdefault(uid, {})[a] = r
        if by_problem:
            _RECORDED_SWEEP_CACHE[cache_key] = by_problem
    except Exception:
        pass
    return by_problem


def _slice_clean_prefix(text: str, target_chars: int) -> str:
    """Slice `text` at a complete newline boundary near `target_chars` so equations never tear mid-line."""
    if target_chars >= len(text):
        return text
    nl = text.rfind("\n", 0, target_chars)
    if nl > max(120, target_chars // 2):
        return text[:nl].rstrip()
    sp = text.rfind(" ", 0, target_chars)
    return text[: (sp if sp > 0 else target_chars)].rstrip()


def _find_problem_by_label_or_id(problem_label: str) -> Dict[str, Any]:
    """Resolve a MATH-500 problem dictionary by dropdown label or embedded `unique_id`."""
    problems = get_math500_problems()
    for p in problems:
        if p["label"] == problem_label:
            return p
    for p in problems:
        if p["unique_id"] and p["unique_id"] in str(problem_label):
            return p
    return problems[0]


def replay_recorded_comparison(
    problem_label: str,
    alpha_left: float,
    alpha_right: float,
    budget_label: str = "32,768 tokens (Paper Protocol — 200 problems)",
    frame_delay_s: float = 0.14,
) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
    """Stream a side-by-side replay from recorded 32,768-token or 4,096-token sweep outputs."""
    selected = _find_problem_by_label_or_id(problem_label)

    budget_val = BUDGET_CHOICES.get(budget_label, 32768)
    by_prob = load_recorded_sweep_pairs(budget=budget_val)
    if selected["unique_id"] not in by_prob:
        alt_budget = 4096 if budget_val == 32768 else 32768
        alt_prob = load_recorded_sweep_pairs(budget=alt_budget)
        if selected["unique_id"] in alt_prob:
            by_prob = alt_prob
            budget_val = alt_budget

    uid = selected["unique_id"]
    gold = selected["answer"]
    prob_md = format_problem_for_display(selected["problem"])
    gold_md = format_gold_for_display(gold)

    if uid not in by_prob:
        msg = f"No pre-recorded sweep entry found for `{uid}`. Click **Stream Live Side-by-Side** to run live inference!"
        h_l = render_repetition_meter_html(alpha_left, 0.0, 0.0, 0, budget_val, 0.0, "idle", False, None, gold, None)
        h_r = render_repetition_meter_html(alpha_right, 0.0, 0.0, 0, budget_val, 0.0, "idle", False, None, gold, None)
        yield prob_md, gold_md, h_l, msg, h_r, msg
        return

    avail = by_prob[uid]
    al_key = min(avail.keys(), key=lambda x: abs(x - float(alpha_left)))
    ar_key = min(avail.keys(), key=lambda x: abs(x - float(alpha_right)))
    rec_l = avail[al_key]
    rec_r = avail[ar_key]

    full_l = str(rec_l["text"])
    full_r = str(rec_r["text"])
    # Pace visual streaming relative to total token count so faster branch finishes earlier while slower branch keeps streaming
    tok_max_l = max(int(rec_l["n_tokens"]), 1)
    tok_max_r = max(int(rec_r["n_tokens"]), 1)
    max_toks_seen = max(tok_max_l, tok_max_r, 1)
    n_frames = 20

    for frame in range(1, n_frames + 1):
        frac = frame / n_frames
        cur_tok_budget = max(128, int(max_toks_seen * frac))

        done_l_now = (frame == n_frames) or (cur_tok_budget >= tok_max_l)
        done_r_now = (frame == n_frames) or (cur_tok_budget >= tok_max_r)

        frac_l = min(1.0, cur_tok_budget / tok_max_l)
        frac_r = min(1.0, cur_tok_budget / tok_max_r)

        cur_l = full_l if done_l_now else _slice_clean_prefix(full_l, max(240, int(len(full_l) * frac_l)))
        cur_r = full_r if done_r_now else _slice_clean_prefix(full_r, max(240, int(len(full_r) * frac_r)))

        tok_l = tok_max_l if done_l_now else min(tok_max_l, cur_tok_budget)
        tok_r = tok_max_r if done_r_now else min(tok_max_r, cur_tok_budget)

        rep_l = float(rec_l["rep_8gram_ratio"]) if done_l_now else repeated_8gram_ratio(cur_l)
        tail_l = float(rec_l["tail_rep_8gram_ratio"]) if done_l_now else tail_repeated_8gram_ratio(cur_l)
        rep_r = float(rec_r["rep_8gram_ratio"]) if done_r_now else repeated_8gram_ratio(cur_r)
        tail_r = float(rec_r["tail_rep_8gram_ratio"]) if done_r_now else tail_repeated_8gram_ratio(cur_r)

        status_l = "done" if done_l_now else "streaming"
        status_r = "done" if done_r_now else "streaming"
        hit_l = bool(rec_l["hit_cap"]) if done_l_now else False
        hit_r = bool(rec_r["hit_cap"]) if done_r_now else False
        pred_l = rec_l.get("pred_strict", rec_l.get("pred")) if done_l_now else extract_last_boxed(cur_l)
        pred_r = rec_r.get("pred_strict", rec_r.get("pred")) if done_r_now else extract_last_boxed(cur_r)
        ok_l = bool(rec_l.get("correct_strict", rec_l.get("correct"))) if done_l_now else None
        ok_r = bool(rec_r.get("correct_strict", rec_r.get("correct"))) if done_r_now else None

        max_new_l = int(rec_l.get("max_new_tokens", budget_val))
        max_new_r = int(rec_r.get("max_new_tokens", budget_val))

        h_l = render_repetition_meter_html(
            al_key, rep_l, tail_l, tok_l, max_new_l, 85.0, status_l, hit_l, pred_l, gold, ok_l
        )
        h_r = render_repetition_meter_html(
            ar_key, rep_r, tail_r, tok_r, max_new_r, 85.0, status_r, hit_r, pred_r, gold, ok_r
        )
        disp_l = format_stream_for_display(cur_l, is_streaming=(not done_l_now))
        disp_r = format_stream_for_display(cur_r, is_streaming=(not done_r_now))
        yield prob_md, gold_md, h_l, disp_l, h_r, disp_r
        if frame < n_frames and frame_delay_s > 0:
            time.sleep(frame_delay_s)


def on_problem_select(problem_label: str) -> Tuple[str, str, str, str]:
    p = _find_problem_by_label_or_id(problem_label)
    return (
        p["problem"],
        p["answer"],
        format_problem_for_display(p["problem"]),
        format_gold_for_display(p["answer"]),
    )


def build_demo(default_max_tokens: int = 32768) -> gr.Blocks:
    _, _, hw_desc = detect_hardware()
    problems = get_math500_problems()
    recorded_32k = load_recorded_sweep_pairs(budget=32768)
    recorded_4k = load_recorded_sweep_pairs(budget=4096)

    default_budget_label = (
        "32,768 tokens (Paper Protocol — 200 problems)"
        if recorded_32k
        else "4,096 tokens (Fast Budget — 50 problems)"
    )
    active_recorded = recorded_32k if recorded_32k else recorded_4k

    default_prob = problems[0]
    preferred_ids = (
        "test/counting_and_probability/389.json",
        "test/algebra/1936.json",
        "test/algebra/1332.json",
    )
    by_uid = {p["unique_id"]: p for p in problems}
    for uid in preferred_ids:
        runs = active_recorded.get(uid, {})
        if (
            uid in by_uid
            and 0.0 in runs
            and 1.0 in runs
            and runs[1.0].get("correct_strict", runs[1.0].get("correct"))
            and not runs[0.0].get("correct_strict", runs[0.0].get("correct"))
        ):
            default_prob = by_uid[uid]
            break
    else:
        for p in problems:
            runs = active_recorded.get(p["unique_id"], {})
            if (
                0.0 in runs
                and 1.0 in runs
                and runs[1.0].get("correct_strict", runs[1.0].get("correct"))
                and not runs[0.0].get("correct_strict", runs[0.0].get("correct"))
            ):
                default_prob = p
                break

    max_tok_init = int(max(128, min(32768, default_max_tokens)))
    init_left_html = render_repetition_meter_html(
        0.0, 0.0, 0.0, 0, max_tok_init, 0.0, "idle", False, None, default_prob["answer"], None
    )
    init_right_html = render_repetition_meter_html(
        1.0, 0.0, 0.0, 0, max_tok_init, 0.0, "idle", False, None, default_prob["answer"], None
    )

    with gr.Blocks(title="Watch the Loop Break — Weight-Space Programming") as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.Markdown(
            """<div class="hero-header">
<h1>Watch the Loop Break: Weight-Space Programming Explorer</h1>
<p style="margin:0;opacity:0.92;font-size:1.02rem;">
Comparing unedited <strong>Qwen3.5-0.8B (Base model)</strong> against the <strong>2-layer weight-space edit</strong> (131,072 weights = 0.015% of the model changed). Watch how the base model gets trapped in a thinking loop while the edited model solves and stops cleanly.
</p>
</div>""",
            latex_delimiters=KATEX_DELIMITERS,
        )

        with gr.Row():
            prob_dropdown = gr.Dropdown(
                choices=[p["label"] for p in problems],
                value=default_prob["label"],
                label="Choose a MATH-500 Problem",
                scale=4,
            )
            budget_dropdown = gr.Dropdown(
                choices=list(BUDGET_CHOICES.keys()),
                value=default_budget_label,
                label="Recorded Trace Budget",
                scale=2,
            )

        with gr.Row():
            prob_display = gr.Markdown(
                value=format_problem_for_display(default_prob["problem"]),
                latex_delimiters=KATEX_DELIMITERS,
                elem_classes=["problem-card"],
                scale=4,
            )
            gold_display = gr.Markdown(
                value=format_gold_for_display(default_prob["answer"]),
                latex_delimiters=KATEX_DELIMITERS,
                elem_classes=["gold-card"],
                scale=1,
            )

        with gr.Row():
            replay_btn = gr.Button("⚡ Replay Recorded Side-by-Side Comparison", variant="primary", scale=2)
            stream_btn = gr.Button("🚀 Stream Live Inference on Local Hardware", variant="secondary", scale=1)

        with gr.Accordion(f"Model Blend & Custom Problem Controls (Hardware: {hw_desc})", open=False):
            with gr.Row():
                prob_input = gr.Textbox(
                    value=default_prob["problem"],
                    label="Custom Problem Statement (for Live Inference)",
                    lines=2,
                    scale=4,
                )
                gold_box = gr.Textbox(
                    value=default_prob["answer"],
                    label="Gold Reference Answer",
                    scale=1,
                )
            with gr.Row():
                alpha_left_slider = gr.Slider(
                    minimum=-0.5,
                    maximum=2.0,
                    value=0.0,
                    step=0.05,
                    label="Left Model Edit Strength (0.00× = Base)",
                )
                alpha_right_slider = gr.Slider(
                    minimum=-0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="Right Model Edit Strength (1.00× = Published Edit)",
                )
                max_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=32768,
                    value=max_tok_init,
                    step=128,
                    label="Max Thinking Tokens (Cap)",
                )
            with gr.Row():
                temp_slider = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                top_k_slider = gr.Slider(1, 100, value=20, step=1, label="Top-k")
                pres_pen_slider = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="Presence Penalty")
                thinking_chk = gr.Checkbox(value=True, label="Thinking Mode")
                seed_num = gr.Number(value=0, precision=0, label="Seed")

        with gr.Row():
            with gr.Column(scale=1):
                left_meter = gr.HTML(value=init_left_html)
                left_out = gr.Markdown(
                    value="*Click **Replay Recorded Side-by-Side Comparison** to watch the Base model trace.*",
                    latex_delimiters=KATEX_DELIMITERS,
                    height=360,
                    line_breaks=True,
                    elem_id="left-stream-md",
                    elem_classes=["stream-panel", "stream-panel-left"],
                )
            with gr.Column(scale=1):
                right_meter = gr.HTML(value=init_right_html)
                right_out = gr.Markdown(
                    value="*Click **Replay Recorded Side-by-Side Comparison** to watch the Edited model trace.*",
                    latex_delimiters=KATEX_DELIMITERS,
                    height=360,
                    line_breaks=True,
                    elem_id="right-stream-md",
                    elem_classes=["stream-panel", "stream-panel-right"],
                )

        prob_dropdown.change(
            fn=on_problem_select,
            inputs=[prob_dropdown],
            outputs=[prob_input, gold_box, prob_display, gold_display],
        )

        stream_btn.click(
            fn=stream_side_by_side,
            inputs=[
                prob_input,
                gold_box,
                alpha_left_slider,
                alpha_right_slider,
                max_tokens_slider,
                temp_slider,
                top_p_slider,
                top_k_slider,
                pres_pen_slider,
                thinking_chk,
                seed_num,
            ],
            outputs=[left_meter, left_out, right_meter, right_out],
            show_progress="hidden",
        )

        replay_btn.click(
            fn=replay_recorded_comparison,
            inputs=[prob_dropdown, alpha_left_slider, alpha_right_slider, budget_dropdown],
            outputs=[prob_display, gold_display, left_meter, left_out, right_meter, right_out],
            show_progress="hidden",
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the watch-the-loop-break Gradio app")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default: 7860)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Default max new tokens (default: 32768)")
    args = parser.parse_args()

    demo = build_demo(default_max_tokens=args.max_tokens)
    # TODO(security): For multi-user non-local deployment, add OAuth/MFA authentication and rate limiting.
    # Binding strictly to 127.0.0.1 per secure web testing guidelines.
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        css=APP_CSS,
        head=APP_HEAD_JS,
    )


if __name__ == "__main__":
    main()

