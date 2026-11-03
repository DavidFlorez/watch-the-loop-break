"""Gradio side-by-side streaming app for `watch-the-loop-break`.

Streams base (`alpha=0.0`) and weight-edited (`alpha=1.0`, or any custom `alpha`) answers
side by side on a chosen `HuggingFaceH4/MATH-500` problem, with:
- Live repetition meter based on the repeated 8-gram ratio (`repeated_8gram_ratio`)
- Prominent flag when an answer hits the token cap vs. finishes naturally with `\\boxed{}`
- Hardware auto-detection (CUDA, MPS, or CPU)
"""

from __future__ import annotations

import argparse
import html
import json
import os
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
        # Fallback built-in MATH-500 examples if offline
        _MATH500_CACHE = [
            {
                "idx": 0,
                "label": "[000]Precalculus (L2) | test/precalculus/807.json — Convert the point (0,3) in rectangular coordinates to polar...",
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

    # Color scale for repetition meter
    if rep_pct < 15.0:
        bar_color = "#16a34a"
        loop_label = "LOW REPETITION (Productive Reasoning)"
        badge_bg = "#dcfce7"
        badge_fg = "#166534"
    elif rep_pct < 38.0:
        bar_color = "#d97706"
        loop_label = "MODERATE REPETITION (Re-checking / Circling)"
        badge_bg = "#fef3c7"
        badge_fg = "#92400e"
    else:
        bar_color = "#dc2626"
        loop_label = "HIGH REPETITION (Reasoning Loop Detected!)"
        badge_bg = "#fee2e2"
        badge_fg = "#991b1b"

    # Cap flag banner
    if hit_cap:
        cap_banner = (
            '<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
            'background:#fef2f2;border:2px solid #dc2626;color:#991b1b;font-weight:700;font-size:13px;">'
            f'🚨 TOKEN CAP HIT ({int(n_tokens)} / {int(max_tokens)} tokens) — Generation truncated before finishing!'
            "</div>"
        )
    elif status == "done":
        if is_correct is True:
            cap_banner = (
                '<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
                'background:#f0fdf4;border:1.5px solid #16a34a;color:#166534;font-weight:700;font-size:13px;">'
                f"✅ FINISHED BEFORE CAP ({int(n_tokens)} / {int(max_tokens)} tokens) — Matches Gold Answer!"
                "</div>"
            )
        elif pred_box is not None:
            cap_banner = (
                '<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
                'background:#fffbeb;border:1.5px solid #d97706;color:#92400e;font-weight:600;font-size:13px;">'
                f"⚠️ FINISHED ({int(n_tokens)} / {int(max_tokens)} tokens) — Boxed answer extracted"
                "</div>"
            )
        else:
            cap_banner = (
                '<div style="margin-top:8px;padding:8px 12px;border-radius:6px;'
                'background:#f8fafc;border:1.5px solid #94a3b8;color:#334155;font-weight:600;font-size:13px;">'
                f"⏹️ FINISHED ({int(n_tokens)} / {int(max_tokens)} tokens) — No \\boxed{{}} answer found"
                "</div>"
            )
    elif status == "streaming":
        cap_banner = (
            '<div style="margin-top:8px;padding:7px 12px;border-radius:6px;'
            'background:#eff6ff;border:1px solid #93c5fd;color:#1e40af;font-weight:600;font-size:12.5px;">'
            f"⚡ Streaming live... ({int(n_tokens)} / {int(max_tokens)} tokens | {tok_per_s:.1f} tok/s)"
            "</div>"
        )
    else:
        cap_banner = (
            '<div style="margin-top:8px;padding:7px 12px;border-radius:6px;'
            'background:#f8fafc;border:1px solid #e2e8f0;color:#64748b;font-size:12.5px;">'
            "Ready — select a MATH-500 problem and click Stream Side-by-Side"
            "</div>"
        )

    safe_pred = html.escape(str(pred_box)) if pred_box is not None else "—"
    safe_gold = html.escape(str(gold_answer)) if gold_answer else "—"

    return f"""
<div style="border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;background:#ffffff;box-shadow:0 1px 2px rgba(0,0,0,0.04);font-family:system-ui,-apple-system,sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-weight:700;font-size:14px;color:#0f172a;">Weight Blend α = {alpha:.2f}</span>
    <span style="padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:700;background:{badge_bg};color:{badge_fg};">
      {html.escape(loop_label)}
    </span>
  </div>

  <div style="margin:6px 0;">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#334155;margin-bottom:3px;">
      <span><b>Repeated 8-Gram Ratio:</b> {rep_pct:.1f}%</span>
      <span><b>Tail (240w) 8-Gram Rep:</b> {tail_pct:.1f}%</span>
    </div>
    <div style="width:100%;height:11px;background:#f1f5f9;border-radius:999px;overflow:hidden;border:1px solid #cbd5e1;">
      <div style="width:{rep_pct:.1f}%;height:100%;background:{bar_color};transition:width 0.15s ease;"></div>
    </div>
  </div>

  <div style="margin:6px 0;">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#475569;margin-bottom:3px;">
      <span><b>Tokens:</b> {int(n_tokens)} / {int(max_tokens)} ({tok_pct:.0f}%)</span>
      <span><b>Boxed:</b> <code>{safe_pred}</code> (Gold: <code>{safe_gold}</code>)</span>
    </div>
    <div style="width:100%;height:6px;background:#f1f5f9;border-radius:999px;overflow:hidden;">
      <div style="width:{tok_pct:.1f}%;height:100%;background:#3b82f6;transition:width 0.15s ease;"></div>
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

    scores = scores / float(temperature)

    if top_k > 0 and top_k < scores.shape[-1]:
        topk_vals, _ = torch.topk(scores, int(top_k))
        min_topk = topk_vals[-1]
        scores = torch.where(scores < min_topk, torch.full_like(scores, float("-inf")), scores)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > float(top_p)
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scores[indices_to_remove] = float("-inf")

    probs = torch.softmax(scores, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


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
    """Stream answers for `alpha_left` and `alpha_right` side by side in lockstep.

    Uses zero-copy parameter tensor swapping between left and right weight blends on a
    single loaded model with two independent KV caches (`past_left`, `past_right`),
    ensuring thread safety on CUDA, Apple MPS, and CPU while using only 1x model RAM.
    """
    # Input validation
    problem_clean = (problem_text or "").strip()
    if not problem_clean:
        err_html = render_repetition_meter_html(0.0, 0.0, 0.0, 0, max_new_tokens, 0.0, "idle", False, None, gold_answer, None)
        yield err_html, "Please enter a non-empty math problem.", err_html, "Please enter a non-empty math problem."
        return
    if len(problem_clean) > 8000:
        problem_clean = problem_clean[:8000]

    alpha_l = float(max(-2.0, min(3.0, alpha_left)))
    alpha_r = float(max(-2.0, min(3.0, alpha_right)))
    max_toks = int(max(32, min(8192, max_new_tokens)))
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

    # Pre-compute the two blended tensors on the target device for both alphas
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

    # Prefill Left
    apply_weights_fast(left_blended)
    out_l = model(input_ids=prompt_ids, use_cache=True)
    past_l = out_l.past_key_values
    gen_l: List[int] = []
    next_tok_l = _sample_next_token(out_l.logits[:, -1, :], gen_l, temp, tp, tk, pp)
    gen_l.append(next_tok_l)
    done_l = next_tok_l in eos_ids

    # Prefill Right
    torch.manual_seed(int(seed))
    apply_weights_fast(right_blended)
    out_r = model(input_ids=prompt_ids, use_cache=True)
    past_r = out_r.past_key_values
    gen_r: List[int] = []
    next_tok_r = _sample_next_token(out_r.logits[:, -1, :], gen_r, temp, tp, tk, pp)
    gen_r.append(next_tok_r)
    done_r = next_tok_r in eos_ids

    emit_every = 6  # Update UI every 6 generated steps for smooth streaming

    for step in range(1, max_toks):
        if done_l and done_r:
            break

        if not done_l:
            apply_weights_fast(left_blended)
            inp_l = torch.tensor([[gen_l[-1]]], device=device, dtype=torch.long)
            out_l = model(input_ids=inp_l, past_key_values=past_l, use_cache=True)
            past_l = out_l.past_key_values
            nt_l = _sample_next_token(out_l.logits[:, -1, :], gen_l, temp, tp, tk, pp)
            gen_l.append(nt_l)
            if nt_l in eos_ids:
                done_l = True

        if not done_r:
            apply_weights_fast(right_blended)
            inp_r = torch.tensor([[gen_r[-1]]], device=device, dtype=torch.long)
            out_r = model(input_ids=inp_r, past_key_values=past_r, use_cache=True)
            past_r = out_r.past_key_values
            nt_r = _sample_next_token(out_r.logits[:, -1, :], gen_r, temp, tp, tk, pp)
            gen_r.append(nt_r)
            if nt_r in eos_ids:
                done_r = True

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
            yield html_l, text_l, html_r, text_r


def load_recorded_sweep_pairs(jsonl_path: str = "assets/sweep_per_problem.jsonl") -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Load recorded sweep outputs indexed by `unique_id -> {alpha -> record}`."""
    by_problem: Dict[str, Dict[float, Dict[str, Any]]] = {}
    if not os.path.exists(jsonl_path):
        return by_problem
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                uid = str(r["unique_id"])
                a = float(r["alpha"])
                by_problem.setdefault(uid, {})[a] = r
    except Exception:
        pass
    return by_problem


def replay_recorded_comparison(
    problem_label: str,
    alpha_left: float,
    alpha_right: float,
) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
    """Stream a side-by-side replay from recorded 4096-token sweep outputs for instant inspection."""
    problems = get_math500_problems()
    selected = problems[0]
    for p in problems:
        if p["label"] == problem_label:
            selected = p
            break

    by_prob = load_recorded_sweep_pairs()
    uid = selected["unique_id"]
    gold = selected["answer"]
    prob_text = selected["problem"]

    if uid not in by_prob:
        msg = f"No pre-recorded sweep entry found for {uid}. Use 'Stream Live Side-by-Side' to run live inference!"
        h_l = render_repetition_meter_html(alpha_left, 0.0, 0.0, 0, 4096, 0.0, "idle", False, None, gold, None)
        h_r = render_repetition_meter_html(alpha_right, 0.0, 0.0, 0, 4096, 0.0, "idle", False, None, gold, None)
        yield prob_text, gold, h_l, msg, h_r, msg
        return

    avail = by_prob[uid]
    # Pick closest recorded alpha
    al_key = min(avail.keys(), key=lambda x: abs(x - float(alpha_left)))
    ar_key = min(avail.keys(), key=lambda x: abs(x - float(alpha_right)))
    rec_l = avail[al_key]
    rec_r = avail[ar_key]

    words_l = rec_l["text"].split(" ")
    words_r = rec_r["text"].split(" ")
    max_len = max(len(words_l), len(words_r), 1)
    n_frames = 18

    for frame in range(1, n_frames + 1):
        frac = frame / n_frames
        cur_l = " ".join(words_l[: max(1, int(len(words_l) * frac))])
        cur_r = " ".join(words_r[: max(1, int(len(words_r) * frac))])
        tok_l = int(rec_l["n_tokens"] * frac)
        tok_r = int(rec_r["n_tokens"] * frac)
        is_last = frame == n_frames

        rep_l = rec_l["rep_8gram_ratio"] if is_last else repeated_8gram_ratio(cur_l)
        tail_l = rec_l["tail_rep_8gram_ratio"] if is_last else tail_repeated_8gram_ratio(cur_l)
        rep_r = rec_r["rep_8gram_ratio"] if is_last else repeated_8gram_ratio(cur_r)
        tail_r = rec_r["tail_rep_8gram_ratio"] if is_last else tail_repeated_8gram_ratio(cur_r)

        status = "done" if is_last else "streaming"
        hit_l = bool(rec_l["hit_cap"]) if is_last else False
        hit_r = bool(rec_r["hit_cap"]) if is_last else False
        pred_l = rec_l["pred"] if is_last else extract_last_boxed(cur_l)
        pred_r = rec_r["pred"] if is_last else extract_last_boxed(cur_r)
        ok_l = bool(rec_l["correct"]) if is_last else None
        ok_r = bool(rec_r["correct"]) if is_last else None

        h_l = render_repetition_meter_html(
            al_key, rep_l, tail_l, tok_l, rec_l.get("max_new_tokens", 4096), 85.0, status, hit_l, pred_l, gold, ok_l
        )
        h_r = render_repetition_meter_html(
            ar_key, rep_r, tail_r, tok_r, rec_r.get("max_new_tokens", 4096), 85.0, status, hit_r, pred_r, gold, ok_r
        )
        yield prob_text, gold, h_l, cur_l, h_r, cur_r
        if not is_last:
            time.sleep(0.06)


def on_problem_select(problem_label: str) -> Tuple[str, str]:
    for p in get_math500_problems():
        if p["label"] == problem_label:
            return p["problem"], p["answer"]
    return "", ""


def build_demo() -> gr.Blocks:
    _, _, hw_desc = detect_hardware()
    problems = get_math500_problems()
    recorded = load_recorded_sweep_pairs()
    # Default to a problem present in recorded sweep if available
    default_prob = problems[0]
    for p in problems:
        if p["unique_id"] in recorded:
            default_prob = p
            break

    init_left_html = render_repetition_meter_html(
        0.0, 0.0, 0.0, 0, 4096, 0.0, "idle", False, None, default_prob["answer"], None
    )
    init_right_html = render_repetition_meter_html(
        1.0, 0.0, 0.0, 0, 4096, 0.0, "idle", False, None, default_prob["answer"], None
    )

    with gr.Blocks(title="Watch the Loop Break — Weight-Space Programming") as demo:
        gr.Markdown(
            f"""# 🧠 Watch the Loop Break: Weight-Space Programming Explorer
Reproducing & extending **[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)** on `Qwen/Qwen3.5-0.8B` vs. `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.
Only **2 tensors** (`layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` — 64 columns each, 0.016% of parameters) differ between Base ($\alpha=0$) and Edited ($\alpha=1$).
Use `blend(alpha)` to set $W(\alpha) = W_{{\text{{base}}}} + \alpha(W_{{\text{{edit}}}} - W_{{\text{{base}}}})$ and watch the repetition loop break in real time.
**Hardware detected:** `{hw_desc}`"""
        )

        with gr.Row():
            prob_dropdown = gr.Dropdown(
                choices=[p["label"] for p in problems],
                value=default_prob["label"],
                label="Choose a MATH-500 Problem (HuggingFaceH4/MATH-500)",
                scale=4,
            )
            gold_box = gr.Textbox(
                value=default_prob["answer"],
                label="Gold Reference Answer",
                scale=1,
            )

        prob_input = gr.Textbox(
            value=default_prob["problem"],
            label="Problem Statement (editable)",
            lines=2,
        )

        with gr.Row():
            alpha_left_slider = gr.Slider(
                minimum=-0.5,
                maximum=2.0,
                value=0.0,
                step=0.05,
                label="Left Model α (0.0 = Base Qwen3.5-0.8B)",
            )
            alpha_right_slider = gr.Slider(
                minimum=-0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Right Model α (1.0 = Weight-Space Programmed)",
            )
            max_tokens_slider = gr.Slider(
                minimum=128,
                maximum=4096,
                value=4096,
                step=128,
                label="Max New Tokens (Cap)",
            )

        with gr.Accordion("Sampling Parameters (Paper Protocol Defaults)", open=False):
            with gr.Row():
                temp_slider = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                top_k_slider = gr.Slider(1, 100, value=20, step=1, label="Top-k")
                pres_pen_slider = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="Presence Penalty")
                thinking_chk = gr.Checkbox(value=True, label="Thinking Mode (<think>)")
                seed_num = gr.Number(value=0, precision=0, label="Seed")

        with gr.Row():
            stream_btn = gr.Button("🚀 Stream Live Side-by-Side (blend(α))", variant="primary", scale=2)
            replay_btn = gr.Button("⚡ Replay Recorded 4096-Token Sweep Run", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                left_meter = gr.HTML(value=init_left_html, label="Left Repetition & Cap Meter")
                left_out = gr.Textbox(
                    label="Left Stream Output",
                    lines=20,
                    max_lines=24,
                )
            with gr.Column(scale=1):
                right_meter = gr.HTML(value=init_right_html, label="Right Repetition & Cap Meter")
                right_out = gr.Textbox(
                    label="Right Stream Output",
                    lines=20,
                    max_lines=24,
                )

        prob_dropdown.change(
            fn=on_problem_select,
            inputs=[prob_dropdown],
            outputs=[prob_input, gold_box],
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
        )

        replay_btn.click(
            fn=replay_recorded_comparison,
            inputs=[prob_dropdown, alpha_left_slider, alpha_right_slider],
            outputs=[prob_input, gold_box, left_meter, left_out, right_meter, right_out],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the watch-the-loop-break Gradio app")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default: 7860)")
    args = parser.parse_args()

    demo = build_demo()
    # TODO(security): For multi-user non-local deployment, add OAuth/MFA authentication and rate limiting.
    # Binding strictly to 127.0.0.1 per secure web testing guidelines.
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
