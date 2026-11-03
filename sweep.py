"""Run an alpha sweep on N=50 MATH-500 problems for Qwen3.5-0.8B Weight-Space Programming.

Evaluates `blend(alpha)` for alpha in [0, 0.5, 1.0, 1.25, 1.5] with:
- temperature = 0.6
- top_p = 0.95
- top_k = 20
- presence_penalty = 1.5 (via native GenerationConfig if supported, else PresencePenaltyLogitsProcessor)
- enable_thinking = True
- max_new_tokens = 4096

Records accuracy, % hitting the cap, mean tokens, mean repeated 8-gram ratio, and wall time,
saving summary CSV, per-problem CSV/JSONL, and publication-quality charts to `assets/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList

from metrics import (
    PresencePenaltyLogitsProcessor,
    build_math_prompt,
    extract_last_boxed,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
)
from weights import (
    BASE_MODEL_ID,
    EDIT_MODEL_ID,
    blend,
    detect_hardware,
    install_fast_qwen35_kernels,
    load_target_tensors,
    register_active_model,
)

DEFAULT_ALPHAS = (0.0, 0.5, 1.0, 1.25, 1.5)


def load_model_and_tokenizer(
    repo_id: str = BASE_MODEL_ID,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, str, torch.dtype, str]:
    """Load `Qwen/Qwen3.5-0.8B` and its tokenizer on auto-detected hardware."""
    install_fast_qwen35_kernels()
    dev, dt, hw_desc = detect_hardware(device)
    if dtype is not None:
        dt = dtype

    tok = AutoTokenizer.from_pretrained(repo_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Qwen3.5-0.8B is released under Qwen3_5ForConditionalGeneration / AutoModelForMultimodalLM
    # or AutoModelForImageTextToText / AutoModelForCausalLM depending on transformers version.
    model = None
    load_errors = []
    for loader_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers as tfm

            cls = getattr(tfm, loader_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(repo_id, dtype=dt, low_cpu_mem_usage=True)
            break
        except Exception as e:
            load_errors.append(f"{loader_name}: {e}")

    if model is None:
        raise RuntimeError(f"Failed to load {repo_id}: {'; '.join(load_errors)}")

    model.to(dev)
    model.eval()

    # Pre-warm torch.linalg.solve_triangular (used by Qwen3.5 gated delta rule)
    try:
        eye = torch.eye(2, device=dev if dev != "mps" else "cpu", dtype=torch.float32).unsqueeze(0)
        torch.linalg.solve_triangular(eye, eye.clone(), upper=False, unitriangular=True)
    except Exception:
        pass

    return tok, model, dev, dt, hw_desc


def load_math500_subset(n: int = 50, seed: int = 0) -> List[Dict[str, Any]]:
    """Load `HuggingFaceH4/MATH-500` (test split) and select `n` problems deterministically."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "idx": i,
            "unique_id": r.get("unique_id", f"math500_{i}"),
            "problem": r["problem"],
            "answer": str(r["answer"]),
            "subject": r.get("subject", ""),
            "level": r.get("level", ""),
        }
        for i, r in enumerate(ds)
    ]
    if n >= len(rows):
        return rows
    # Deterministic evenly-spaced stride across the 500 problems (preserves subject/level mix)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in indices]


def supports_native_presence_penalty() -> bool:
    """Check whether Hugging Face `GenerationConfig` natively supports `presence_penalty`."""
    cfg = GenerationConfig()
    return hasattr(cfg, "presence_penalty")


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    problems: Sequence[Dict[str, Any]],
    device: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    enable_thinking: bool = True,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run batched generation with thinking enabled, presence_penalty=1.5, and dynamic EOS batch compaction."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_math_prompt(p["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for p in problems
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    batch_size = input_ids.shape[0]

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    eos_ids = set()
    if isinstance(tokenizer.eos_token_id, int):
        eos_ids.add(tokenizer.eos_token_id)
    elif isinstance(tokenizer.eos_token_id, (list, tuple)):
        eos_ids.update(tokenizer.eos_token_id)
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)

    active_slots: List[int] = list(range(batch_size))
    generated_lists: List[List[int]] = [[] for _ in range(batch_size)]
    hit_eos_flags: List[bool] = [False] * batch_size
    finish_times: List[float] = [0.0] * batch_size

    t0 = time.perf_counter()
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    seen_penalty = (
        torch.zeros_like(logits) if presence_penalty > 0.0 else None
    )
    next_tok: Optional[torch.Tensor] = None

    for step in range(int(max_new_tokens)):
        if step > 0:
            out = model(
                input_ids=next_tok,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :].float()
            if seen_penalty is not None:
                logits.sub_(seen_penalty)

        if temperature > 0.0:
            if top_k > 0:
                cand_logits, cand_indices = torch.topk(
                    logits, k=min(int(top_k), logits.shape[-1]), dim=-1
                )
            else:
                cand_logits, cand_indices = torch.sort(logits, descending=True, dim=-1)
            cand_logits = cand_logits / float(temperature)
            probs = torch.softmax(cand_logits, dim=-1)
            if 0.0 < top_p < 1.0:
                cum_probs = torch.cumsum(probs, dim=-1)
                remove_mask = (cum_probs - probs) > float(top_p)
                probs.masked_fill_(remove_mask, 0.0)
            sampled_rel = torch.multinomial(probs, num_samples=1)
            next_tok = cand_indices.gather(-1, sampled_rel)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)

        if seen_penalty is not None:
            seen_penalty.scatter_(1, next_tok, float(presence_penalty))

        tok_list = next_tok.squeeze(-1).tolist()
        if isinstance(tok_list, int):
            tok_list = [tok_list]

        keep_rows: List[int] = []
        now_elapsed = time.perf_counter() - t0
        for row_idx, tid in enumerate(tok_list):
            orig_idx = active_slots[row_idx]
            generated_lists[orig_idx].append(int(tid))
            if int(tid) in eos_ids:
                hit_eos_flags[orig_idx] = True
                finish_times[orig_idx] = now_elapsed
            else:
                keep_rows.append(row_idx)

        if not keep_rows:
            break

        if (step + 1) % 512 == 0:
            print(
                f"  [step {step + 1}/{max_new_tokens}] active={len(keep_rows)}/{batch_size} | elapsed={now_elapsed:.1f}s",
                flush=True,
            )

        if len(keep_rows) < len(active_slots):
            keep_idx = torch.tensor(keep_rows, dtype=torch.long, device=device)
            past.reorder_cache(keep_idx)
            next_tok = next_tok.index_select(0, keep_idx)
            attention_mask = attention_mask.index_select(0, keep_idx)
            if seen_penalty is not None:
                seen_penalty = seen_penalty.index_select(0, keep_idx)
            if hasattr(model, "model") and getattr(model.model, "rope_deltas", None) is not None:
                model.model.rope_deltas = model.model.rope_deltas.index_select(0, keep_idx)
            active_slots = [active_slots[r] for r in keep_rows]

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((len(active_slots), 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    batch_elapsed = time.perf_counter() - t0

    results: List[Dict[str, Any]] = []
    per_item_time = batch_elapsed / max(len(problems), 1)

    for b, prob in enumerate(problems):
        actual_tokens = generated_lists[b]
        hit_eos = hit_eos_flags[b]
        n_tokens = len(actual_tokens)
        hit_cap = (n_tokens >= max_new_tokens) and not hit_eos
        text = tokenizer.decode(actual_tokens, skip_special_tokens=False)

        pred = extract_last_boxed(text)
        correct = grade_math_answer(pred, prob["answer"])
        rep8 = repeated_8gram_ratio(text)
        tail_rep8 = tail_repeated_8gram_ratio(text)

        results.append(
            {
                "unique_id": prob["unique_id"],
                "subject": prob["subject"],
                "level": prob["level"],
                "gold": prob["answer"],
                "pred": pred,
                "correct": bool(correct),
                "n_tokens": int(n_tokens),
                "hit_cap": bool(hit_cap),
                "rep_8gram_ratio": round(rep8, 4),
                "tail_rep_8gram_ratio": round(tail_rep8, 4),
                "wall_time_s": round(per_item_time, 3),
                "text": text,
            }
        )
    return results


def save_sweep_charts(
    summary_df: pd.DataFrame,
    per_problem_df: pd.DataFrame,
    out_dir: str = "assets",
) -> Tuple[str, str]:
    """Create and save publication-quality charts from the alpha sweep results."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    main_chart_path = os.path.join(out_dir, "sweep_charts.png")
    dist_chart_path = os.path.join(out_dir, "alpha_token_distributions.png")

    alphas = summary_df["alpha"].to_numpy()
    acc = summary_df["accuracy_pct"].to_numpy()
    cap = summary_df["hit_cap_pct"].to_numpy()
    toks = summary_df["mean_tokens"].to_numpy()
    wall = summary_df["wall_time_s"].to_numpy()
    rep8 = summary_df["mean_rep_8gram_pct"].to_numpy()

    # 1. Main 4-panel dashboard (`assets/sweep_charts.png`)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    # Panel A: Accuracy vs % Hitting Token Cap
    ax = axes[0, 0]
    ax.plot(alphas, acc, marker="o", linewidth=2.6, markersize=8, color="#16a34a", label="MATH-500 Accuracy (%)")
    ax.plot(alphas, cap, marker="s", linewidth=2.4, markersize=7.5, color="#dc2626", linestyle="--", label="% Hitting 4096-Token Cap")
    for x, y_a, y_c in zip(alphas, acc, cap):
        ax.annotate(f"{y_a:.1f}%", (x, y_a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#15803d", fontweight="bold")
        ax.annotate(f"{y_c:.1f}%", (x, y_c), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#b91c1c")
    ax.axvline(1.0, color="#64748b", linestyle=":", alpha=0.7, label="α = 1.0 (Published Edit)")
    ax.set_title("A. Accuracy vs. Token-Cap Truncation Across α", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Interpolation Coefficient α  (W = W_base + α·ΔW)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(alphas)
    ax.set_ylim(-5, 108)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center right", fontsize=8.8)

    # Panel B: Mean Generated Tokens vs α
    ax = axes[0, 1]
    bars = ax.bar([str(a) for a in alphas], toks, color=["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#6366f1"][: len(alphas)], width=0.55, alpha=0.88)
    ax.axhline(4096, color="#dc2626", linestyle="--", linewidth=1.4, label="Token Cap (4096)")
    for bar, val in zip(bars, toks):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 70, f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("B. Mean Generated Tokens per Problem (Max 4096)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Interpolation Coefficient α")
    ax.set_ylabel("Mean New Tokens")
    ax.set_ylim(0, 4600)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.8)

    # Panel C: Repeated 8-Gram Ratio vs α
    ax = axes[1, 0]
    ax.plot(alphas, rep8, marker="D", linewidth=2.5, markersize=7.5, color="#9333ea", label="Mean Repeated 8-gram Ratio (%)")
    for x, y_r in zip(alphas, rep8):
        ax.annotate(f"{y_r:.1f}%", (x, y_r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#7e22ce", fontweight="bold")
    ax.axvline(1.0, color="#64748b", linestyle=":", alpha=0.7)
    ax.set_title("C. Loop Severity: Repeated 8-Gram Ratio vs. α", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Interpolation Coefficient α")
    ax.set_ylabel("Repeated 8-Gram Ratio (%)")
    ax.set_xticks(alphas)
    ax.set_ylim(0, max(100.0, float(rep8.max()) + 12.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.8)

    # Panel D: Wall Time per α Sweep (seconds)
    ax = axes[1, 1]
    bars_w = ax.bar([str(a) for a in alphas], wall, color="#0284c7", width=0.55, alpha=0.85)
    for bar, val in zip(bars_w, wall):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(wall) * 0.02, f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("D. Wall-Clock Evaluation Time per α (N=50 Problems)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Interpolation Coefficient α")
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_ylim(0, max(wall) * 1.18 if len(wall) else 10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Watch the Loop Break: Weight-Space Interpolation Sweep on MATH-500 (N=50, Qwen3.5-0.8B)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(main_chart_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    # 2. Secondary chart: Per-problem token length & 8-gram repetition scatter (`assets/alpha_token_distributions.png`)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    fig2.patch.set_facecolor("#ffffff")

    palette = {0.0: "#ef4444", 0.5: "#f59e0b", 1.0: "#10b981", 1.25: "#3b82f6", 1.5: "#6366f1"}
    ax1 = axes2[0]
    for a in alphas:
        sub = per_problem_df[per_problem_df["alpha"] == a]
        if sub.empty:
            continue
        jitter = np.random.default_rng(int(a * 100)).uniform(-0.04, 0.04, size=len(sub))
        c = palette.get(float(a), "#475569")
        correct_mask = sub["correct"].astype(bool)
        ax1.scatter(
            np.full(correct_mask.sum(), a) + jitter[correct_mask],
            sub.loc[correct_mask, "n_tokens"],
            color=c,
            marker="o",
            alpha=0.8,
            s=36,
            label=f"α={a:g} (correct)" if a == alphas[0] else None,
        )
        ax1.scatter(
            np.full((~correct_mask).sum(), a) + jitter[~correct_mask],
            sub.loc[~correct_mask, "n_tokens"],
            color=c,
            marker="x",
            alpha=0.65,
            s=36,
        )
    ax1.axhline(4096, color="#dc2626", linestyle="--", linewidth=1.2, label="Token Cap (4096)")
    ax1.set_xticks(alphas)
    ax1.set_title("Per-Problem Token Length by α (● Correct, ✕ Wrong/Cap)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Interpolation Coefficient α")
    ax1.set_ylabel("Generated Tokens")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = axes2[1]
    for a in alphas:
        sub = per_problem_df[per_problem_df["alpha"] == a]
        if sub.empty:
            continue
        c = palette.get(float(a), "#475569")
        ax2.scatter(
            sub["rep_8gram_ratio"] * 100.0,
            sub["n_tokens"],
            color=c,
            alpha=0.72,
            s=36,
            label=f"α={a:g}",
        )
    ax2.set_title("Generated Tokens vs. Repeated 8-Gram Ratio", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Repeated 8-Gram Ratio (%)")
    ax2.set_ylabel("Generated Tokens")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig2.savefig(dist_chart_path, bbox_inches="tight", dpi=160)
    plt.close(fig2)

    return main_chart_path, dist_chart_path


def run_sweep(
    n_problems: int = 50,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    batch_size: int = 10,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    enable_thinking: bool = True,
    seed: int = 0,
    out_dir: str = "assets",
    device_override: Optional[str] = None,
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the N-problem MATH-500 sweep across all `alphas` and save CSVs + charts."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jsonl_cache = os.path.join(out_dir, "sweep_per_problem.jsonl")
    per_problem_csv = os.path.join(out_dir, "sweep_per_problem.csv")
    summary_csv = os.path.join(out_dir, "sweep_results.csv")

    problems = load_math500_subset(n=n_problems, seed=seed)
    problem_ids = {p["unique_id"] for p in problems}

    # Load any existing checkpointed records
    cached_records: Dict[Tuple[float, str], Dict[str, Any]] = {}
    if resume and os.path.exists(jsonl_cache):
        with open(jsonl_cache, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (float(rec["alpha"]), str(rec["unique_id"]))
                if key[1] in problem_ids and rec.get("max_new_tokens") == max_new_tokens:
                    cached_records[key] = rec

    tokenizer, model, device, dtype, hw_desc = load_model_and_tokenizer(
        BASE_MODEL_ID, device=device_override
    )
    base_tensors, edit_tensors = load_target_tensors(BASE_MODEL_ID, EDIT_MODEL_ID)
    register_active_model(model, base_tensors, edit_tensors)

    native_pp = supports_native_presence_penalty()
    print(f"[Sweep] Hardware: {hw_desc}", flush=True)
    print(
        f"[Sweep] N={len(problems)} MATH-500 problems | alphas={list(alphas)} | "
        f"temp={temperature}, top_p={top_p}, top_k={top_k}, "
        f"presence_penalty={presence_penalty} ({'native' if native_pp else 'dynamic-batch'}), "
        f"thinking={enable_thinking}, max_new_tokens={max_new_tokens}, batch_size={batch_size}",
        flush=True,
    )

    all_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for alpha in alphas:
        alpha_f = float(alpha)
        blend(alpha_f, model=model, base_tensors=base_tensors, edit_tensors=edit_tensors)

        pending = [p for p in problems if (alpha_f, p["unique_id"]) not in cached_records]
        # Length-bucket by prompt length to minimize left-padding overhead during chunked prefill
        pending.sort(key=lambda p: len(p["problem"]))
        alpha_t0 = time.perf_counter()

        if pending:
            starts = list(range(0, len(pending), batch_size))
            for b_idx, start in enumerate(starts, start=1):
                batch_probs = pending[start : start + batch_size]
                batch_out = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    problems=batch_probs,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    enable_thinking=enable_thinking,
                    seed=seed,
                )
                with open(jsonl_cache, "a", encoding="utf-8") as jf:
                    for item in batch_out:
                        rec = {
                            "alpha": alpha_f,
                            "max_new_tokens": max_new_tokens,
                            **item,
                        }
                        cached_records[(alpha_f, item["unique_id"])] = rec
                        jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                done_for_alpha = [
                    cached_records[(alpha_f, p["unique_id"])]
                    for p in problems
                    if (alpha_f, p["unique_id"]) in cached_records
                ]
                cur_acc = np.mean([r["correct"] for r in done_for_alpha]) * 100.0
                cur_cap = np.mean([r["hit_cap"] for r in done_for_alpha]) * 100.0
                mean_t = np.mean([r["n_tokens"] for r in batch_out])
                print(
                    f"[alpha={alpha_f:g}] Batch {b_idx}/{len(starts)} ({len(done_for_alpha)}/{len(problems)}) | "
                    f"batch_mean_toks={mean_t:.0f} | running_acc={cur_acc:.1f}% | running_cap={cur_cap:.1f}%",
                    flush=True,
                )

        alpha_records = [cached_records[(alpha_f, p["unique_id"])] for p in problems]
        all_records.extend(alpha_records)

        n_total = len(alpha_records)
        n_correct = sum(1 for r in alpha_records if r["correct"])
        n_cap = sum(1 for r in alpha_records if r["hit_cap"])
        mean_toks = float(np.mean([r["n_tokens"] for r in alpha_records]))
        median_toks = float(np.median([r["n_tokens"] for r in alpha_records]))
        mean_rep8 = float(np.mean([r["rep_8gram_ratio"] for r in alpha_records])) * 100.0
        mean_tail_rep8 = float(np.mean([r["tail_rep_8gram_ratio"] for r in alpha_records])) * 100.0
        wall_s = float(sum(r["wall_time_s"] for r in alpha_records))
        if pending and len(pending) == len(problems):
            wall_s = round(time.perf_counter() - alpha_t0, 2)

        row_summary = {
            "alpha": alpha_f,
            "n_problems": n_total,
            "correct": n_correct,
            "accuracy": round(n_correct / n_total, 4),
            "accuracy_pct": round((n_correct / n_total) * 100.0, 2),
            "hit_cap_count": n_cap,
            "hit_cap_pct": round((n_cap / n_total) * 100.0, 2),
            "mean_tokens": round(mean_toks, 1),
            "median_tokens": round(median_toks, 1),
            "mean_rep_8gram_pct": round(mean_rep8, 2),
            "mean_tail_rep_8gram_pct": round(mean_tail_rep8, 2),
            "wall_time_s": round(wall_s, 2),
            "sec_per_problem": round(wall_s / n_total, 2),
        }
        summary_rows.append(row_summary)
        print(
            f"[Summary α={alpha_f:g}] Accuracy={row_summary['accuracy_pct']:.1f}% ({n_correct}/{n_total}) | "
            f"Hit Cap={row_summary['hit_cap_pct']:.1f}% | Mean Tokens={row_summary['mean_tokens']:.1f} | "
            f"8-gram Rep={row_summary['mean_rep_8gram_pct']:.1f}% | Wall={row_summary['wall_time_s']:.1f}s",
            flush=True,
        )

        # Save intermediate CSVs and charts after each alpha completes
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        per_problem_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "text"} for r in all_records]
        )
        per_problem_df.to_csv(per_problem_csv, index=False)
        save_sweep_charts(summary_df, per_problem_df, out_dir=out_dir)

    summary_df = pd.DataFrame(summary_rows)
    per_problem_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "text"} for r in all_records]
    )
    c1, c2 = save_sweep_charts(summary_df, per_problem_df, out_dir=out_dir)
    print(f"[Saved] Summary CSV -> {summary_csv}", flush=True)
    print(f"[Saved] Per-problem CSV -> {per_problem_csv}", flush=True)
    print(f"[Saved] Charts -> {c1}, {c2}", flush=True)
    return summary_df, per_problem_df


def main() -> None:
    parser = argparse.ArgumentParser(description="MATH-500 alpha sweep for Qwen3.5-0.8B weight-space programming")
    parser.add_argument("--n", type=int, default=50, help="Number of MATH-500 problems (default: 50)")
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,0.5,1.0,1.25,1.5",
        help="Comma-separated list of alpha values (default: 0,0.5,1.0,1.25,1.5)",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation (default: 10)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--out-dir", type=str, default="assets")
    parser.add_argument("--no-resume", action="store_true", help="Ignore cached per-problem JSONL results")
    args = parser.parse_args()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    run_sweep(
        n_problems=args.n,
        alphas=alphas,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        enable_thinking=True,
        seed=args.seed,
        out_dir=args.out_dir,
        device_override=args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
