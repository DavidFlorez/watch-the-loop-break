"""Single-alpha vLLM MATH-500 evaluation runner with forced-answer pass and locked atomic resume."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from gpu.common import (
    FORCE_ANSWER_SUFFIX,
    FORCED_MAX_TOKENS,
    SAMPLING_DEFAULTS,
    STOP_TOKEN_IDS,
    atomic_write_gzip_jsonl,
    atomic_write_text,
    check_min_compute_capability,
    cleanup_ckpt_dir,
    collect_environment_metadata,
    create_vllm_engine,
    file_lock,
    make_scratch_ckpt_dir,
    make_tokens_prompt,
    release_vllm_engine,
    utc_now_iso,
)
from gpu.make_ckpt import build_checkpoint
from metrics import (
    build_math_prompt,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
    wilson_ci,
)
from sweep import load_math500_subset, save_sweep_charts
from weights import BASE_MODEL_ID, EDIT_MODEL_ID, TARGET_TENSORS

ResumeKey = Tuple[float, str, int, int]


def _record_key(rec: Dict[str, Any]) -> ResumeKey:
    return (
        float(rec["alpha"]),
        str(rec["unique_id"]),
        int(rec.get("sample_idx", 0)),
        int(rec["max_new_tokens"]),
    )


def _load_existing_records(jsonl_gz_path: Path) -> Dict[ResumeKey, Dict[str, Any]]:
    records: Dict[ResumeKey, Dict[str, Any]] = {}
    if not jsonl_gz_path.is_file():
        return records
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[_record_key(rec)] = rec
    return records


def resolve_dtype_choice(
    dtype_arg: str,
    out_root: Union[str, Path],
    gpu_tag: str,
) -> str:
    """Resolve `--dtype` (`float16`, `float32`, or `auto` via `assets/gpu_smoke_<gpu>.json`)."""
    choice = (dtype_arg or "float16").strip().lower()
    if choice in ("float16", "float32"):
        return choice
    if choice == "auto":
        smoke_path = Path(out_root) / f"gpu_smoke_{gpu_tag}.json"
        if smoke_path.is_file():
            with open(smoke_path, "r", encoding="utf-8") as f:
                smoke_data = json.load(f)
            rec = smoke_data.get("recommended_dtype")
            if rec in ("float16", "float32"):
                print(f"[vLLM Sweep] Resolved dtype='{rec}' from {smoke_path}", flush=True)
                return rec
        print(
            f"[vLLM Sweep] Warning: {smoke_path} not found or incomplete; defaulting dtype='float32'",
            flush=True,
        )
        return "float32"
    raise ValueError(f"Unsupported dtype '{dtype_arg}'; expected float16, float32, or auto.")


def _make_sampling_params(
    *,
    n: int,
    max_tokens: int,
    seed: int,
    temperature: float = SAMPLING_DEFAULTS["temperature"],
    top_p: float = SAMPLING_DEFAULTS["top_p"],
    top_k: int = SAMPLING_DEFAULTS["top_k"],
    presence_penalty: float = SAMPLING_DEFAULTS["presence_penalty"],
) -> Any:
    try:
        from vllm import SamplingParams

        try:
            return SamplingParams(
                n=int(n),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                presence_penalty=float(presence_penalty),
                max_tokens=int(max_tokens),
                seed=int(seed),
                stop_token_ids=list(STOP_TOKEN_IDS),
                skip_special_tokens=False,
            )
        except TypeError:
            return SamplingParams(
                n=int(n),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                presence_penalty=float(presence_penalty),
                max_tokens=int(max_tokens),
                seed=int(seed),
            )
    except ImportError as exc:
        raise RuntimeError(
            "vllm is not installed. Run inside the GPU environment with `requirements-gpu.txt` installed."
        ) from exc


def build_summary_dataframe(
    all_records: Sequence[Dict[str, Any]],
    problems: Sequence[Dict[str, Any]],
    *,
    budget: int,
    samples: int,
    gpu_tag: str,
    dtype: str,
    vllm_version: str,
) -> pd.DataFrame:
    """Compute per-alpha summary rows (compatible with `sweep.py` plus strict/forced metrics & Wilson CIs)."""
    n_problems = len(problems)
    problem_ids = {p["unique_id"] for p in problems}
    expected_per_alpha = n_problems * int(samples)

    filtered = [
        r
        for r in all_records
        if int(r["max_new_tokens"]) == int(budget) and str(r["unique_id"]) in problem_ids
    ]
    candidate_alphas = sorted({float(r["alpha"]) for r in filtered})
    summary_rows: List[Dict[str, Any]] = []

    for a_val in candidate_alphas:
        a_recs = [r for r in filtered if float(r["alpha"]) == a_val]
        seen_pairs = {(str(r["unique_id"]), int(r.get("sample_idx", 0))) for r in a_recs}
        if len(seen_pairs) < expected_per_alpha:
            # Skip incomplete alphas in `sweep_results.csv` until all (problem, sample_idx) finish
            continue

        n_evals = len(a_recs)
        n_strict_corr = sum(1 for r in a_recs if bool(r["correct_strict"]))
        n_forced_corr = sum(1 for r in a_recs if bool(r["correct_forced"]))
        n_cap_cnt = sum(1 for r in a_recs if bool(r["hit_cap"]))
        n_closed_think = sum(
            1 for r in a_recs if bool(r.get("closed_think", "</think>" in str(r.get("text", ""))))
        )

        acc_strict_frac = n_strict_corr / max(1, n_evals)
        acc_forced_frac = n_forced_corr / max(1, n_evals)
        cap_frac = n_cap_cnt / max(1, n_evals)

        acc_lo, acc_hi = wilson_ci(n_strict_corr, n_evals)
        forced_lo, forced_hi = wilson_ci(n_forced_corr, n_evals)
        cap_lo, cap_hi = wilson_ci(n_cap_cnt, n_evals)

        m_toks = float(np.mean([int(r["n_tokens"]) for r in a_recs]))
        med_toks = float(np.median([int(r["n_tokens"]) for r in a_recs]))
        m_forced_toks = float(np.mean([int(r.get("forced_tokens", 0)) for r in a_recs]))
        m_rep8 = float(np.mean([float(r["rep_8gram_ratio"]) for r in a_recs])) * 100.0
        m_tail_rep8 = float(np.mean([float(r["tail_rep_8gram_ratio"]) for r in a_recs])) * 100.0
        wall_s = float(sum(float(r["wall_time_s"]) for r in a_recs))
        total_out_toks = sum(int(r["n_tokens"]) + int(r.get("forced_tokens", 0)) for r in a_recs)

        summary_rows.append(
            {
                "alpha": a_val,
                "n_problems": n_problems,
                "n_samples": int(samples),
                "total_evaluations": n_evals,
                "correct": int(n_strict_corr),
                "accuracy": round(acc_strict_frac, 4),
                "accuracy_pct": round(acc_strict_frac * 100.0, 2),
                "accuracy_ci95_low": round(acc_lo * 100.0, 2),
                "accuracy_ci95_high": round(acc_hi * 100.0, 2),
                "hit_cap_count": int(n_cap_cnt),
                "hit_cap_pct": round(cap_frac * 100.0, 2),
                "hit_cap_ci95_low": round(cap_lo * 100.0, 2),
                "hit_cap_ci95_high": round(cap_hi * 100.0, 2),
                "forced_correct": int(n_forced_corr),
                "forced_accuracy": round(acc_forced_frac, 4),
                "forced_accuracy_pct": round(acc_forced_frac * 100.0, 2),
                "forced_accuracy_ci95_low": round(forced_lo * 100.0, 2),
                "forced_accuracy_ci95_high": round(forced_hi * 100.0, 2),
                "mean_tokens": round(m_toks, 1),
                "median_tokens": round(med_toks, 1),
                "mean_forced_tokens": round(m_forced_toks, 1),
                "closed_think_pct": round((n_closed_think / max(1, n_evals)) * 100.0, 2),
                "mean_rep_8gram_pct": round(m_rep8, 2),
                "mean_tail_rep_8gram_pct": round(m_tail_rep8, 2),
                "wall_time_s": round(wall_s, 2),
                "sec_per_problem": round(wall_s / max(1, n_problems), 2),
                "output_tok_per_s": round(total_out_toks / max(wall_s, 1e-6), 2),
                "backend": "vllm",
                "gpu": gpu_tag,
                "dtype": dtype,
                "vllm_version": vllm_version,
            }
        )

    if not summary_rows:
        return pd.DataFrame()
    return pd.DataFrame(summary_rows).sort_values("alpha").reset_index(drop=True)


def run_vllm_alpha(
    *,
    alpha: float,
    budget: int = 8192,
    n_problems: int = 50,
    samples: int = 1,
    seed: int = 0,
    dtype: str = "float16",
    gpu_tag: Optional[str] = None,
    out_root: str = "assets",
    enforce_eager: bool = False,
    gpu_mem: float = 0.85,
    ckpt_dir: Optional[str] = None,
    keep_ckpt: bool = False,
    llm_cls: Optional[Any] = None,
    sampling_params_factory: Optional[Any] = None,
    tokenizer_override: Optional[Any] = None,
    require_gpu: bool = True,
) -> Dict[str, Any]:
    """Run Pass 1 (normal generation) + Pass 2 (forced answer on capped traces) for a single `alpha`."""
    alpha_f = float(alpha)
    check_min_compute_capability(min_cc=7.5, require_gpu=require_gpu)
    env_meta = collect_environment_metadata(gpu_tag_override=gpu_tag)
    resolved_gpu = env_meta["gpu_tag"]
    resolved_dtype = resolve_dtype_choice(dtype, out_root=out_root, gpu_tag=resolved_gpu)

    out_dir = Path(out_root) / f"budget_{int(budget)}" / f"gpu-{resolved_gpu}-{resolved_dtype}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".write.lock"
    jsonl_gz_path = out_dir / "sweep_per_problem.jsonl.gz"
    per_problem_csv = out_dir / "sweep_per_problem.csv"
    summary_csv = out_dir / "sweep_results.csv"
    meta_path = out_dir / "meta.json"

    problems = load_math500_subset(n=n_problems, seed=seed)
    problem_rank = {p["unique_id"]: idx for idx, p in enumerate(problems)}

    # Check if all samples for this alpha & budget are already complete before touching weights or GPU
    with file_lock(lock_path):
        existing = _load_existing_records(jsonl_gz_path)

    pending_problems: List[Dict[str, Any]] = []
    missing_sample_indices_by_uid: Dict[str, List[int]] = {}
    for p in problems:
        uid = str(p["unique_id"])
        missing_s = [
            s_idx
            for s_idx in range(int(samples))
            if (alpha_f, uid, s_idx, int(budget)) not in existing
        ]
        if missing_s:
            pending_problems.append(p)
            missing_sample_indices_by_uid[uid] = missing_s

    owned_scratch_ckpt: Optional[Path] = None
    new_records: List[Dict[str, Any]] = []
    alpha_run_meta: Optional[Dict[str, Any]] = None
    engine_meta: Optional[Dict[str, Any]] = None
    llm = None

    try:
        if pending_problems:
            if ckpt_dir is not None:
                ckpt_path = Path(ckpt_dir).resolve()
                if not ckpt_path.is_dir():
                    raise FileNotFoundError(f"Provided --ckpt-dir does not exist: {ckpt_path}")
            else:
                ckpt_path = make_scratch_ckpt_dir(alpha_f, prefix="vllm")
                owned_scratch_ckpt = ckpt_path
                print(f"[vLLM Sweep] Building scratch checkpoint for alpha={alpha_f:g} -> {ckpt_path}", flush=True)
                build_checkpoint(alpha=alpha_f, out_dir=ckpt_path)

            if tokenizer_override is not None:
                tokenizer = tokenizer_override
            else:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))

            prompts_str: List[str] = []
            prompt_ids_list: List[List[int]] = []
            for p in pending_problems:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_math_prompt(p["problem"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                assert formatted.endswith("<think>\n"), (
                    f"Prompt must end with '<think>\\n', got {formatted[-40:]!r}"
                )
                prompts_str.append(formatted)
                prompt_ids_list.append(list(tokenizer.encode(formatted, add_special_tokens=False)))

            max_model_len = int(budget) + 2048
            print(
                f"[vLLM Sweep] Initializing engine | alpha={alpha_f:g} | budget={budget} | "
                f"N={len(pending_problems)}x{samples} | dtype={resolved_dtype} | gpu={resolved_gpu}",
                flush=True,
            )
            llm, engine_meta = create_vllm_engine(
                ckpt_dir=ckpt_path,
                dtype=resolved_dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_mem,
                seed=seed,
                enforce_eager=enforce_eager,
                llm_cls=llm_cls,
            )
            vllm_version = engine_meta["vllm_version"]

            # Pass 1: Normal generation with presence_penalty=1.5, temperature=0.6, top_p=0.95, top_k=20
            sp_pass1 = (
                sampling_params_factory(
                    n=int(samples),
                    max_tokens=int(budget),
                    seed=int(seed),
                    **SAMPLING_DEFAULTS,
                )
                if sampling_params_factory is not None
                else _make_sampling_params(n=int(samples), max_tokens=int(budget), seed=int(seed))
            )

            t0_p1 = time.perf_counter()
            outputs_p1 = llm.generate(prompts_str, sp_pass1)
            pass1_wall_s = time.perf_counter() - t0_p1

            staged_items: List[Dict[str, Any]] = []
            capped_indices: List[int] = []
            pass1_out_tokens = 0

            for p, p_ids, req_out in zip(pending_problems, prompt_ids_list, outputs_p1):
                uid = str(p["unique_id"])
                needed_samples = set(missing_sample_indices_by_uid[uid])
                for s_idx, comp in enumerate(req_out.outputs):
                    if s_idx not in needed_samples:
                        continue
                    out_ids = list(comp.token_ids)
                    n_toks = len(out_ids)
                    pass1_out_tokens += n_toks
                    finish_reason = str(getattr(comp, "finish_reason", "stop"))
                    hit_cap = finish_reason == "length"
                    text = (
                        comp.text
                        if getattr(comp, "text", None) is not None
                        else tokenizer.decode(out_ids, skip_special_tokens=False)
                    )
                    closed_think = "</think>" in text
                    pred = extract_last_boxed(text)
                    corr = bool(grade_math_answer(pred, p["answer"]))
                    pred_strict = extract_last_boxed_strict(text)
                    corr_strict = bool(grade_math_answer(pred_strict, p["answer"]))
                    rep8 = round(repeated_8gram_ratio(text), 4)
                    tail_rep8 = round(tail_repeated_8gram_ratio(text), 4)

                    item_dict = {
                        "problem": p,
                        "prompt_ids": p_ids,
                        "sample_idx": int(s_idx),
                        "out_ids": out_ids,
                        "n_tokens": n_toks,
                        "finish_reason": finish_reason,
                        "hit_cap": hit_cap,
                        "text": text,
                        "closed_think": closed_think,
                        "pred": pred,
                        "correct": corr,
                        "pred_strict": pred_strict,
                        "correct_strict": corr_strict,
                    }
                    staged_idx = len(staged_items)
                    staged_items.append(item_dict)
                    if hit_cap:
                        capped_indices.append(staged_idx)

            per_p1_wall = pass1_wall_s / max(1, len(staged_items))

            # Pass 2: Forced-answer pass on the same LLM & alpha for every sample that hit `budget`
            pass2_wall_s = 0.0
            pass2_out_tokens = 0
            if capped_indices:
                force_suffix_ids = list(tokenizer.encode(FORCE_ANSWER_SUFFIX, add_special_tokens=False))
                forced_inputs: List[Any] = []
                for idx_s in capped_indices:
                    it = staged_items[idx_s]
                    combined_ids = it["prompt_ids"] + it["out_ids"] + force_suffix_ids
                    forced_inputs.append(make_tokens_prompt(combined_ids))

                sp_pass2 = (
                    sampling_params_factory(
                        n=1,
                        max_tokens=FORCED_MAX_TOKENS,
                        seed=int(seed),
                        **SAMPLING_DEFAULTS,
                    )
                    if sampling_params_factory is not None
                    else _make_sampling_params(n=1, max_tokens=FORCED_MAX_TOKENS, seed=int(seed))
                )

                print(
                    f"[vLLM Sweep] Running forced-answer Pass 2 on {len(capped_indices)} capped sample(s) "
                    f"(max_tokens={FORCED_MAX_TOKENS}) ...",
                    flush=True,
                )
                t0_p2 = time.perf_counter()
                outputs_p2 = llm.generate(forced_inputs, sp_pass2)
                pass2_wall_s = time.perf_counter() - t0_p2
                per_p2_wall = pass2_wall_s / max(1, len(capped_indices))

                for idx_s, req_p2 in zip(capped_indices, outputs_p2):
                    it = staged_items[idx_s]
                    comp2 = req_p2.outputs[0]
                    f_ids = list(comp2.token_ids)
                    f_toks = len(f_ids)
                    pass2_out_tokens += f_toks
                    f_text = (
                        comp2.text
                        if getattr(comp2, "text", None) is not None
                        else tokenizer.decode(f_ids, skip_special_tokens=False)
                    )
                    pred_f = extract_last_boxed(f_text)
                    corr_f = bool(grade_math_answer(pred_f, it["problem"]["answer"]))

                    it["forced_tokens"] = f_toks
                    it["pred_forced"] = pred_f
                    it["correct_forced"] = corr_f
                    it["wall_time_s"] = round(per_p1_wall + per_p2_wall, 3)

            for it in staged_items:
                if not it["hit_cap"]:
                    it["forced_tokens"] = 0
                    it["pred_forced"] = it["pred_strict"]
                    it["correct_forced"] = it["correct_strict"]
                    it["wall_time_s"] = round(per_p1_wall, 3)

                p = it["problem"]
                new_records.append(
                    {
                        "alpha": alpha_f,
                        "max_new_tokens": int(budget),
                        "unique_id": str(p["unique_id"]),
                        "sample_idx": int(it["sample_idx"]),
                        "subject": str(p["subject"]),
                        "level": int(p["level"]),
                        "gold": str(p["answer"]),
                        "pred": it["pred"],
                        "correct": bool(it["correct"]),
                        "pred_strict": it["pred_strict"],
                        "correct_strict": bool(it["correct_strict"]),
                        "pred_forced": it["pred_forced"],
                        "correct_forced": bool(it["correct_forced"]),
                        "forced_tokens": int(it["forced_tokens"]),
                        "n_tokens": int(it["n_tokens"]),
                        "hit_cap": bool(it["hit_cap"]),
                        "finish_reason": str(it["finish_reason"]),
                        "closed_think": bool(it["closed_think"]),
                        "rep_8gram_ratio": round(repeated_8gram_ratio(it["text"]), 4),
                        "tail_rep_8gram_ratio": round(tail_repeated_8gram_ratio(it["text"]), 4),
                        "wall_time_s": float(it["wall_time_s"]),
                        "backend": "vllm",
                        "gpu": resolved_gpu,
                        "dtype": resolved_dtype,
                        "vllm_version": vllm_version,
                        "text": it["text"],
                    }
                )

            total_wall = round(pass1_wall_s + pass2_wall_s, 2)
            total_out_toks = int(pass1_out_tokens + pass2_out_tokens)
            alpha_run_meta = {
                "alpha": alpha_f,
                "n_generated_samples": len(staged_items),
                "n_capped_forced": len(capped_indices),
                "pass1_wall_s": round(pass1_wall_s, 2),
                "pass2_wall_s": round(pass2_wall_s, 2),
                "total_wall_s": total_wall,
                "pass1_output_tokens": int(pass1_out_tokens),
                "pass2_output_tokens": int(pass2_out_tokens),
                "total_output_tokens": total_out_toks,
                "output_tok_per_s": round(total_out_toks / max(total_wall, 1e-6), 2),
                "updated_at_utc": utc_now_iso(),
            }
        else:
            print(
                f"[vLLM Sweep] All {n_problems}x{samples} samples for alpha={alpha_f:g}, "
                f"budget={budget} already cached in {jsonl_gz_path}; refreshing summaries.",
                flush=True,
            )
    finally:
        release_vllm_engine(llm)
        if owned_scratch_ckpt is not None and not keep_ckpt:
            print(f"[vLLM Sweep] Cleaning up scratch checkpoint {owned_scratch_ckpt}", flush=True)
            cleanup_ckpt_dir(owned_scratch_ckpt)

    # Locked atomic merge & summary/chart regeneration (safe under parallel 2-GPU execution)
    with file_lock(lock_path):
        merged = _load_existing_records(jsonl_gz_path)
        for rec in new_records:
            merged[_record_key(rec)] = rec

        sorted_records = sorted(
            merged.values(),
            key=lambda r: (
                float(r["alpha"]),
                problem_rank.get(str(r["unique_id"]), 999999),
                str(r["unique_id"]),
                int(r.get("sample_idx", 0)),
            ),
        )
        atomic_write_gzip_jsonl(jsonl_gz_path, sorted_records)

        if sorted_records:
            p_rows = [{k: v for k, v in r.items() if k != "text"} for r in sorted_records]
            p_df = pd.DataFrame(p_rows)
            atomic_write_text(per_problem_csv, p_df.to_csv(index=False))

            vllm_ver_str = str(sorted_records[0].get("vllm_version") or env_meta["package_versions"].get("vllm") or "unknown")
            s_df = build_summary_dataframe(
                sorted_records,
                problems,
                budget=int(budget),
                samples=int(samples),
                gpu_tag=resolved_gpu,
                dtype=resolved_dtype,
                vllm_version=vllm_ver_str,
            )
            if not s_df.empty:
                atomic_write_text(summary_csv, s_df.to_csv(index=False))
                save_sweep_charts(s_df, p_df, out_dir=str(out_dir), max_new_tokens=int(budget))
        else:
            s_df = pd.DataFrame()
            vllm_ver_str = str(env_meta["package_versions"].get("vllm") or "unknown")

        # Update meta.json preserving other parallel alphas' timing telemetry
        existing_meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}

        alpha_stats = dict(existing_meta.get("alpha_stats", {}))
        if alpha_run_meta is not None:
            alpha_stats[f"{alpha_f:g}"] = alpha_run_meta

        agg_wall = round(sum(float(v.get("total_wall_s", 0.0)) for v in alpha_stats.values()), 2)
        agg_toks = int(sum(int(v.get("total_output_tokens", 0)) for v in alpha_stats.values()))
        agg_tps = round(agg_toks / max(agg_wall, 1e-6), 2) if agg_wall > 0 else 0.0

        prior_engine = dict(existing_meta.get("engine_config", {}))
        merged_fallbacks = list(prior_engine.get("fallbacks_triggered", []))
        if engine_meta is not None:
            for fb in engine_meta.get("fallbacks_triggered", []):
                if fb not in merged_fallbacks:
                    merged_fallbacks.append(fb)
            final_engine_cfg = {**engine_meta, "fallbacks_triggered": merged_fallbacks}
            final_engine_cfg.pop("model", None)
        else:
            final_engine_cfg = prior_engine

        meta_payload = {
            "updated_at_utc": utc_now_iso(),
            "backend": "vllm",
            "model_base": BASE_MODEL_ID,
            "model_edit": EDIT_MODEL_ID,
            "target_tensors": list(TARGET_TENSORS),
            "budget": int(budget),
            "n_problems": int(n_problems),
            "samples": int(samples),
            "seed": int(seed),
            "gpu_tag": resolved_gpu,
            "dtype": resolved_dtype,
            "vllm_version": vllm_ver_str,
            "sampling_pass1": {
                **SAMPLING_DEFAULTS,
                "n": int(samples),
                "max_tokens": int(budget),
                "seed": int(seed),
                "stop_token_ids": list(STOP_TOKEN_IDS),
                "enable_thinking": True,
            },
            "sampling_pass2_forced": {
                **SAMPLING_DEFAULTS,
                "n": 1,
                "max_tokens": FORCED_MAX_TOKENS,
                "seed": int(seed),
                "stop_token_ids": list(STOP_TOKEN_IDS),
                "force_suffix": FORCE_ANSWER_SUFFIX,
            },
            "engine_config": final_engine_cfg,
            "alpha_stats": alpha_stats,
            "aggregate_wall_time_s": agg_wall,
            "aggregate_output_tokens": agg_toks,
            "aggregate_output_tok_per_s": agg_tps,
            "environment": env_meta,
        }
        atomic_write_text(meta_path, json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n")

    if not s_df.empty:
        row_a = s_df[s_df["alpha"] == alpha_f]
        if not row_a.empty:
            r0 = row_a.iloc[0]
            print(
                f"[vLLM Sweep] Done alpha={alpha_f:g} | strict_acc={r0['accuracy_pct']:.2f}% | "
                f"forced_acc={r0['forced_accuracy_pct']:.2f}% | hit_cap={r0['hit_cap_pct']:.2f}% | "
                f"mean_tok={r0['mean_tokens']:.1f} | tok/s={r0['output_tok_per_s']:.1f}",
                flush=True,
            )

    return {
        "out_dir": str(out_dir),
        "alpha": alpha_f,
        "budget": int(budget),
        "gpu": resolved_gpu,
        "dtype": resolved_dtype,
        "new_records_count": len(new_records),
        "summary_rows": int(len(s_df)),
        "meta_path": str(meta_path),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run single-alpha MATH-500 vLLM evaluation with Pass 2 forced answer on capped traces."
    )
    parser.add_argument("--alpha", type=float, required=True, help="Interpolation coefficient alpha.")
    parser.add_argument("--budget", type=int, default=8192, help="Pass 1 max_tokens budget (default: 8192).")
    parser.add_argument("--n", type=int, default=50, help="Number of MATH-500 problems (default: 50).")
    parser.add_argument("--samples", type=int, default=1, help="Samples per problem (default: 1).")
    parser.add_argument("--seed", type=int, default=0, help="Dataset & sampling seed (default: 0).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "float32", "auto"),
        help="vLLM model dtype (default: float16; or 'auto' from smoke test JSON).",
    )
    parser.add_argument("--gpu-tag", type=str, default=None, help="Override GPU tag (e.g. 't4').")
    parser.add_argument("--out-root", type=str, default="assets", help="Root assets directory (default: assets).")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM.")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (default: 0.85).")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional pre-built checkpoint directory (if omitted, builds & cleans scratch checkpoint).",
    )
    parser.add_argument(
        "--keep-ckpt",
        action="store_true",
        help="Do not delete auto-created scratch checkpoint on exit.",
    )
    args = parser.parse_args(argv)

    run_vllm_alpha(
        alpha=args.alpha,
        budget=args.budget,
        n_problems=args.n,
        samples=args.samples,
        seed=args.seed,
        dtype=args.dtype,
        gpu_tag=args.gpu_tag,
        out_root=args.out_root,
        enforce_eager=args.enforce_eager,
        gpu_mem=args.gpu_mem,
        ckpt_dir=args.ckpt_dir,
        keep_ckpt=args.keep_ckpt,
    )


if __name__ == "__main__":
    main()
