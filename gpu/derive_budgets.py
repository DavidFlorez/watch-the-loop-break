"""Derive nested smaller token-budget runs (4k/8k/16k/32k) from a 32k vLLM sweep."""

from __future__ import annotations

import os

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

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

import pandas as pd

from gpu.common import (
    FORCE_ANSWER_SUFFIX,
    FORCE_BOX_OPEN_SUFFIX,
    FORCED_MAX_TOKENS,
    SAMPLING_DEFAULTS,
    STOP_TOKEN_IDS,
    atomic_write_gzip_jsonl,
    atomic_write_text,
    check_min_compute_capability,
    cleanup_ckpt_dir,
    collect_environment_metadata,
    create_vllm_engine,
    make_scratch_ckpt_dir,
    make_tokens_prompt,
    release_vllm_engine,
    utc_now_iso,
)
from gpu.make_ckpt import build_checkpoint
from gpu.vllm_sweep import (
    _format_prompt_for_problem,
    _load_existing_records,
    _make_sampling_params,
    _record_key,
    _safe_compute_loop_onset,
    build_summary_dataframe,
)
from metrics import (
    extract_forced_boxed,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
)
from sweep import load_math500_subset, save_sweep_charts
from weights import BASE_MODEL_ID, EDIT_MODEL_ID, TARGET_TENSORS

DEFAULT_DERIVED_BUDGETS: Tuple[int, ...] = (4096, 8192, 16384, 32768)


def parse_budgets_arg(budgets_str: str) -> List[int]:
    vals: List[int] = []
    for tok in (budgets_str or "").split(","):
        cleaned = tok.strip()
        if not cleaned:
            continue
        b = int(cleaned)
        if b <= 0:
            raise ValueError(f"Budget must be positive, got {b}")
        if b not in vals:
            vals.append(b)
    if not vals:
        raise ValueError("At least one budget is required.")
    return sorted(vals)


def parse_alphas_arg(alphas_str: str) -> Optional[List[float]]:
    if not alphas_str or not alphas_str.strip():
        return None
    vals: List[float] = []
    for tok in alphas_str.split(","):
        cleaned = tok.strip()
        if not cleaned:
            continue
        a = float(cleaned)
        if a not in vals:
            vals.append(a)
    return vals


def resolve_32k_source_dir(src_path: Union[str, Path], preferred_subdir: Optional[str] = None) -> Path:
    """Resolve a 32k run directory from either a direct leaf path or a root containing `budget_32768/`."""
    p = Path(src_path)
    if (p / "sweep_per_problem.jsonl.gz").is_file():
        return p

    b32 = p / "budget_32768"
    if b32.is_dir():
        if preferred_subdir and (b32 / preferred_subdir / "sweep_per_problem.jsonl.gz").is_file():
            return b32 / preferred_subdir
        candidates = sorted(
            [d for d in b32.iterdir() if d.is_dir() and d.name.startswith("gpu-") and (d / "sweep_per_problem.jsonl.gz").is_file()]
        )
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not find 32k `sweep_per_problem.jsonl.gz` under {p} (or {p / 'budget_32768/gpu-*'})."
    )


def _derive_single_prefix_record(
    rec32: Dict[str, Any],
    budget: int,
    tokenizer: Any,
    full_ids_cache: Dict[Tuple[float, str, int], Tuple[List[int], int]],
    *,
    gpu_tag: str,
    dtype: str,
    vllm_version: str,
) -> Tuple[Dict[str, Any], List[int], bool]:
    """Truncate a single 32k Pass-1 trace to `budget` and recompute strict/loop/text metrics."""
    b_int = int(budget)
    alpha_f = float(rec32["alpha"])
    uid = str(rec32["unique_id"])
    s_idx = int(rec32.get("sample_idx", 0))
    cache_key = (alpha_f, uid, s_idx)

    raw_text = str(rec32.get("text", ""))
    finish_32 = str(rec32.get("finish_reason", "stop"))

    if cache_key not in full_ids_cache:
        raw_out_ids = rec32.get("out_ids")
        if raw_out_ids is not None:
            ids_list = [int(t) for t in raw_out_ids]
            orig_len = len(ids_list)
        else:
            ids_list = [int(t) for t in tokenizer.encode(raw_text, add_special_tokens=False)]
            orig_len = int(rec32.get("n_tokens", len(ids_list)))
        full_ids_cache[cache_key] = (ids_list, orig_len)

    full_ids, full_n_tokens = full_ids_cache[cache_key]

    finished_within_b = (finish_32 == "stop") and (full_n_tokens <= b_int)
    hit_cap_b = not finished_within_b

    if not hit_cap_b:
        sub_ids = list(full_ids)
        sub_n_tokens = full_n_tokens
        finish_reason_b = "stop"
        sub_text = raw_text if raw_text else tokenizer.decode(sub_ids, skip_special_tokens=False)
    else:
        sub_ids = list(full_ids[:b_int])
        sub_n_tokens = len(sub_ids) if rec32.get("out_ids") is not None else min(full_n_tokens, b_int)
        finish_reason_b = "length"
        if b_int >= 32768 and raw_text and full_n_tokens <= b_int:
            sub_text = raw_text
        else:
            sub_text = tokenizer.decode(sub_ids, skip_special_tokens=False)

    closed_think_b = "</think>" in sub_text
    gold = str(rec32["gold"])
    pred_b = extract_last_boxed(sub_text)
    corr_b = bool(grade_math_answer(pred_b, gold))
    pred_strict_b = None if hit_cap_b else extract_last_boxed_strict(sub_text)
    corr_strict_b = False if hit_cap_b else bool(grade_math_answer(pred_strict_b, gold))
    loop_onset_b = _safe_compute_loop_onset(sub_text, tokenizer, token_ids=sub_ids)

    expected_mode = ("open_box" if closed_think_b else "close_think") if hit_cap_b else None
    derived_rec: Dict[str, Any] = {
        "alpha": alpha_f,
        "max_new_tokens": b_int,
        "unique_id": uid,
        "sample_idx": s_idx,
        "subject": str(rec32["subject"]),
        "level": int(rec32["level"]),
        "gold": gold,
        "pred": pred_b,
        "correct": corr_b,
        "pred_strict": pred_strict_b,
        "correct_strict": corr_strict_b,
        "pred_forced": pred_strict_b if not hit_cap_b else None,
        "correct_forced": corr_strict_b if not hit_cap_b else False,
        "forced_tokens": 0,
        "forced_text": None,
        "forced_mode": expected_mode,
        "n_tokens": int(sub_n_tokens),
        "hit_cap": bool(hit_cap_b),
        "finish_reason": finish_reason_b,
        "closed_think": bool(closed_think_b),
        "loop_onset_token": loop_onset_b,
        "rep_8gram_ratio": round(repeated_8gram_ratio(sub_text), 4),
        "tail_rep_8gram_ratio": round(tail_repeated_8gram_ratio(sub_text), 4),
        "wall_time_s": float(rec32.get("wall_time_s", 0.0)) if not hit_cap_b else 0.0,
        "backend": str(rec32.get("backend", "vllm")),
        "gpu": gpu_tag,
        "dtype": dtype,
        "vllm_version": vllm_version,
        "out_ids": sub_ids,
        "text": sub_text,
    }
    return derived_rec, sub_ids, hit_cap_b


def verify_derived_against_root(
    derived_root: Union[str, Path],
    ref_root: Union[str, Path],
    *,
    budgets: Sequence[int] = (4096, 8192, 16384),
    gpu_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare derived budgets against reference `budget_<B>/gpu-*` CSVs and per-problem traces."""
    d_root = Path(derived_root)
    r_root = Path(ref_root)
    per_budget: Dict[int, Dict[str, Any]] = {}
    all_csv_exact_legacy = True
    all_csv_exact_finished = True
    total_text_mismatches = 0
    total_cap_mismatches = 0
    total_closed_think_mismatches = 0
    total_traces_checked = 0

    for b in budgets:
        b_int = int(b)
        ref_b_parent = r_root / f"budget_{b_int}"
        if not ref_b_parent.is_dir():
            continue
        if gpu_subdir and (ref_b_parent / gpu_subdir).is_dir():
            ref_dir = ref_b_parent / gpu_subdir
        else:
            cands = sorted([d for d in ref_b_parent.iterdir() if d.is_dir() and d.name.startswith("gpu-")])
            if not cands:
                continue
            ref_dir = cands[0]

        sub_name = ref_dir.name
        der_dir = d_root / f"budget_{b_int}" / sub_name
        ref_records = _load_existing_records(ref_dir / "sweep_per_problem.jsonl.gz")
        der_records = _load_existing_records(der_dir / "sweep_per_problem.jsonl.gz")
        ref_csv = pd.read_csv(ref_dir / "sweep_results.csv")
        der_csv = pd.read_csv(der_dir / "sweep_results.csv")

        text_mm = 0
        cap_mm = 0
        ct_mm = 0
        legacy_strict_by_alpha: Dict[float, List[bool]] = {}
        for key, ref_rec in ref_records.items():
            total_traces_checked += 1
            der_rec = der_records.get(key)
            if der_rec is None:
                text_mm += 1
                cap_mm += 1
                ct_mm += 1
                continue
            if str(der_rec.get("text", "")) != str(ref_rec.get("text", "")):
                text_mm += 1
            if bool(der_rec.get("hit_cap")) != bool(ref_rec.get("hit_cap")):
                cap_mm += 1
            if bool(der_rec.get("closed_think")) != bool(ref_rec.get("closed_think")):
                ct_mm += 1

            raw_ps = extract_last_boxed_strict(str(der_rec.get("text", "")))
            raw_cs = bool(grade_math_answer(raw_ps, str(der_rec["gold"])))
            legacy_strict_by_alpha.setdefault(float(key[0]), []).append(raw_cs)

        total_text_mismatches += text_mm
        total_cap_mismatches += cap_mm
        total_closed_think_mismatches += ct_mm

        alpha_diffs: List[Dict[str, Any]] = []
        for _, r_row in ref_csv.iterrows():
            a_val = float(r_row["alpha"])
            d_sub = der_csv[der_csv["alpha"] == a_val]
            if d_sub.empty:
                all_csv_exact_legacy = False
                all_csv_exact_finished = False
                continue
            d_row = d_sub.iloc[0]
            leg_vals = legacy_strict_by_alpha.get(a_val, [])
            leg_strict_pct = round((sum(1 for x in leg_vals if x) / max(1, len(leg_vals))) * 100.0, 2)

            strict_fin_match = float(d_row["accuracy_pct"]) == float(r_row["accuracy_pct"])
            strict_leg_match = leg_strict_pct == float(r_row["accuracy_pct"])
            cap_match = float(d_row["hit_cap_pct"]) == float(r_row["hit_cap_pct"])
            ct_match = float(d_row["closed_think_pct"]) == float(r_row["closed_think_pct"])

            if not (strict_leg_match and cap_match and ct_match):
                all_csv_exact_legacy = False
            if not (strict_fin_match and cap_match and ct_match):
                all_csv_exact_finished = False

            if not strict_fin_match or not cap_match or not ct_match:
                alpha_diffs.append(
                    {
                        "alpha": a_val,
                        "derived_strict_finished_pct": float(d_row["accuracy_pct"]),
                        "derived_strict_unclosed_pct": leg_strict_pct,
                        "ref_strict_pct": float(r_row["accuracy_pct"]),
                        "derived_cap_pct": float(d_row["hit_cap_pct"]),
                        "ref_cap_pct": float(r_row["hit_cap_pct"]),
                        "derived_closed_think_pct": float(d_row["closed_think_pct"]),
                        "ref_closed_think_pct": float(r_row["closed_think_pct"]),
                    }
                )

        per_budget[b_int] = {
            "n_traces": len(ref_records),
            "text_mismatches": text_mm,
            "cap_mismatches": cap_mm,
            "closed_think_mismatches": ct_mm,
            "alpha_diffs": alpha_diffs,
        }

    return {
        "total_traces_checked": total_traces_checked,
        "total_text_mismatches": total_text_mismatches,
        "total_cap_mismatches": total_cap_mismatches,
        "total_closed_think_mismatches": total_closed_think_mismatches,
        "exact_csv_match_legacy_strict": all_csv_exact_legacy,
        "exact_csv_match_finished_strict": all_csv_exact_finished,
        "per_budget": per_budget,
    }


def derive_budgets_from_32k(
    *,
    src_32k_dir: Union[str, Path],
    out_root: Union[str, Path],
    budgets: Sequence[int] = DEFAULT_DERIVED_BUDGETS,
    alphas: Optional[Sequence[float]] = None,
    cached_forced_root: Optional[Union[str, Path]] = None,
    run_forced: bool = False,
    ckpt_dir: Optional[Union[str, Path]] = None,
    keep_ckpt: bool = False,
    dtype: str = "auto",
    gpu_mem: float = 0.85,
    enforce_eager: bool = False,
    seed: int = 0,
    verify_against_root: Optional[Union[str, Path]] = None,
    llm_cls: Optional[Any] = None,
    sampling_params_factory: Optional[Any] = None,
    tokenizer_override: Optional[Any] = None,
    require_gpu: bool = False,
) -> Dict[str, Any]:
    """Derive token budgets in `budgets` by truncating a 32k vLLM sweep and optionally running Pass 2."""
    resolved_src = resolve_32k_source_dir(src_32k_dir)
    src_records = _load_existing_records(resolved_src / "sweep_per_problem.jsonl.gz")
    if not src_records:
        raise RuntimeError(f"No records found in {resolved_src / 'sweep_per_problem.jsonl.gz'}")

    src_meta: Dict[str, Any] = {}
    src_meta_file = resolved_src / "meta.json"
    if src_meta_file.is_file():
        src_meta = json.loads(src_meta_file.read_text(encoding="utf-8"))

    first_rec = next(iter(src_records.values()))
    gpu_tag = str(src_meta.get("gpu_tag") or first_rec.get("gpu") or "rtx-pro-6000")
    resolved_dtype = str(src_meta.get("dtype") or first_rec.get("dtype") or "bfloat16")
    vllm_version = str(src_meta.get("vllm_version") or first_rec.get("vllm_version") or "unknown")
    samples = int(src_meta.get("samples") or (max(int(r.get("sample_idx", 0)) for r in src_records.values()) + 1))

    available_alphas = sorted({float(r["alpha"]) for r in src_records.values()})
    target_alphas = sorted({float(a) for a in alphas}) if alphas is not None else available_alphas

    # Reconstruct ordered problem list matching the 32k run's problem set
    src_uids = sorted({str(r["unique_id"]) for r in src_records.values()})
    max_n = max(len(src_uids), int(src_meta.get("n_problems", len(src_uids))))
    full_subset = load_math500_subset(n=max_n, seed=seed)
    subset_uid_set = {str(p["unique_id"]) for p in full_subset}
    if set(src_uids).issubset(subset_uid_set):
        src_uid_set = set(src_uids)
        problems = [p for p in full_subset if str(p["unique_id"]) in src_uid_set]
    else:
        all_500 = load_math500_subset(n=500, seed=seed)
        by_uid_all = {str(p["unique_id"]): p for p in all_500}
        problems = [
            by_uid_all.get(
                uid,
                {
                    "unique_id": uid,
                    "problem": "",
                    "subject": str(next(r["subject"] for r in src_records.values() if r["unique_id"] == uid)),
                    "level": int(next(r["level"] for r in src_records.values() if r["unique_id"] == uid)),
                    "answer": str(next(r["gold"] for r in src_records.values() if r["unique_id"] == uid)),
                },
            )
            for uid in src_uids
        ]

    problem_by_uid = {str(p["unique_id"]): p for p in problems}
    problem_rank = {str(p["unique_id"]): idx for idx, p in enumerate(problems)}
    n_problems = len(problems)

    tokenizer = tokenizer_override
    if tokenizer is None:
        from transformers import AutoTokenizer

        tok_src = str(ckpt_dir) if ckpt_dir is not None else BASE_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(tok_src)

    sub_name = f"gpu-{gpu_tag}-{resolved_dtype}"
    full_ids_cache: Dict[Tuple[float, str, int], Tuple[List[int], int]] = {}
    ordered_budgets = sorted({int(b) for b in budgets})

    # Preload any cached forced records per budget (from existing out_root or cached_forced_root)
    cached_by_budget: Dict[int, Dict[Tuple[float, str, int, int], Dict[str, Any]]] = {}
    for b_int in ordered_budgets:
        merged_cache: Dict[Tuple[float, str, int, int], Dict[str, Any]] = {}
        for root_cand in (cached_forced_root, out_root):
            if not root_cand:
                continue
            cand_jsonl = Path(root_cand) / f"budget_{b_int}" / sub_name / "sweep_per_problem.jsonl.gz"
            if cand_jsonl.is_file():
                merged_cache.update(_load_existing_records(cand_jsonl))
        cached_by_budget[b_int] = merged_cache

    # Build Pass-1 derived records per (alpha, budget)
    derived_by_alpha_budget: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    sub_ids_lookup: Dict[Tuple[float, int, str, int], List[int]] = {}
    pending_forced_by_alpha: Dict[float, List[Tuple[int, Dict[str, Any]]]] = {a: [] for a in target_alphas}

    for a_val in target_alphas:
        alpha_recs_32 = [
            r for r in src_records.values() if float(r["alpha"]) == a_val and str(r["unique_id"]) in problem_by_uid
        ]
        alpha_recs_32.sort(
            key=lambda r: (problem_rank.get(str(r["unique_id"]), 999999), int(r.get("sample_idx", 0)))
        )

        for b_int in ordered_budgets:
            b_rows: List[Dict[str, Any]] = []
            b_cache = cached_by_budget.get(b_int, {})
            for r32 in alpha_recs_32:
                d_rec, sub_ids, hit_cap_b = _derive_single_prefix_record(
                    r32,
                    b_int,
                    tokenizer,
                    full_ids_cache,
                    gpu_tag=gpu_tag,
                    dtype=resolved_dtype,
                    vllm_version=vllm_version,
                )
                uid = str(d_rec["unique_id"])
                s_idx = int(d_rec["sample_idx"])
                r_key = _record_key(d_rec)
                sub_ids_lookup[(a_val, b_int, uid, s_idx)] = sub_ids

                if hit_cap_b:
                    expected_mode = "open_box" if d_rec["closed_think"] else "close_think"
                    # Check 32k source first when deriving 32k, then cached_by_budget
                    reused = False
                    cand_sources = []
                    if b_int >= 32768 and int(r32.get("max_new_tokens", 0)) == b_int:
                        cand_sources.append(r32)
                    if r_key in b_cache:
                        cand_sources.append(b_cache[r_key])

                    for c_rec in cand_sources:
                        has_forced_data = (
                            (c_rec.get("forced_text") is not None)
                            or (int(c_rec.get("forced_tokens", 0)) > 0)
                            or (c_rec.get("pred_forced") is not None)
                        )
                        c_mode = c_rec.get("forced_mode")
                        # When live GPU forced pass is enabled, re-force any legacy closed-</think> capped trace
                        mode_ok = (c_mode == expected_mode) or (
                            not run_forced and ckpt_dir is None and has_forced_data
                        )
                        if has_forced_data and mode_ok:
                            d_rec["forced_tokens"] = int(c_rec.get("forced_tokens", 0))
                            d_rec["forced_text"] = c_rec.get("forced_text")
                            d_rec["forced_mode"] = c_mode or expected_mode
                            d_rec["pred_forced"] = c_rec.get("pred_forced")
                            d_rec["correct_forced"] = bool(c_rec.get("correct_forced", False))
                            d_rec["wall_time_s"] = float(c_rec.get("wall_time_s", d_rec["wall_time_s"]))
                            reused = True
                            break

                    if not reused and (run_forced or ckpt_dir is not None):
                        pending_forced_by_alpha[a_val].append((b_int, d_rec))

                b_rows.append(d_rec)
            derived_by_alpha_budget[(a_val, b_int)] = b_rows

    # Optionally run Pass-2 forced completions on truncated prefixes that need them
    forced_wall_by_alpha: Dict[float, float] = {}
    forced_count_by_alpha: Dict[float, int] = {}
    if run_forced or ckpt_dir is not None:
        check_min_compute_capability(min_cc=7.5, require_gpu=require_gpu)
        force_close_ids = list(tokenizer.encode(FORCE_ANSWER_SUFFIX, add_special_tokens=False))
        force_box_ids = list(tokenizer.encode(FORCE_BOX_OPEN_SUFFIX, add_special_tokens=False))
        prompt_ids_cache: Dict[str, List[int]] = {}

        for a_val in target_alphas:
            items_to_force = pending_forced_by_alpha.get(a_val, [])
            if not items_to_force:
                continue

            if ckpt_dir is not None and len(target_alphas) > 1:
                raise ValueError(
                    "A single `--ckpt-dir` corresponds to one alpha; pass `--alpha` or omit `--ckpt-dir` to build per-alpha checkpoints."
                )

            owned_ckpt: Optional[Path] = None
            llm = None
            try:
                if ckpt_dir is not None:
                    use_ckpt = Path(ckpt_dir).resolve()
                else:
                    owned_ckpt = make_scratch_ckpt_dir(a_val, prefix="derive")
                    use_ckpt = owned_ckpt
                    build_checkpoint(alpha=a_val, out_dir=use_ckpt)

                max_b = max(ordered_budgets)
                llm, eng_meta = create_vllm_engine(
                    ckpt_dir=use_ckpt,
                    dtype=resolved_dtype if dtype == "auto" else dtype,
                    max_model_len=max_b + 2048,
                    gpu_memory_utilization=gpu_mem,
                    seed=seed,
                    enforce_eager=enforce_eager,
                    llm_cls=llm_cls,
                )
                if eng_meta.get("vllm_version") and eng_meta["vllm_version"] != "unknown":
                    vllm_version = str(eng_meta["vllm_version"])

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

                forced_inputs: List[Any] = []
                expected_modes: List[str] = []
                for b_int, d_rec in items_to_force:
                    uid = str(d_rec["unique_id"])
                    s_idx = int(d_rec["sample_idx"])
                    if uid not in prompt_ids_cache:
                        fmt = _format_prompt_for_problem(tokenizer, problem_by_uid[uid]["problem"])
                        prompt_ids_cache[uid] = list(tokenizer.encode(fmt, add_special_tokens=False))
                    sub_ids = sub_ids_lookup[(a_val, b_int, uid, s_idx)]
                    f_mode = "open_box" if d_rec["closed_think"] else "close_think"
                    expected_modes.append(f_mode)
                    sfx_ids = force_box_ids if f_mode == "open_box" else force_close_ids
                    forced_inputs.append(make_tokens_prompt(prompt_ids_cache[uid] + sub_ids + sfx_ids))

                print(
                    f"[Derive Budgets] Running Pass-2 forced answers for alpha={a_val:g} on {len(items_to_force)} truncated prefix(es) ...",
                    flush=True,
                )
                t0 = time.perf_counter()
                outputs_p2 = llm.generate(forced_inputs, sp_pass2)
                wall_p2 = time.perf_counter() - t0
                forced_wall_by_alpha[a_val] = round(wall_p2, 2)
                forced_count_by_alpha[a_val] = len(items_to_force)
                per_p2_wall = wall_p2 / max(1, len(items_to_force))

                for (_, d_rec), f_mode, req_p2 in zip(items_to_force, expected_modes, outputs_p2):
                    comp2 = req_p2.outputs[0]
                    f_ids = list(comp2.token_ids)
                    f_toks = len(f_ids)
                    f_text = (
                        comp2.text
                        if getattr(comp2, "text", None) is not None
                        else tokenizer.decode(f_ids, skip_special_tokens=False)
                    )
                    pred_f = extract_forced_boxed(f_text, forced_mode=f_mode)
                    corr_f = bool(grade_math_answer(pred_f, d_rec["gold"]))
                    d_rec["forced_tokens"] = f_toks
                    d_rec["forced_text"] = f_text
                    d_rec["forced_mode"] = f_mode
                    d_rec["pred_forced"] = pred_f
                    d_rec["correct_forced"] = corr_f
                    d_rec["wall_time_s"] = round(float(d_rec.get("wall_time_s", 0.0)) + per_p2_wall, 3)
            finally:
                release_vllm_engine(llm)
                if owned_ckpt is not None and not keep_ckpt:
                    cleanup_ckpt_dir(owned_ckpt)

    # Write output directories per budget (merging safely with any other alphas already present)
    env_meta = src_meta.get("environment") or collect_environment_metadata(gpu_tag_override=gpu_tag)
    written_dirs: Dict[int, str] = {}

    for b_int in ordered_budgets:
        out_dir = Path(out_root) / f"budget_{b_int}" / sub_name
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "sweep_per_problem.jsonl.gz"
        per_prob_csv = out_dir / "sweep_per_problem.csv"
        summary_csv = out_dir / "sweep_results.csv"
        meta_path = out_dir / "meta.json"

        merged_b = _load_existing_records(jsonl_path)
        for a_val in target_alphas:
            for d_rec in derived_by_alpha_budget[(a_val, b_int)]:
                merged_b[_record_key(d_rec)] = d_rec

        sorted_b = sorted(
            merged_b.values(),
            key=lambda r: (
                float(r["alpha"]),
                problem_rank.get(str(r["unique_id"]), 999999),
                str(r["unique_id"]),
                int(r.get("sample_idx", 0)),
            ),
        )
        atomic_write_gzip_jsonl(jsonl_path, sorted_b)

        p_rows = [{k: v for k, v in r.items() if k not in ("text", "out_ids", "forced_text")} for r in sorted_b]
        p_df = pd.DataFrame(p_rows)
        atomic_write_text(per_prob_csv, p_df.to_csv(index=False))

        s_df = build_summary_dataframe(
            sorted_b,
            problems,
            budget=b_int,
            samples=samples,
            gpu_tag=gpu_tag,
            dtype=resolved_dtype,
            vllm_version=vllm_version,
        )
        if not s_df.empty:
            atomic_write_text(summary_csv, s_df.to_csv(index=False))
            save_sweep_charts(s_df, p_df, out_dir=str(out_dir), max_new_tokens=b_int)

        meta_b = {
            **src_meta,
            "updated_at_utc": utc_now_iso(),
            "derived_from_32k": str(resolved_src),
            "budget": b_int,
            "n_problems": n_problems,
            "samples": samples,
            "seed": seed,
            "gpu_tag": gpu_tag,
            "dtype": resolved_dtype,
            "vllm_version": vllm_version,
            "forced_pass2_wall_s_by_alpha": {f"{k:g}": v for k, v in forced_wall_by_alpha.items()},
            "forced_pass2_count_by_alpha": {f"{k:g}": v for k, v in forced_count_by_alpha.items()},
            "environment": env_meta,
        }
        atomic_write_text(meta_path, json.dumps(meta_b, indent=2, ensure_ascii=False) + "\n")
        written_dirs[b_int] = str(out_dir)

    verification = None
    if verify_against_root is not None:
        verification = verify_derived_against_root(
            out_root,
            verify_against_root,
            budgets=[b for b in ordered_budgets if b < 32768],
            gpu_subdir=sub_name,
        )

    return {
        "source_32k_dir": str(resolved_src),
        "out_root": str(out_root),
        "budgets": ordered_budgets,
        "alphas": target_alphas,
        "n_problems": n_problems,
        "written_dirs": written_dirs,
        "forced_count_by_alpha": {f"{k:g}": v for k, v in forced_count_by_alpha.items()},
        "forced_wall_s_by_alpha": {f"{k:g}": v for k, v in forced_wall_by_alpha.items()},
        "verification": verification,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive nested smaller token-budget runs (4k/8k/16k/32k) from a 32k vLLM sweep."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to 32k run directory (or root containing `budget_32768/gpu-*`).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Destination root directory where `budget_<B>/gpu-<gpu>-<dtype>/` will be written.",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="4096,8192,16384,32768",
        help="Comma-separated target budgets to derive (default: 4096,8192,16384,32768).",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="",
        help="Optional comma-separated subset of alphas to derive (default: all alphas in `--src`).",
    )
    parser.add_argument(
        "--cached-forced-root",
        type=str,
        default=None,
        help="Optional existing assets root to reuse cached Pass-2 forced completions.",
    )
    parser.add_argument(
        "--run-forced",
        action="store_true",
        help="Run live Pass-2 forced continuations via vLLM on truncated capped prefixes.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional pre-built vLLM checkpoint directory (for single-alpha live Pass 2).",
    )
    parser.add_argument(
        "--keep-ckpt",
        action="store_true",
        help="Keep auto-created scratch checkpoints when `--run-forced` builds per-alpha checkpoints.",
    )
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify-against",
        type=str,
        default=None,
        help="Optional reference assets root (e.g. `assets`) to verify derived 4k/8k/16k CSVs & traces.",
    )
    args = parser.parse_args(argv)

    result = derive_budgets_from_32k(
        src_32k_dir=args.src,
        out_root=args.out_root,
        budgets=parse_budgets_arg(args.budgets),
        alphas=parse_alphas_arg(args.alphas),
        cached_forced_root=args.cached_forced_root,
        run_forced=args.run_forced,
        ckpt_dir=args.ckpt_dir,
        keep_ckpt=args.keep_ckpt,
        dtype=args.dtype,
        gpu_mem=args.gpu_mem,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        verify_against_root=args.verify_against,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
