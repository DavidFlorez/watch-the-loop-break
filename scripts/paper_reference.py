"""Fetch and compute published Takehi et al. (2026) MATH-500 reference metrics on exact & truncated sets."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from gpu.common import atomic_write_text
from metrics import extract_last_boxed_strict, grade_math_answer, wilson_ci
from sweep import load_math500_subset
from weights import BASE_MODEL_ID

UPSTREAM_EVAL_BASE_URL = (
    "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/eval"
)
UPSTREAM_SHARDS: Tuple[int, ...] = tuple(range(10))
UPSTREAM_CONDITIONS: Tuple[Tuple[str, float, str], ...] = (
    ("unedited", 0.0, "Paper Base (unedited, α=0.0)"),
    ("p50c", 1.0, "Paper Edited (p50c, α=1.0)"),
)
TRUNCATION_BUDGETS: Tuple[int, ...] = (4096, 8192, 16384, 32768)


def _fetch_url_bytes(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def fetch_upstream_condition_logs(
    condition: str,
    *,
    base_url: str = UPSTREAM_EVAL_BASE_URL,
    shards: Sequence[int] = UPSTREAM_SHARDS,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Fetch `math500_per_problem.json` (all 500) and `math500_first_samples.jsonl.gz` (200) for `condition`."""
    per_problem: Dict[str, Dict[str, Any]] = {}
    first_samples: Dict[str, Dict[str, Any]] = {}

    def _download_shard(shard_idx: int) -> Tuple[List[Dict[str, Any]], bytes]:
        shard_tag = f"math500_s{int(shard_idx)}of10"
        pp_url = f"{base_url}/{condition}/parts/{shard_tag}/math500_per_problem.json"
        fs_url = f"{base_url}/{condition}/parts/{shard_tag}/math500_first_samples.jsonl.gz"
        pp_raw = json.loads(_fetch_url_bytes(pp_url).decode("utf-8"))
        fs_bytes = _fetch_url_bytes(fs_url)
        return pp_raw, fs_bytes

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(shards)))) as ex:
        for pp_list, fs_bytes in ex.map(_download_shard, shards):
            for item in pp_list:
                per_problem[str(item["id"])] = item
            with gzip.GzipFile(fileobj=io.BytesIO(fs_bytes), mode="rb") as gz:
                for raw_line in gz:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    first_samples[str(rec["id"])] = rec

    return per_problem, first_samples


def compute_avg_at_k_row(
    *,
    cohort: str,
    condition: str,
    alpha: float,
    label: str,
    per_problem_map: Dict[str, Dict[str, Any]],
    target_ids: Sequence[str],
) -> Dict[str, Any]:
    """Compute exact published `avg@4` accuracy across `target_ids`."""
    matched_items = [per_problem_map[uid] for uid in target_ids if uid in per_problem_map]
    n_probs = len(matched_items)
    total_correct = 0
    total_samples = 0
    first_sample_correct = 0
    first_sample_unboxed = 0

    for item in matched_items:
        corr_vec = [int(x) for x in item.get("correct", [])]
        total_correct += sum(corr_vec)
        total_samples += len(corr_vec)
        if corr_vec and corr_vec[0] == 1:
            first_sample_correct += 1
        if item.get("pred_first") is None:
            first_sample_unboxed += 1

    avg4_frac = (total_correct / total_samples) if total_samples > 0 else 0.0
    lo_f, hi_f = wilson_ci(total_correct, total_samples)
    cut_first_pct = round((first_sample_unboxed / n_probs) * 100.0, 2) if n_probs > 0 else 0.0

    return {
        "cohort": cohort,
        "metric_type": "avg_at_4",
        "condition": condition,
        "alpha": float(alpha),
        "label": label,
        "budget": 32768,
        "n_problems": n_probs,
        "n_samples_per_problem": (total_samples // n_probs) if n_probs > 0 else 0,
        "total_evaluations": total_samples,
        "correct_count": total_correct,
        "accuracy_pct": round(avg4_frac * 100.0, 2),
        "accuracy_ci95_low": round(lo_f * 100.0, 2),
        "accuracy_ci95_high": round(hi_f * 100.0, 2),
        "first_sample_accuracy_pct": round((first_sample_correct / n_probs) * 100.0, 2) if n_probs > 0 else 0.0,
        "hit_cap_or_unboxed_pct": cut_first_pct,
        "status": "exact (published avg@4)",
    }


def evaluate_first_sample_truncations(
    first_samples_map: Dict[str, Dict[str, Any]],
    tokenizer: Any,
    *,
    cohort_sets: Sequence[Tuple[str, Sequence[str]]],
    condition: str,
    alpha: float,
    label: str,
    budgets: Sequence[int] = TRUNCATION_BUDGETS,
) -> List[Dict[str, Any]]:
    """Re-tokenize `first_samples` text with Qwen tokenizer and evaluate strict pass rate under each budget."""
    annotated: Dict[str, Dict[str, Any]] = {}
    for uid, rec in first_samples_map.items():
        text = str(rec.get("text", ""))
        gold = str(rec.get("answer", ""))
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        closed_think = "</think>" in text
        pred_strict = extract_last_boxed_strict(text)
        boxed_ok = bool(grade_math_answer(pred_strict, gold))
        annotated[uid] = {
            "n_tokens": n_tok,
            "closed_think": closed_think,
            "boxed_ok": boxed_ok,
        }

    rows: List[Dict[str, Any]] = []
    for cohort_name, id_list in cohort_sets:
        matched_ids = [uid for uid in id_list if uid in annotated]
        n_probs = len(matched_ids)
        if n_probs == 0:
            continue

        for b in budgets:
            b_int = int(b)
            n_finished = 0
            n_correct_under_b = 0
            for uid in matched_ids:
                info = annotated[uid]
                n_tok = int(info["n_tokens"])
                # A 32k-capped paper sample lacks `</think>` (even if BPE re-tokenization is 32766..32768)
                finished_under_b = (
                    (info["closed_think"] and n_tok <= b_int)
                    if b_int >= 32768
                    else (n_tok <= b_int)
                )
                if finished_under_b:
                    n_finished += 1
                    if info["boxed_ok"]:
                        n_correct_under_b += 1

            acc_frac = n_correct_under_b / n_probs
            lo_f, hi_f = wilson_ci(n_correct_under_b, n_probs)
            cap_pct = round(((n_probs - n_finished) / n_probs) * 100.0, 2)

            rows.append(
                {
                    "cohort": cohort_name,
                    "metric_type": "first_sample_truncated",
                    "condition": condition,
                    "alpha": float(alpha),
                    "label": label,
                    "budget": b_int,
                    "n_problems": n_probs,
                    "n_samples_per_problem": 1,
                    "total_evaluations": n_probs,
                    "correct_count": n_correct_under_b,
                    "accuracy_pct": round(acc_frac * 100.0, 2),
                    "accuracy_ci95_low": round(lo_f * 100.0, 2),
                    "accuracy_ci95_high": round(hi_f * 100.0, 2),
                    "first_sample_accuracy_pct": round(acc_frac * 100.0, 2),
                    "hit_cap_or_unboxed_pct": cap_pct,
                    "status": "approximate (re-tokenized)",
                }
            )
    return rows


def build_paper_reference_table(
    *,
    out_csv: Optional[Union[str, Path]] = "assets/paper_reference.csv",
    tokenizer_override: Optional[Any] = None,
    condition_logs_override: Optional[Dict[str, Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]]] = None,
) -> pd.DataFrame:
    """Compute exact `avg@4` and re-tokenized first-sample budget curves for Takehi et al. (2026)."""
    tokenizer = tokenizer_override
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    ids_50 = [str(p["unique_id"]) for p in load_math500_subset(n=50, seed=0)]
    ids_100 = [str(p["unique_id"]) for p in load_math500_subset(n=100, seed=0)]
    ids_200 = [str(p["unique_id"]) for p in load_math500_subset(n=200, seed=0)]
    ids_500 = [str(p["unique_id"]) for p in load_math500_subset(n=500, seed=0)]

    all_rows: List[Dict[str, Any]] = []

    for cond, alpha_val, label in UPSTREAM_CONDITIONS:
        if condition_logs_override is not None and cond in condition_logs_override:
            pp_map, fs_map = condition_logs_override[cond]
        else:
            pp_map, fs_map = fetch_upstream_condition_logs(cond)

        fs_ids_sorted = sorted(fs_map.keys())
        avg_cohorts: List[Tuple[str, Sequence[str]]] = [
            ("our_n50", ids_50),
            ("our_n100_dose", ids_100),
            ("our_n200_core", ids_200),
            ("paper_first_samples_n200", fs_ids_sorted),
            ("math500_all_500", ids_500),
        ]
        for cohort_name, id_list in avg_cohorts:
            all_rows.append(
                compute_avg_at_k_row(
                    cohort=cohort_name,
                    condition=cond,
                    alpha=alpha_val,
                    label=label,
                    per_problem_map=pp_map,
                    target_ids=id_list,
                )
            )

        trunc_cohorts: List[Tuple[str, Sequence[str]]] = [
            ("paper_first_samples_n200", fs_ids_sorted),
            ("our_n200_overlap", ids_200),
            ("our_n100_overlap", ids_100),
            ("our_n50_overlap", ids_50),
        ]
        all_rows.extend(
            evaluate_first_sample_truncations(
                fs_map,
                tokenizer,
                cohort_sets=trunc_cohorts,
                condition=cond,
                alpha=alpha_val,
                label=label,
            )
        )

    df = pd.DataFrame(all_rows)
    if out_csv is not None:
        atomic_write_text(out_csv, df.to_csv(index=False))
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute Takehi et al. (2026) exact avg@4 and approximate re-tokenized budget curves."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="assets/paper_reference.csv",
        help="Destination CSV path (default: assets/paper_reference.csv).",
    )
    args = parser.parse_args(argv)

    df = build_paper_reference_table(out_csv=args.out)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
