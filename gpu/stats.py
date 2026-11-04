"""Paired exact McNemar hypothesis tests and Wilson 95% CIs across GPU alpha/budget runs."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from gpu.common import atomic_write_text
from metrics import exact_mcnemar_test, wilson_ci


def _load_per_problem_by_alpha(jsonl_gz_path: Path) -> Dict[float, Dict[Tuple[str, int], Dict[str, Any]]]:
    """Load `sweep_per_problem.jsonl.gz` (or CSV fallback) grouped by `alpha -> (unique_id, sample_idx)`."""
    grouped: Dict[float, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    if jsonl_gz_path.is_file():
        with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                a_f = float(rec["alpha"])
                key = (str(rec["unique_id"]), int(rec.get("sample_idx", 0)))
                grouped.setdefault(a_f, {})[key] = rec
        return grouped

    csv_path = jsonl_gz_path.parent / "sweep_per_problem.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rec = row.to_dict()
            a_f = float(rec["alpha"])
            key = (str(rec["unique_id"]), int(rec.get("sample_idx", 0)))
            grouped.setdefault(a_f, {})[key] = rec
    return grouped


def _discover_budget_dirs(root: Path) -> Dict[int, Path]:
    """Return `{budget: gpu_run_dir}` for all `budget_<B>/gpu-*` under `root`."""
    found: Dict[int, Path] = {}
    if not root.is_dir():
        return found
    for b_dir in sorted(root.iterdir()):
        if not b_dir.is_dir() or not b_dir.name.startswith("budget_"):
            continue
        suffix = b_dir.name.replace("budget_", "")
        if not suffix.isdigit():
            continue
        b_int = int(suffix)
        gpu_cands = sorted(
            [
                d
                for d in b_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("gpu-")
                and ((d / "sweep_per_problem.jsonl.gz").is_file() or (d / "sweep_per_problem.csv").is_file())
            ]
        )
        if gpu_cands:
            found[b_int] = gpu_cands[0]
    return found


def compute_paired_stats_for_maps(
    *,
    cohort: str,
    budget: int,
    alpha_maps: Dict[float, Dict[Tuple[str, int], Dict[str, Any]]],
    metrics_list: Sequence[str] = ("strict", "forced"),
    required_ids: Optional[Sequence[Tuple[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Compute Wilson 95% CIs and two-sided exact McNemar p-values for all `alpha_a < alpha_b` pairs."""
    alphas = sorted(alpha_maps.keys())
    if len(alphas) < 2:
        return []

    req_set = set(required_ids) if required_ids is not None else None
    rows: List[Dict[str, Any]] = []

    for metric in metrics_list:
        field = "correct_strict" if metric == "strict" else "correct_forced"
        for i in range(len(alphas)):
            for j in range(i + 1, len(alphas)):
                a_val = alphas[i]
                b_val = alphas[j]
                map_a = alpha_maps[a_val]
                map_b = alpha_maps[b_val]
                common_keys = sorted(k for k in map_a.keys() if k in map_b and (req_set is None or k in req_set))
                if not common_keys:
                    continue

                vec_a = [bool(map_a[k].get(field, False)) for k in common_keys]
                vec_b = [bool(map_b[k].get(field, False)) for k in common_keys]
                mc = exact_mcnemar_test(vec_a, vec_b)

                n_p = int(mc["n_paired"])
                corr_a = int(sum(1 for x in vec_a if x))
                corr_b = int(sum(1 for x in vec_b if x))
                lo_a, hi_a = wilson_ci(corr_a, n_p)
                lo_b, hi_b = wilson_ci(corr_b, n_p)

                rows.append(
                    {
                        "cohort": cohort,
                        "budget": int(budget),
                        "metric": metric,
                        "alpha_a": a_val,
                        "alpha_b": b_val,
                        "n_paired": n_p,
                        "correct_a": corr_a,
                        "accuracy_a_pct": round((corr_a / n_p) * 100.0, 2),
                        "ci95_low_a_pct": round(lo_a * 100.0, 2),
                        "ci95_high_a_pct": round(hi_a * 100.0, 2),
                        "correct_b": corr_b,
                        "accuracy_b_pct": round((corr_b / n_p) * 100.0, 2),
                        "ci95_low_b_pct": round(lo_b * 100.0, 2),
                        "ci95_high_b_pct": round(hi_b * 100.0, 2),
                        "diff_b_minus_a_pp": mc["diff_b_minus_a_pp"],
                        "n11_both_correct": mc["n11_both_correct"],
                        "n10_a_only": mc["n10_a_only"],
                        "n01_b_only": mc["n01_b_only"],
                        "n00_both_wrong": mc["n00_both_wrong"],
                        "discordant": mc["discordant"],
                        "mcnemar_p_value": round(float(mc["p_value"]), 6),
                        "significant_0_05": bool(float(mc["p_value"]) < 0.05),
                        "significant_0_01": bool(float(mc["p_value"]) < 0.01),
                    }
                )
    return rows


def generate_gpu_stats(
    assets_root: Union[str, Path] = "assets",
    out_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Compute paired Wilson CIs + exact McNemar statistics across all available GPU cohorts and save CSV."""
    root = Path(assets_root)
    target_csv = Path(out_csv) if out_csv is not None else (root / "gpu_stats.csv")
    all_rows: List[Dict[str, Any]] = []

    # 1. Committed N=50 runs directly under `assets/budget_<B>/gpu-*`
    n50_dirs = _discover_budget_dirs(root)
    for b_int, run_dir in sorted(n50_dirs.items()):
        a_maps = _load_per_problem_by_alpha(run_dir / "sweep_per_problem.jsonl.gz")
        n_tag = max((len(v) for v in a_maps.values()), default=0)
        cohort_name = f"n{n_tag}_{run_dir.name}"
        all_rows.extend(
            compute_paired_stats_for_maps(cohort=cohort_name, budget=b_int, alpha_maps=a_maps)
        )

    # 2. Core N=200 runs under `assets/gpu_n200/budget_<B>/gpu-*`
    core_root = root / "gpu_n200"
    core_dirs = _discover_budget_dirs(core_root)
    for b_int, run_dir in sorted(core_dirs.items()):
        a_maps = _load_per_problem_by_alpha(run_dir / "sweep_per_problem.jsonl.gz")
        n_tag = max((len(v) for v in a_maps.values()), default=0)
        cohort_name = f"n{n_tag}_{run_dir.name}"
        all_rows.extend(
            compute_paired_stats_for_maps(cohort=cohort_name, budget=b_int, alpha_maps=a_maps)
        )

    # 3. Combined 6-alpha dose cohort on the first N=100 problems (core[:100] + dose[:100])
    dose_root = root / "gpu_dose_n100"
    dose_dirs = _discover_budget_dirs(dose_root)
    for b_int, dose_dir in sorted(dose_dirs.items()):
        dose_maps = _load_per_problem_by_alpha(dose_dir / "sweep_per_problem.jsonl.gz")
        if not dose_maps:
            continue
        first_alpha = next(iter(dose_maps.keys()))
        dose_keys = sorted(dose_maps[first_alpha].keys())
        combined_maps = dict(dose_maps)
        if b_int in core_dirs:
            core_maps = _load_per_problem_by_alpha(core_dirs[b_int] / "sweep_per_problem.jsonl.gz")
            for a_val, m in core_maps.items():
                combined_maps.setdefault(a_val, m)
        n_dose = len(dose_keys)
        cohort_name = f"n{n_dose}_dose_{dose_dir.name}"
        all_rows.extend(
            compute_paired_stats_for_maps(
                cohort=cohort_name,
                budget=b_int,
                alpha_maps=combined_maps,
                required_ids=dose_keys,
            )
        )

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["cohort", "budget", "metric", "alpha_a", "alpha_b"]).reset_index(drop=True)
    atomic_write_text(target_csv, df.to_csv(index=False))
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute Wilson 95% CIs and exact two-sided McNemar tests across GPU alpha pairs."
    )
    parser.add_argument("--assets-root", type=str, default="assets", help="Assets directory (default: assets).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: <assets-root>/gpu_stats.csv).",
    )
    args = parser.parse_args(argv)

    df = generate_gpu_stats(assets_root=args.assets_root, out_csv=args.out)
    if not df.empty:
        print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
