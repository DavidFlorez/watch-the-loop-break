"""FP16 vs FP32 vLLM numerical smoke test on 8 MATH-500 problems for Qwen3.5-0.8B."""

from __future__ import annotations

import os

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu.common import (
    STOP_TOKEN_IDS,
    atomic_write_text,
    check_min_compute_capability,
    cleanup_ckpt_dir,
    collect_environment_metadata,
    create_vllm_engine,
    detect_nan_junk,
    make_scratch_ckpt_dir,
    release_vllm_engine,
    utc_now_iso,
)
from gpu.make_ckpt import build_checkpoint
from metrics import build_math_prompt
from sweep import load_math500_subset

SMOKE_ALPHAS: Tuple[float, float] = (0.0, 1.0)
PREFIX_CHECK_LEN = 128
MAX_ALLOWED_PROMPT_DISAGREEMENTS = 2


def _make_greedy_sampling_params(max_tokens: int, seed: int) -> Any:
    try:
        from vllm import SamplingParams

        try:
            return SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                presence_penalty=0.0,
                max_tokens=int(max_tokens),
                seed=int(seed),
                stop_token_ids=list(STOP_TOKEN_IDS),
                skip_special_tokens=False,
            )
        except TypeError:
            return SamplingParams(
                n=1,
                temperature=0.0,
                max_tokens=int(max_tokens),
                seed=int(seed),
            )
    except ImportError as exc:
        raise RuntimeError(
            "vllm is not installed. Run on a GPU runtime with `requirements-gpu.txt` installed."
        ) from exc


def _first_diff_index(seq_a: Sequence[int], seq_b: Sequence[int], limit: int = PREFIX_CHECK_LEN) -> Optional[int]:
    """Return first index `< limit` where `seq_a` and `seq_b` differ (or length mismatch), else `None`."""
    sub_a = list(seq_a[:limit])
    sub_b = list(seq_b[:limit])
    if sub_a == sub_b:
        return None
    common = min(len(sub_a), len(sub_b))
    for i in range(common):
        if sub_a[i] != sub_b[i]:
            return i
    return common


def evaluate_smoke_outputs(
    problems: Sequence[Dict[str, Any]],
    runs_by_key: Dict[tuple[float, str], List[Dict[str, Any]]],
    prefix_len: int = PREFIX_CHECK_LEN,
    max_allowed_disagreements: int = MAX_ALLOWED_PROMPT_DISAGREEMENTS,
) -> Dict[str, Any]:
    """Evaluate NaN/junk and 128-token greedy prefix agreement between `float16` and `float32`."""
    fp16_nan_failures: List[Dict[str, Any]] = []
    fp32_nan_failures: List[Dict[str, Any]] = []

    for dtype_label, target_list in (("float16", fp16_nan_failures), ("float32", fp32_nan_failures)):
        for alpha in SMOKE_ALPHAS:
            outputs = runs_by_key.get((float(alpha), dtype_label), [])
            if len(outputs) != len(problems):
                target_list.append(
                    {
                        "alpha": float(alpha),
                        "unique_id": "__count_mismatch__",
                        "reason": f"expected {len(problems)} outputs, got {len(outputs)}",
                    }
                )
            for prob, out in zip(problems, outputs):
                toks = out.get("token_ids", [])
                txt = out.get("text", "")
                reason = detect_nan_junk(txt, token_ids=toks)
                if reason is not None:
                    target_list.append(
                        {
                            "alpha": float(alpha),
                            "unique_id": prob["unique_id"],
                            "reason": reason,
                        }
                    )

    per_prompt: List[Dict[str, Any]] = []
    a0_fp16 = runs_by_key.get((0.0, "float16"), [])
    a0_fp32 = runs_by_key.get((0.0, "float32"), [])
    a1_fp16 = runs_by_key.get((1.0, "float16"), [])
    a1_fp32 = runs_by_key.get((1.0, "float32"), [])

    for i, prob in enumerate(problems):
        ids_16_0 = list(a0_fp16[i].get("token_ids", [])) if i < len(a0_fp16) else []
        ids_32_0 = list(a0_fp32[i].get("token_ids", [])) if i < len(a0_fp32) else []
        ids_16_1 = list(a1_fp16[i].get("token_ids", [])) if i < len(a1_fp16) else []
        ids_32_1 = list(a1_fp32[i].get("token_ids", [])) if i < len(a1_fp32) else []

        diff_0 = _first_diff_index(ids_16_0, ids_32_0, limit=prefix_len)
        diff_1 = _first_diff_index(ids_16_1, ids_32_1, limit=prefix_len)
        a0_match = diff_0 is None
        a1_match = diff_1 is None
        disagreed = (not a0_match) or (not a1_match)

        per_prompt.append(
            {
                "unique_id": prob["unique_id"],
                "a0_match": a0_match,
                "a1_match": a1_match,
                "disagreed": disagreed,
                "a0_first_diff_idx": diff_0,
                "a1_first_diff_idx": diff_1,
                "fp16_a0_len": len(ids_16_0),
                "fp32_a0_len": len(ids_32_0),
                "fp16_a1_len": len(ids_16_1),
                "fp32_a1_len": len(ids_32_1),
            }
        )

    prompt_disagreements = sum(1 for item in per_prompt if item["disagreed"])
    a0_disagreements = sum(1 for item in per_prompt if not item["a0_match"])
    a1_disagreements = sum(1 for item in per_prompt if not item["a1_match"])

    failure_reasons: List[str] = []
    if fp16_nan_failures:
        failure_reasons.append(
            f"fp16 emitted NaN/corrupt output on {len(fp16_nan_failures)} trace(s): {fp16_nan_failures[0]['reason']}"
        )
    if prompt_disagreements > max_allowed_disagreements:
        failure_reasons.append(
            f"fp16 disagreed with fp32 within first {prefix_len} greedy tokens on "
            f"{prompt_disagreements}/{len(problems)} prompts (max allowed: {max_allowed_disagreements})"
        )

    fp16_passed = len(failure_reasons) == 0
    recommended_dtype = "float16" if fp16_passed else "float32"

    return {
        "fp16_passed": fp16_passed,
        "recommended_dtype": recommended_dtype,
        "failure_reasons": failure_reasons,
        "fp16_nan_failures": fp16_nan_failures,
        "fp32_nan_failures": fp32_nan_failures,
        "prompt_disagreements": prompt_disagreements,
        "alpha0_prefix_disagreements": a0_disagreements,
        "alpha1_prefix_disagreements": a1_disagreements,
        "per_prompt": per_prompt,
    }


def evaluate_native_bf16_smoke_outputs(
    problems: Sequence[Dict[str, Any]],
    runs_by_key: Dict[tuple[float, str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Validate native `bfloat16` outputs on Ampere/Ada+ (`CC >= 8.0`) GPUs against NaN/corruption."""
    bf16_nan_failures: List[Dict[str, Any]] = []
    for alpha in SMOKE_ALPHAS:
        outputs = runs_by_key.get((float(alpha), "bfloat16"), [])
        if len(outputs) != len(problems):
            bf16_nan_failures.append(
                {
                    "alpha": float(alpha),
                    "unique_id": "__count_mismatch__",
                    "reason": f"expected {len(problems)} outputs, got {len(outputs)}",
                }
            )
        for prob, out in zip(problems, outputs):
            toks = out.get("token_ids", [])
            txt = out.get("text", "")
            reason = detect_nan_junk(txt, token_ids=toks)
            if reason is not None:
                bf16_nan_failures.append(
                    {
                        "alpha": float(alpha),
                        "unique_id": prob["unique_id"],
                        "reason": reason,
                    }
                )

    a0_bf16 = runs_by_key.get((0.0, "bfloat16"), [])
    a1_bf16 = runs_by_key.get((1.0, "bfloat16"), [])
    per_prompt: List[Dict[str, Any]] = []
    for i, prob in enumerate(problems):
        ids_0 = list(a0_bf16[i].get("token_ids", [])) if i < len(a0_bf16) else []
        ids_1 = list(a1_bf16[i].get("token_ids", [])) if i < len(a1_bf16) else []
        per_prompt.append(
            {
                "unique_id": prob["unique_id"],
                "a0_match": True,
                "a1_match": True,
                "disagreed": False,
                "a0_first_diff_idx": None,
                "a1_first_diff_idx": None,
                "bf16_a0_len": len(ids_0),
                "bf16_a1_len": len(ids_1),
            }
        )

    failure_reasons: List[str] = []
    if bf16_nan_failures:
        failure_reasons.append(
            f"bfloat16 emitted NaN/corrupt output on {len(bf16_nan_failures)} trace(s): {bf16_nan_failures[0]['reason']}"
        )

    bf16_passed = len(failure_reasons) == 0
    recommended_dtype = "bfloat16" if bf16_passed else "float32"

    return {
        "native_bfloat16": True,
        "bf16_passed": bf16_passed,
        "fp16_passed": bf16_passed,
        "recommended_dtype": recommended_dtype,
        "failure_reasons": failure_reasons,
        "bf16_nan_failures": bf16_nan_failures,
        "fp16_nan_failures": [],
        "fp32_nan_failures": [],
        "prompt_disagreements": 0,
        "alpha0_prefix_disagreements": 0,
        "alpha1_prefix_disagreements": 0,
        "per_prompt": per_prompt,
    }


def run_smoke_test(
    n_problems: int = 8,
    max_tokens: int = 1024,
    seed: int = 0,
    out_root: str = "assets",
    gpu_tag: Optional[str] = None,
    enforce_eager: bool = False,
    gpu_mem: float = 0.85,
    llm_cls: Optional[Any] = None,
    sampling_params_factory: Optional[Any] = None,
    tokenizer_override: Optional[Any] = None,
    ckpt_dirs_override: Optional[Dict[float, Path]] = None,
    require_gpu: bool = True,
    native_bf16_override: Optional[bool] = None,
    force_fp16_check: bool = False,
) -> Dict[str, Any]:
    """Execute the 8-problem greedy smoke test and write `assets/gpu_smoke_<gpu>.json`.

    On CC >= 8.0 GPUs (e.g., L4/A100), validates native `bfloat16` unless `force_fp16_check=True`.
    On Turing (`CC 7.5`, e.g., T4), runs the FP16 vs FP32 prefix agreement & NaN smoke test.
    """
    check_min_compute_capability(min_cc=7.5, require_gpu=require_gpu)
    env_meta = collect_environment_metadata(gpu_tag_override=gpu_tag)
    resolved_gpu_tag = env_meta["gpu_tag"]
    use_native_bf16 = (
        False
        if force_fp16_check
        else (
            bool(native_bf16_override)
            if native_bf16_override is not None
            else bool(env_meta.get("native_bfloat16"))
        )
    )

    problems = load_math500_subset(n=n_problems, seed=seed)
    t_start = time.perf_counter()

    owned_ckpts: List[Path] = []
    ckpt_paths: Dict[float, Path] = {}
    try:
        if ckpt_dirs_override is not None:
            ckpt_paths = {float(k): Path(v) for k, v in ckpt_dirs_override.items()}
        else:
            for alpha in SMOKE_ALPHAS:
                cdir = make_scratch_ckpt_dir(alpha, prefix="smoke")
                owned_ckpts.append(cdir)
                print(f"[Smoke] Building scratch checkpoint for alpha={alpha:g} -> {cdir}", flush=True)
                build_checkpoint(alpha=alpha, out_dir=cdir)
                ckpt_paths[alpha] = cdir

        if tokenizer_override is not None:
            tokenizer = tokenizer_override
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_paths[0.0]))

        prompts: List[str] = []
        for p in problems:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_math_prompt(p["problem"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            assert formatted.endswith("<think>\n"), (
                f"Expected prompt to end with '<think>\\n', got {formatted[-40:]!r}"
            )
            prompts.append(formatted)

        sampling_params = (
            sampling_params_factory(max_tokens=max_tokens, seed=seed)
            if sampling_params_factory is not None
            else _make_greedy_sampling_params(max_tokens=max_tokens, seed=seed)
        )

        runs_by_key: Dict[tuple[float, str], List[Dict[str, Any]]] = {}
        engine_meta: Dict[str, Any] = {}
        max_model_len = int(max_tokens) + 2048
        dtypes_to_check = ("bfloat16",) if use_native_bf16 else ("float16", "float32")

        for dtype_str in dtypes_to_check:
            for alpha in SMOKE_ALPHAS:
                print(
                    f"[Smoke] Running alpha={alpha:g}, dtype={dtype_str}, N={len(prompts)}, "
                    f"temperature=0.0, max_tokens={max_tokens} ...",
                    flush=True,
                )
                llm = None
                try:
                    llm, e_meta = create_vllm_engine(
                        ckpt_dir=ckpt_paths[alpha],
                        dtype=dtype_str,
                        max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_mem,
                        seed=seed,
                        enforce_eager=enforce_eager,
                        llm_cls=llm_cls,
                    )
                    engine_meta[f"alpha_{alpha:g}_{dtype_str}"] = e_meta
                    req_outputs = llm.generate(prompts, sampling_params)
                    parsed: List[Dict[str, Any]] = []
                    for ro in req_outputs:
                        comp = ro.outputs[0]
                        toks = list(comp.token_ids)
                        txt = (
                            comp.text
                            if getattr(comp, "text", None) is not None
                            else tokenizer.decode(toks, skip_special_tokens=False)
                        )
                        parsed.append(
                            {
                                "token_ids": toks,
                                "text": txt,
                                "finish_reason": getattr(comp, "finish_reason", "stop"),
                            }
                        )
                    runs_by_key[(alpha, dtype_str)] = parsed
                finally:
                    release_vllm_engine(llm)

        if use_native_bf16:
            eval_res = evaluate_native_bf16_smoke_outputs(problems=problems, runs_by_key=runs_by_key)
        else:
            eval_res = evaluate_smoke_outputs(
                problems=problems,
                runs_by_key=runs_by_key,
                prefix_len=PREFIX_CHECK_LEN,
                max_allowed_disagreements=MAX_ALLOWED_PROMPT_DISAGREEMENTS,
            )
            eval_res["native_bfloat16"] = False
        wall_s = round(time.perf_counter() - t_start, 2)

        report: Dict[str, Any] = {
            "timestamp_utc": utc_now_iso(),
            "gpu_tag": resolved_gpu_tag,
            "gpu_name": env_meta["gpu_name"],
            "compute_capability": env_meta.get("compute_capability"),
            "n_problems": len(problems),
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "seed": int(seed),
            "prefix_check_tokens": PREFIX_CHECK_LEN,
            "max_allowed_prompt_disagreements": MAX_ALLOWED_PROMPT_DISAGREEMENTS,
            "wall_time_s": wall_s,
            **eval_res,
            "engine_meta": engine_meta,
            "environment": env_meta,
        }

        out_path = Path(out_root) / f"gpu_smoke_{resolved_gpu_tag}.json"
        import json

        atomic_write_text(out_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        status_word = "PASSED" if report["fp16_passed"] else "FAILED"
        print(
            f"[Smoke] {status_word} | GPU={resolved_gpu_tag} ({env_meta['gpu_name']}) | "
            f"native_bf16={use_native_bf16} | prompt_disagreements={report['prompt_disagreements']}/{len(problems)} | "
            f"recommended_dtype={report['recommended_dtype']} | saved -> {out_path}",
            flush=True,
        )
        if report["failure_reasons"]:
            for reason in report["failure_reasons"]:
                print(f"  - Reason: {reason}", flush=True)
        return report
    finally:
        for cdir in owned_ckpts:
            cleanup_ckpt_dir(cdir)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run vLLM numerical smoke test on 8 MATH-500 problems (native bfloat16 on CC>=8.0, or FP16 vs FP32 on T4)."
    )
    parser.add_argument("--n", type=int, default=8, help="Number of MATH-500 problems to check (default: 8).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max greedy tokens (default: 1024).")
    parser.add_argument("--seed", type=int, default=0, help="Dataset & sampling seed (default: 0).")
    parser.add_argument("--out-root", type=str, default="assets", help="Directory for gpu_smoke_<gpu>.json.")
    parser.add_argument("--gpu-tag", type=str, default=None, help="Override normalized GPU tag (e.g. 't4', 'l4').")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM.")
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (default: 0.85).")
    parser.add_argument(
        "--force-fp16-check",
        action="store_true",
        help="Force FP16 vs FP32 comparison even on CC >= 8.0 GPUs with native bfloat16.",
    )
    args = parser.parse_args(argv)

    run_smoke_test(
        n_problems=args.n,
        max_tokens=args.max_tokens,
        seed=args.seed,
        out_root=args.out_root,
        gpu_tag=args.gpu_tag,
        enforce_eager=args.enforce_eager,
        gpu_mem=args.gpu_mem,
        force_fp16_check=args.force_fp16_check,
    )


if __name__ == "__main__":
    main()
