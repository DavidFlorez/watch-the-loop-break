"""Unit and integration tests for nested budgets, forced open-box recovery, loop onset, McNemar stats, and paper reference."""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu.common import FORCE_ANSWER_SUFFIX, FORCE_BOX_OPEN_SUFFIX
from gpu.derive_budgets import derive_budgets_from_32k
from gpu.stats import generate_gpu_stats
from gpu.vllm_sweep import run_vllm_alpha
from metrics import (
    LOOP_ONSET_STRIDE_TOKENS,
    LOOP_ONSET_THRESHOLD,
    LOOP_ONSET_WINDOW_TOKENS,
    compute_loop_onset_from_token_ids,
    compute_loop_onset_token,
    exact_mcnemar_test,
    extract_forced_boxed,
)
from scripts.generate_article_charts import generate_all_charts
from scripts.paper_reference import (
    build_paper_reference_table,
    compute_avg_at_k_row,
    evaluate_first_sample_truncations,
)
from sweep import load_math500_subset


class _WordPieceMockTokenizer:
    """Deterministic integer-to-token mock tokenizer for offline unit testing."""

    def __init__(self) -> None:
        self._id_to_tok: Dict[int, str] = {}
        self._tok_to_id: Dict[str, int] = {}
        self.chat_template = "mock"

    def _tid(self, piece: str) -> int:
        if piece not in self._tok_to_id:
            tid = len(self._tok_to_id) + 10
            self._tok_to_id[piece] = tid
            self._id_to_tok[tid] = piece
        return self._tok_to_id[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        return [self._tid(piece) for piece in text.split(" ")]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return " ".join(self._id_to_tok.get(int(t), f"tok{int(t)}") for t in token_ids)

    def batch_decode(self, sequences: Sequence[Sequence[int]], skip_special_tokens: bool = False) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class _SequencedMockLLM:
    """Mock vLLM engine with programmable Pass-1 and Pass-2 outputs per prompt index."""

    def __init__(
        self,
        tokenizer: _WordPieceMockTokenizer,
        pass1_specs: List[Dict[str, Any]],
        pass2_specs: List[Dict[str, Any]],
    ) -> None:
        self.tok = tokenizer
        self.pass1_specs = list(pass1_specs)
        self.pass2_specs = list(pass2_specs)
        self.calls: List[Dict[str, Any]] = []
        self._p2_cursor = 0

    def generate(self, prompts: List[Any], sampling_params: Any) -> List[Any]:
        is_tokens_prompt = bool(prompts and isinstance(prompts[0], dict) and "prompt_token_ids" in prompts[0])
        self.calls.append(
            {
                "n_prompts": len(prompts),
                "is_tokens_prompt": is_tokens_prompt,
                "prompts": list(prompts),
            }
        )
        results = []
        if not is_tokens_prompt:
            for spec in self.pass1_specs:
                comp = SimpleNamespace(
                    token_ids=list(spec["token_ids"]),
                    text=spec.get("text"),
                    finish_reason=spec.get("finish_reason", "stop"),
                )
                results.append(SimpleNamespace(outputs=[comp]))
        else:
            for _ in prompts:
                spec = self.pass2_specs[self._p2_cursor]
                self._p2_cursor += 1
                comp = SimpleNamespace(
                    token_ids=list(spec["token_ids"]),
                    text=spec.get("text"),
                    finish_reason=spec.get("finish_reason", "stop"),
                )
                results.append(SimpleNamespace(outputs=[comp]))
        return results


class TestNestedBudgetsAndDerivedRuns(unittest.TestCase):
    """Verify subset prefix nesting, real N=50 prefix invariance, and `derive_budgets.py`."""

    def test_load_math500_subset_strictly_nested(self) -> None:
        s50 = load_math500_subset(50, seed=0)
        s100 = load_math500_subset(100, seed=0)
        s200 = load_math500_subset(200, seed=0)
        self.assertEqual([p["unique_id"] for p in s50], [p["unique_id"] for p in s100[:50]])
        self.assertEqual([p["unique_id"] for p in s100], [p["unique_id"] for p in s200[:100]])

    def test_committed_gpu_traces_are_exact_prefixes_of_32k(self) -> None:
        ref_root = REPO_ROOT / "assets"
        b32_jsonl = ref_root / "budget_32768" / "gpu-rtx-pro-6000-bfloat16" / "sweep_per_problem.jsonl.gz"
        if not b32_jsonl.is_file():
            self.skipTest("Committed 32k GPU results not found.")

        recs_32: Dict[tuple, Dict[str, Any]] = {}
        with gzip.open(b32_jsonl, "rt", encoding="utf-8") as gz:
            for line in gz:
                if line.strip():
                    r = json.loads(line)
                    recs_32[(float(r["alpha"]), str(r["unique_id"]), int(r.get("sample_idx", 0)))] = r

        self.assertEqual(len(recs_32), 150)

        checked = 0
        for b in (4096, 8192, 16384):
            b_jsonl = ref_root / f"budget_{b}" / "gpu-rtx-pro-6000-bfloat16" / "sweep_per_problem.jsonl.gz"
            self.assertTrue(b_jsonl.is_file(), f"Missing {b_jsonl}")
            with gzip.open(b_jsonl, "rt", encoding="utf-8") as gz:
                for line in gz:
                    if not line.strip():
                        continue
                    rb = json.loads(line)
                    key = (float(rb["alpha"]), str(rb["unique_id"]), int(rb.get("sample_idx", 0)))
                    self.assertIn(key, recs_32)
                    r32 = recs_32[key]
                    tb = str(rb["text"])
                    t32 = str(r32["text"])
                    self.assertTrue(
                        t32.startswith(tb),
                        f"Trace mismatch at budget={b} for {key}: prefix length {len(tb)} vs {len(t32)}",
                    )
                    if not bool(rb["hit_cap"]):
                        self.assertFalse(bool(r32["hit_cap"]))
                        self.assertEqual(int(rb["n_tokens"]), int(r32["n_tokens"]))
                        self.assertEqual(tb, t32)
                    else:
                        self.assertEqual(int(rb["n_tokens"]), b)
                    checked += 1
        self.assertEqual(checked, 450)

    def test_derive_budgets_offline_and_mock_live_pass2(self) -> None:
        tok = _WordPieceMockTokenizer()
        probs = load_math500_subset(2, seed=0)
        gold0 = str(probs[0]["answer"])
        gold1 = str(probs[1]["answer"])

        # Problem 0: 10 tokens with </think> at index 6 -> finishes inside 12 & 32k, capped BEFORE </think> at 6
        p0_words = ["think", "step", "one", "two", "three", "four", "</think>", "answer", "is", f"\\boxed{{{gold0}}}"]
        p0_ids = tok.encode(" ".join(p0_words), add_special_tokens=False)
        self.assertEqual(len(p0_ids), 10)

        # Problem 1: 20 tokens; emits </think> at token 10, then continues prose until cap (open_box at 12 & 32k, close_think at 6)
        p1_words = [f"w{i}" for i in range(9)] + ["</think>"] + [f"prose{i}" for i in range(10)]
        p1_ids = tok.encode(" ".join(p1_words), add_special_tokens=False)
        self.assertEqual(len(p1_ids), 20)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src" / "budget_32768" / "gpu-t4-float16"
            src_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = tmp_path / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            rec32_0 = {
                "alpha": 1.0,
                "max_new_tokens": 32768,
                "unique_id": probs[0]["unique_id"],
                "sample_idx": 0,
                "subject": probs[0]["subject"],
                "level": probs[0]["level"],
                "gold": gold0,
                "pred": gold0,
                "correct": True,
                "pred_strict": gold0,
                "correct_strict": True,
                "pred_forced": gold0,
                "correct_forced": True,
                "forced_tokens": 0,
                "forced_text": None,
                "forced_mode": None,
                "n_tokens": len(p0_ids),
                "hit_cap": False,
                "finish_reason": "stop",
                "closed_think": True,
                "rep_8gram_ratio": 0.0,
                "tail_rep_8gram_ratio": 0.0,
                "wall_time_s": 0.5,
                "backend": "vllm",
                "gpu": "t4",
                "dtype": "float16",
                "vllm_version": "0.33.0",
                "out_ids": p0_ids,
                "text": " ".join(p0_words),
            }
            rec32_1 = {
                "alpha": 1.0,
                "max_new_tokens": 32768,
                "unique_id": probs[1]["unique_id"],
                "sample_idx": 0,
                "subject": probs[1]["subject"],
                "level": probs[1]["level"],
                "gold": gold1,
                "pred": None,
                "correct": False,
                "pred_strict": None,
                "correct_strict": False,
                "pred_forced": None,
                "correct_forced": False,
                "forced_tokens": 0,
                "forced_text": None,
                "forced_mode": None,
                "n_tokens": len(p1_ids),
                "hit_cap": True,
                "finish_reason": "length",
                "closed_think": True,
                "rep_8gram_ratio": 0.0,
                "tail_rep_8gram_ratio": 0.0,
                "wall_time_s": 1.5,
                "backend": "vllm",
                "gpu": "t4",
                "dtype": "float16",
                "vllm_version": "0.33.0",
                "out_ids": p1_ids,
                "text": " ".join(p1_words),
            }
            with gzip.open(src_dir / "sweep_per_problem.jsonl.gz", "wt", encoding="utf-8") as gz:
                for r in (rec32_0, rec32_1):
                    gz.write(json.dumps(r) + "\n")
            (src_dir / "meta.json").write_text(
                json.dumps(
                    {"n_problems": 2, "samples": 1, "gpu_tag": "t4", "dtype": "float16", "vllm_version": "0.33.0"}
                ),
                encoding="utf-8",
            )

            p2_specs = [
                {"token_ids": [900], "text": f"\\boxed{{{gold0}}}"},
                {"token_ids": [901], "text": f"\\boxed{{{gold1}}}"},
                {"token_ids": [902], "text": f"{gold1}}}"},
                {"token_ids": [903], "text": f"{gold1}}}"},
            ]
            created_engines: List[_SequencedMockLLM] = []

            def _mock_llm_factory(**kwargs: Any) -> _SequencedMockLLM:
                eng = _SequencedMockLLM(tok, pass1_specs=[], pass2_specs=p2_specs)
                created_engines.append(eng)
                return eng

            res = derive_budgets_from_32k(
                src_32k_dir=src_dir,
                out_root=tmp_path / "derived",
                budgets=[6, 12, 32768],
                run_forced=True,
                ckpt_dir=ckpt_dir,
                dtype="float16",
                llm_cls=_mock_llm_factory,
                sampling_params_factory=lambda **kw: kw,
                tokenizer_override=tok,
                require_gpu=False,
            )
            self.assertEqual(res["forced_count_by_alpha"]["1"], 4)
            self.assertEqual(len(created_engines), 1)
            p2_prompts = created_engines[0].calls[0]["prompts"]
            close_suffix_ids = tok.encode(FORCE_ANSWER_SUFFIX, add_special_tokens=False)
            open_suffix_ids = tok.encode(FORCE_BOX_OPEN_SUFFIX, add_special_tokens=False)
            self.assertEqual(p2_prompts[0]["prompt_token_ids"][-len(close_suffix_ids) :], close_suffix_ids)
            self.assertEqual(p2_prompts[1]["prompt_token_ids"][-len(close_suffix_ids) :], close_suffix_ids)
            self.assertEqual(p2_prompts[2]["prompt_token_ids"][-len(open_suffix_ids) :], open_suffix_ids)
            self.assertEqual(p2_prompts[3]["prompt_token_ids"][-len(open_suffix_ids) :], open_suffix_ids)

            b12_csv = pd.read_csv(tmp_path / "derived" / "budget_12" / "gpu-t4-float16" / "sweep_results.csv")
            self.assertEqual(float(b12_csv.iloc[0]["accuracy_pct"]), 50.0)
            self.assertEqual(float(b12_csv.iloc[0]["forced_accuracy_pct"]), 100.0)
            self.assertEqual(float(b12_csv.iloc[0]["hit_cap_pct"]), 50.0)
            self.assertEqual(float(b12_csv.iloc[0]["closed_think_pct"]), 100.0)


class TestForcedOpenBoxAndVllmSweep(unittest.TestCase):
    """Test `open_box` vs `close_think` forced pass and `out_ids`/`forced_text` persistence."""

    def test_extract_forced_boxed_open_box_and_close_think(self) -> None:
        self.assertEqual(extract_forced_boxed("28}\n\nDone.", forced_mode="open_box"), "28")
        self.assertEqual(
            extract_forced_boxed("\\frac{\\sqrt{3}}{3}}\\n<|im_end|>", forced_mode="open_box"),
            "\\frac{\\sqrt{3}}{3}",
        )
        self.assertEqual(
            extract_forced_boxed("Therefore \\boxed{-14}.", forced_mode="close_think"),
            "-14",
        )

    def test_vllm_sweep_open_box_forced_mode_and_seed_jsonl_reforce(self) -> None:
        tok = _WordPieceMockTokenizer()
        probs = load_math500_subset(2, seed=0)
        gold0 = str(probs[0]["answer"])
        gold1 = str(probs[1]["answer"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            seed_path = tmp_path / "legacy_32k.jsonl.gz"
            legacy_rec = {
                "alpha": 1.0,
                "max_new_tokens": 32768,
                "unique_id": probs[0]["unique_id"],
                "sample_idx": 0,
                "subject": probs[0]["subject"],
                "level": probs[0]["level"],
                "gold": gold0,
                "pred": None,
                "correct": False,
                "pred_strict": None,
                "correct_strict": False,
                "pred_forced": None,
                "correct_forced": False,
                "forced_tokens": 12,
                "n_tokens": 32768,
                "hit_cap": True,
                "finish_reason": "length",
                "closed_think": True,
                "rep_8gram_ratio": 0.1,
                "tail_rep_8gram_ratio": 0.0,
                "wall_time_s": 5.0,
                "backend": "vllm",
                "gpu": "rtx-pro-6000",
                "dtype": "bfloat16",
                "vllm_version": "0.33.0",
                "text": "thinking done </think> rambling without box",
            }
            with gzip.open(seed_path, "wt", encoding="utf-8") as gz:
                gz.write(json.dumps(legacy_rec) + "\n")

            p1_txt = "thought </think> still explaining without final box"
            p1_ids = tok.encode(p1_txt, add_special_tokens=False)

            pass1_specs = [{"token_ids": p1_ids, "text": p1_txt, "finish_reason": "length"}]
            pass2_specs = [
                {"token_ids": [801], "text": f"{gold0}}}", "finish_reason": "stop"},
                {"token_ids": [802], "text": f"{gold1}}}", "finish_reason": "stop"},
            ]
            created: List[_SequencedMockLLM] = []

            def _mock_llm(**kw: Any) -> _SequencedMockLLM:
                eng = _SequencedMockLLM(tok, pass1_specs=pass1_specs, pass2_specs=pass2_specs)
                created.append(eng)
                return eng

            res = run_vllm_alpha(
                alpha=1.0,
                budget=32768,
                n_problems=2,
                samples=1,
                seed=0,
                dtype="bfloat16",
                gpu_tag="rtx-pro-6000",
                out_root=str(tmp_path / "out"),
                ckpt_dir=str(ckpt_dir),
                seed_jsonl=str(seed_path),
                llm_cls=_mock_llm,
                sampling_params_factory=lambda **kw: kw,
                tokenizer_override=tok,
                require_gpu=False,
            )
            self.assertEqual(len(created), 1)
            open_sfx_ids = tok.encode(FORCE_BOX_OPEN_SUFFIX, add_special_tokens=False)
            self.assertTrue(created[0].calls[0]["is_tokens_prompt"])
            self.assertEqual(created[0].calls[0]["prompts"][0]["prompt_token_ids"][-len(open_sfx_ids) :], open_sfx_ids)
            self.assertFalse(created[0].calls[1]["is_tokens_prompt"])
            self.assertTrue(created[0].calls[2]["is_tokens_prompt"])
            self.assertEqual(created[0].calls[2]["prompts"][0]["prompt_token_ids"][-len(open_sfx_ids) :], open_sfx_ids)

            out_jsonl = Path(res["out_dir"]) / "sweep_per_problem.jsonl.gz"
            with gzip.open(out_jsonl, "rt", encoding="utf-8") as gz:
                saved = [json.loads(line) for line in gz if line.strip()]
            self.assertEqual(len(saved), 2)
            for rec in saved:
                self.assertEqual(rec["forced_mode"], "open_box")
                self.assertTrue(rec["correct_forced"])
                self.assertFalse(rec["correct_strict"])
                self.assertIn("out_ids", rec)
                self.assertIsInstance(rec["out_ids"], list)
                self.assertIsNotNone(rec["forced_text"])


class TestLoopOnsetMcNemarAndPaperReference(unittest.TestCase):
    """Test sliding-window `loop_onset_token`, exact McNemar test, and paper reference helpers."""

    def test_loop_onset_sliding_window(self) -> None:
        self.assertEqual(LOOP_ONSET_WINDOW_TOKENS, 1024)
        self.assertEqual(LOOP_ONSET_STRIDE_TOKENS, 256)
        self.assertAlmostEqual(LOOP_ONSET_THRESHOLD, 0.50)

        tok = _WordPieceMockTokenizer()
        unique_prefix = [f"u{i}" for i in range(600)]
        loop_unit = ["wait", "if", "toppings", "are", "distinct", "then", "valid", "case"]
        loop_tail = (loop_unit * 260)[:2048]
        full_text = " ".join(unique_prefix + loop_tail)
        ids = tok.encode(full_text, add_special_tokens=False)

        onset = compute_loop_onset_from_token_ids(ids, tok)
        self.assertEqual(onset, 256)
        self.assertEqual(compute_loop_onset_token(full_text, tok), 256)

        clean_ids = tok.encode(" ".join(f"distinct_tok_{i}" for i in range(2500)), add_special_tokens=False)
        self.assertIsNone(compute_loop_onset_from_token_ids(clean_ids, tok))

        from scripts.generate_article_charts import _get_or_compute_loop_onset

        rep_words = " ".join(["wait if toppings are distinct then valid case"] * 200)
        self.assertIsNone(_get_or_compute_loop_onset({"loop_onset_token": None, "text": rep_words, "n_tokens": 1600}))
        self.assertEqual(_get_or_compute_loop_onset({"loop_onset_token": 5120, "text": rep_words}), 5120)
        self.assertEqual(_get_or_compute_loop_onset({"text": rep_words, "n_tokens": 1600}), 0)


    def test_exact_mcnemar_test_values(self) -> None:
        a = [True] * 3 + [False] * 17 + [False] * 30
        b = [True] * 3 + [True] * 17 + [False] * 30
        res = exact_mcnemar_test(a, b)
        self.assertEqual(res["n11_both_correct"], 3)
        self.assertEqual(res["n10_a_only"], 0)
        self.assertEqual(res["n01_b_only"], 17)
        self.assertEqual(res["n00_both_wrong"], 30)
        self.assertEqual(res["discordant"], 17)
        self.assertAlmostEqual(res["p_value"], 2.0 ** (-16), places=12)

        tie = exact_mcnemar_test([True, False, True], [True, False, True])
        self.assertEqual(tie["p_value"], 1.0)

    def test_gpu_stats_and_paper_reference_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "stats.csv"
            df = generate_gpu_stats(REPO_ROOT / "assets", out_csv=out_csv)
            self.assertGreaterEqual(len(df), 24)
            sub = df[
                (df["cohort"] == "n50_gpu-rtx-pro-6000-bfloat16")
                & (df["budget"] == 32768)
                & (df["metric"] == "strict")
                & (df["alpha_a"] == 0.0)
                & (df["alpha_b"] == 1.0)
            ]
            self.assertEqual(len(sub), 1)
            self.assertEqual(int(sub.iloc[0]["n01_b_only"]), 17)
            self.assertTrue(bool(sub.iloc[0]["significant_0_01"]))
            with open(out_csv, "r", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), len(df))

        tok = _WordPieceMockTokenizer()
        probs = load_math500_subset(2, seed=0)
        uid0 = str(probs[0]["unique_id"])
        uid1 = str(probs[1]["unique_id"])

        mock_pp = {
            uid0: {"id": uid0, "correct": [1, 1, 1, 0], "pred_first": "42"},
            uid1: {"id": uid1, "correct": [1, 0, 0, 0], "pred_first": None},
        }
        mock_fs = {
            uid0: {"id": uid0, "answer": str(probs[0]["answer"]), "text": f"step one two three </think> final \\boxed{{{probs[0]['answer']}}}"},
            uid1: {"id": uid1, "answer": str(probs[1]["answer"]), "text": " ".join([f"loop{i}" for i in range(32768)])},
        }
        avg_row = compute_avg_at_k_row(
            cohort="test_cohort",
            condition="p50c",
            alpha=1.0,
            label="Test",
            per_problem_map=mock_pp,
            target_ids=[uid0, uid1],
        )
        self.assertEqual(avg_row["accuracy_pct"], 50.0)
        self.assertEqual(avg_row["first_sample_accuracy_pct"], 100.0)

        trunc_rows = evaluate_first_sample_truncations(
            mock_fs,
            tok,
            cohort_sets=[("test_cohort", [uid0, uid1])],
            condition="p50c",
            alpha=1.0,
            label="Test",
            budgets=[4096, 32768],
        )
        self.assertEqual(len(trunc_rows), 2)
        self.assertEqual(trunc_rows[0]["status"], "approximate (re-tokenized)")
        self.assertEqual(trunc_rows[0]["accuracy_pct"], 50.0)
        self.assertEqual(trunc_rows[1]["accuracy_pct"], 50.0)

        with tempfile.TemporaryDirectory() as tmp:
            ref_csv = Path(tmp) / "paper_ref.csv"
            ref_df = build_paper_reference_table(
                out_csv=ref_csv,
                tokenizer_override=tok,
                condition_logs_override={
                    "unedited": (mock_pp, mock_fs),
                    "p50c": (mock_pp, mock_fs),
                },
            )
            self.assertFalse(ref_df.empty)
            self.assertTrue((ref_df["status"].isin(["exact (published avg@4)", "approximate (re-tokenized)"])).all())

    def test_article_chart_generation_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Populate isolated assets dir with weight_verification.json and synthetic 32k N=200 + 6-dose N=100 data
            ver_src = REPO_ROOT / "assets/weight_verification.json"
            if ver_src.is_file():
                (tmp_path / "weight_verification.json").write_text(ver_src.read_text(encoding="utf-8"), encoding="utf-8")

            core_dir = tmp_path / "gpu_n200/budget_32768/gpu-rtx-pro-6000-bfloat16"
            dose_dir = tmp_path / "gpu_dose_n100/budget_32768/gpu-rtx-pro-6000-bfloat16"
            core_dir.mkdir(parents=True, exist_ok=True)
            dose_dir.mkdir(parents=True, exist_ok=True)

            core_rows = []
            core_jsonl = []
            for a in (0.0, 1.0, 1.25):
                corr = 12 if a == 0.0 else (96 if a == 1.0 else 104)
                fcorr = 40 if a == 0.0 else (100 if a == 1.0 else 104)
                hcap = 188 if a == 0.0 else 72
                core_rows.append(
                    {
                        "alpha": a,
                        "n_problems": 200,
                        "n_samples": 1,
                        "total_evaluations": 200,
                        "correct": corr,
                        "accuracy": corr / 200.0,
                        "accuracy_pct": 100.0 * corr / 200.0,
                        "accuracy_ci95_low": 3.0 if a == 0.0 else 41.0,
                        "accuracy_ci95_high": 11.0 if a == 0.0 else 58.0,
                        "hit_cap_count": hcap,
                        "hit_cap_pct": 100.0 * hcap / 200.0,
                        "hit_cap_ci95_low": 88.0 if a == 0.0 else 29.0,
                        "hit_cap_ci95_high": 97.0 if a == 0.0 else 43.0,
                        "forced_correct": fcorr,
                        "forced_accuracy": fcorr / 200.0,
                        "forced_accuracy_pct": 100.0 * fcorr / 200.0,
                        "forced_accuracy_ci95_low": 15.0 if a == 0.0 else 43.0,
                        "forced_accuracy_ci95_high": 26.0 if a == 0.0 else 59.0,
                        "mean_tokens": 31000.0 if a == 0.0 else 21000.0,
                        "median_tokens": 32768.0 if a == 0.0 else 22000.0,
                        "mean_forced_tokens": 450.0 if a == 0.0 else 180.0,
                        "closed_think_pct": 6.0 if a == 0.0 else 75.0,
                        "mean_rep_8gram_pct": 72.0 if a == 0.0 else 15.0,
                        "mean_tail_rep_8gram_pct": 85.0 if a == 0.0 else 10.0,
                        "wall_time_s": 1200.0,
                        "sec_per_problem": 6.0,
                        "output_tok_per_s": 4600.0,
                        "backend": "vllm",
                        "gpu": "rtx-pro-6000",
                        "dtype": "bfloat16",
                        "vllm_version": "0.33.0",
                    }
                )
                for i in range(4):
                    uid = f"prob_{i}"
                    core_jsonl.append(
                        {
                            "alpha": a,
                            "budget": 32768,
                            "unique_id": uid,
                            "sample_idx": 0,
                            "hit_cap": (a == 0.0),
                            "finish_reason": "length" if a == 0.0 else "stop",
                            "correct_strict": (a > 0.0),
                            "correct_forced": (a > 0.0),
                            "loop_onset_token": 5120 if a == 0.0 else (18000 if i == 0 else None),
                            "n_tokens": 32768 if a == 0.0 else 3844,
                            "rep_8gram_ratio": 0.83 if a == 0.0 else 0.01,
                            "tail_rep_8gram_ratio": 0.95 if a == 0.0 else 0.0,
                            "text": "step one </think> \\boxed{28}",
                        }
                    )
            pd.DataFrame(core_rows).to_csv(core_dir / "sweep_results.csv", index=False)
            with gzip.open(core_dir / "sweep_per_problem.jsonl.gz", "wt", encoding="utf-8") as gz:
                for rec in core_jsonl:
                    gz.write(json.dumps(rec) + "\n")

            with gzip.open(dose_dir / "sweep_per_problem.jsonl.gz", "wt", encoding="utf-8") as gz:
                for a in (0.5, 0.75, 1.5):
                    for i in range(4):
                        gz.write(
                            json.dumps(
                                {
                                    "alpha": a,
                                    "budget": 32768,
                                    "unique_id": f"prob_{i}",
                                    "sample_idx": 0,
                                    "hit_cap": False,
                                    "finish_reason": "stop",
                                    "correct_strict": True,
                                    "correct_forced": True,
                                    "loop_onset_token": None,
                                    "n_tokens": 4200,
                                    "text": "solved </think> \\boxed{28}",
                                }
                            )
                            + "\n"
                        )

            summary = generate_all_charts(assets_root=tmp_path, article_dir=tmp_path / "article", with_tokenizer=False)
            self.assertEqual(summary["n_core"], 200)
            self.assertGreater(summary["stats_rows"], 0)
            expected_charts = {
                "cover_5x2": (1500, 600),
                "headline_budget_curve": (1600, 1000),
                "headline_budget_curve_full": (1600, 1000),
                "hero_trace": (1600, 1000),
                "outcomes_32k": (1600, 1000),
                "outcomes_32k_core": (1600, 1000),
                "outcomes_dose": (1600, 1000),
                "loop_onset": (1600, 1000),
                "loop_onset_full": (1600, 1000),
                "edit_diagram": (1600, 1000),
            }
            self.assertEqual(set(summary["article_charts"].keys()), set(expected_charts.keys()))
            for name, path_str in summary["article_charts"].items():
                p = Path(path_str)
                self.assertTrue(p.is_file(), f"Missing chart {name}")
                with Image.open(p) as im:
                    self.assertEqual(im.size, expected_charts[name], f"{name} size mismatch: got {im.size}")


if __name__ == "__main__":
    unittest.main()
