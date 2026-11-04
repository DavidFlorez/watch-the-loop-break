"""Unit and integration tests for `weights.py`, `metrics.py`, `sweep.py`, `app.py`, `gpu/`, and `notebooks/gpu_sweep.ipynb`."""

from __future__ import annotations

import ast
import gzip
import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from app import render_repetition_meter_html
from gpu.aggregate import GPU_NOTES_END, GPU_NOTES_START, run_aggregation
from gpu.common import (
    FORCE_ANSWER_SUFFIX,
    create_vllm_engine,
    detect_nan_junk,
    normalize_gpu_tag,
)
from gpu.make_ckpt import build_checkpoint
from gpu.smoke_test import evaluate_smoke_outputs, run_smoke_test
from gpu.vllm_sweep import run_vllm_alpha
from metrics import (
    PresencePenaltyLogitsProcessor,
    build_math_prompt,
    extract_last_boxed,
    extract_last_boxed_strict,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
    wilson_ci,
)
from sweep import save_sweep_charts
from weights import (
    EXPECTED_SHA256,
    TARGET_TENSORS,
    bf16_sha256,
    blend,
    detect_hardware,
    load_target_tensors,
)


class DummyQwenLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "15": torch.nn.ModuleDict(
                    {
                        "mlp": torch.nn.ModuleDict(
                            {"down_proj": torch.nn.Linear(3584, 1024, bias=False, dtype=torch.bfloat16)}
                        )
                    }
                ),
                "19": torch.nn.ModuleDict(
                    {
                        "mlp": torch.nn.ModuleDict(
                            {"down_proj": torch.nn.Linear(3584, 1024, bias=False, dtype=torch.bfloat16)}
                        )
                    }
                ),
            }
        )


class DummyQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = DummyQwenLanguageModel()


class DummyQwenTop(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DummyQwenModel()


class DummyWordTokenizer:
    """Lightweight deterministic word-level tokenizer for CPU unit tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<think>": 1, "</think>": 2}

    def _tok_id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab) + 10
        return self._vocab[tok]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        content = messages[-1]["content"]
        base = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        if enable_thinking:
            base += "<think>\n"
        return base

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._tok_id(tok) for tok in text.replace("\n", " \n ").split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        rev = {v: k for k, v in self._vocab.items()}
        return " ".join(rev.get(int(t), f"t{t}") for t in token_ids)


class TestWeightsAndBlend(unittest.TestCase):
    def test_detect_hardware_returns_valid_triple(self) -> None:
        device, dtype, desc = detect_hardware()
        self.assertIn(device, ("cuda", "mps", "cpu"))
        self.assertIn(dtype, (torch.bfloat16, torch.float16, torch.float32))
        self.assertTrue(len(desc) > 0)

    def test_blend_endpoints_and_linearity(self) -> None:
        base_t, edit_t = load_target_tensors()
        self.assertEqual(set(base_t.keys()), set(TARGET_TENSORS))
        self.assertEqual(set(edit_t.keys()), set(TARGET_TENSORS))

        b0 = blend(0.0, base_tensors=base_t, edit_tensors=edit_t)
        b1 = blend(1.0, base_tensors=base_t, edit_tensors=edit_t)
        b05 = blend(0.5, base_tensors=base_t, edit_tensors=edit_t)
        b125 = blend(1.25, base_tensors=base_t, edit_tensors=edit_t)
        b15 = blend(1.5, base_tensors=base_t, edit_tensors=edit_t)

        for name in TARGET_TENSORS:
            self.assertEqual(bf16_sha256(b0[name]), EXPECTED_SHA256[name]["base"])
            self.assertEqual(bf16_sha256(b1[name]), EXPECTED_SHA256[name]["edit"])
            diff_05 = (b05[name].float() - b0[name].float()).abs().max().item()
            diff_full = (b1[name].float() - b0[name].float()).abs().max().item()
            self.assertAlmostEqual(diff_05 / diff_full, 0.5, places=2)
            diff_125 = (b125[name].float() - b0[name].float()).abs().max().item()
            self.assertAlmostEqual(diff_125 / diff_full, 1.25, places=2)
            diff_15 = (b15[name].float() - b0[name].float()).abs().max().item()
            self.assertAlmostEqual(diff_15 / diff_full, 1.5, places=2)

    def test_blend_updates_model_in_place(self) -> None:
        base_t, edit_t = load_target_tensors()
        dummy = DummyQwenTop()
        blend(1.0, model=dummy, base_tensors=base_t, edit_tensors=edit_t)
        for name in TARGET_TENSORS:
            param = dict(dummy.named_parameters())[name]
            self.assertEqual(bf16_sha256(param), EXPECTED_SHA256[name]["edit"])


class TestMetricsAndGrading(unittest.TestCase):
    def test_repeated_8gram_ratio_unique_vs_looping(self) -> None:
        unique_text = " ".join(f"tok{i}" for i in range(60))
        self.assertEqual(repeated_8gram_ratio(unique_text), 0.0)

        phrase = "let us check if x equals two now"
        loop_text = " ".join([phrase] * 25)
        ratio = repeated_8gram_ratio(loop_text)
        self.assertGreater(ratio, 0.90)
        self.assertGreater(tail_repeated_8gram_ratio(loop_text), 0.90)

    def test_presence_penalty_processor_applies_once_per_unique_token(self) -> None:
        proc = PresencePenaltyLogitsProcessor(presence_penalty=1.5, prompt_length=3)
        input_ids = torch.tensor([[10, 20, 30, 5, 5, 6]], dtype=torch.long)
        scores = torch.zeros((1, 50), dtype=torch.float32)
        out = proc(input_ids, scores)
        self.assertAlmostEqual(float(out[0, 5]), -1.5, places=5)
        self.assertAlmostEqual(float(out[0, 6]), -1.5, places=5)
        self.assertAlmostEqual(float(out[0, 10]), 0.0, places=5)

    def test_extract_and_grade_math(self) -> None:
        sample = "<think>First wait...</think>Therefore the radius is \\boxed{\\frac{3}{2}}."
        pred = extract_last_boxed(sample)
        self.assertEqual(pred, "\\frac{3}{2}")
        self.assertEqual(extract_last_boxed_strict(sample), "\\frac{3}{2}")
        self.assertTrue(grade_math_answer(pred, "1.5"))
        self.assertTrue(grade_math_answer(pred, "\\dfrac{3}{2}"))
        self.assertFalse(grade_math_answer(pred, "2"))
        self.assertIn("\\boxed{}", build_math_prompt("Find x."))

    def test_extract_last_boxed_strict_has_no_think_fallback(self) -> None:
        closed_no_box_after = "<think>Wait I think it is \\boxed{42}.</think>Now let me summarize without a box."
        self.assertEqual(extract_last_boxed(closed_no_box_after), "42")
        self.assertIsNone(extract_last_boxed_strict(closed_no_box_after))

        unclosed_trace = "<think>Still reasoning... maybe \\boxed{7}"
        self.assertEqual(extract_last_boxed_strict(unclosed_trace), "7")

    def test_wilson_ci_bounds(self) -> None:
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))
        lo0, hi0 = wilson_ci(0, 50)
        self.assertEqual(lo0, 0.0)
        self.assertAlmostEqual(hi0, 0.07135, places=3)
        lo_mid, hi_mid = wilson_ci(25, 50)
        self.assertAlmostEqual((lo_mid + hi_mid) / 2.0, 0.5, places=3)
        self.assertLess(lo_mid, 0.5)
        self.assertGreater(hi_mid, 0.5)

    def test_render_repetition_meter_escapes_html(self) -> None:
        rendered = render_repetition_meter_html(
            alpha=1.0,
            rep_ratio=0.12,
            tail_rep_ratio=0.05,
            n_tokens=400,
            max_tokens=4096,
            tok_per_s=85.0,
            status="done",
            hit_cap=False,
            pred_box="<script>alert(1)</script>",
            gold_answer="42",
            is_correct=False,
        )
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)


class TestGpuExecutionPath(unittest.TestCase):
    def test_gpu_tag_normalization_and_nan_detector(self) -> None:
        self.assertEqual(normalize_gpu_tag("Tesla T4"), "t4")
        self.assertEqual(normalize_gpu_tag("NVIDIA A100-SXM4-40GB"), "a100")
        self.assertEqual(normalize_gpu_tag("NVIDIA GeForce RTX 4090"), "rtx4090")

        self.assertIsNotNone(detect_nan_junk("", token_ids=[1, 2]))
        self.assertIsNotNone(detect_nan_junk("valid", token_ids=[]))
        self.assertIsNotNone(detect_nan_junk("step 1 \ufffd corrupted"))
        self.assertIsNotNone(detect_nan_junk("prefix " + ("語" * 10)))
        self.assertIsNotNone(detect_nan_junk("nan " * 8))
        self.assertIsNone(detect_nan_junk("<think>Let r = 2π and x ≤ 5.</think>\\boxed{10}", token_ids=[1, 2, 3]))

    def test_vllm_engine_fallbacks_logged(self) -> None:
        seen_kwargs: list[dict] = []

        def mock_llm_factory(**kwargs):
            seen_kwargs.append(dict(kwargs))
            if "language_model_only" in kwargs:
                raise TypeError("unexpected keyword argument 'language_model_only'")
            if not kwargs.get("enforce_eager", False):
                raise RuntimeError("torch.compile inductor backend failed")
            return types.SimpleNamespace(llm_engine=None)

        llm, meta = create_vllm_engine(
            ckpt_dir="/tmp/fake_ckpt",
            dtype="float16",
            max_model_len=10240,
            enforce_eager=False,
            llm_cls=mock_llm_factory,
        )
        self.assertIsNotNone(llm)
        self.assertEqual(len(seen_kwargs), 3)
        self.assertEqual(meta["limit_mm_per_prompt"], {"image": 0, "video": 0})
        self.assertFalse(meta["language_model_only"])
        self.assertTrue(meta["enforce_eager"])
        self.assertEqual(len(meta["fallbacks_triggered"]), 2)

    def test_make_ckpt_preserves_non_targets_and_verifies_sha256(self) -> None:
        base_t, edit_t = load_target_tensors()
        extra_tensor = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            snap_dir = tmp_path / "snapshot"
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "hidden_size": 1024}),
                encoding="utf-8",
            )
            (snap_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
                encoding="utf-8",
            )
            shard_dict = {
                TARGET_TENSORS[0]: base_t[TARGET_TENSORS[0]],
                TARGET_TENSORS[1]: base_t[TARGET_TENSORS[1]],
                "model.embed_tokens.weight": extra_tensor,
            }
            save_file(
                shard_dict,
                str(snap_dir / "model.safetensors-00001-of-00001.safetensors"),
                metadata={"format": "pt"},
            )

            out_a1 = tmp_path / "ckpt_a1"
            with mock.patch("gpu.make_ckpt.snapshot_download", return_value=str(snap_dir)):
                res = build_checkpoint(alpha=1.0, out_dir=out_a1)

            self.assertEqual(res["architectures"], ["Qwen3_5ForConditionalGeneration"])
            loaded_out = load_file(str(out_a1 / "model.safetensors-00001-of-00001.safetensors"), device="cpu")
            self.assertTrue(torch.equal(loaded_out["model.embed_tokens.weight"], extra_tensor))
            self.assertEqual(bf16_sha256(loaded_out[TARGET_TENSORS[0]]), EXPECTED_SHA256[TARGET_TENSORS[0]]["edit"])
            self.assertEqual(bf16_sha256(loaded_out[TARGET_TENSORS[1]]), EXPECTED_SHA256[TARGET_TENSORS[1]]["edit"])

            out_a0 = tmp_path / "ckpt_a0"
            with mock.patch("gpu.make_ckpt.snapshot_download", return_value=str(snap_dir)):
                build_checkpoint(alpha=0.0, out_dir=out_a0)
            loaded_0 = load_file(str(out_a0 / "model.safetensors-00001-of-00001.safetensors"), device="cpu")
            self.assertEqual(bf16_sha256(loaded_0[TARGET_TENSORS[0]]), EXPECTED_SHA256[TARGET_TENSORS[0]]["base"])

    def test_smoke_test_pass_and_fail_thresholds(self) -> None:
        problems = [{"unique_id": f"p/{i}", "problem": f"Compute {i}+1"} for i in range(8)]
        tok = DummyWordTokenizer()

        # Case 1: 2 disagreements (<= 2) -> PASS float16
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt0 = tmp_path / "c0"
            ckpt1 = tmp_path / "c1"
            ckpt0.mkdir()
            ckpt1.mkdir()

            def make_llm_pass(model: str, dtype: str, **kwargs):
                is_a1 = model.endswith("c1")

                def generate(prompts, sp):
                    outs = []
                    for idx, _ in enumerate(prompts):
                        base_ids = [100 + idx, 200 + idx, 300 + idx, 400 + idx]
                        if dtype == "float16" and is_a1 and idx < 2:
                            base_ids = list(base_ids)
                            base_ids[3] = 999
                        txt = "<think>ok</think>\\boxed{1}"
                        outs.append(
                            types.SimpleNamespace(
                                outputs=[types.SimpleNamespace(token_ids=base_ids, text=txt, finish_reason="stop")]
                            )
                        )
                    return outs

                return types.SimpleNamespace(generate=generate, llm_engine=None)

            with mock.patch("gpu.smoke_test.load_math500_subset", return_value=problems):
                report = run_smoke_test(
                    n_problems=8,
                    out_root=str(tmp_path / "assets"),
                    gpu_tag="t4",
                    llm_cls=make_llm_pass,
                    sampling_params_factory=lambda **kw: types.SimpleNamespace(**kw),
                    tokenizer_override=tok,
                    ckpt_dirs_override={0.0: ckpt0, 1.0: ckpt1},
                    require_gpu=False,
                )
            self.assertTrue(report["fp16_passed"])
            self.assertEqual(report["recommended_dtype"], "float16")
            self.assertEqual(report["prompt_disagreements"], 2)
            self.assertTrue((tmp_path / "assets" / "gpu_smoke_t4.json").is_file())

        # Case 2: 3 disagreements (> 2) and NaN junk check -> FAIL float16 -> float32
        base_seq = list(range(10, 150))
        runs_bad = {
            (0.0, "float32"): [{"token_ids": base_seq, "text": "<think>ok</think>\\boxed{2}"} for _ in range(8)],
            (1.0, "float32"): [{"token_ids": base_seq, "text": "<think>ok</think>\\boxed{2}"} for _ in range(8)],
            (0.0, "float16"): [
                {
                    "token_ids": ([999] + base_seq[1:]) if i < 3 else base_seq,
                    "text": "<think>ok \ufffd</think>\\boxed{2}" if i == 0 else "<think>ok</think>\\boxed{2}",
                }
                for i in range(8)
            ],
            (1.0, "float16"): [{"token_ids": base_seq, "text": "<think>ok</think>\\boxed{2}"} for _ in range(8)],
        }
        fail_eval = evaluate_smoke_outputs(problems, runs_bad)
        self.assertFalse(fail_eval["fp16_passed"])
        self.assertEqual(fail_eval["recommended_dtype"], "float32")
        self.assertEqual(fail_eval["prompt_disagreements"], 3)
        self.assertEqual(len(fail_eval["fp16_nan_failures"]), 1)

    def test_vllm_sweep_two_pass_resume_and_multi_budget_aggregation(self) -> None:
        problems = [
            {"unique_id": f"alg/{i}", "problem": f"Find {i}+1.", "answer": str(i + 1), "subject": "Algebra", "level": 1}
            for i in range(4)
        ]
        tok = DummyWordTokenizer()
        suffix_ids = tok.encode(FORCE_ANSWER_SUFFIX, add_special_tokens=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt = tmp_path / "ckpt_a1"
            ckpt.mkdir()
            out_root = tmp_path / "assets"

            def make_mock_vllm(model: str, dtype: str, **kwargs):
                def generate(prompts, sp):
                    # Pass 1: string prompts
                    if isinstance(prompts[0], str):
                        req_list = []
                        for p_idx, _ in enumerate(prompts):
                            comps = []
                            for s_idx in range(sp.n):
                                if p_idx < 3:
                                    txt = f"<think>work</think>\\boxed{{{p_idx + 1}}}"
                                    comps.append(
                                        types.SimpleNamespace(
                                            token_ids=[1, 10 + p_idx, 2, 20 + s_idx],
                                            text=txt,
                                            finish_reason="stop",
                                        )
                                    )
                                else:
                                    # Hits cap with wrong unclosed box inside <think>
                                    txt = "<think>wait maybe \\boxed{999} " + ("loop here now " * 20)
                                    comps.append(
                                        types.SimpleNamespace(
                                            token_ids=[1, 50, 51, 52],
                                            text=txt,
                                            finish_reason="length",
                                        )
                                    )
                            req_list.append(types.SimpleNamespace(outputs=comps))
                        return req_list

                    # Pass 2: forced continuation TokensPrompt ending with FORCE_ANSWER_SUFFIX ids
                    req_list = []
                    for tp in prompts:
                        p_ids = tp["prompt_token_ids"] if isinstance(tp, dict) else tp.prompt_token_ids
                        self.assertEqual(p_ids[-len(suffix_ids) :], suffix_ids)
                        f_txt = "Based on the above, \\boxed{4}."
                        req_list.append(
                            types.SimpleNamespace(
                                outputs=[types.SimpleNamespace(token_ids=[70, 71, 72], text=f_txt, finish_reason="stop")]
                            )
                        )
                    return req_list

                return types.SimpleNamespace(generate=generate, llm_engine=None)

            for alpha_val in (0.0, 1.0):
                with mock.patch("gpu.vllm_sweep.load_math500_subset", return_value=problems):
                    run_vllm_alpha(
                        alpha=alpha_val,
                        budget=8192,
                        n_problems=4,
                        samples=2,
                        seed=0,
                        dtype="float16",
                        gpu_tag="t4",
                        out_root=str(out_root),
                        ckpt_dir=str(ckpt),
                        llm_cls=make_mock_vllm,
                        sampling_params_factory=lambda **kw: types.SimpleNamespace(**kw),
                        tokenizer_override=tok,
                        require_gpu=False,
                    )

            run_dir = out_root / "budget_8192" / "gpu-t4-float16"
            s_df = pd.read_csv(run_dir / "sweep_results.csv")
            p_df = pd.read_csv(run_dir / "sweep_per_problem.csv")
            self.assertEqual(len(s_df), 2)
            self.assertNotIn("text", p_df.columns)
            # 3/4 strict (75.0%) vs 4/4 forced (100.0%) across 8 evaluations (4 problems x 2 samples)
            self.assertAlmostEqual(float(s_df.iloc[0]["accuracy_pct"]), 75.0, places=2)
            self.assertAlmostEqual(float(s_df.iloc[0]["forced_accuracy_pct"]), 100.0, places=2)
            self.assertAlmostEqual(float(s_df.iloc[0]["hit_cap_pct"]), 25.0, places=2)
            self.assertTrue((run_dir / "sweep_charts.png").is_file())
            self.assertTrue((run_dir / "alpha_token_distributions.png").is_file())
            self.assertTrue((run_dir / "meta.json").is_file())

            with gzip.open(run_dir / "sweep_per_problem.jsonl.gz", "rt", encoding="utf-8") as gz:
                jsonl_rows = [json.loads(line) for line in gz if line.strip()]
            self.assertEqual(len(jsonl_rows), 16)

            # Resume check: re-running alpha=1.0 should generate 0 new rows
            with mock.patch("gpu.vllm_sweep.load_math500_subset", return_value=problems):
                res_resume = run_vllm_alpha(
                    alpha=1.0,
                    budget=8192,
                    n_problems=4,
                    samples=2,
                    seed=0,
                    dtype="float16",
                    gpu_tag="t4",
                    out_root=str(out_root),
                    ckpt_dir=str(ckpt),
                    llm_cls=make_mock_vllm,
                    sampling_params_factory=lambda **kw: types.SimpleNamespace(**kw),
                    tokenizer_override=tok,
                    require_gpu=False,
                )
            self.assertEqual(res_resume["new_records_count"], 0)

            # Copy 4k MPS CSV + NOTES.md and run cross-budget aggregation
            shutil.copy2("assets/sweep_results.csv", out_root / "sweep_results.csv")
            notes_copy = tmp_path / "NOTES.md"
            shutil.copy2("NOTES.md", notes_copy)
            agg_df = run_aggregation(out_root=str(out_root), notes_path=str(notes_copy))
            self.assertTrue((out_root / "budget_curve_gpu.png").is_file())
            self.assertIn(4096, set(agg_df["budget"]))
            self.assertIn(8192, set(agg_df["budget"]))
            notes_text = notes_copy.read_text(encoding="utf-8")
            self.assertIn(GPU_NOTES_START, notes_text)
            self.assertIn(GPU_NOTES_END, notes_text)
            # Re-running aggregation is idempotent
            run_aggregation(out_root=str(out_root), notes_path=str(notes_copy))
            notes_text_2 = notes_copy.read_text(encoding="utf-8")
            self.assertEqual(notes_text_2.count(GPU_NOTES_START), 1)

    def test_gpu_notebook_valid_and_outputs_cleared(self) -> None:
        nb_path = Path("notebooks/gpu_sweep.ipynb")
        self.assertTrue(nb_path.is_file())
        raw_nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(raw_nb["nbformat"], 4)
        self.assertGreaterEqual(len(raw_nb["cells"]), 7)
        code_cells = [c for c in raw_nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 7)
        for cell in code_cells:
            self.assertEqual(cell.get("outputs"), [])
            self.assertIsNone(cell.get("execution_count"))
            src = "".join(cell["source"])
            ast.parse(src)


if __name__ == "__main__":
    unittest.main()
