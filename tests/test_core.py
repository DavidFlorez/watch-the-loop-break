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

if not os.environ.get("MPLBACKEND") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

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

    def test_app_katex_and_markdown_formatting(self) -> None:
        from app import (
            format_gold_for_display,
            format_problem_for_display,
            format_stream_for_display,
        )

        prob = format_problem_for_display(r"Find \(\frac{8 \times 7}{2}\) and put it in \boxed{28}.")
        self.assertNotIn(r"\(", prob)
        self.assertNotIn(r"\)", prob)
        self.assertIn(r"$\frac{8 \times 7}{2}$", prob)
        self.assertIn(r"$\boxed{28}$", prob)

        gold = format_gold_for_display(r"\left( 3, \frac{\pi}{2} \right)")
        self.assertIn("$$\n\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}\n$$", gold)

        raw_stream = (
            "<think>\n"
            "Let us compute:\n"
            "\\[ C(8, 2) = \\frac{8 \\times 7}{2} = 28 \\]\n"
            "    * Check case 1: $A$ and $B$\n"
            "    * Check case 2: $B$ and $A$\n"
            "</think>\n\n"
            "Thus, there are 28 different pizzas.\n"
            "\\boxed{28}"
        )
        formatted = format_stream_for_display(raw_stream, is_streaming=False)
        self.assertNotIn("<think>", formatted)
        self.assertNotIn("</think>", formatted)
        self.assertIn("Finished Thinking", formatted)
        self.assertIn("$$\n\\boxed{28}\n$$", formatted)
        self.assertNotIn("\n    * Check case 1", formatted)
        self.assertIn("\n  * Check case 1", formatted)

        partial = format_stream_for_display("We compute $\\frac{8 \\times 7", is_streaming=True)
        self.assertEqual(partial, "We compute")

    def test_app_replay_recorded_comparison_32k_hero(self) -> None:
        from app import get_math500_problems, replay_recorded_comparison

        problems = get_math500_problems()
        hero = next(
            (p for p in problems if p["unique_id"] == "test/counting_and_probability/389.json"),
            None,
        )
        self.assertIsNotNone(hero)
        frames = list(
            replay_recorded_comparison(
                hero["label"],
                alpha_left=0.0,
                alpha_right=1.0,
                budget_label="32,768 tokens (Paper Protocol — 200 problems)",
                frame_delay_s=0.0,
            )
        )
        self.assertGreaterEqual(len(frames), 2)
        prob_md, gold_md, h_l, disp_l, h_r, disp_r = frames[-1]
        self.assertIn("8 different toppings", prob_md)
        self.assertIn("\\boxed{28}", gold_md)
        self.assertIn("RAN OUT OF TOKENS", h_l)
        self.assertIn("32,768 / 32,768", h_l)
        self.assertIn("FINISHED CLEANLY", h_r)
        self.assertIn("3,843 / 32,768", h_r)
        self.assertNotIn("<think>", disp_l)
        self.assertNotIn("</think>", disp_l)
        self.assertNotIn("<think>", disp_r)
        self.assertNotIn("</think>", disp_r)
        self.assertNotIn("`\\boxed", disp_r)
        self.assertIn("$$\n\\boxed{28}\n$$", disp_r)


class TestGpuExecutionPath(unittest.TestCase):
    def test_gpu_tag_normalization_and_nan_detector(self) -> None:
        self.assertEqual(normalize_gpu_tag("Tesla T4"), "t4")
        self.assertEqual(normalize_gpu_tag("NVIDIA A100-SXM4-40GB"), "a100")
        self.assertEqual(normalize_gpu_tag("NVIDIA GeForce RTX 4090"), "rtx4090")
        self.assertEqual(
            normalize_gpu_tag("NVIDIA RTX PRO 6000 Blackwell Server Edition"),
            "rtx-pro-6000",
        )

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
        joined_sources = []
        for cell in code_cells:
            self.assertEqual(cell.get("outputs"), [])
            self.assertIsNone(cell.get("execution_count"))
            src = "".join(cell["source"])
            ast.parse(src)
            joined_sources.append(src)
        full_nb_src = "\n".join(joined_sources)
        self.assertIn("VENV_PYTHON", full_nb_src)
        self.assertIn("--torch-backend=auto", full_nb_src)
        self.assertNotIn("os.kill(os.getpid(), 9)", full_nb_src)

    def test_native_bfloat16_support_and_auto_dtype(self) -> None:
        from gpu.common import supports_native_bfloat16
        from gpu.smoke_test import run_smoke_test
        from gpu.vllm_sweep import resolve_dtype_choice

        self.assertTrue(supports_native_bfloat16({"compute_capability": 8.9}))
        self.assertTrue(supports_native_bfloat16({"compute_capability": 8.0}))
        self.assertFalse(supports_native_bfloat16({"compute_capability": 7.5}))
        self.assertFalse(supports_native_bfloat16({"compute_capability": None}))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.assertEqual(
                resolve_dtype_choice("bfloat16", tmp_path, "l4", native_bf16_override=True),
                "bfloat16",
            )
            self.assertEqual(
                resolve_dtype_choice("auto", tmp_path, "l4", native_bf16_override=True),
                "bfloat16",
            )
            self.assertEqual(
                resolve_dtype_choice("auto", tmp_path, "t4", native_bf16_override=False),
                "float32",
            )

            problems = [
                {
                    "unique_id": f"test/algebra/{i}.json",
                    "problem": f"Compute {i}+{i}",
                    "answer": str(2 * i),
                    "subject": "Algebra",
                    "level": 1,
                }
                for i in range(4)
            ]
            ckpt_0 = tmp_path / "c0"
            ckpt_1 = tmp_path / "c1"
            ckpt_0.mkdir()
            ckpt_1.mkdir()

            class DummyTok:
                def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                    return "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n"

            class DummyEngine:
                def __init__(self, **kwargs):
                    self.dtype = kwargs["dtype"]

                def generate(self, prompts, sp):
                    outs = []
                    for _ in prompts:
                        txt = f"<think>\nok ({self.dtype})\n</think>\n\n\\boxed{{2}}"
                        outs.append(
                            types.SimpleNamespace(
                                outputs=[
                                    types.SimpleNamespace(
                                        token_ids=[1, 2, 3],
                                        text=txt,
                                        finish_reason="stop",
                                    )
                                ]
                            )
                        )
                    return outs

            with (
                mock.patch("gpu.smoke_test.load_math500_subset", return_value=problems),
                mock.patch(
                    "gpu.smoke_test.collect_environment_metadata",
                    return_value={
                        "gpu_tag": "l4",
                        "gpu_name": "NVIDIA L4",
                        "compute_capability": 8.9,
                        "native_bfloat16": True,
                        "nvidia_smi": {"available": True, "compute_capability": 8.9, "gpus": []},
                        "pip_freeze_filtered": [],
                        "package_versions": {},
                    },
                ),
            ):
                rep = run_smoke_test(
                    n_problems=4,
                    max_tokens=128,
                    out_root=str(tmp_path),
                    llm_cls=DummyEngine,
                    sampling_params_factory=lambda **kw: types.SimpleNamespace(**kw),
                    tokenizer_override=DummyTok(),
                    ckpt_dirs_override={0.0: ckpt_0, 1.0: ckpt_1},
                    require_gpu=False,
                    native_bf16_override=True,
                )

            self.assertTrue(rep["native_bfloat16"])
            self.assertTrue(rep["bf16_passed"])
            self.assertEqual(rep["recommended_dtype"], "bfloat16")
            self.assertEqual(
                resolve_dtype_choice("auto", tmp_path, "l4", native_bf16_override=True),
                "bfloat16",
            )

    def test_gcp_runner_helpers_and_dry_run(self) -> None:
        from gpu.gcp_run import (
            _is_safe_result_relpath,
            build_create_instance_cmd,
            build_gcloud_env,
            build_remote_runner_script,
            check_region_gpu_quota,
            create_upload_bundle,
            merge_downloaded_outputs,
            run_on_gcp,
        )

        # 1. Attribution env & non-interactive prompts
        env = build_gcloud_env({"FOO": "bar"})
        self.assertEqual(env["CLOUDSDK_CORE_DISABLE_PROMPTS"], "1")
        self.assertIn("datacloud.antigravity", env["CLOUDSDK_METRICS_ENVIRONMENT"].split())

        # 2. Quota inspection (Spot L4 vs T4 fallback)
        quotas = [
            {"metric": "PREEMPTIBLE_NVIDIA_L4_GPUS", "limit": 0.0, "usage": 0.0},
            {"metric": "PREEMPTIBLE_NVIDIA_T4_GPUS", "limit": 1.0, "usage": 0.0},
        ]
        self.assertEqual(check_region_gpu_quota(quotas, "l4", is_spot=True), 0.0)
        self.assertEqual(check_region_gpu_quota(quotas, "t4", is_spot=True), 1.0)

        # 3. Create VM command & cost safety flags
        cmd = build_create_instance_cmd(
            vm_name="wsp-test-vm",
            project="valid-project-123",
            zone="us-central1-a",
            gpu_key="l4",
            provisioning="spot",
            max_minutes=45,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("--machine-type=g2-standard-4", cmd_str)
        self.assertIn("--accelerator=type=nvidia-l4,count=1", cmd_str)
        self.assertIn("--provisioning-model=SPOT", cmd_str)
        self.assertIn("--max-run-duration=2700s", cmd_str)
        self.assertIn("--instance-termination-action=DELETE", cmd_str)
        with self.assertRaises(ValueError):
            build_create_instance_cmd(
                vm_name="../bad-name",
                project="valid-project-123",
                zone="us-central1-a",
                gpu_key="l4",
            )

        # 4. Remote runner script uses isolated uv venv & cleans scratch
        script = build_remote_runner_script(
            budgets=[8192, 16384],
            alphas=[0.0, 1.0],
            n_problems=50,
            samples=1,
            seed=0,
            dtype="auto",
            gpu_mem=0.85,
            enforce_eager=False,
            smoke_only=False,
        )
        self.assertIn("--torch-backend=auto", script)
        self.assertIn("/opt/wsp/venv", script)
        self.assertIn("rm -rf /tmp/wsp_ckpt", script)

        # 5. Path safety & upload bundle exclusions
        self.assertTrue(_is_safe_result_relpath("assets/budget_8192/gpu-l4-bfloat16/sweep_results.csv"))
        self.assertTrue(_is_safe_result_relpath("NOTES.md"))
        self.assertFalse(_is_safe_result_relpath("../secret.txt"))
        self.assertFalse(_is_safe_result_relpath("assets/budget_8192/model.safetensors"))
        self.assertFalse(_is_safe_result_relpath("assets/sweep_results.csv"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            staging = tmp_path / "stage"
            (staging / "assets" / "budget_8192" / "gpu-l4-bfloat16").mkdir(parents=True)
            (staging / "assets" / "budget_8192" / "gpu-l4-bfloat16" / "sweep_results.csv").write_text("alpha\n0.0\n", encoding="utf-8")
            (staging / "assets" / "budget_8192" / "gpu-l4-bfloat16" / "weights.safetensors").write_bytes(b"123")
            dst_repo = tmp_path / "repo"
            dst_repo.mkdir()
            copied = merge_downloaded_outputs(staging, dst_repo)
            self.assertEqual(copied, ["assets/budget_8192/gpu-l4-bfloat16/sweep_results.csv"])
            self.assertFalse((dst_repo / "assets" / "budget_8192" / "gpu-l4-bfloat16" / "weights.safetensors").exists())

            bundle_path = tmp_path / "bundle.tar.gz"
            create_upload_bundle(Path.cwd(), runner_script=script, bundle_path=bundle_path)
            self.assertTrue(bundle_path.is_file())

        # 6. Offline dry-run smoke check
        plan = run_on_gcp(project="valid-project-123", gpu="auto", dry_run=True)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["machine_type"], "g2-standard-4")

    def test_colab_runner_helpers_and_dry_run(self) -> None:
        import tarfile
        from gpu.colab_run import (
            build_colab_env,
            build_colab_remote_driver,
            build_colab_shell_script,
            extract_tar_safely,
            parse_gpu_candidates,
            resolve_ordered_budgets,
            run_on_colab,
            session_exists,
            validate_session_name,
        )
        from gpu.gcp_run import _BytesIO

        self.assertEqual(validate_session_name("wsp-g4"), "wsp-g4")
        with self.assertRaises(ValueError):
            validate_session_name("../bad")

        self.assertEqual(parse_gpu_candidates("g4", "l4,a100,t4"), ["G4", "L4", "A100", "T4"])
        self.assertEqual(
            resolve_ordered_budgets([8192, 16384], include_4k=True, include_32k=True),
            [4096, 8192, 16384, 32768],
        )

        env = build_colab_env({"FOO": "bar"})
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

        # Orphan '[?] ...' CPU session must never match named session lookup
        sample_sessions = [
            "[?] m-s-usc1a10-2huy3qnedtjxn | Hardware: CPU | Shape: Standard | Variant: DEFAULT",
            "wsp-g4 | m-s-usw2a1-abc | Hardware: G4 | Shape: Standard | Variant: DEFAULT",
        ]
        self.assertTrue(session_exists("wsp-g4", sample_sessions))
        self.assertFalse(session_exists("wsp-l4", sample_sessions))
        self.assertFalse(session_exists("m-s-usc1a10-2huy3qnedtjxn", sample_sessions))

        shell_script = build_colab_shell_script(
            budgets=[8192, 16384],
            alphas=[0.0, 1.0, 1.25],
            n_problems=50,
            samples=1,
            seed=0,
            dtype="auto",
            gpu_mem=0.85,
            enforce_eager=False,
            smoke_only=False,
            include_4k=True,
            include_32k=True,
        )
        self.assertIn("--torch-backend=auto", shell_script)
        self.assertIn("/content/wsp/venv", shell_script)
        self.assertIn("BUDGETS=(4096 8192 16384 32768)", shell_script)

        driver_code = build_colab_remote_driver()
        ast.parse(driver_code)

        # Safe tar extraction filters out .safetensors and protected MPS root assets
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            arch = tmp_path / "wsp_out_summary.tar.gz"
            with tarfile.open(arch, "w:gz") as tf:
                for name, data in (
                    ("assets/gpu_smoke_rtx-pro-6000.json", b"{}"),
                    ("assets/sweep_results.csv", b"corrupt"),
                    ("assets/budget_8192/model.safetensors", b"weights"),
                ):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tf.addfile(info, fileobj=_BytesIO(data))
            staging = tmp_path / "stage"
            extract_tar_safely(arch, staging)
            self.assertTrue((staging / "assets" / "gpu_smoke_rtx-pro-6000.json").is_file())
            self.assertFalse((staging / "assets" / "sweep_results.csv").exists())
            self.assertFalse((staging / "assets" / "budget_8192" / "model.safetensors").exists())

        plan = run_on_colab(
            gpu="G4",
            include_4k=True,
            include_32k=True,
            dry_run=True,
        )
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["session_name"], "wsp-g4")
        self.assertEqual(plan["gpu_candidates"], ["G4"])
        self.assertEqual(plan["budgets"], [4096, 8192, 16384, 32768])
        self.assertGreaterEqual(plan["exec_timeout_s"], 7200)


if __name__ == "__main__":
    unittest.main()
