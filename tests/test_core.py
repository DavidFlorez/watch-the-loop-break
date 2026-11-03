"""Unit tests for `weights.py`, `metrics.py`, and `app.py`."""

from __future__ import annotations

import unittest

import torch

from app import render_repetition_meter_html
from metrics import (
    PresencePenaltyLogitsProcessor,
    build_math_prompt,
    extract_last_boxed,
    grade_math_answer,
    repeated_8gram_ratio,
    tail_repeated_8gram_ratio,
)
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
        self.assertTrue(grade_math_answer(pred, "1.5"))
        self.assertTrue(grade_math_answer(pred, "\\dfrac{3}{2}"))
        self.assertFalse(grade_math_answer(pred, "2"))
        self.assertIn("\\boxed{}", build_math_prompt("Find x."))

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


if __name__ == "__main__":
    unittest.main()
