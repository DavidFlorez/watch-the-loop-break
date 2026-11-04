"""Repetition metrics (repeated 8-gram ratio), presence penalty processor, and MATH-500 answer grading."""

from __future__ import annotations

import functools
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import LogitsProcessor

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_math_prompt(problem: str) -> str:
    """Format a math problem with Qwen3.5's recommended boxed-answer instruction."""
    return problem.strip() + "\n" + MATH_INSTRUCTION


def repeated_8gram_ratio(text: str, n: int = 8) -> float:
    """Compute the repeated n-gram ratio (default 8-gram) over whitespace-delimited tokens.

    Defined as:
        (total_ngrams - unique_ngrams) / total_ngrams = 1.0 - (unique_ngrams / total_ngrams)

    Returns 0.0 when fewer than `n` tokens are present.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    total = len(words) - n + 1
    unique = len({tuple(words[i : i + n]) for i in range(total)})
    return float(total - unique) / float(total)


def tail_repeated_8gram_ratio(text: str, n: int = 8, tail_words: int = 240) -> float:
    """Compute the repeated 8-gram ratio on the trailing `tail_words` words of `text`."""
    words = text.split()
    if len(words) < n:
        return 0.0
    tail = words[-tail_words:] if len(words) > tail_words else words
    if len(tail) < n:
        return 0.0
    total = len(tail) - n + 1
    unique = len({tuple(tail[i : i + n]) for i in range(total)})
    return float(total - unique) / float(total)


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Apply OpenAI / vLLM-style additive `presence_penalty` to tokens already generated.

    For each sequence `b` in the batch, finds unique token IDs generated after `prompt_length`
    and subtracts `presence_penalty` from their logits:
        logits[b, token_id] -= presence_penalty
    """

    def __init__(self, presence_penalty: float, prompt_length: int) -> None:
        if presence_penalty < 0:
            raise ValueError(f"presence_penalty must be >= 0, got {presence_penalty}")
        self.presence_penalty = float(presence_penalty)
        self.prompt_length = int(prompt_length)
        self._penalty_buf: Optional[torch.Tensor] = None
        self._last_seq_len: int = int(prompt_length)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[1]
        if self.presence_penalty == 0.0 or cur_len <= self.prompt_length:
            return scores

        if (
            self._penalty_buf is None
            or self._penalty_buf.shape != scores.shape
            or self._penalty_buf.device != scores.device
            or cur_len <= self._last_seq_len
        ):
            self._penalty_buf = torch.zeros_like(scores)
            gen_tokens = input_ids[:, self.prompt_length :]
            self._penalty_buf.scatter_(1, gen_tokens, self.presence_penalty)
        else:
            new_tokens = input_ids[:, self._last_seq_len :]
            self._penalty_buf.scatter_(1, new_tokens, self.presence_penalty)

        self._last_seq_len = cur_len
        return scores - self._penalty_buf


# ---------------------------------------------------------------------------
# Boxed answer extraction & grading (compatible with RikiyaT/lm_understand-logs)
# ---------------------------------------------------------------------------

_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text_after_think(text: str) -> str:
    """Cut off the `<think>...</think>` reasoning block if closed; otherwise inspect full text."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _box_at(text: str, idx: int) -> Optional[str]:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the content of the last `\\boxed{...}` or `\\fbox{...}` in `text`.

    First inspects text after the last `</think>` if present, falling back to full text if no box follows.
    """
    # First look after </think> if present; fallback to full text if no box after </think>
    t_after = answer_text_after_think(text)
    for candidate in (t_after, text):
        idx = candidate.rfind("\\boxed")
        if idx < 0:
            idx = candidate.rfind("\\fbox")
        if idx >= 0:
            val = _box_at(candidate, idx)
            if val is not None:
                return val
    return None


def extract_last_boxed_strict(text: str) -> Optional[str]:
    """Paper-style strict boxed extraction: only inspect text after the last `</think>` if present, no fallback."""
    t_after = answer_text_after_think(text)
    idx = t_after.rfind("\\boxed")
    if idx < 0:
        idx = t_after.rfind("\\fbox")
    if idx < 0:
        return None
    return _box_at(t_after, idx)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute Wilson score confidence interval (default 95%) as `(low_frac, high_frac)` in `[0.0, 1.0]`."""
    if total <= 0:
        return (0.0, 0.0)
    n_int = int(total)
    k = max(0, min(int(successes), n_int))
    n = float(n_int)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * ((p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) ** 0.5)) / denom
    low = 0.0 if k == 0 else max(0.0, center - margin)
    high = 1.0 if k == n_int else min(1.0, center + margin)
    return (low, high)


def _strip_math_string(string: str) -> str:
    """Normalize LaTeX math strings (Hendrycks MATH equivalence normalization)."""
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "")
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string


def is_equiv_str(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return _strip_math_string(pred) == _strip_math_string(gold)
    except Exception:
        return pred.strip() == gold.strip()


@functools.lru_cache(maxsize=4096)
def _mv_parse(expr: str, timeout: Optional[int]):
    import math_verify

    return math_verify.parse("\\boxed{" + expr + "}", parsing_timeout=timeout)


def grade_math_answer(pred: Optional[str], gold: Optional[str]) -> bool:
    """Grade extracted `\\boxed{}` prediction against `gold` using `math-verify` + `is_equiv` fallback."""
    if pred is None or gold is None:
        return False
    if is_equiv_str(pred, gold):
        return True
    try:
        import math_verify

        timeout = 5 if threading.current_thread() is threading.main_thread() else None
        g_parsed = _mv_parse(gold, timeout)
        p_parsed = _mv_parse(pred, timeout)
        if g_parsed and p_parsed and math_verify.verify(g_parsed, p_parsed, timeout_seconds=timeout):
            return True
    except Exception:
        pass
    return False
