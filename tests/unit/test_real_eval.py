"""Unit tests for cycle-3 C3.8 Phase A real-benchmark evaluators.

Covers ``evaluate_mmlu`` / ``evaluate_hellaswag`` / ``evaluate_mega_v2``
with mocked models so the suite stays network-free and <1 s total.

Each test constructs a tiny deterministic ``model`` + ``tokenizer``
pair that returns hand-engineered logits, calls the evaluator, and
asserts the shape + key invariants of the returned dict.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from harness.real_benchmarks.hellaswag import evaluate_hellaswag
from harness.real_benchmarks.mega_v2_eval import evaluate_mega_v2
from harness.real_benchmarks.mmlu import evaluate_mmlu


# --------------------------------------------------------------------------
# Mock tokenizer + model.
# --------------------------------------------------------------------------


class _StubTokenizer:
    """Deterministic tokenizer : maps each character to a vocab id.

    Vocab size is fixed at 128 (ASCII). Letters A/B/C/D → 65/66/67/68.
    This lets us predict exactly which vocab column the evaluator
    should read for letter-argmax scoring.
    """

    vocab_size: int = 128

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # bounded to vocab_size so we stay deterministic even on
        # multi-byte characters.
        return [ord(c) % self.vocab_size for c in text][:64]


class _StubModel:
    """Callable that returns deterministic logits.

    ``forward_mode`` :

    - ``"letter_A"`` : boosts the logit of the 'A' letter token
      (vocab 65) so the MMLU letter-argmax picks A. Ground truth
      matches any MMLU record with ``answer == 0``.
    - ``"uniform"`` : returns a zero logit tensor — random argmax.
    - ``"first_ending"`` : boosts log-prob of the first ending's
      first token for HellaSwag so argmax prefers ending 0.
    """

    def __init__(self, forward_mode: str = "uniform") -> None:
        self.forward_mode = forward_mode

    def __call__(self, token_ids: mx.array) -> mx.array:
        arr = np.asarray(token_ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        bsz, seq = arr.shape
        vocab = _StubTokenizer.vocab_size
        logits = np.zeros((bsz, seq, vocab), dtype=np.float32)
        if self.forward_mode == "letter_A":
            # Bias letter 'A' = id 65 on the last position.
            logits[:, -1, 65] = 10.0
        elif self.forward_mode == "first_ending":
            # No-op — all positions share the same distribution, so
            # shortest ending tokens dominate sum log-prob. Tests
            # using this mode pass different-length endings.
            pass
        return mx.array(logits)


# --------------------------------------------------------------------------
# MMLU evaluator test
# --------------------------------------------------------------------------


def test_evaluate_mmlu_returns_accuracy_dict(tmp_path: Path) -> None:
    """``evaluate_mmlu`` picks letter A → accuracy = fraction with answer=0."""
    # Write a 4-record fixture : 2 records with answer=0 (A), 2 with others.
    fixture = tmp_path / "mmlu.jsonl"
    rows = [
        {
            "question": "Q1",
            "choices": ["Correct", "Wrong", "Wrong", "Wrong"],
            "answer": 0,  # A
            "subject": "t",
        },
        {
            "question": "Q2",
            "choices": ["Correct", "Wrong", "Wrong", "Wrong"],
            "answer": 0,  # A
            "subject": "t",
        },
        {
            "question": "Q3",
            "choices": ["Wrong", "Correct", "Wrong", "Wrong"],
            "answer": 1,  # B
            "subject": "t",
        },
        {
            "question": "Q4",
            "choices": ["Wrong", "Wrong", "Correct", "Wrong"],
            "answer": 2,  # C
            "subject": "t",
        },
    ]
    with fixture.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    model = _StubModel(forward_mode="letter_A")
    tokenizer = _StubTokenizer()
    result = evaluate_mmlu(
        model,
        tokenizer,
        n_samples=4,
        seed=0,
        fixture_path=fixture,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"accuracy", "n"}
    # Model always predicts A → 2 of 4 records correct.
    assert result["n"] == 4
    assert result["accuracy"] == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------
# HellaSwag evaluator test
# --------------------------------------------------------------------------


def test_evaluate_hellaswag_returns_accuracy_dict(tmp_path: Path) -> None:
    """``evaluate_hellaswag`` returns the expected dict shape + valid range."""
    fixture = tmp_path / "hs.jsonl"
    rows = [
        {
            "ctx": "The child opened the door and",
            "endings": [" ran outside.", " XX", " YY", " ZZ"],
            "label": 0,
            "activity_label": "kids",
        },
        {
            "ctx": "She turned the key in the lock and",
            "endings": [" AA", " BB", " heard a click.", " CC"],
            "label": 2,
            "activity_label": "home",
        },
    ]
    with fixture.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    model = _StubModel(forward_mode="uniform")
    tokenizer = _StubTokenizer()
    result = evaluate_hellaswag(
        model,
        tokenizer,
        n_samples=2,
        seed=0,
        fixture_path=fixture,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"accuracy", "n"}
    assert result["n"] == 2
    assert 0.0 <= result["accuracy"] <= 1.0


# --------------------------------------------------------------------------
# mega-v2 evaluator test
# --------------------------------------------------------------------------


def test_evaluate_mega_v2_returns_accuracy_dict(tmp_path: Path) -> None:
    """``evaluate_mega_v2`` returns ``{accuracy, nll, n}`` with finite values."""
    fixture = tmp_path / "mv2.jsonl"
    rows = [
        {
            "id": "mv2-0001",
            "context": "Two plus two equals",
            "expected": " four.",
            "domain": "math",
        },
        {
            "id": "mv2-0002",
            "context": "The sun rises in the",
            "expected": " east.",
            "domain": "world_facts",
        },
        {
            "id": "mv2-0003",
            "context": "Water freezes at zero degrees",
            "expected": " Celsius.",
            "domain": "science",
        },
    ]
    with fixture.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    model = _StubModel(forward_mode="uniform")
    tokenizer = _StubTokenizer()
    result = evaluate_mega_v2(
        model,
        tokenizer,
        n_samples=3,
        seed=0,
        fixture_path=fixture,
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"accuracy", "nll", "n"}
    assert result["n"] == 3
    assert np.isfinite(result["nll"])
    assert 0.0 <= result["accuracy"] <= 1.0


# --------------------------------------------------------------------------
# Auxiliary : evaluator accepts a wrapper with .model attribute
# (mirrors QwenMLXFP16Wrapper surface).
# --------------------------------------------------------------------------


def test_evaluator_accepts_wrapper_with_model_attribute(tmp_path: Path) -> None:
    """Wrapper-style ``.model`` attribute is unwrapped by evaluators."""
    class _Wrapper:
        def __init__(self) -> None:
            self.model = _StubModel(forward_mode="letter_A")

    fixture = tmp_path / "mmlu.jsonl"
    with fixture.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "question": "Q",
                    "choices": ["Correct", "W", "W", "W"],
                    "answer": 0,
                    "subject": "t",
                }
            )
            + "\n"
        )

    result = evaluate_mmlu(
        _Wrapper(), _StubTokenizer(), n_samples=1, seed=0, fixture_path=fixture
    )
    assert result["accuracy"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Auxiliary : fallback fixtures used when no caller path is provided
# (exercises the in-module hand-authored HellaSwag + mega-v2 fallback).
# --------------------------------------------------------------------------


def test_hellaswag_evaluator_uses_builtin_fallback(tmp_path: Path) -> None:
    """Evaluator runs end-to-end even when no fixture path is given.

    ``fixture_path`` kept None + ``_DEFAULT_HELLASWAG_FALLBACK``
    missing → the 8-row in-module fallback materialises the
    records. Validates the pipeline path, not empirical accuracy.
    """
    from harness.real_benchmarks import hellaswag as hs_mod

    missing = tmp_path / "nothing.jsonl"
    original = hs_mod._DEFAULT_HELLASWAG_FALLBACK
    hs_mod._DEFAULT_HELLASWAG_FALLBACK = missing
    try:
        result = evaluate_hellaswag(
            _StubModel(forward_mode="uniform"),
            _StubTokenizer(),
            n_samples=4,
            seed=0,
        )
    finally:
        hs_mod._DEFAULT_HELLASWAG_FALLBACK = original
    assert result["n"] == 4
    assert 0.0 <= result["accuracy"] <= 1.0


def test_mega_v2_evaluator_uses_builtin_fallback(tmp_path: Path) -> None:
    """mega-v2 evaluator + in-module fallback → well-formed dict."""
    from harness.real_benchmarks import mega_v2_eval as mv2_mod

    missing = tmp_path / "nothing.jsonl"
    original = mv2_mod._DEFAULT_MEGA_V2_FALLBACK
    mv2_mod._DEFAULT_MEGA_V2_FALLBACK = missing
    try:
        result = evaluate_mega_v2(
            _StubModel(forward_mode="uniform"),
            _StubTokenizer(),
            n_samples=3,
            seed=0,
        )
    finally:
        mv2_mod._DEFAULT_MEGA_V2_FALLBACK = original
    assert result["n"] == 3
    assert np.isfinite(result["nll"])
