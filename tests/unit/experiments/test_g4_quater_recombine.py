"""Unit tests for the three RECOMBINE strategies (G4-quater Step 3
H4-C placebo control)."""
from __future__ import annotations

import numpy as np

from experiments.g4_quater_test.recombine_strategies import (
    sample_synthetic_latents,
)


def test_mog_returns_n_synthetic() -> None:
    latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    out = sample_synthetic_latents(
        strategy="mog",
        latents=latents,
        labels=labels,
        n_synthetic=4,
        seed=0,
    )
    assert out is not None
    assert out["x"].shape == (4, 16)
    assert out["y"].shape == (4,)


def test_ae_returns_reconstructed_latents() -> None:
    latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    out = sample_synthetic_latents(
        strategy="ae",
        latents=latents,
        labels=labels,
        n_synthetic=4,
        seed=0,
    )
    assert out is not None
    assert out["x"].shape == (4, 16)
    assert out["y"].shape == (4,)


def test_none_returns_none() -> None:
    latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    out = sample_synthetic_latents(
        strategy="none",
        latents=latents,
        labels=labels,
        n_synthetic=4,
        seed=0,
    )
    assert out is None


def test_mog_empty_latents_returns_none() -> None:
    latents = np.zeros((0, 16), dtype=np.float32)
    labels = np.array([], dtype=np.int64)
    out = sample_synthetic_latents(
        strategy="mog",
        latents=latents,
        labels=labels,
        n_synthetic=4,
        seed=0,
    )
    assert out is None


def test_mog_single_class_returns_none() -> None:
    latents = np.random.RandomState(0).randn(10, 16).astype(np.float32)
    labels = np.zeros(10, dtype=np.int64)
    out = sample_synthetic_latents(
        strategy="mog",
        latents=latents,
        labels=labels,
        n_synthetic=4,
        seed=0,
    )
    assert out is None


def test_mog_deterministic_under_seed() -> None:
    latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    a = sample_synthetic_latents(
        strategy="mog", latents=latents, labels=labels,
        n_synthetic=4, seed=42,
    )
    b = sample_synthetic_latents(
        strategy="mog", latents=latents, labels=labels,
        n_synthetic=4, seed=42,
    )
    assert a is not None and b is not None
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["y"], b["y"])
