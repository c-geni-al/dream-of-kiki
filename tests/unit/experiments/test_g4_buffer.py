"""Unit tests for the curated β buffer feeding the G4 dream coupling."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO


def test_buffer_starts_empty() -> None:
    buf = BetaBufferFIFO(capacity=128)
    assert len(buf) == 0


def test_push_increments_size_until_capacity() -> None:
    buf = BetaBufferFIFO(capacity=4)
    buf.push(np.zeros(8, dtype=np.float32), 0)
    buf.push(np.zeros(8, dtype=np.float32), 1)
    assert len(buf) == 2


def test_push_evicts_oldest_at_capacity() -> None:
    buf = BetaBufferFIFO(capacity=2)
    a = np.array([1.0] * 4, dtype=np.float32)
    b = np.array([2.0] * 4, dtype=np.float32)
    c = np.array([3.0] * 4, dtype=np.float32)
    buf.push(a, 0)
    buf.push(b, 1)
    buf.push(c, 0)  # evicts a
    assert len(buf) == 2
    items = buf.snapshot()
    np.testing.assert_array_equal(items[0]["x"], b)
    np.testing.assert_array_equal(items[1]["x"], c)


def test_sample_seeded_reproducible() -> None:
    buf = BetaBufferFIFO(capacity=16)
    rng = np.random.default_rng(0)
    for i in range(10):
        buf.push(rng.standard_normal(4).astype(np.float32), i % 2)
    s1 = buf.sample(n=4, seed=42)
    s2 = buf.sample(n=4, seed=42)
    assert len(s1) == 4
    for r1, r2 in zip(s1, s2, strict=True):
        np.testing.assert_array_equal(r1["x"], r2["x"])
        assert r1["y"] == r2["y"]


def test_sample_respects_buffer_size() -> None:
    """Sample more than buffer size returns buffer size."""
    buf = BetaBufferFIFO(capacity=8)
    buf.push(np.zeros(4, dtype=np.float32), 0)
    buf.push(np.ones(4, dtype=np.float32), 1)
    s = buf.sample(n=10, seed=0)
    assert len(s) == 2  # min(n_request, len(buffer))


def test_sample_empty_buffer_returns_empty_list() -> None:
    buf = BetaBufferFIFO(capacity=8)
    assert buf.sample(n=4, seed=0) == []


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BetaBufferFIFO(capacity=0)
