"""Conformance test for invariant S2 — no NaN/Inf in W_scratch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


def test_s2_invariant_blocks_nan_post_op() -> None:
    """S2 must abort swap when W_scratch contains NaN."""
    fake_post_op_weights = np.array([0.1, math.nan, 0.5])
    with pytest.raises(FiniteGuardError):
        check_finite(fake_post_op_weights)


def test_s2_invariant_passes_clean_post_op() -> None:
    """S2 should pass through valid weights silently."""
    fake_post_op_weights = np.array([0.05, -0.12, 0.3, 0.0])
    check_finite(fake_post_op_weights)
