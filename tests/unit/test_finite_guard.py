"""Unit tests for S2 finite guard (no NaN/Inf in weights)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


def test_check_finite_accepts_clean_array() -> None:
    weights = np.array([0.1, -0.5, 1.2, 0.0])
    check_finite(weights)  # No exception


def test_check_finite_rejects_nan() -> None:
    weights = np.array([0.1, math.nan, 0.5])
    with pytest.raises(FiniteGuardError, match="NaN"):
        check_finite(weights)


def test_check_finite_rejects_inf() -> None:
    weights = np.array([0.1, math.inf, 0.5])
    with pytest.raises(FiniteGuardError, match="Inf"):
        check_finite(weights)
    weights = np.array([0.1, -math.inf, 0.5])
    with pytest.raises(FiniteGuardError, match="Inf"):
        check_finite(weights)


def test_check_finite_rejects_above_w_max() -> None:
    weights = np.array([0.1, 1e9])
    with pytest.raises(FiniteGuardError, match="bound"):
        check_finite(weights, w_max=1e6)


def test_check_finite_handles_dict_of_arrays() -> None:
    weights = {
        "layer1": np.array([0.1, 0.2]),
        "layer2": np.array([math.nan, 0.5]),
    }
    with pytest.raises(FiniteGuardError, match="layer2"):
        check_finite(weights)
