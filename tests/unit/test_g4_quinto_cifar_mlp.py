"""Unit tests for G4HierarchicalCIFARClassifier (port of deeper MLP)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_quinto_test.cifar_mlp_classifier import (
    G4HierarchicalCIFARClassifier,
)

H = (256, 128, 64, 32)


def _clf() -> G4HierarchicalCIFARClassifier:
    return G4HierarchicalCIFARClassifier(
        in_dim=3072, hidden=H, n_classes=2, seed=0
    )


def test_predict_logits_shape() -> None:
    x = (
        np.random.default_rng(0)
        .standard_normal((4, 3072))
        .astype(np.float32)
    )
    assert _clf().predict_logits(x).shape == (4, 2)


def test_latent_returns_h3_dim() -> None:
    x = (
        np.random.default_rng(0)
        .standard_normal((4, 3072))
        .astype(np.float32)
    )
    # h3 = third hidden width (64) per pre-reg §5 sizing.
    assert _clf().latent(x).shape == (4, 64)


def test_restructure_step_zero_factor_is_noop() -> None:
    clf = _clf()
    w = np.asarray(clf._l3.weight).copy()
    clf.restructure_step(factor=0.0, seed=0)
    np.testing.assert_array_equal(np.asarray(clf._l3.weight), w)


def test_restructure_step_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _clf().restructure_step(factor=-0.1, seed=0)


def test_downscale_bounds() -> None:
    clf = _clf()
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=1.1)
