"""Unit tests for the G4HierarchicalClassifier (Plan G4-ter Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_ter_hp_sweep.dream_wrap_hier import (
    G4HierarchicalClassifier,
)


def test_classifier_has_three_linear_layers() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    # Public attributes for RESTRUCTURE site identification.
    assert clf.hidden_1 == 32
    assert clf.hidden_2 == 16
    assert clf.n_classes == 2


def test_predict_logits_shape() -> None:
    clf = G4HierarchicalClassifier(
        in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, seed=7
    )
    x = np.zeros((4, 784), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (4, 2)


def test_seed_determinism() -> None:
    a = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    b = G4HierarchicalClassifier(
        in_dim=10, hidden_1=4, hidden_2=3, n_classes=2, seed=42
    )
    x = np.ones((1, 10), dtype=np.float32)
    np.testing.assert_array_equal(a.predict_logits(x), b.predict_logits(x))
