"""Hierarchical MLX MLP classifier + dream-episode wrapper for G4-ter.

Architecture: input -> Linear(in_dim, hidden_1) -> ReLU ->
Linear(hidden_1, hidden_2) -> ReLU -> Linear(hidden_2, n_classes).

Compared to ``experiments.g4_split_fmnist.dream_wrap.G4Classifier``
the hierarchy exposes a *middle* hidden layer (hidden_2) that is
addressable as a RESTRUCTURE site (perturb its weight tensor without
touching input projection nor output classifier) and a latent
representation (hidden_2 activations) that is addressable as a
RECOMBINE site (Gaussian-MoG synthetic-latent injection).

DR-0 accountability is automatic: every call to ``dream_episode_hier``
appends one EpisodeLogEntry to ``profile.runtime.log`` regardless of
handler outcome.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-ter-pilot.md sec 2-3
    docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class BetaRecordHier(TypedDict, total=False):
    """One curated episodic exemplar for the hierarchical head.

    Compared to ``BetaRecord`` (G4-bis), adds an optional
    ``latent`` field that holds the hidden_2 activation captured at
    push time, used as the support set for the RECOMBINE Gaussian-
    MoG sampler.
    """

    x: list[float]
    y: int
    latent: list[float] | None


@dataclass
class G4HierarchicalClassifier:
    """Hierarchical MLP classifier for Split-FMNIST 2-class tasks.

    Layers : Linear(in_dim, hidden_1) -> ReLU -> Linear(hidden_1,
    hidden_2) -> ReLU -> Linear(hidden_2, n_classes). Deterministic
    under a fixed ``seed`` via ``mx.random.seed`` at construction.
    """

    in_dim: int
    hidden_1: int
    hidden_2: int
    n_classes: int
    seed: int
    _l1: nn.Linear = field(init=False, repr=False)
    _l2: nn.Linear = field(init=False, repr=False)
    _l3: nn.Linear = field(init=False, repr=False)
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._l1 = nn.Linear(self.in_dim, self.hidden_1)
        self._l2 = nn.Linear(self.hidden_1, self.hidden_2)
        self._l3 = nn.Linear(self.hidden_2, self.n_classes)
        self._model = nn.Sequential(
            self._l1, nn.ReLU(), self._l2, nn.ReLU(), self._l3
        )
        mx.eval(self._model.parameters())

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return hidden_2 activations shape ``(N, hidden_2)``.

        Used by the beta buffer to capture per-record latents at push
        time for the RECOMBINE Gaussian-MoG sampler.
        """
        h1 = nn.relu(self._l1(mx.array(x)))
        h2 = nn.relu(self._l2(h1))
        mx.eval(h2)
        return np.asarray(h2)
