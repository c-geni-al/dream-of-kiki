"""Opt-in observability protocols for substrate-side measurements.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§5 (invariants K family).

These Protocols are **not** part of the DR-3 Conformance Criterion
condition (1) (the 8 mandatory primitives in
:mod:`kiki_oniric.core.primitives` are). They are opt-in surfaces a
substrate may implement when it can expose a measurement relevant
to a measurement-class invariant (currently: K2 phase-coupling).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PhaseCouplingObservable(Protocol):
    """Substrate emits a signal usable for SO x fast-spindle coupling.

    Empirical anchor: eLife 2025 Bayesian meta-analysis
    (BibTeX `elife2025bayesian`) — coupling strength 0.33 in 95 %
    CI [0.27, 0.39]; Bayes factor > 58 vs. null; Egger
    publication-bias test p = 0.59 on the phase branch.

    Returns
    -------
    so_phase, spindle_amplitude : 1-D float32 arrays of identical length.
        ``so_phase[i]`` is the instantaneous phase (radians,
        wrapped to [-pi, pi]) of the slow-oscillation channel at
        sample ``i``. ``spindle_amplitude[i]`` is the instantaneous
        amplitude of the fast-spindle envelope at sample ``i``.
    fs : float
        Sampling rate in Hz, > 0. Used only for documentation /
        downstream estimators; the K2 estimator works in samples.
    """

    def emit_phase_coupling_signal(
        self, n_samples: int, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]: ...
