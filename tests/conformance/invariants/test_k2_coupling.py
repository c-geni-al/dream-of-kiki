"""K-coupling invariant — SO x fast-spindle phase-coupling measurement.

Reference: docs/invariants/registry.md (K2),
docs/proofs/k2-coupling-evidence.md, framework-C spec §5,
empirical anchor `elife2025bayesian` (eLife 2025,
coupling strength 0.33 [0.27, 0.39]).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from kiki_oniric.core.observables import PhaseCouplingObservable


def _is_protocol(cls: type) -> bool:
    return Protocol in getattr(cls, "__mro__", ())


def test_phase_coupling_observable_is_runtime_checkable() -> None:
    """Protocol must be @runtime_checkable so isinstance() works."""
    assert runtime_checkable(PhaseCouplingObservable) is PhaseCouplingObservable


def test_phase_coupling_observable_is_protocol() -> None:
    """Structural test mirrors test_dr3_substrate.test_all_8_protocols_declared."""
    assert _is_protocol(PhaseCouplingObservable)
