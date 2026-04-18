"""S2 finite guard — no NaN/Inf, |w| ≤ w_max in weights.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.2
Invariant S2 — BLOCKING. Enforced before swap (pre-step 2).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray


DEFAULT_W_MAX = 1e6


class FiniteGuardError(Exception):
    """Raised when S2 invariant is violated (NaN, Inf, or |w| > w_max)."""


def check_finite(
    weights: NDArray | Mapping[str, NDArray],
    w_max: float = DEFAULT_W_MAX,
) -> None:
    """Verify all weights are finite and bounded.

    Accepts either a single array or a dict-of-arrays (e.g.,
    layer-keyed model weights).

    Raises FiniteGuardError on first violation, with location
    information when possible (key name for dict-of-arrays).
    """
    if isinstance(weights, Mapping):
        for key, arr in weights.items():
            try:
                check_finite(arr, w_max=w_max)
            except FiniteGuardError as exc:
                raise FiniteGuardError(
                    f"layer {key!r}: {exc}"
                ) from exc
        return

    arr = np.asarray(weights)

    if np.isnan(arr).any():
        raise FiniteGuardError("contains NaN")

    if np.isinf(arr).any():
        raise FiniteGuardError("contains Inf")

    abs_max = float(np.abs(arr).max()) if arr.size else 0.0
    if abs_max > w_max:
        raise FiniteGuardError(
            f"max |w| = {abs_max} exceeds bound {w_max}"
        )
