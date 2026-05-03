"""Sharon 2025 SO-trough biomarker calibration utilities.

Provides a substrate-agnostic proxy reader for the
``so_trough_amplitude_factor`` field on dream profiles, plus the
three anchor constants extracted (qualitatively) from the
Sharon et al. 2025 hd-EEG dataset (N=55: 21 healthy older /
28 aMCI / 6 AD).

The publication does not report absolute SO-trough amplitudes in µV
— only a monotone gradient of cognitive performance with slow-wave
coherence across the three groups. The constants below are therefore
unit-arbitrary ratios anchored on the healthy-older arm = 1.0:

* ``SHARON_2025_HEALTHY_OLDER_ANCHOR = 1.0`` — intact slow-wave coherence.
* ``SHARON_2025_AMCI_MIDPOINT = 0.45`` — informed placeholder for the
  aMCI cohort (midpoint between healthy anchor and AD floor; final
  empirical value lands at G2 P_min pilot, ``scripts/pilot_g2.py``).
* ``SHARON_2025_AD_FLOOR = 0.20`` — informed placeholder for the AD
  cohort. Not currently consumed by any profile; reserved for a
  future P_pathological extension or sensitivity analysis.

Reference: Sharon et al., Alzheimer's & Dementia 2025
(sharon2025alzdementia in docs/papers/paper1/references.bib).
Calibration narrative: docs/papers/paper1/methodology.md §6.6.
"""
from __future__ import annotations

from typing import Protocol

SHARON_2025_HEALTHY_OLDER_ANCHOR: float = 1.0
SHARON_2025_AMCI_MIDPOINT: float = 0.45
SHARON_2025_AD_FLOOR: float = 0.20


class _HasSOFactor(Protocol):
    """Local structural type for objects exposing the SO-trough factor.

    Defined inline to avoid a circular import with the concrete
    ``PMinProfile`` / ``PEquProfile`` / ``PMaxProfile`` classes —
    any duck-typed object with a numeric
    ``so_trough_amplitude_factor`` attribute satisfies the contract.
    """

    so_trough_amplitude_factor: float


def compute_so_amplitude_proxy(profile: _HasSOFactor) -> float:
    """Read ``so_trough_amplitude_factor`` from a profile instance.

    Substrate-agnostic accessor used by DR-4-adjacent monotonicity
    tests and by the harness when reporting per-profile biomarker
    proxies.

    Parameters
    ----------
    profile :
        A dream profile instance — typically ``PMinProfile``,
        ``PEquProfile``, or ``PMaxProfile``. Any object exposing a
        float ``so_trough_amplitude_factor`` attribute is accepted.

    Returns
    -------
    float
        The calibration factor in arbitrary ratio units (anchor 1.0
        = healthy-older Sharon 2025 baseline).

    Raises
    ------
    TypeError
        If ``profile`` does not expose a numeric
        ``so_trough_amplitude_factor`` attribute.
    """
    factor = getattr(profile, "so_trough_amplitude_factor", None)
    if factor is None or not isinstance(factor, (int, float)):
        raise TypeError(
            f"{type(profile).__name__} does not expose a numeric "
            "so_trough_amplitude_factor attribute"
        )
    return float(factor)
