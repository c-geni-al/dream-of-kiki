"""Sharon 2025 SO-trough biomarker calibration tests for P_min / P_equ / P_max.

Reference: Sharon et al., Alzheimer's & Dementia 2025 (sharon2025alzdementia
in docs/papers/paper1/references.bib). hd-EEG, N=55 (21 healthy older /
28 aMCI / 6 AD). Cognitive performance decreases monotonically with
slow-wave trough amplitude and frontocentral synchronization.

These tests verify the qualitative calibration of the
``so_trough_amplitude_factor`` field on each profile: an informed
placeholder whose final empirical value lands at G2 / G4 pilots
(cf. ``scripts/pilot_g2.py``).
"""
from __future__ import annotations

import math

from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.profiles.so_calibration import (
    SHARON_2025_AD_FLOOR,
    SHARON_2025_AMCI_MIDPOINT,
    SHARON_2025_HEALTHY_OLDER_ANCHOR,
    compute_so_amplitude_proxy,
)


def test_p_min_so_trough_amplitude_factor_default_value() -> None:
    """P_min default factor = 0.45 (aMCI midpoint, Sharon 2025)."""
    profile = PMinProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 0.45)


def test_p_equ_so_trough_amplitude_factor_default_value() -> None:
    """P_equ default factor = 1.0 (healthy-older anchor, Sharon 2025)."""
    profile = PEquProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


def test_p_max_so_trough_amplitude_factor_default_value() -> None:
    """P_max default factor = 1.0 (intact substrate, healthy-young anchor)."""
    profile = PMaxProfile()
    assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


def test_compute_so_amplitude_proxy_reads_p_min() -> None:
    """compute_so_amplitude_proxy returns the field value on P_min."""
    profile = PMinProfile()
    assert math.isclose(compute_so_amplitude_proxy(profile), 0.45)


def test_compute_so_amplitude_proxy_reads_p_equ_and_p_max() -> None:
    """compute_so_amplitude_proxy returns 1.0 on healthy anchors."""
    assert math.isclose(compute_so_amplitude_proxy(PEquProfile()), 1.0)
    assert math.isclose(compute_so_amplitude_proxy(PMaxProfile()), 1.0)


def test_compute_so_amplitude_proxy_rejects_non_profile() -> None:
    """compute_so_amplitude_proxy raises TypeError on missing attribute."""
    import pytest

    class _NotAProfile:
        pass

    with pytest.raises(TypeError, match="so_trough_amplitude_factor"):
        compute_so_amplitude_proxy(_NotAProfile())  # type: ignore[arg-type]


def test_sharon_2025_anchor_constants() -> None:
    """Module-level constants pin the Sharon 2025 anchor values."""
    assert math.isclose(SHARON_2025_HEALTHY_OLDER_ANCHOR, 1.0)
    assert math.isclose(SHARON_2025_AMCI_MIDPOINT, 0.45)
    assert math.isclose(SHARON_2025_AD_FLOOR, 0.20)
