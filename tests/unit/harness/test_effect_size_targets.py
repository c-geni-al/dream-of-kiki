"""Unit tests for empirical effect-size targets (Hu 2020 + Javadi 2024).

Targets are typed, frozen constants encoding published meta-analytic
Hedges' g and 95% CIs. Every constant must be immutable and resolve
to a real BibTeX key in docs/papers/paper1/references.bib.
"""
from dataclasses import FrozenInstanceError

import pytest

from harness.benchmarks.effect_size_targets import EffectSizeTarget


def test_target_constructs_with_all_fields() -> None:
    target = EffectSizeTarget(
        name="dummy_overall",
        hedges_g=0.29,
        ci_low=0.21,
        ci_high=0.38,
        sample_size_n=2004,
        k_studies=91,
        source_bibtex_key="hu2020tmr",
        profile_target="P_equ",
        stratum=None,
    )
    assert target.name == "dummy_overall"
    assert target.hedges_g == 0.29


def test_target_is_frozen() -> None:
    target = EffectSizeTarget(
        name="dummy",
        hedges_g=0.29,
        ci_low=0.21,
        ci_high=0.38,
        sample_size_n=2004,
        k_studies=91,
        source_bibtex_key="hu2020tmr",
        profile_target="P_equ",
        stratum=None,
    )
    with pytest.raises(FrozenInstanceError):
        target.hedges_g = 0.99  # type: ignore[misc]


def test_target_rejects_inverted_ci() -> None:
    """ci_low must be <= hedges_g <= ci_high (sanity, not stat rule)."""
    with pytest.raises(ValueError, match="ci_low.*ci_high"):
        EffectSizeTarget(
            name="bad",
            hedges_g=0.29,
            ci_low=0.50,   # > ci_high — invalid
            ci_high=0.10,
            sample_size_n=10,
            k_studies=1,
            source_bibtex_key="hu2020tmr",
            profile_target="P_equ",
            stratum=None,
        )


def test_target_rejects_g_outside_ci() -> None:
    with pytest.raises(ValueError, match="hedges_g.*ci"):
        EffectSizeTarget(
            name="bad",
            hedges_g=0.99,    # outside [0.21, 0.38]
            ci_low=0.21,
            ci_high=0.38,
            sample_size_n=10,
            k_studies=1,
            source_bibtex_key="hu2020tmr",
            profile_target="P_equ",
            stratum=None,
        )
