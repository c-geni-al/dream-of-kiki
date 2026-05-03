"""Unit tests for G4-sexto aggregator (H6-A / H6-B / H6-C conjunction)."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.g4_sexto_test.aggregator import aggregate_g4_sexto_verdict


def _step1_payload(*, fail_to_reject: bool) -> dict[str, object]:
    return {
        "verdict": {
            "h6a_recombine_strategy": {
                "n_p_max_mog": 30,
                "n_p_max_none": 30,
                "mean_p_max_mog": 0.50,
                "mean_p_max_none": 0.50,
                "welch_t": 0.0,
                "welch_p_two_sided": 0.99 if fail_to_reject else 0.001,
                "alpha_per_test": 0.0167,
                "fail_to_reject_h0": fail_to_reject,
                "h6a_recombine_empty_confirmed": fail_to_reject,
                "hedges_g_mog_vs_none": 0.0,
            }
        },
    }


def _step2_payload(*, fail_to_reject: bool) -> dict[str, object]:
    return {
        "verdict": {
            "h6b_recombine_strategy": {
                "n_p_max_mog": 30,
                "n_p_max_none": 30,
                "mean_p_max_mog": 0.30,
                "mean_p_max_none": 0.30,
                "welch_t": 0.0,
                "welch_p_two_sided": 0.99 if fail_to_reject else 0.001,
                "alpha_per_test": 0.0167,
                "fail_to_reject_h0": fail_to_reject,
                "h6b_recombine_empty_confirmed": fail_to_reject,
                "hedges_g_mog_vs_none": 0.0,
            }
        },
    }


def _write(p: Path, payload: dict[str, object]) -> None:
    p.write_text(json.dumps(payload))


def test_both_confirmed_yields_h6c(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=True))
    s2 = tmp_path / "step2.json"
    _write(s2, _step2_payload(fail_to_reject=True))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is True
    assert v["summary"]["h6b_confirmed"] is True
    assert v["summary"]["h6c_confirmed"] is True
    assert v["summary"]["h6c_partial"] is False
    assert v["summary"]["h6b_deferred"] is False


def test_only_h6a_yields_partial(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=True))
    s2 = tmp_path / "step2.json"
    _write(s2, _step2_payload(fail_to_reject=False))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is True
    assert v["summary"]["h6b_confirmed"] is False
    assert v["summary"]["h6c_confirmed"] is False
    assert v["summary"]["h6c_partial"] is True
    assert v["summary"]["h6b_deferred"] is False


def test_only_h6b_yields_partial(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=False))
    s2 = tmp_path / "step2.json"
    _write(s2, _step2_payload(fail_to_reject=True))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is False
    assert v["summary"]["h6b_confirmed"] is True
    assert v["summary"]["h6c_confirmed"] is False
    assert v["summary"]["h6c_partial"] is True


def test_both_falsified_yields_break(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=False))
    s2 = tmp_path / "step2.json"
    _write(s2, _step2_payload(fail_to_reject=False))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is False
    assert v["summary"]["h6b_confirmed"] is False
    assert v["summary"]["h6c_confirmed"] is False
    assert v["summary"]["h6c_partial"] is False


def test_step2_deferred_yields_deferred(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=True))
    v = aggregate_g4_sexto_verdict(s1, None)
    assert v["summary"]["h6b_deferred"] is True
    assert v["summary"]["h6c_confirmed"] is False
    assert v["summary"]["h6a_confirmed"] is True
    # H6-C is not "partial" when the second leg is deferred —
    # it is "deferred" (an open empirical question, not a
    # falsification or scope-bound resolution).
    assert v["summary"]["h6c_partial"] is False


def test_step2_deferred_when_path_does_not_exist(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=True))
    missing = tmp_path / "missing-step2.json"
    v = aggregate_g4_sexto_verdict(s1, missing)
    # missing step2 path also triggers the deferred branch.
    assert v["summary"]["h6b_deferred"] is True
    assert v["summary"]["h6c_confirmed"] is False


def test_h5c_to_h6c_extension_only_when_both_confirmed(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"
    _write(s1, _step1_payload(fail_to_reject=True))
    v_def = aggregate_g4_sexto_verdict(s1, None)
    assert v_def["summary"]["h5c_to_h6c_universality_extension"] is False
    s2 = tmp_path / "step2.json"
    _write(s2, _step2_payload(fail_to_reject=True))
    v_full = aggregate_g4_sexto_verdict(s1, s2)
    assert v_full["summary"]["h5c_to_h6c_universality_extension"] is True
