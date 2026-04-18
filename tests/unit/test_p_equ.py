"""Unit tests for P_equ profile skeleton (β+δ → 1+3+4)."""
from __future__ import annotations

from kiki_oniric.profiles.p_equ import PEquProfile


def test_p_equ_can_be_instantiated() -> None:
    profile = PEquProfile()
    assert profile is not None


def test_p_equ_marks_unimplemented_ops() -> None:
    profile = PEquProfile()
    assert "restructure" in profile.unimplemented_ops
    assert "recombine" in profile.unimplemented_ops


def test_p_equ_status_is_skeleton() -> None:
    profile = PEquProfile()
    assert profile.status == "skeleton"
