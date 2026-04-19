"""Unit tests for ``harness.real_models.base_model_registry``.

Cycle-3 pre-cycle-3 lock #2 : SHA-256 pins for the Qwen3.5 scale
axis ``{1.5B, 7B, 35B}``. Tests enforce the R1 reproducibility
contract (bit-stable ``(c_version, profile, seed, commit_sha)``
→ ``run_id``) by asserting every pin is well-formed. No live
network access — live verification is covered by ``verify_all(
live=True)`` which is opt-in and outside CI.
"""
from __future__ import annotations

import re

import pytest

from harness.real_models.base_model_registry import (
    REGISTRY,
    BaseModelPin,
    get_pin,
    verify_all,
)

_SHA40_RE = re.compile(r"^[a-f0-9]{40}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

_EXPECTED_SLOTS: tuple[str, ...] = (
    "qwen3p5-1p5b",
    "qwen3p5-7b",
    "qwen3p5-35b",
)


def test_registry_has_all_three_scale_slots() -> None:
    """Scale axis ``N ∈ {1.5B, 7B, 35B}`` is fully occupied.

    Cycle-3 spec §4 H5 requires evidence at all three scales ;
    missing any slot would block the stratified power budget.
    """
    for slot in _EXPECTED_SLOTS:
        assert slot in REGISTRY, f"missing scale slot : {slot}"


@pytest.mark.parametrize("slot", _EXPECTED_SLOTS)
def test_every_entry_is_well_formed(slot: str) -> None:
    """Every pin has non-empty ``repo_id`` + 40-char revision SHA.

    Enforces R1 : the pin is the hashable key in the reproducibility
    contract, so a malformed entry silently corrupts ``run_id``
    lineage.
    """
    pin = REGISTRY[slot]
    assert isinstance(pin, BaseModelPin)
    assert pin.repo_id, f"empty repo_id for {slot}"
    assert "/" in pin.repo_id, f"repo_id for {slot} must be org/name"
    assert _SHA40_RE.match(pin.revision_sha), (
        f"revision_sha for {slot} is not 40-char lowercase hex : "
        f"{pin.revision_sha!r}"
    )
    assert pin.scale_params > 0


@pytest.mark.parametrize("slot", _EXPECTED_SLOTS)
def test_file_sha256_when_present_is_64_hex(slot: str) -> None:
    """``file_sha256`` is either ``None`` or a 64-char hex digest."""
    pin = REGISTRY[slot]
    if pin.file_sha256 is not None:
        assert _SHA256_RE.match(pin.file_sha256), (
            f"file_sha256 for {slot} is not 64-char lowercase hex"
        )


def test_get_pin_returns_registered_entry() -> None:
    """``get_pin`` returns the same object as direct lookup."""
    pin = get_pin("qwen3p5-1p5b")
    assert pin is REGISTRY["qwen3p5-1p5b"]


def test_get_pin_raises_keyerror_with_available_slots() -> None:
    """Unknown slot raises ``KeyError`` listing available slots."""
    with pytest.raises(KeyError) as excinfo:
        get_pin("no-such-slot")
    msg = str(excinfo.value)
    assert "no-such-slot" in msg
    assert "qwen3p5-1p5b" in msg


def test_verify_all_returns_mapping_of_bools() -> None:
    """``verify_all`` (local-only) returns ``{slot: True}`` for all."""
    results = verify_all(live=False)
    assert set(results.keys()) == set(REGISTRY.keys())
    assert all(isinstance(v, bool) for v in results.values())
    # Every bundled pin must pass self-consistency (regressions
    # here indicate a bad commit of a malformed pin).
    assert all(results.values()), (
        f"self-consistency regression : {results}"
    )


def test_base_model_pin_is_frozen() -> None:
    """Pin dataclass is frozen so registry entries stay immutable."""
    pin = get_pin("qwen3p5-1p5b")
    with pytest.raises(Exception):
        pin.repo_id = "hacked"  # type: ignore[misc]
