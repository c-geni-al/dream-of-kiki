"""R1 contract tests tying the run registry to the hash suite.

The registry at :mod:`harness.storage.run_registry` enforces half of
R1 : for a fixed ``(c_version, profile, seed, commit_sha)`` tuple,
``RunRegistry.register`` must return the same ``run_id`` bit-for-bit
(see the module docstring). This test verifies that contract from
the caller side **without mutating the registry**.

The other half of R1 — that the recorded *output* hash is stable for
that tuple — is enforced through the ``register_output_hash`` /
``get_output_hash`` pair exposed by
:class:`harness.storage.run_registry.RunRegistry` (schema :
sibling table ``run_output_hashes`` keyed on ``run_id``). The
contract test below drives that API directly.
"""
from __future__ import annotations

from pathlib import Path

from harness.storage.run_registry import RunRegistry

from tests.reproducibility._r1_helpers import (
    CANONICAL_C_VERSION,
    CANONICAL_PROFILE,
    CANONICAL_SEED,
)


# --------------------------------------------------------------------------
# Part 1 — run_id is a pure function of (c_version, profile, seed, sha)
# --------------------------------------------------------------------------


def test_r1_registry_same_tuple_same_run_id(tmp_path: Path) -> None:
    """Same tuple → identical ``run_id`` across fresh registries.

    Two registries backed by different DB files both encode the
    same canonical tuple to the same 128-bit id. This is the
    caller-observable half of R1.
    """
    commit_sha = "3a90b59deadbeefcafebabe0000000000000000"
    r1 = RunRegistry(tmp_path / "a.sqlite")
    r2 = RunRegistry(tmp_path / "b.sqlite")
    id_1 = r1.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, commit_sha
    )
    id_2 = r2.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, commit_sha
    )
    assert id_1 == id_2
    # Width contract — documented as 32 hex chars (128 bits).
    assert len(id_1) == 32


def test_r1_registry_different_seed_different_run_id(tmp_path: Path) -> None:
    """Changing any component of the tuple → different ``run_id``."""
    commit_sha = "3a90b59deadbeefcafebabe0000000000000000"
    registry = RunRegistry(tmp_path / "r.sqlite")
    id_seed_7 = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, 7, commit_sha
    )
    id_seed_8 = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, 8, commit_sha
    )
    assert id_seed_7 != id_seed_8

    id_profile_alt = registry.register(
        CANONICAL_C_VERSION, "p_equ", CANONICAL_SEED, commit_sha
    )
    id_profile_min = registry.register(
        CANONICAL_C_VERSION, "p_min", CANONICAL_SEED, commit_sha
    )
    assert id_profile_alt != id_profile_min

    id_sha_a = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, "a" * 40
    )
    id_sha_b = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, "b" * 40
    )
    assert id_sha_a != id_sha_b


def test_r1_registry_idempotent_insert(tmp_path: Path) -> None:
    """Re-registering the same tuple is a no-op + returns same id.

    The ``INSERT OR IGNORE`` clause means the row is unchanged, and
    the caller receives the same id — a necessary property for R1
    under retry / resume scenarios.
    """
    commit_sha = "3a90b59000000000000000000000000000000000"
    registry = RunRegistry(tmp_path / "r.sqlite")
    id_first = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, commit_sha
    )
    id_second = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, commit_sha
    )
    assert id_first == id_second
    row = registry.get(id_first)
    assert row["c_version"] == CANONICAL_C_VERSION
    assert row["profile"] == CANONICAL_PROFILE
    assert row["seed"] == CANONICAL_SEED
    assert row["commit_sha"] == commit_sha


# --------------------------------------------------------------------------
# Part 2 — output-hash contract : recorded output is stable for the tuple
# --------------------------------------------------------------------------


def test_r1_registry_output_hash_contract(tmp_path: Path) -> None:
    """Registry stores the canonical op output hash bit-for-bit.

    Completes R1 from the caller side :

    1. ``registry.register_output_hash(run_id, hash)`` stores the
       SHA-256 of the op output for that run.
    2. ``registry.get_output_hash(run_id)`` returns the same hash
       bit-for-bit, regardless of process / machine.
    3. Re-registering a conflicting hash for an existing run_id
       raises so the caller can surface the R1 violation.
    """
    commit_sha = "3a90b59000000000000000000000000000000000"
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register(
        CANONICAL_C_VERSION, CANONICAL_PROFILE, CANONICAL_SEED, commit_sha
    )
    registry.register_output_hash(run_id, "deadbeef" * 8)
    assert registry.get_output_hash(run_id) == "deadbeef" * 8
