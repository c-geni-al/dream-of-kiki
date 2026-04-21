"""Helpers for R1 bit-exact reproducibility tests.

Exposes the canonical serialization + hash routine and the
compare-or-bootstrap logic that reads / writes
``golden_hashes.json``.

Two modes :

* **Bootstrap** — the JSON has no entry for the test name : write
  the computed hash with ``"status": "pending_review"`` and pass.
  The user promotes to ``"accepted"`` manually.
* **Verify** — the JSON has an ``"accepted"`` entry : compare the
  computed hash ; fail loudly on mismatch.

Any ``"pending_review"`` or unknown status is treated as bootstrap
so repeated runs keep writing the current hash (useful while
drafting) but do not pass a stale value as a golden.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

GOLDEN_PATH = Path(__file__).parent / "golden_hashes.json"

# Canonical scenario parameters — every test in this suite MUST use
# these constants so that the JSON is coherent across ops.
CANONICAL_SEED = 7
CANONICAL_N_EPISODES = 3
CANONICAL_C_VERSION = "C-v0.7.0+UNSTABLE"
CANONICAL_PROFILE = "p_min"


def _load_golden() -> dict[str, dict[str, Any]]:
    """Load ``golden_hashes.json`` ; return ``{}`` if absent."""
    if not GOLDEN_PATH.exists():
        return {}
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(
            f"golden_hashes.json must be a top-level object, got "
            f"{type(data).__name__}"
        )
    return data


def _save_golden(data: dict[str, dict[str, Any]]) -> None:
    """Persist ``data`` back to ``golden_hashes.json`` (sorted keys)."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _mlx_version() -> str:
    """Return the installed mlx version or ``"unknown"``."""
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    """Return the short git HEAD sha, or ``"unknown"``.

    Honours ``$GIT_COMMIT`` when the repo is not a git checkout
    (e.g. CI artifact) so the workflow can inject the commit.
    """
    override = os.environ.get("GIT_COMMIT")
    if override:
        return override[:12]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha or "unknown"


def canonical_hash(payload: Any) -> str:
    """SHA-256 over a canonically-serialized ``payload``.

    The payload is JSON-encoded with ``sort_keys=True`` and the
    default `json` separators to keep serialization stable across
    Python minor versions. Floats are emitted via ``repr`` semantics
    so the exact IEEE-754 bit pattern is preserved (``json.dumps``
    uses ``float.__repr__`` under the hood).
    """
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def tensor_to_list(arr: Any) -> list[Any]:
    """Convert an mlx / numpy array to a nested Python list.

    Uses ``numpy.asarray`` which supports both backends via the
    buffer protocol. Floats are left as ``float`` so
    ``json.dumps`` produces their stable repr.
    """
    import numpy as np  # lazy : helper may be imported for path only

    result: list[Any] = np.asarray(arr).tolist()
    return result


def compare_or_bootstrap(
    test_name: str,
    computed_hash: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Compare ``computed_hash`` against the golden entry for ``test_name``.

    * No entry → write ``"status": "pending_review"`` and pass.
    * ``"accepted"`` entry with matching hash → pass.
    * ``"accepted"`` entry with mismatching hash → ``AssertionError``.
    * Any other status → refresh with the new hash,
      ``"status": "pending_review"``, and pass.
    """
    data = _load_golden()
    entry = data.get(test_name)
    metadata: dict[str, Any] = {
        "hash": computed_hash,
        "status": "pending_review",
        "mlx_version": _mlx_version(),
        "commit": _git_commit(),
    }
    if extra:
        metadata.update(extra)

    if entry is None:
        data[test_name] = metadata
        _save_golden(data)
        return

    status = entry.get("status")
    golden = entry.get("hash")
    if status == "accepted":
        if golden != computed_hash:
            raise AssertionError(
                f"R1 golden hash mismatch for {test_name!r} :\n"
                f"  accepted : {golden}\n"
                f"  computed : {computed_hash}\n"
                f"  mlx      : {metadata['mlx_version']}\n"
                f"  commit   : {metadata['commit']}"
            )
        return

    # pending_review (or unknown status) : refresh the payload so
    # the user always reviews the latest computed value.
    data[test_name] = metadata
    _save_golden(data)


__all__ = [
    "CANONICAL_C_VERSION",
    "CANONICAL_N_EPISODES",
    "CANONICAL_PROFILE",
    "CANONICAL_SEED",
    "GOLDEN_PATH",
    "canonical_hash",
    "compare_or_bootstrap",
    "tensor_to_list",
]
