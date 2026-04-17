"""Typed Protocol signatures for the 8 primitives of framework C.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1

These protocols satisfy condition (1) of the DR-3 Conformance
Criterion: any substrate instantiating the framework must implement
these typed signatures (see §6.2).

Awake → Dream:
  - AlphaStreamProtocol  (α raw traces, P_max only)
  - BetaBufferProtocol   (β curated episodic buffer, all profiles)
  - GammaSnapshotProtocol (γ weights-only snapshot, fallback)
  - DeltaLatentsProtocol (δ hierarchical latents, P_equ/P_max)

Dream → Awake:
  - WeightDeltaChannel     (canal 1, applied via swap)
  - LatentSampleChannel    (canal 2, data augmenter queue)
  - HierarchyChangeChannel (canal 3, atomic apply at swap time)
  - AttentionPriorChannel  (canal 4, copied at swap or live RO)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ==================== Awake → Dream ====================


@runtime_checkable
class AlphaStreamProtocol(Protocol):
    """α — Raw traces firehose (P_max only).

    Storage: ring buffer mmap, LIFO rotation.
    Complexity: O(tokens * model_size) per forward pass.
    """

    def append_trace(
        self,
        tokens: NDArray[np.int32],
        activations: NDArray[np.float32],
        attention_patterns: NDArray[np.float32],
        prediction_errors: NDArray[np.float32],
    ) -> None: ...

    def iter_traces(self) -> Iterator[dict[str, NDArray]]: ...


@runtime_checkable
class BetaBufferProtocol(Protocol):
    """β — Curated episodic buffer (all profiles).

    Storage: SQLite with index on saillance + consumed_by.
    Complexity: O(1) append when saillance > threshold.
    """

    def append_record(
        self,
        context: str,
        outcome: str,
        saillance_score: float,
        timestamp: datetime,
    ) -> int: ...

    def fetch_unconsumed(self, limit: int) -> list[dict]: ...

    def mark_consumed(self, record_ids: list[int], de_id: str) -> None: ...


@runtime_checkable
class GammaSnapshotProtocol(Protocol):
    """γ — Weights-only snapshot (fallback / diagnostic)."""

    def get_checkpoint_path(self) -> Path: ...

    def get_checkpoint_sha256(self) -> str: ...


@runtime_checkable
class DeltaLatentsProtocol(Protocol):
    """δ — Hierarchical latent snapshots (P_equ, P_max).

    Storage: ring buffer N=256.
    Complexity: O(sum species_dim) per snapshot.
    """

    def snapshot(
        self,
        species_activations: dict[str, NDArray[np.float32]],
    ) -> int: ...

    def get_recent(
        self, n: int
    ) -> list[dict[str, NDArray[np.float32]]]: ...


# ==================== Dream → Awake ====================


@runtime_checkable
class WeightDeltaChannel(Protocol):
    """Canal 1 — Weight delta (applied via swap).

    Constraint: must satisfy S1 (retained non-regression) + S2
    (finite values).
    """

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None: ...


@runtime_checkable
class LatentSampleChannel(Protocol):
    """Canal 2 — Latent samples queue (awake data augmenter).

    Constraint: must satisfy I3 (distributional drift bounded).
    """

    def enqueue(
        self,
        species: str,
        latent_vector: NDArray[np.float32],
        provenance: str,
    ) -> None: ...

    def dequeue(self) -> dict | None: ...


@runtime_checkable
class HierarchyChangeChannel(Protocol):
    """Canal 3 — Topology diff (atomic apply at swap).

    Constraint: must satisfy S3 (topology valid).
    """

    def apply_diff(self, diff: list[tuple[str, dict]]) -> None: ...


@runtime_checkable
class AttentionPriorChannel(Protocol):
    """Canal 4 — Attention prior (copy at swap or live RO).

    Constraint: must satisfy S4 (attention bounded).
    """

    def set_prior(self, prior: NDArray[np.float32]) -> None: ...

    def get_prior(self) -> NDArray[np.float32] | None: ...
