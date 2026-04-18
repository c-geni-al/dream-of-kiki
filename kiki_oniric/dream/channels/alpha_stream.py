"""α-stream — raw forward-pass traces firehose ring buffer.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
    Awake → Dream channel α (P_max only).

Records are typed (TraceRecord dataclass) and validated for
finiteness before append (S2 invariant propagation). Capacity is
fixed at construction ; FIFO or LIFO read order selectable.

Cycle 2 C2.5 : skeleton implementation for P_max wiring.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


class AlphaStreamError(Exception):
    """Raised when an α-stream operation violates invariants."""


@dataclass(frozen=True)
class TraceRecord:
    """One raw forward-pass trace (α channel element).

    Per framework spec §2.1 : (tokens, activations, attention,
    errors) per forward pass. All arrays are float/int numpy.
    """

    tokens: NDArray[np.int32]
    activations: NDArray[np.float32]
    attention: NDArray[np.float32]
    errors: NDArray[np.float32]


class AlphaStreamBuffer:
    """Ring buffer with bounded retention for α-stream traces.

    Implementation: `collections.deque(maxlen=capacity)` for
    constant-time append + automatic eviction when full.

    Finiteness guard (S2) runs on append ; malformed records raise
    AlphaStreamError immediately rather than silently contaminating
    the buffer. Read order is FIFO (insertion order) or LIFO
    (reverse) per construction arg.
    """

    def __init__(
        self,
        capacity: int,
        order: Literal["fifo", "lifo"] = "fifo",
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                f"capacity must be > 0, got {capacity}"
            )
        if order not in ("fifo", "lifo"):
            raise ValueError(
                f"order must be 'fifo' or 'lifo', got {order!r}"
            )
        self._capacity = capacity
        self._order = order
        self._buffer: deque[TraceRecord] = deque(maxlen=capacity)

    def append(self, record: TraceRecord) -> None:
        """Append a trace record to the buffer.

        Raises AlphaStreamError if record contains NaN/Inf values
        in any array (S2 invariant propagation).
        """
        try:
            check_finite(record.activations)
            check_finite(record.attention)
            check_finite(record.errors)
        except FiniteGuardError as exc:
            raise AlphaStreamError(
                f"α-stream record not finite: {exc}"
            ) from exc
        self._buffer.append(record)

    def snapshot(self) -> list[TraceRecord]:
        """Return a snapshot of currently held records.

        Order depends on construction : fifo = insertion order,
        lifo = reversed (most recent first).
        """
        items = list(self._buffer)
        if self._order == "lifo":
            items.reverse()
        return items

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def order(self) -> str:
        return self._order
