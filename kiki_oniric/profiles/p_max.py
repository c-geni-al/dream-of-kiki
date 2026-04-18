"""P_max profile — maximalist consolidation skeleton (S16.1).

Channels: α + β + δ → 1 + 2 + 3 + 4 (full input + full output set).
Operations: {replay, downscale, restructure, recombine_full}.

Skeleton state: declares its target ops + channels via metadata
fields so that DR-4 chain inclusion (P_equ ⊆ P_max) can be tested
without wiring handlers. Real wiring lands cycle 2 alongside
α-stream firehose ingestion + ATTENTION_PRIOR canal-4 emission.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kiki_oniric.dream.episode import Operation, OutputChannel


@dataclass
class PMaxProfile:
    """Maximalist profile skeleton — implementation tracked cycle 2.

    Declares `target_ops` + `target_channels_out` so DR-4 axiom
    test can verify the chain inclusion P_equ ⊆ P_max even before
    handlers are wired. `recombine_full` is the lone op marked
    unimplemented (covers both wiring + the recombine variant
    upgrade from the C-Hobson light variant in P_equ).
    """

    status: str = "skeleton"
    unimplemented_ops: list[str] = field(
        default_factory=lambda: ["recombine_full"]
    )
    target_ops: set[Operation] = field(
        default_factory=lambda: {
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        }
    )
    target_channels_out: set[OutputChannel] = field(
        default_factory=lambda: {
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.LATENT_SAMPLE,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.ATTENTION_PRIOR,
        }
    )
