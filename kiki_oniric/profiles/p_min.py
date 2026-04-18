"""P_min profile — minimal publishable consolidation.

Channels: β → 1 (curated buffer in, weight delta out).
Operations: {replay, downscale}.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


@dataclass
class PMinProfile:
    """Minimal profile: replay + downscale handlers wired."""

    runtime: DreamRuntime = field(default_factory=DreamRuntime)
    replay_state: ReplayOpState = field(default_factory=ReplayOpState)
    downscale_state: DownscaleOpState = field(
        default_factory=DownscaleOpState
    )

    def __post_init__(self) -> None:
        self.runtime.register_handler(
            Operation.REPLAY, replay_handler(self.replay_state)
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE, downscale_handler(self.downscale_state)
        )
