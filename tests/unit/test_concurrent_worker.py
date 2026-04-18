"""Unit tests for concurrent dream worker skeleton (S14.1)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.concurrent import (
    ConcurrentDreamWorker,
    QueueFullError,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


def _noop_handler(_episode: DreamEpisode) -> None:
    return None


def make_episode(ep_id: str) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=100, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def test_worker_initializes_with_runtime() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _noop_handler)
    worker = ConcurrentDreamWorker(runtime=runtime, queue_size=4)
    assert worker.runtime is runtime
    assert worker.queue_size == 4
    assert worker.pending_count == 0


def test_worker_submit_returns_future_resolves_to_log_entry() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _noop_handler)
    worker = ConcurrentDreamWorker(runtime=runtime, queue_size=4)
    future = worker.submit(make_episode("de-cw0"))
    # Future API: result() returns the log entry
    entry = future.result()
    assert isinstance(entry, EpisodeLogEntry)
    assert entry.episode_id == "de-cw0"
    assert entry.completed is True


def test_worker_drain_returns_all_pending_log_entries() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _noop_handler)
    worker = ConcurrentDreamWorker(runtime=runtime, queue_size=8)
    futures = [
        worker.submit(make_episode(f"de-cw-{i}")) for i in range(3)
    ]
    entries = worker.drain()
    assert len(entries) == 3
    assert [e.episode_id for e in entries] == [
        "de-cw-0", "de-cw-1", "de-cw-2"
    ]
    assert all(f.done() for f in futures)


def test_worker_queue_full_raises() -> None:
    """When skeleton queue is at capacity, submit raises QueueFullError.

    Skeleton enforces capacity even though it executes synchronously
    (forward-compat: real async worker may block instead).
    """
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _noop_handler)
    worker = ConcurrentDreamWorker(
        runtime=runtime, queue_size=2, sync_drain=False
    )
    worker.submit(make_episode("de-cw-q0"))
    worker.submit(make_episode("de-cw-q1"))
    with pytest.raises(QueueFullError):
        worker.submit(make_episode("de-cw-q2"))
