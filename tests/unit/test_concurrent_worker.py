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


def test_worker_deferred_drain_aggregates_failures_and_continues() -> None:
    """Deferred drain must process ALL pending futures even when one
    handler raises (DR-0 accountability). The first exception is
    re-raised after the drain loop completes.
    """

    def fail_on_q1(episode: DreamEpisode) -> None:
        if episode.episode_id == "de-cw-fail-1":
            raise RuntimeError("boom")
        return None

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, fail_on_q1)
    worker = ConcurrentDreamWorker(
        runtime=runtime, queue_size=4, sync_drain=False
    )
    f0 = worker.submit(make_episode("de-cw-fail-0"))
    f1 = worker.submit(make_episode("de-cw-fail-1"))
    f2 = worker.submit(make_episode("de-cw-fail-2"))

    with pytest.raises(RuntimeError, match="boom"):
        worker.drain()

    # All futures resolved despite the middle failure (DR-0).
    assert f0.done() and f1.done() and f2.done()
    assert f1.exception() is not None
    assert f0.exception() is None
    assert f2.exception() is None
    # Runtime log preserves the full submission-order trace.
    assert len(runtime.log) == 3
    assert [e.episode_id for e in runtime.log] == [
        "de-cw-fail-0", "de-cw-fail-1", "de-cw-fail-2"
    ]
