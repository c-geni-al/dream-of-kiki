"""ConcurrentDreamWorker — skeleton for async dream-episode dispatch.

S14.1 cycle-1: single-threaded synchronous execution behind a
Future-based API. The API contract (`submit() -> Future`,
`drain() -> list[EpisodeLogEntry]`, `pending_count`,
`QueueFullError` on capacity) is the substantive deliverable —
enables forward-compat with cycle-2 async/threading swap.

`sync_drain=True` (default): submit() executes the episode
immediately and returns a resolved Future. Pending count never
exceeds capacity in practice.

`sync_drain=False`: submit() enqueues without execution; queue
capacity is enforced. Episodes execute on `.drain()` call. Used
by the queue-full test.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §7
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


class QueueFullError(Exception):
    """Raised when submit() exceeds queue_size and worker is in
    deferred-drain mode."""


@dataclass
class ConcurrentDreamWorker:
    """Skeleton concurrent dream worker.

    Real concurrent execution (asyncio/threading) lands cycle 2.
    For cycle 1, sync execution behind Future API enables clients
    to write forward-compatible code.
    """

    runtime: DreamRuntime
    queue_size: int = 128
    sync_drain: bool = True
    _pending: list[tuple[DreamEpisode, Future]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, episode: DreamEpisode) -> Future:
        """Submit a dream-episode for execution.

        Returns a Future that resolves to the EpisodeLogEntry once
        the episode has been executed (immediately in sync_drain
        mode, on drain() in deferred mode).

        Raises QueueFullError if the queue is at capacity in
        deferred mode.
        """
        future: Future = Future()
        if self.sync_drain:
            self._execute_one(episode, future)
            return future

        if len(self._pending) >= self.queue_size:
            raise QueueFullError(
                f"queue full: {len(self._pending)} >= "
                f"{self.queue_size}"
            )
        self._pending.append((episode, future))
        return future

    def drain(self) -> list[EpisodeLogEntry]:
        """Execute all pending episodes (deferred mode) and return
        the log entries in submission order.

        In sync_drain mode the queue is always empty, so this
        returns the most recent N log entries from runtime.log
        where N matches the number of futures resolved since the
        last drain.
        """
        if self.sync_drain:
            # All episodes already executed; return the log
            # snapshot accumulated by runtime (caller's
            # responsibility to interpret).
            return list(self.runtime.log)

        entries: list[EpisodeLogEntry] = []
        while self._pending:
            episode, future = self._pending.pop(0)
            self._execute_one(episode, future)
            entries.append(future.result())
        return entries

    def _execute_one(
        self, episode: DreamEpisode, future: Future
    ) -> None:
        """Run a single episode through the runtime, resolve the
        future to the log entry."""
        log_len_before = len(self.runtime.log)
        try:
            self.runtime.execute(episode)
        except Exception as exc:
            future.set_exception(exc)
            return
        # The runtime appended one entry; pick it up.
        entry = self.runtime.log[log_len_before]
        future.set_result(entry)
