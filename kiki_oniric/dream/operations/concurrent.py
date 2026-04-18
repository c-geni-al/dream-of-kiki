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

Preserves : DR-0 (every executed episode appends an
`EpisodeLogEntry` to the runtime log), invariant K1 (per-DE
budget enforced upstream by `DreamRuntime.execute`), and the
queue-capacity invariant K-QUEUE (cycle-1 local : pending count
never exceeds `queue_size` in deferred mode). The capacity
guard raises `QueueFullError` annotated with K-QUEUE so the
invariant registry can track the breach.

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
    _resolved_since_last_drain: int = field(
        default=0, init=False, repr=False
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
                f"K-QUEUE: queue full: {len(self._pending)} >= "
                f"{self.queue_size} — violates queue-capacity "
                f"invariant"
            )
        self._pending.append((episode, future))
        return future

    def drain(self) -> list[EpisodeLogEntry]:
        """Execute all pending episodes (deferred mode) and return
        the log entries in submission order.

        In sync_drain mode the queue is always empty, so this
        returns the most recent N log entries from `runtime.log`
        where N matches the number of futures resolved since the
        last `drain()` call. The drain counter resets to 0 after
        the slice is returned ; callers that need the full log
        history should read `runtime.log` directly.
        """
        if self.sync_drain:
            n = self._resolved_since_last_drain
            self._resolved_since_last_drain = 0
            if n == 0:
                return []
            return list(self.runtime.log[-n:])

        entries: list[EpisodeLogEntry] = []
        while self._pending:
            episode, future = self._pending.pop(0)
            self._execute_one(episode, future)
            entries.append(future.result())
        # Deferred drains return the just-executed entries
        # directly ; the sync counter is irrelevant in this mode.
        self._resolved_since_last_drain = 0
        return entries

    def _execute_one(
        self, episode: DreamEpisode, future: Future
    ) -> None:
        """Run a single episode through the runtime, resolve the
        future to the log entry.

        Defends against runtimes that append zero or multiple
        entries per `execute()` (DR-0 invariant : exactly one
        per call) — the future surfaces a clear `RuntimeError`
        rather than an opaque `IndexError`.
        """
        log_len_before = len(self.runtime.log)
        try:
            self.runtime.execute(episode)
        except Exception as exc:
            future.set_exception(exc)
            return
        delta = len(self.runtime.log) - log_len_before
        if delta == 1:
            entry = self.runtime.log[log_len_before]
            future.set_result(entry)
            self._resolved_since_last_drain += 1
        elif delta == 0:
            future.set_exception(RuntimeError(
                "DR-0: runtime.execute did not append any log "
                "entry for episode"
            ))
        else:
            future.set_exception(RuntimeError(
                f"DR-0: runtime.execute appended {delta} entries, "
                f"expected exactly 1"
            ))
