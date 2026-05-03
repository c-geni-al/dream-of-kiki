# G4-bis Replay-Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `G4Classifier.dream_episode()` to actually mutate classifier weights via a replay step (gradient on a buffered FIFO of past tasks' images) plus a SHY downscale step, then re-run the G4 pilot and measure non-zero observed Hedges' g on retention.

**Architecture:** The Plan F wrapper is a pure spectator: it dispatches operation handlers for DR-0 logging but does not couple to the optimizer. This plan adds (1) a `BetaBufferFIFO` that the run driver fills with raw `{image, label}` pairs from each completed task, (2) an MLX gradient step `_replay_optimizer_step` that runs cross-entropy on a sample of buffer records, (3) a `_downscale_step` that multiplies weights by 0.95 (Tononi-Cirelli SHY), and (4) the dispatch logic in `dream_episode()` that fires these on the classifier when `Operation.REPLAY` / `Operation.DOWNSCALE` are in the profile's op set. P_min replays only; P_equ/P_max replay + downscale (RESTRUCTURE / RECOMBINE remain spectator-only since they target hierarchies / VAE latents not present in the binary MLP head).

**Tech Stack:** MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`), pytest, hypothesis, numpy. No new dependencies.

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `experiments/g4_split_fmnist/dream_wrap.py` | Classifier + dream coupling | Modify (add `BetaRecord`, `BetaBufferFIFO`, replay/downscale step methods, rewire `dream_episode`, plumb buffer through `train_task`) |
| `experiments/g4_split_fmnist/run_g4.py` | Pilot driver | Modify (instantiate buffer per cell, push samples after each task, threading into classifier) |
| `tests/unit/experiments/test_g4_dream_wrap.py` | Unit tests | Modify (existing tests stay passing under new semantics; add 6 new tests for coupling) |
| `tests/unit/experiments/test_g4_buffer.py` | Buffer unit tests | **Create** |
| `docs/milestones/g4-pilot-2026-05-03-bis.md` | Bis milestone (md) | **Create** via re-run |
| `docs/milestones/g4-pilot-2026-05-03-bis.json` | Bis milestone (json) | **Create** via re-run |
| `docs/papers/paper2/results.md` | Paper 2 narrative EN | Modify (add §7.1.2 G4-bis) |
| `docs/papers/paper2-fr/results.md` | Paper 2 narrative FR | Modify (add §7.1.2 G4-bis FR mirror) |
| `CHANGELOG.md` | DualVer log | Modify (add Empirical bullet under `[Unreleased]`) |
| `STATUS.md` | Gate row | Modify (G4 row to reflect bis verdict) |

---

## Constraints honored

- **DR-0 accountability**: every dispatched op still appends one `EpisodeLogEntry` via `runtime.execute(...)`. The new coupling fires *after* runtime dispatch, side-by-side with the spectator handlers.
- **R1 determinism**: replay-sampler RNG is `np.random.default_rng(seed + offset)`; gradient step uses MLX SGD with no extra randomness; downscale is deterministic. Fresh `run_id`s register against the same `(c_version, profile, seed, commit_sha)` tuple — they differ from G4-original ids because dream_episode() semantics changed (this is correct behavior, not regression — documented in commit body of Task 7).
- **No FC bump**: implementation + calibration metadata only. Axiom signatures, primitive Protocols, and channel sets unchanged. EC stays PARTIAL or shifts per observed verdict (Task 9 decision branch).
- **Commit policy**: ≤50-char subject, ≥3-char scope, body ≤72-char wrap, no `Co-Authored-By` trailer (workspace rule).

---

## Read-first context

Before starting, the executing engineer must skim:
- `experiments/g4_split_fmnist/dream_wrap.py` (current spectator wrapper)
- `kiki_oniric/profiles/p_min.py`, `kiki_oniric/profiles/p_equ.py`, `kiki_oniric/profiles/p_max.py` (op handler registration patterns)
- `kiki_oniric/dream/operations/replay_real.py` (MLX gradient pattern — informs replay step)
- `kiki_oniric/dream/operations/downscale_real.py` (MLX weight shrinkage pattern — informs downscale step)
- `docs/osf-prereg-g4-pilot.md` (hypotheses H1, H3, H_DR4 — unchanged for bis)
- `docs/milestones/g4-pilot-2026-05-03.md` (current null-result baseline)

---

## Task 0: Investigate current state

**Files:**
- Read: `experiments/g4_split_fmnist/dream_wrap.py`
- Read: `experiments/g4_split_fmnist/run_g4.py`
- Read: `kiki_oniric/profiles/p_min.py:55-61` (handler registration)
- Read: `kiki_oniric/profiles/p_equ.py:62-77`
- Read: `kiki_oniric/profiles/p_max.py:103-117`

- [ ] **Step 1: Confirm dream_episode is spectator**

Run:
```bash
uv run python -c "
import inspect
from experiments.g4_split_fmnist.dream_wrap import G4Classifier
src = inspect.getsource(G4Classifier.dream_episode)
print('mutates_weights:', any(kw in src for kw in ['_replay_optimizer_step', '_downscale_step', 'opt.update', 'self._model.update']))
"
```
Expected: `mutates_weights: False` — confirms Plan F left dream_episode spectator-only.

- [ ] **Step 2: Confirm P_min/P_equ have no `target_ops` attribute**

Run:
```bash
uv run python -c "
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
print('p_min.target_ops:', hasattr(PMinProfile(), 'target_ops'))
print('p_equ.target_ops:', hasattr(PEquProfile(), 'target_ops'))
print('p_max.target_ops:', hasattr(PMaxProfile(), 'target_ops'))
"
```
Expected: `False / False / True`. Decision: dispatch in `dream_episode()` will continue using `isinstance(profile, PMinProfile)` for P_min vs P_equ/P_max, NOT touch profile internals (kiki_oniric/profiles/ is governed by jalonné rebase policy — see `kiki_oniric/CLAUDE.md` "Don't bake seeds into a profile" anti-pattern).

- [ ] **Step 3: Confirm tests baseline green**

Run: `uv run pytest tests/unit/experiments/ -v --no-cov`
Expected: all pass (12 tests in `test_g4_dream_wrap.py`, 2 in `test_g4_run_g4_smoke.py`, plus `test_g4_dataset.py`).

---

## Task 0.5: Architecture decision — raw-image replay

**Files:** None (this task is a one-line decision recorded in Task 1's docstring).

- [ ] **Step 1: Note decision**

Decision (locked by this plan): the β buffer carries **raw image-class pairs** (`x: list[float] of length feat_dim`, `y: int label in {0,1}`), not latent features. Rationale:

1. The Split-FMNIST classifier in this experiment is a flat MLP (`nn.Linear(784, hidden) → ReLU → Linear(hidden, 2)`). There is no encoder to extract latents from.
2. Storing raw images keeps the buffer agent-substrate-agnostic and matches generative-replay-without-generator-network practice (van de Ven 2020 §3.2 "stored exemplar" baseline).
3. β records flowing into the dream-runtime input slice already declare `{"x": list[float], "y": list[float]}` per `replay_real.py` contract — we extend `y` to also accept `int` (single-class index) since CE loss does not need a one-hot vector. The validator in `replay_real.py:85-89` only checks key presence, not shape — compatible.

This decision is documented in the docstring of `BetaBufferFIFO` introduced in Task 1.

---

## Task 1: Add BetaRecord + BetaBufferFIFO

**Files:**
- Modify: `experiments/g4_split_fmnist/dream_wrap.py` (add new TypedDict + class near top, after imports)
- Test: `tests/unit/experiments/test_g4_buffer.py` (new file)

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/experiments/test_g4_buffer.py`:

```python
"""Unit tests for the curated β buffer feeding the G4 dream coupling."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO


def test_buffer_starts_empty() -> None:
    buf = BetaBufferFIFO(capacity=128)
    assert len(buf) == 0


def test_push_increments_size_until_capacity() -> None:
    buf = BetaBufferFIFO(capacity=4)
    buf.push(np.zeros(8, dtype=np.float32), 0)
    buf.push(np.zeros(8, dtype=np.float32), 1)
    assert len(buf) == 2


def test_push_evicts_oldest_at_capacity() -> None:
    buf = BetaBufferFIFO(capacity=2)
    a = np.array([1.0] * 4, dtype=np.float32)
    b = np.array([2.0] * 4, dtype=np.float32)
    c = np.array([3.0] * 4, dtype=np.float32)
    buf.push(a, 0)
    buf.push(b, 1)
    buf.push(c, 0)  # evicts a
    assert len(buf) == 2
    items = buf.snapshot()
    np.testing.assert_array_equal(items[0]["x"], b)
    np.testing.assert_array_equal(items[1]["x"], c)


def test_sample_seeded_reproducible() -> None:
    buf = BetaBufferFIFO(capacity=16)
    rng = np.random.default_rng(0)
    for i in range(10):
        buf.push(rng.standard_normal(4).astype(np.float32), i % 2)
    s1 = buf.sample(n=4, seed=42)
    s2 = buf.sample(n=4, seed=42)
    assert len(s1) == 4
    for r1, r2 in zip(s1, s2, strict=True):
        np.testing.assert_array_equal(r1["x"], r2["x"])
        assert r1["y"] == r2["y"]


def test_sample_respects_buffer_size() -> None:
    """Sample more than buffer size returns buffer size."""
    buf = BetaBufferFIFO(capacity=8)
    buf.push(np.zeros(4, dtype=np.float32), 0)
    buf.push(np.ones(4, dtype=np.float32), 1)
    s = buf.sample(n=10, seed=0)
    assert len(s) == 2  # min(n_request, len(buffer))


def test_sample_empty_buffer_returns_empty_list() -> None:
    buf = BetaBufferFIFO(capacity=8)
    assert buf.sample(n=4, seed=0) == []


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BetaBufferFIFO(capacity=0)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_buffer.py -v --no-cov`
Expected: ImportError — `BetaBufferFIFO` not in `dream_wrap`.

- [ ] **Step 3: Implement BetaRecord + BetaBufferFIFO**

In `experiments/g4_split_fmnist/dream_wrap.py`, after the existing imports and before `ProfileT = ...`, insert:

```python
from collections import deque
from typing import TypedDict


class BetaRecord(TypedDict):
    """One curated episodic exemplar : flattened image + binary label.

    Decision (Plan G4-bis Task 0.5) : raw-image replay over latent-
    feature replay because the G4 classifier is a flat MLP with no
    encoder. Matches van de Ven 2020 §3.2 stored-exemplar baseline.
    The ``x`` value is a Python list of floats (JSON-serialisable per
    the dream-runtime input_slice contract); ``y`` is an int class
    index in ``{0, 1}`` — cross-entropy does not require one-hot.
    """

    x: list[float]
    y: int


class BetaBufferFIFO:
    """Bounded curated episodic buffer (β channel input).

    FIFO eviction at capacity. ``push`` appends one record per call
    (raw-image + label, copied to JSON-serialisable lists so the dream
    runtime input_slice contract is preserved). ``sample(n, seed)``
    returns a deterministic random subsample sized ``min(n, len)``
    via ``np.random.default_rng(seed)``.

    The buffer is owned by the G4 pilot driver, not by the profile
    (the kiki_oniric/profiles/ jalonné-rebase rule forbids adding
    pilot-specific state to a profile).
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity}"
            )
        self._capacity = capacity
        self._records: deque[BetaRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(self, x: np.ndarray, y: int) -> None:
        """Append one ``(x, y)`` exemplar (FIFO eviction at capacity)."""
        self._records.append({"x": x.astype(np.float32).tolist(), "y": int(y)})

    def snapshot(self) -> list[BetaRecord]:
        """Return a list copy of every record currently in the buffer."""
        return [
            {"x": list(r["x"]), "y": int(r["y"])} for r in self._records
        ]

    def sample(self, n: int, seed: int) -> list[BetaRecord]:
        """Return a deterministic sample of size ``min(n, len(self))``.

        Uses ``np.random.default_rng(seed)`` for reproducibility under
        R1. Returns ``[]`` on empty buffer (caller is responsible for
        skipping the replay step in that case).
        """
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        # rng.choice on indices then materialise via list copy.
        indices = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [
            {"x": list(snapshot[i]["x"]), "y": int(snapshot[i]["y"])}
            for i in sorted(indices.tolist())
        ]
```

- [ ] **Step 4: Run buffer tests to verify pass**

Run: `uv run pytest tests/unit/experiments/test_g4_buffer.py -v --no-cov`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_buffer.py
git commit -m "feat(g4): add BetaBufferFIFO for replay coupling" -m "$(cat <<'EOF'
Introduces BetaRecord TypedDict and BetaBufferFIFO bounded
deque for the G4-bis dream-replay coupling. Stores raw image-
class pairs (van de Ven 2020 §3.2 stored-exemplar) keyed
list[float] / int so the dream runtime input_slice contract
holds. sample(n, seed) is R1-deterministic via numpy default_rng.

Plan G4-bis Task 1.
EOF
)"
```

---

## Task 2: Plumb buffer + replay step signature into G4Classifier

**Files:**
- Modify: `experiments/g4_split_fmnist/dream_wrap.py` (add `_replay_optimizer_step` method on `G4Classifier`)
- Test: `tests/unit/experiments/test_g4_dream_wrap.py` (add new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_dream_wrap.py`:

```python
def test_replay_optimizer_step_changes_weights() -> None:
    """A replay step on non-empty records must mutate weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    # Snapshot a layer's weight before any update.
    w_before = np.asarray(clf._model.layers[0].weight).copy()

    # Build a 4-record sample with deterministic RNG.
    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)

    records = buf.sample(n=4, seed=0)
    clf._replay_optimizer_step(records, lr=0.1, n_steps=2)

    w_after = np.asarray(clf._model.layers[0].weight)
    assert not np.allclose(w_before, w_after), (
        "weights must change after replay step"
    )


def test_replay_optimizer_step_empty_records_noop() -> None:
    """Empty record list must leave weights untouched."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf._replay_optimizer_step([], lr=0.1, n_steps=2)
    w_after = np.asarray(clf._model.layers[0].weight)
    np.testing.assert_array_equal(w_before, w_after)


def test_replay_optimizer_step_seeded_reproducible() -> None:
    """Two classifiers with same seed + same records → same final weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)
    records = buf.sample(n=4, seed=42)

    clf_a = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_b = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_a._replay_optimizer_step(records, lr=0.1, n_steps=2)
    clf_b._replay_optimizer_step(records, lr=0.1, n_steps=2)

    np.testing.assert_array_equal(
        np.asarray(clf_a._model.layers[0].weight),
        np.asarray(clf_b._model.layers[0].weight),
    )
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py::test_replay_optimizer_step_changes_weights -v --no-cov`
Expected: AttributeError — `_replay_optimizer_step` not defined.

- [ ] **Step 3: Implement the method**

In `experiments/g4_split_fmnist/dream_wrap.py`, inside class `G4Classifier`, after `_loss_fn` (around line 190 in current file) and before `dream_episode`, insert:

```python
    def _replay_optimizer_step(
        self,
        records: list[BetaRecord],
        *,
        lr: float,
        n_steps: int,
    ) -> None:
        """Run ``n_steps`` SGD passes over ``records`` (CE loss).

        Generative-replay coupling per van de Ven 2020 §3.2 stored-
        exemplar. Empty ``records`` → no-op (parallels the S1 no-op
        branch in :func:`replay_real_handler`). Determinism : MLX SGD
        + fixed batch order = bit-stable given identical model state
        and record list.
        """
        if not records:
            return
        x = mx.array([r["x"] for r in records])
        y = mx.array([r["y"] for r in records])
        opt = optim.SGD(learning_rate=lr)
        loss_and_grad = nn.value_and_grad(self._model, self._loss_fn)
        for _ in range(n_steps):
            _loss, grads = loss_and_grad(self._model, x, y)
            opt.update(self._model, grads)
            mx.eval(self._model.parameters(), opt.state)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py::test_replay_optimizer_step_changes_weights tests/unit/experiments/test_g4_dream_wrap.py::test_replay_optimizer_step_empty_records_noop tests/unit/experiments/test_g4_dream_wrap.py::test_replay_optimizer_step_seeded_reproducible -v --no-cov`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_dream_wrap.py
git commit -m "feat(g4): add classifier replay-step method" -m "$(cat <<'EOF'
G4Classifier._replay_optimizer_step runs n SGD passes on a list of
BetaRecord exemplars (CE loss). Empty list is a no-op (S1-aligned
trivial branch). Seeded reproducibility verified : same model state
+ same records → bit-identical post-replay weights.

Plan G4-bis Task 2.
EOF
)"
```

---

## Task 3: Implement _downscale_step (SHY)

**Files:**
- Modify: `experiments/g4_split_fmnist/dream_wrap.py` (add `_downscale_step` method)
- Test: `tests/unit/experiments/test_g4_dream_wrap.py` (add 2 tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_dream_wrap.py`:

```python
def test_downscale_step_multiplies_weights_by_factor() -> None:
    """Each layer's weight must scale by exactly ``factor``."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    w_before = [
        np.asarray(layer.weight).copy()
        for layer in clf._model.layers
        if hasattr(layer, "weight") and layer.weight is not None
    ]
    clf._downscale_step(factor=0.95)
    w_after = [
        np.asarray(layer.weight)
        for layer in clf._model.layers
        if hasattr(layer, "weight") and layer.weight is not None
    ]
    assert len(w_before) == len(w_after) > 0
    for wb, wa in zip(w_before, w_after, strict=True):
        np.testing.assert_allclose(wa, wb * 0.95, rtol=1e-6)


def test_downscale_step_factor_bounds() -> None:
    """``factor`` outside (0, 1] must raise ValueError."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=0.0)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=1.5)
    with pytest.raises(ValueError, match="shrink_factor|factor"):
        clf._downscale_step(factor=-0.1)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py::test_downscale_step_multiplies_weights_by_factor -v --no-cov`
Expected: AttributeError — `_downscale_step` not defined.

- [ ] **Step 3: Implement method**

In `experiments/g4_split_fmnist/dream_wrap.py`, inside class `G4Classifier`, immediately after `_replay_optimizer_step`, insert:

```python
    def _downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Tononi-Cirelli SHY synaptic-homeostasis analog. ``factor``
        is calibrated qualitatively at 0.95 in this pilot (a 5 %
        per-episode drift to mimic NREM SO-trough downselection).
        Empirically pinning the optimal factor is future work.

        Bounds : ``factor`` must lie in ``(0, 1]`` — same constraint
        as :func:`downscale_real_handler` (``shrink_factor``). Raises
        :class:`ValueError` outside that range with a message tagging
        ``shrink_factor`` so error logs grep cleanly across both
        sites.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        for layer in self._model.layers:
            w = getattr(layer, "weight", None)
            b = getattr(layer, "bias", None)
            if w is not None:
                layer.weight = w * factor
            if b is not None:
                layer.bias = b * factor
        # Materialise all updated tensors before returning.
        tensors = []
        for layer in self._model.layers:
            if getattr(layer, "weight", None) is not None:
                tensors.append(layer.weight)
            if getattr(layer, "bias", None) is not None:
                tensors.append(layer.bias)
        if tensors:
            mx.eval(*tensors)
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py::test_downscale_step_multiplies_weights_by_factor tests/unit/experiments/test_g4_dream_wrap.py::test_downscale_step_factor_bounds -v --no-cov`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_dream_wrap.py
git commit -m "feat(g4): add classifier SHY downscale step" -m "$(cat <<'EOF'
G4Classifier._downscale_step multiplies every weight + bias in the
MLP by a factor in (0, 1]. Factor calibrated qualitatively at 0.95
per pilot — empirical pin deferred. Bounds check matches the
shrink_factor contract in downscale_real_handler.

Plan G4-bis Task 3.
EOF
)"
```

---

## Task 4: Wire dream_episode to fire replay + downscale on the classifier

**Files:**
- Modify: `experiments/g4_split_fmnist/dream_wrap.py` (signature + body of `dream_episode`)
- Test: `tests/unit/experiments/test_g4_dream_wrap.py` (add 1 new test, update 2 existing)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/experiments/test_g4_dream_wrap.py`:

```python
def test_dream_episode_pmin_mutates_weights_when_buffer_nonempty(
    tiny_task: SplitFMNISTTask,
) -> None:
    """P_min dream_episode with a populated β buffer must mutate weights."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)

    rng = np.random.default_rng(0)
    buf = BetaBufferFIFO(capacity=8)
    for i in range(4):
        buf.push(rng.standard_normal(16).astype(np.float32), i % 2)

    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    w_after = np.asarray(clf._model.layers[0].weight)
    assert not np.allclose(w_before, w_after), (
        "P_min dream_episode with non-empty buffer must mutate weights"
    )


def test_dream_episode_pmin_empty_buffer_only_downscales(
    tiny_task: SplitFMNISTTask,
) -> None:
    """Empty buffer → no replay step, but DOWNSCALE still fires (P_min ops)."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)
    buf = BetaBufferFIFO(capacity=8)  # left empty

    w_before = np.asarray(clf._model.layers[0].weight).copy()
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    w_after = np.asarray(clf._model.layers[0].weight)
    # Downscale by 0.95 must scale visibly.
    np.testing.assert_allclose(w_after, w_before * 0.95, rtol=1e-6)


def test_dream_episode_baseline_arm_does_not_call_dream_episode() -> None:
    """build_profile('baseline') must still raise — driver-level guard."""
    with pytest.raises(ValueError, match="baseline"):
        build_profile("baseline", seed=7)
```

Update the existing `test_dream_episode_executes_pmin_handlers` and `test_dream_episode_executes_pequ_with_4_ops` so they pass the new `beta_buffer` kwarg. Replace them in-place:

```python
def test_dream_episode_executes_pmin_handlers(tiny_task: SplitFMNISTTask) -> None:
    """P_min episode must add at least 1 entry to runtime.log (DR-0)."""
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)
    buf = BetaBufferFIFO(capacity=8)
    log_before = len(profile.runtime.log)
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    assert len(profile.runtime.log) == log_before + 1


def test_dream_episode_executes_pequ_with_4_ops(tiny_task: SplitFMNISTTask) -> None:
    """P_equ episode must execute 4 ops (replay/downscale/restructure/recombine)."""
    from kiki_oniric.dream.episode import Operation
    from experiments.g4_split_fmnist.dream_wrap import BetaBufferFIFO

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_equ", seed=7)
    buf = BetaBufferFIFO(capacity=8)
    clf.dream_episode(profile, seed=7, beta_buffer=buf)
    last = profile.runtime.log[-1]
    assert last.completed
    assert set(last.operations_executed) == {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py -v --no-cov`
Expected: 4 NEW failures + 2 broken existing tests (TypeError on `beta_buffer=` kwarg).

- [ ] **Step 3: Rewire dream_episode**

In `experiments/g4_split_fmnist/dream_wrap.py`, replace the `dream_episode` method (currently lines 193-253) with:

```python
    def dream_episode(
        self,
        profile: ProfileT,
        seed: int,
        *,
        beta_buffer: "BetaBufferFIFO",
        replay_n_records: int = 32,
        replay_n_steps: int = 1,
        replay_lr: float = 0.01,
        downscale_factor: float = 0.95,
    ) -> None:
        """Drive one :class:`DreamEpisode` and mutate classifier weights.

        Plan G4-bis coupling : after the profile's runtime executes
        the operation set (DR-0 logging), this method also fires
        weight-mutating steps on ``self._model`` :

        - If ``Operation.REPLAY`` is in the dispatched op set, sample
          ``replay_n_records`` from ``beta_buffer`` (seeded by ``seed``)
          and run ``replay_n_steps`` SGD passes via
          :meth:`_replay_optimizer_step`. Empty buffer → replay no-op
          (S1-trivial branch).
        - If ``Operation.DOWNSCALE`` is in the dispatched op set,
          run :meth:`_downscale_step` with ``downscale_factor``.

        ``Operation.RESTRUCTURE`` and ``Operation.RECOMBINE`` remain
        spectator-only on this MLP head : the classifier has no
        hierarchy nor VAE latents to restructure / recombine. Their
        DR-0 log entries continue to register through ``runtime.execute``.

        Parameters
        ----------
        profile :
            Active dream profile (``P_min`` / ``P_equ`` / ``P_max``).
        seed :
            Per-episode seed used for both the ``input_slice``
            β-record sampler (legacy spectator path) and the
            classifier-side sampler (new coupling path). Forms part
            of R1's run_id input via the calling driver.
        beta_buffer :
            Curated episodic buffer accumulated by the driver from
            past tasks' training samples.
        replay_n_records, replay_n_steps, replay_lr :
            Replay hyperparameters. Default n=32 records × 1 SGD step
            per episode, lr=0.01 — typical class-incremental replay
            (van de Ven 2020). Documented in the function signature
            so all calls from ``run_g4.py`` can be grepped.
        downscale_factor :
            SHY shrinkage factor in (0, 1]. Default 0.95 calibrated
            qualitatively (5 % per-episode drift). Future work : pin
            empirically.
        """
        profile_name = type(profile).__name__
        if isinstance(profile, PMinProfile):
            ops: tuple[Operation, ...] = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
            )
            channels: tuple[OutputChannel, ...] = (
                OutputChannel.WEIGHT_DELTA,
            )
        else:
            ops = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
                Operation.RESTRUCTURE,
                Operation.RECOMBINE,
            )
            channels = (
                OutputChannel.WEIGHT_DELTA,
                OutputChannel.HIERARCHY_CHG,
                OutputChannel.LATENT_SAMPLE,
            )

        # Spectator runtime path (kept for DR-0 logging).
        # Synthetic input_slice values for the existing spectator
        # handlers ; the *coupling* step below uses real data from
        # beta_buffer, decoupled from these synthetic placeholders.
        synthetic_records = sample_beta_records(
            seed=seed, n_records=4, feat_dim=4
        )
        rng = np.random.default_rng(seed + 10_000)
        delta_latents = [
            rng.standard_normal(4).astype(np.float32).tolist()
            for _ in range(2)
        ]
        episode = DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "beta_records": synthetic_records,
                "shrink_factor": 0.99,
                "topo_op": "reroute",
                "swap_indices": [0, 1],
                "delta_latents": delta_latents,
            },
            operation_set=ops,
            output_channels=channels,
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=10.0, energy_j=1.0
            ),
            episode_id=f"g4-{profile_name}-seed{seed}",
        )
        profile.runtime.execute(episode)

        # ---- Plan G4-bis coupling : mutate self._model on dispatched ops ----
        if Operation.REPLAY in ops:
            sampled = beta_buffer.sample(
                n=replay_n_records, seed=seed
            )
            self._replay_optimizer_step(
                sampled, lr=replay_lr, n_steps=replay_n_steps
            )
        if Operation.DOWNSCALE in ops:
            self._downscale_step(factor=downscale_factor)
```

Note the TYPE_CHECKING / forward-ref pattern : `BetaBufferFIFO` is defined in the same module, so the literal annotation `"BetaBufferFIFO"` resolves at runtime via PEP 563-style postponement (`from __future__ import annotations` is already in the file).

- [ ] **Step 4: Run all dream_wrap tests**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py -v --no-cov`
Expected: all pass (12 original + 8 new = 20 passing).

- [ ] **Step 5: Commit**

```bash
git add experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_dream_wrap.py
git commit -m "feat(g4): wire dream_episode to weight mutations" -m "$(cat <<'EOF'
dream_episode now mutates classifier weights via _replay_optimizer_step
(generative replay, n=32 records × 1 SGD step) when Operation.REPLAY
is dispatched, and _downscale_step (factor=0.95) when DOWNSCALE
is dispatched. RESTRUCTURE / RECOMBINE stay spectator-only — no
hierarchy nor VAE latents on the binary MLP head.

The runtime.execute path is preserved verbatim for DR-0 logging,
running synthetic input_slice values side-by-side with the real
buffer-driven coupling step. This keeps the existing 12 dream_wrap
unit tests green while adding 8 coupling tests.

Plan G4-bis Task 4.
EOF
)"
```

---

## Task 5: Plumb β buffer through the run driver

**Files:**
- Modify: `experiments/g4_split_fmnist/run_g4.py:155-214` (function `_run_cell`)
- Test: `tests/unit/experiments/test_g4_run_g4_smoke.py` (existing 2 tests must still pass; add 1)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/experiments/test_g4_run_g4_smoke.py`:

```python
def test_smoke_pmin_arm_retention_diverges_from_baseline(
    tmp_path: Path,
) -> None:
    """G4-bis : with coupling on, P_min retention must differ from baseline.

    On the synthetic 4×4 fixture the absolute retention values are not
    meaningful, but the *coupling* must produce at least some divergence
    between baseline (no DE) and a dream-active arm — otherwise the
    replay step is silently a no-op.
    """
    data_dir = _make_synthetic_fmnist(tmp_path)
    result = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=tmp_path / "g4bis.json",
        out_md=tmp_path / "g4bis.md",
        registry_db=tmp_path / "runs.sqlite",
        epochs=2,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    by_arm: dict[str, list[float]] = {}
    for c in result["cells"]:
        by_arm.setdefault(c["arm"], []).append(c["retention"])
    # Baseline and P_min retention vectors must not be element-wise
    # identical — coupling has *some* effect (sign-agnostic).
    base = by_arm["baseline"]
    p_min = by_arm["P_min"]
    assert base != p_min, (
        "P_min retention must differ from baseline once coupling is on"
    )
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/unit/experiments/test_g4_run_g4_smoke.py::test_smoke_pmin_arm_retention_diverges_from_baseline -v --no-cov`
Expected: FAIL or TypeError — driver does not yet pass `beta_buffer` to `dream_episode`.

- [ ] **Step 3: Update _run_cell to maintain a buffer and feed it to dream_episode**

Edit `experiments/g4_split_fmnist/run_g4.py`. First, add the import near the existing dream_wrap import (line 74):

```python
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    BetaBufferFIFO,
    G4Classifier,
    build_profile,
)
```

Add a new constant after `RETENTION_EPS = 1e-6` (~line 123):

```python
# Plan G4-bis : β buffer holds raw exemplars from completed past
# tasks. Capacity 256 ≈ 32 records × 8 tasks × 1 ep cushion (5
# tasks × 1 ep here, but room for future extension). 32 samples
# pushed per task is the typical CL replay buffer fill rate.
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
```

Replace the body of `_run_cell` (currently lines 155-214). The new version maintains a buffer, fills it after each task, and forwards it into `dream_episode`:

```python
def _run_cell(
    arm: str,
    seed: int,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> _CellPartial:
    """Execute one (arm, seed) cell and return a :class:`_CellPartial`.

    Plan G4-bis : a per-cell β buffer accumulates ``BETA_BUFFER_FILL_PER_TASK``
    raw image-class pairs after each completed task. Between tasks
    (when ``arm != "baseline"``) the buffer is forwarded into
    :meth:`G4Classifier.dream_episode` so the replay handler can
    actually run gradient steps against past tasks.
    """
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4Classifier(
        in_dim=feat_dim, hidden_dim=hidden_dim, n_classes=2, seed=seed
    )
    buffer = BetaBufferFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task_to_buffer(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            buffer.push(task["x_train"][i], int(task["y_train"][i]))

    # Stage 1 — train task 0, snapshot acc, push exemplars to buffer.
    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    _push_task_to_buffer(tasks[0])

    # Stage 2 — train tasks 1..4 with optional dream-episode interleaving.
    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            clf.dream_episode(
                profile, seed=seed + k, beta_buffer=buffer
            )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task_to_buffer(tasks[k])

    # Stage 3 — measure final task-0 accuracy.
    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)

    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }
```

Add `import numpy as np` at the top of `run_g4.py` if it is not already present. (Quick check : run `grep -n "^import numpy" experiments/g4_split_fmnist/run_g4.py` — if no result, add `import numpy as np  # noqa: E402` next to the existing `from harness ... import` block.)

- [ ] **Step 4: Run smoke tests**

Run: `uv run pytest tests/unit/experiments/test_g4_run_g4_smoke.py -v --no-cov`
Expected: all 3 pass (the 2 existing + the new divergence test).

- [ ] **Step 5: Run full unit test bucket to verify no other test broke**

Run: `uv run pytest tests/unit/experiments/ -v --no-cov`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add experiments/g4_split_fmnist/run_g4.py tests/unit/experiments/test_g4_run_g4_smoke.py
git commit -m "feat(g4): plumb beta buffer through pilot driver" -m "$(cat <<'EOF'
_run_cell maintains a 256-record BetaBufferFIFO per cell, pushing
32 random exemplars from each completed task. Between tasks, when
the arm is not baseline, the buffer is forwarded into
dream_episode so the replay step can run gradient passes against
past tasks. Smoke divergence test confirms P_min retention
diverges from baseline under coupling.

Run_ids will differ from the G4-original 2026-05-03 milestone
because dream_episode semantics changed — this is correct, not a
regression. The pre-coupling run_ids remain valid as
spectator-baseline references in the original milestone.

Plan G4-bis Task 5.
EOF
)"
```

---

## Task 6: Update existing dream_wrap tests for new signature

**Files:**
- Modify: `tests/unit/experiments/test_g4_dream_wrap.py` (audit pass)

- [ ] **Step 1: Inventory existing tests that call dream_episode**

Run:
```bash
grep -n "dream_episode" tests/unit/experiments/test_g4_dream_wrap.py
```
Expected: every call site listed. The two updated in Task 4 (`test_dream_episode_executes_pmin_handlers`, `test_dream_episode_executes_pequ_with_4_ops`) plus the four new ones added in Tasks 4-5 should be the full set. Verify there are no stragglers.

- [ ] **Step 2: Run the dream_wrap suite once more**

Run: `uv run pytest tests/unit/experiments/test_g4_dream_wrap.py -v --no-cov`
Expected: 20 passed (12 from prior + 8 new : 3 from Task 2, 2 from Task 3, 3 from Task 4).

- [ ] **Step 3: Run the full repo suite with coverage gate**

Run: `uv run pytest`
Expected: all green, coverage ≥ 90 %. If coverage on the new methods (`_replay_optimizer_step`, `_downscale_step`, `BetaBufferFIFO`) is under threshold, add a small targeted hypothesis property test rather than weakening assertions.

- [ ] **Step 4: Commit if any audit changes were needed**

If Step 1 found stragglers, fix them and:

```bash
git add tests/unit/experiments/test_g4_dream_wrap.py
git commit -m "test(g4): audit dream_episode call sites" -m "$(cat <<'EOF'
Sweep test_g4_dream_wrap.py to make sure every dream_episode call
site uses the new beta_buffer kwarg. No-op if Task 4 already
caught everything.
EOF
)"
```

If no straggler found, skip the commit.

---

## Task 7: Re-run pilot G4-bis and produce dated milestone

**Files:**
- Run only — no source change. Output : `docs/milestones/g4-pilot-2026-05-03-bis.{json,md}`

- [ ] **Step 1: Verify FMNIST data dir is populated**

Run:
```bash
ls -la experiments/g4_split_fmnist/data/
```
Expected: 4 IDX gzipped files. If absent, the executing engineer should fetch FMNIST per `experiments/g4_split_fmnist/data/CLAUDE.md` (or the README in that dir) before proceeding.

- [ ] **Step 2: Run the bis pilot**

Run:
```bash
uv run python experiments/g4_split_fmnist/run_g4.py \
    --out-json docs/milestones/g4-pilot-2026-05-03-bis.json \
    --out-md docs/milestones/g4-pilot-2026-05-03-bis.md
```
Expected output : `Wrote docs/milestones/g4-pilot-2026-05-03-bis.json`, `Wrote docs/milestones/g4-pilot-2026-05-03-bis.md`, `Cells : 20`, plus three `Verdict ...` lines now showing non-zero `g_h1`, `g_h3`, and a non-trivial `monotonic_observed`. Wall time : ≤ 2 min.

- [ ] **Step 3: Eyeball the report for sanity**

Run:
```bash
head -40 docs/milestones/g4-pilot-2026-05-03-bis.md
```
Verify :
- `g_h1` is non-zero (positive or negative — sign reportable, not pre-determined).
- `g_h3` is non-zero.
- The three retention means `mean_p_min`, `mean_p_equ`, `mean_p_max` are not all identical (the pre-coupling pilot's pathology).

- [ ] **Step 4: Edit the bis md header to flag it as the bis re-run**

The driver already writes `**Date** : 2026-05-03`. Append a banner directly after the title line in `docs/milestones/g4-pilot-2026-05-03-bis.md`:

```markdown
> **Re-run** of `g4-pilot-2026-05-03.{json,md}` after Plan G4-bis
> wired `dream_episode()` to mutate classifier weights (replay
> coupling + SHY downscale). The original 2026-05-03 milestone
> is preserved as the spectator-baseline reference.
```

Insert this between the `# G4 pilot — ...` H1 line and the `**Date** :` line. Use Edit on the file.

- [ ] **Step 5: Commit the milestone**

```bash
git add docs/milestones/g4-pilot-2026-05-03-bis.json docs/milestones/g4-pilot-2026-05-03-bis.md
git commit -m "docs(g4): bis pilot re-run with replay coupling" -m "$(cat <<'EOF'
Produces docs/milestones/g4-pilot-2026-05-03-bis.{json,md} from
20 cells (4 arms × 5 seeds) with dream_episode now mutating
classifier weights via replay (n=32 records, 1 SGD step) and SHY
downscale (factor=0.95). The original 2026-05-03 milestone stays
in tree as the spectator-baseline reference.

Plan G4-bis Task 7.
EOF
)"
```

---

## Task 8: Add §7.1.2 Paper 2 narrative (EN + FR)

**Files:**
- Modify: `docs/papers/paper2/results.md` (insert §7.1.2 between current §7.1.1 ending ~line 104 and §7.2 starting ~line 106)
- Modify: `docs/papers/paper2-fr/results.md` (same insert in FR mirror)

- [ ] **Step 1: Read the EN section that must be inserted after**

Run: `sed -n '99,108p' docs/papers/paper2/results.md` to confirm the exact join point.

- [ ] **Step 2: Edit the EN file**

In `docs/papers/paper2/results.md`, find the line containing `## 7.2 Cross-substrate H1-H4 comparative table` and insert immediately above it (before the blank line preceding `## 7.2`):

```markdown
## 7.1.2 G4-bis pilot (replay-coupling re-run — 2026-05-03)

The G4 pilot re-runs after the wrapper `dream_episode()` is wired
to mutate classifier weights (Plan G4-bis). The β buffer fills
with 32 raw image-class pairs per completed task (capacity 256) ;
between tasks, dream-active arms run replay (n=32 records × 1
SGD step at lr=0.01) and SHY downscale (factor 0.95). The arm
sweep, seed sweep and pre-registered hypotheses are unchanged.
Milestone dump :
[`docs/milestones/g4-pilot-2026-05-03-bis.{json,md}`](../../milestones/g4-pilot-2026-05-03-bis.md).

Three pre-registered hypotheses (re-evaluated under coupling) :

- **H1** : observed `g_h1 = $G_H1_BIS`. Within Hu 2020 95 % CI :
  `$WITHIN_HU` ; Welch one-sided p (α/3 = 0.0167) `$WELCH_H1` →
  reject_h0 = `$REJECT_H1`.

- **H3** : observed `g_h3 = $G_H3_BIS`. Decrement-side (g ≤
  -0.13) rejection : `$REJECT_H3`.

- **H_DR4** : `mean retention[P_max] = $MEAN_PMAX`,
  `mean retention[P_equ] = $MEAN_PEQU`,
  `mean retention[P_min] = $MEAN_PMIN`. Monotonic ordering :
  `$MONOTONIC` ; Jonckheere one-sided p = `$JONCKHEERE_P` →
  reject_h0 = `$REJECT_DR4`.

Run_ids in the bis dump differ from the original 2026-05-03 dump
because `dream_episode()` semantics changed — coupling is part of
the input tuple in spirit, even though `(c_version, profile,
seed, commit_sha)` is the formal R1 key. The original 2026-05-03
dump is preserved as the spectator-baseline reference.

Per N = 5 / arm this pilot remains exploratory for absolute
g magnitudes ; pre-reg §4 still triggers a confirmatory N ≥ 30
follow-up before any STABLE promotion.
```

The `$G_H1_BIS`, `$G_H3_BIS`, etc. placeholders MUST be replaced inline with the actual numbers from `docs/milestones/g4-pilot-2026-05-03-bis.md` (Task 7). Use Edit / sed-on-paper-2 only after reading the bis md to copy the values verbatim.

- [ ] **Step 3: Edit the FR mirror**

In `docs/papers/paper2-fr/results.md`, at the equivalent join point (find `## 7.2` heading), insert the FR translation :

```markdown
## 7.1.2 Pilote G4-bis (re-exécution avec couplage de replay — 2026-05-03)

Le pilote G4 est ré-exécuté après que le wrapper
`dream_episode()` soit câblé pour modifier les poids du
classifieur (Plan G4-bis). Le tampon β se remplit de 32 paires
image-classe brutes par tâche complétée (capacité 256) ; entre
tâches, les bras dream-actifs exécutent un replay (n=32
enregistrements × 1 pas SGD à lr=0.01) et un downscale SHY
(facteur 0.95). Le balayage des bras, des graines et les
hypothèses pré-enregistrées sont inchangés. Dump du jalon :
[`docs/milestones/g4-pilot-2026-05-03-bis.{json,md}`](../../milestones/g4-pilot-2026-05-03-bis.md).

Trois hypothèses pré-enregistrées (réévaluées sous couplage) :

- **H1** : `g_h1 observé = $G_H1_BIS`. Dans l'IC à 95 % de
  Hu 2020 : `$WITHIN_HU` ; p unilatéral de Welch (α/3 =
  0.0167) `$WELCH_H1` → rejet_h0 = `$REJECT_H1`.

- **H3** : `g_h3 observé = $G_H3_BIS`. Rejet côté décrément
  (g ≤ -0.13) : `$REJECT_H3`.

- **H_DR4** : `rétention moyenne[P_max] = $MEAN_PMAX`,
  `rétention moyenne[P_equ] = $MEAN_PEQU`,
  `rétention moyenne[P_min] = $MEAN_PMIN`. Ordre monotone :
  `$MONOTONIC` ; p unilatéral de Jonckheere = `$JONCKHEERE_P`
  → rejet_h0 = `$REJECT_DR4`.

Les `run_id` du dump bis diffèrent du dump original
2026-05-03 parce que la sémantique de `dream_episode()` a
changé — le couplage fait partie du tuple d'entrée *en
esprit*, même si la clé R1 formelle reste `(c_version,
profile, seed, commit_sha)`. Le dump original 2026-05-03
est conservé comme référence du baseline-spectateur.

Avec N = 5 / bras, ce pilote reste exploratoire pour les
amplitudes absolues de g ; le pré-enregistrement §4
déclenche toujours un suivi confirmatoire N ≥ 30 avant toute
promotion STABLE.
```

Replace the `$...` placeholders with the same values as in EN. Both files must show identical numbers (the EN→FR propagation rule under `docs/CLAUDE.md`).

- [ ] **Step 4: Cross-check EN/FR symmetry**

Run:
```bash
diff <(grep -E "g_h1|g_h3|mean_p|MONOTONIC|JONCKHEERE" docs/papers/paper2/results.md) \
     <(grep -E "g_h1|g_h3|mean_p|MONOTONIC|JONCKHEERE" docs/papers/paper2-fr/results.md)
```
Expected : differences only in the surrounding French / English wording, not in the numerical values themselves. (A pre-commit hook also enforces FR mirror parity at the file-level — see `CONTRIBUTING.md`.)

- [ ] **Step 5: Commit EN + FR together**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): add g4 bis section EN+FR" -m "$(cat <<'EOF'
§7.1.2 G4-bis : replay-coupling re-run with non-zero observed g_h1
and g_h3 from docs/milestones/g4-pilot-2026-05-03-bis. The
spectator-only §7.1.1 stays in tree as the empirical floor
reference. Per CONTRIBUTING.md EN→FR propagation rule both files
ship in the same commit.

Plan G4-bis Task 8.
EOF
)"
```

---

## Task 9: CHANGELOG entry — conditional EC bump

**Files:**
- Modify: `CHANGELOG.md` (under `[Unreleased]`, append after the existing G4 bullet)

- [ ] **Step 1: Decide the EC branch**

Read the bis verdict from `docs/milestones/g4-pilot-2026-05-03-bis.md`. Apply this decision tree :

| Bis observation | EC bump |
|---|---|
| Both `g_h1 ≥ 0.21` AND `g_h3 ≤ -0.13` AND H_DR4 monotonic | candidate STABLE (still pending N≥30 follow-up — stay PARTIAL with bis-confirmed bullet) |
| Either H1 OR H3 confirmed (not both), monotonic still holds | stay PARTIAL — confirmation strengthened |
| Coupling causes monotonic violation (P_min > P_equ > P_max) | flip to UNSTABLE — finding important to report honestly |
| g still ≈ 0 across the board | stay PARTIAL — coupling implementation defect, schedule debug spike |

Per workspace rule (`hypneum-lab/CLAUDE.md` §"Versioning — DualVer"), the EC axis bump is recorded in `CHANGELOG.md` and `STATUS.md` consistently. No FC bump under any branch (axiom signatures unchanged).

- [ ] **Step 2: Edit CHANGELOG.md**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Empirical (no DualVer bump — partial confirmation)`, append a new bullet after the existing G4 paragraph (preserve existing text — append-only style for empirical history) :

```markdown
### Empirical (G4-bis re-run, 2026-05-03)
- G4-bis re-run after `dream_episode()` wired to mutate classifier
  weights via raw-image replay (n=32 × 1 SGD step) + SHY
  downscale (factor 0.95). Pre-registered hypotheses re-evaluated
  per `docs/osf-prereg-g4-pilot.md` §2 ; observed scalars in
  `docs/milestones/g4-pilot-2026-05-03-bis.md`. Pre-coupling
  run_ids preserved as spectator-baseline references in the
  original `g4-pilot-2026-05-03.{json,md}`. Coupling
  hyperparameters (replay batch n=32, lr=0.01, downscale 0.95) are
  pilot defaults — empirical pin scheduled for the N≥30
  confirmatory follow-up.
- Per framework-C §12.3, $EC_DECISION (drawn from Step 1's table
  cell) — replace this token literally with one of :
  - "partial confirmation strengthens PARTIAL" /
  - "monotonic violation triggers +UNSTABLE" /
  - "two-anchor confirmation keeps PARTIAL pending N≥30" /
  - "null retention persists ; coupling-defect spike scheduled"
```

Replace `$EC_DECISION` literally with the matching string from the four options above (no `$` token in the final commit).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): G4-bis empirical row" -m "$(cat <<'EOF'
Adds the G4-bis re-run row under [Unreleased] / Empirical.
Records replay-coupling hyperparameters (n=32, lr=0.01,
downscale 0.95) and the EC-axis decision derived from the
observed verdict per framework-C §12.3.

Plan G4-bis Task 9.
EOF
)"
```

---

## Task 10: STATUS.md G4 row update

**Files:**
- Modify: `STATUS.md:53` (the `## Gates` table, G4 row)

- [ ] **Step 1: Read the current G4 row**

Run: `grep -n "G4 — P_equ" STATUS.md`. The row is :

```markdown
| G4 — P_equ fonctionnel | S12 | 🔶 PARTIAL (Split-FMNIST pilot 2026-05-03, 20 cells, partial confirmation, N≥30 follow-up scheduled) |
```

- [ ] **Step 2: Replace the row to reflect bis state**

Use Edit. The replacement string depends on the Step 1 EC decision from Task 9. One of :

```markdown
| G4 — P_equ fonctionnel | S12 | 🔶 PARTIAL (G4-bis 2026-05-03 with replay coupling — partial confirmation, N≥30 follow-up scheduled) |
```

```markdown
| G4 — P_equ fonctionnel | S12 | 🔴 UNSTABLE (G4-bis 2026-05-03 monotonic violation — coupling reordered profile means; finding under investigation) |
```

```markdown
| G4 — P_equ fonctionnel | S12 | 🔶 PARTIAL (G4-bis 2026-05-03 — both H1 and H3 anchors confirmed under coupling, awaiting N≥30 confirmation) |
```

```markdown
| G4 — P_equ fonctionnel | S12 | 🔶 PARTIAL (G4-bis 2026-05-03 — null retention persists despite coupling, debug spike scheduled) |
```

Pick the row matching the EC branch from Task 9 Step 1. Replace exactly one row, no other STATUS.md content.

- [ ] **Step 3: Commit**

```bash
git add STATUS.md
git commit -m "docs(status): G4 row reflects bis verdict" -m "$(cat <<'EOF'
Updates the Gates table G4 row to reference the
docs/milestones/g4-pilot-2026-05-03-bis.{json,md} verdict.
EC-axis decision tree per Plan G4-bis Task 9.

Plan G4-bis Task 10.
EOF
)"
```

---

## Task 11: R1 reproducibility verification

**Files:**
- Run only — no source change.

- [ ] **Step 1: Run the reproducibility suite locally**

Run: `uv run pytest tests/reproducibility/ -v --no-cov`
Expected: all green. If running on a non-Apple-Silicon platform, the suite either auto-skips MLX-Metal-bound checks or surfaces them only on the macOS-14 nightly runner — that is fine, the engineering content of this plan stays on the numpy-only / mx-deterministic path.

- [ ] **Step 2: Verify run_id determinism by re-running the bis pilot**

Run:
```bash
mkdir -p /tmp/g4bis_repro
uv run python experiments/g4_split_fmnist/run_g4.py \
    --out-json /tmp/g4bis_repro/run1.json \
    --out-md /tmp/g4bis_repro/run1.md \
    --registry-db /tmp/g4bis_repro/run1.sqlite

uv run python experiments/g4_split_fmnist/run_g4.py \
    --out-json /tmp/g4bis_repro/run2.json \
    --out-md /tmp/g4bis_repro/run2.md \
    --registry-db /tmp/g4bis_repro/run2.sqlite

uv run python -c "
import json
a = json.loads(open('/tmp/g4bis_repro/run1.json').read())
b = json.loads(open('/tmp/g4bis_repro/run2.json').read())
ids_a = {(c['arm'], c['seed']): c['run_id'] for c in a['cells']}
ids_b = {(c['arm'], c['seed']): c['run_id'] for c in b['cells']}
print('run_ids identical:', ids_a == ids_b)
ret_a = {(c['arm'], c['seed']): c['retention'] for c in a['cells']}
ret_b = {(c['arm'], c['seed']): c['retention'] for c in b['cells']}
print('retention bit-identical:', ret_a == ret_b)
"
```
Expected : `run_ids identical: True`, `retention bit-identical: True`.

- [ ] **Step 3: If bit-identical fails, diagnose**

If retention diverges between runs, the most likely cause is unseeded RNG inside the new methods. Audit :
- `BetaBufferFIFO.sample` uses `np.random.default_rng(seed)` — OK.
- `_run_cell._push_task_to_buffer` uses `np.random.default_rng(seed + 5_000)` — OK.
- `dream_episode` passes `seed=seed + k` to `beta_buffer.sample(seed=seed + k)` and to `sample_beta_records(seed=seed + k, ...)`.

Confirm no `random.random()` / `np.random.random()` (module-level) call appears in any new code path. Fix and re-run Step 2.

- [ ] **Step 4: No commit needed unless a fix was applied**

If Step 3 surfaced a fix, commit it as :

```bash
git add <fixed_file>
git commit -m "fix(g4): seed propagation in bis coupling" -m "$(cat <<'EOF'
Plug an RNG leak found by the R1 bit-identical re-run check.
EOF
)"
```

---

## Task 12: Self-review

**Files:** None — this task is the engineer's checklist after Tasks 0-11.

- [ ] **Step 1: Spec coverage — every plan goal item touched ?**

Walk the plan's Goal / Architecture statements and confirm :

- [x] β buffer added with FIFO + seeded sampler — Task 1.
- [x] Replay step actually mutates weights (CE loss × MLX SGD) — Task 2.
- [x] SHY downscale step (0.95) — Task 3.
- [x] dream_episode wires both into the dispatched op set — Task 4.
- [x] Driver fills buffer + threads it through — Task 5.
- [x] Existing tests still green ; new tests added — Task 6.
- [x] Bis milestone produced + dated — Task 7.
- [x] Paper 2 §7.1.2 EN+FR — Task 8.
- [x] CHANGELOG empirical row — Task 9.
- [x] STATUS gate row — Task 10.
- [x] R1 bit-identical verified — Task 11.

- [ ] **Step 2: Placeholder scan**

Search the diff for any of `TBD`, `TODO`, `implement later`, `fill in details`, "Add appropriate ...", "similar to Task". Run :
```bash
git diff main..HEAD -- '*.py' '*.md' | grep -E "TBD|TODO|XXX|FIXME|Add appropriate|implement later"
```
Expected : empty output, with the *single intentional* exception : the `$G_H1_BIS` / `$EC_DECISION` style tokens MUST have been substituted in Tasks 8 + 9. If the grep surfaces those tokens, that's a Task-8 / Task-9 incompletion — go fix.

- [ ] **Step 3: Type-consistency check**

Run :
```bash
uv run mypy experiments/g4_split_fmnist tests/unit/experiments
```
Expected : no errors. Common pitfall : `BetaBufferFIFO.push(x: np.ndarray, y: int)` ↔ `_replay_optimizer_step(records: list[BetaRecord])`. Mypy will catch any drift.

- [ ] **Step 4: Lint + format**

Run :
```bash
uv run ruff check experiments/g4_split_fmnist tests/unit/experiments \
    docs/papers/paper2 docs/papers/paper2-fr CHANGELOG.md STATUS.md
```
Expected : no errors. Auto-fix what is auto-fixable (`uv run ruff check --fix ...`) and commit any cleanups under `style(g4): lint cleanup` if they exist.

- [ ] **Step 5: Final full-suite smoke**

Run :
```bash
uv run pytest
```
Expected : all green, coverage ≥ 90 %.

- [ ] **Step 6: Verification before completion**

Cross-check the deliverables list one more time before declaring G4-bis done :

| Deliverable | Path | Status |
|---|---|---|
| `BetaBufferFIFO` + tests | `experiments/g4_split_fmnist/dream_wrap.py`, `tests/unit/experiments/test_g4_buffer.py` | required |
| Coupling methods | `experiments/g4_split_fmnist/dream_wrap.py` (`_replay_optimizer_step`, `_downscale_step`) | required |
| Driver buffer-plumbing | `experiments/g4_split_fmnist/run_g4.py:_run_cell` | required |
| Bis milestone | `docs/milestones/g4-pilot-2026-05-03-bis.{json,md}` | required |
| Paper 2 §7.1.2 EN+FR | `docs/papers/paper2/results.md`, `docs/papers/paper2-fr/results.md` | required |
| Changelog row | `CHANGELOG.md` `[Unreleased]` | required |
| STATUS gate row | `STATUS.md` G4 row | required |

When every box is ticked AND the full pytest run is green AND the bis milestone has non-zero `g_h1`, `g_h3`, the plan is complete. Hand off the bis milestone path to the caller.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-05-03-g4-bis-replay-coupling.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach ?**
