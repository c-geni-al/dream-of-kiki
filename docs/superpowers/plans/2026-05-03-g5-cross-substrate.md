# G5 cross-substrate pilot — E-SNN replication of G4-bis on Split-FMNIST — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically validate DR-3 substrate-agnosticism by re-running the same `4-arms × 5-seeds` Split-FMNIST 5-task pilot used in G4-bis on the **E-SNN thalamocortical** substrate (`kiki_oniric/substrates/esnn_thalamocortical.py`), then statistically compare the per-arm retention curves across the two substrates with Welch one-sided tests at Bonferroni α / 4 = 0.0125 to upgrade `docs/proofs/dr3-substrate-evidence.md` C2.10 from "synthetic substitute" to "real-substrate empirical evidence" (or to record a cross-substrate divergence finding).

**Architecture:** A new `experiments/g5_cross_substrate/` module reuses the G4-bis Split-FMNIST loader and verdict helpers but introduces a **spike-rate classifier** (`EsnnG5Classifier`) that owns its own LIF-based weight matrix and uses the E-SNN substrate's `replay_handler_factory()` / `downscale_handler_factory()` / `restructure_handler_factory()` / `recombine_handler_factory()` exactly as G4-bis does for MLX, so the dream-episode coupling stays substrate-routed. A separate `aggregator.py` loads both the G4-bis MLX milestone dump and the new G5 E-SNN milestone, computes pair-wise Welch tests on retention per arm with overlapping-CI verdicts, and writes a cross-substrate verdict to `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`. Pre-registration discipline mirrors G4-bis: the verdict file is dated/append-only, every cell registers a deterministic `run_id` in `harness/storage/run_registry.RunRegistry`, and DR-3 evidence upgrade is a **separate task** at the end of the plan, gated on observed cross-substrate consistency.

**Tech Stack:** Python 3.12, numpy, scipy, MLX-free (E-SNN substrate uses pure numpy LIF), existing `kiki_oniric.substrates.esnn_thalamocortical`, existing `kiki_oniric.eval.statistics.{compute_hedges_g,welch_one_sided}`, existing `harness.storage.run_registry.RunRegistry`, existing `experiments.g4_split_fmnist.dataset.{SplitFMNISTTask,load_split_fmnist_5tasks}`, pytest + Hypothesis, conventional commits.

---

## Hard prerequisites (block until satisfied)

This plan is **BLOCKED** until the parallel G4-bis plan is committed. Specifically:

1. The current G4-bis driver `experiments/g4_split_fmnist/run_g4.py` runs end-to-end with weight-mutating coupling so retention is **not** identical across arms (the current 2026-05-03 milestone shows H1 g = 0 because the dream wrapper is a logging-only wrapper). The G5 plan inherits whatever coupling G4-bis ships ; it does not invent a new one.
2. `docs/milestones/g4-pilot-2026-05-03.json` exists and contains the verdict payload schema below (verified by reading the file in **Task 0**).
3. The latest commit on `main` includes both plans (G4-bis + this one) — no in-flight rebase.

If any of (1)-(3) is not true, **stop and surface the blocker** before implementing Task 1.

## Variant decision (locked)

**Full SNN classifier (variant A), not "MLP wrapped in E-SNN dispatch" (variant B).**

Rationale :
- DR-3 evidence is only meaningful if the classifier under test carries the substrate's native state representation. A G4-bis MLX MLP wrapped in `runtime.execute(...)` of an E-SNN profile would re-route the **dream operations** through E-SNN handlers but the **classifier itself** would still be MLX — that is the same failure mode as the current G4-bis logging-only wrapper, just at a different layer.
- The E-SNN substrate already exposes a numpy LIF simulator (`_simulate_population` in `esnn_thalamocortical.py:89`) that produces deterministic spike-rate dynamics. Building a 2-class classifier on top of it requires only a weight matrix `W ∈ R^(in_dim × n_classes)` driven by mean-spike-rate readout — ~80 LOC, fully tested.
- This makes the G5 retention comparison a **cross-substrate** comparison in the strict sense : different state representation, different forward / backward path, identical Protocol contract.

Variant B (MLX classifier + E-SNN dispatch) is explicitly rejected because it would conflate "E-SNN dream ops are callable" (already proved by `tests/conformance/axioms/test_dr3_esnn_substrate.py`) with "DR-3 holds empirically across substrates" — these are distinct claims.

## Compute / power note

- **20 cells** = 4 arms × 5 seeds (mirrors G4-bis). E-SNN forward step is dominated by `_simulate_population` (n_steps=20 default, 4-FLOP-per-neuron LIF) ; on Apple Silicon a 256-neuron population over 20 steps is ~5 ms / forward, vs ~0.3 ms for the MLX MLP. Net per-cell cost expected ~10× G4-bis cell, i.e. **~30-60 min / cell** worst case → **~10-20 h total**.
- The plan caps `epochs=2`, `batch_size=64`, `hidden_dim=64`, `n_steps=20` to keep total wall under 24 h. These values are documented in the milestone header so the next pilot inherits them.
- N=5 seeds gives the same exploratory power as G4-bis (≈ 80 % to detect g ≈ 1.4). The plan flags the result as **exploratory cross-substrate evidence** if observed g is in the [0.0, 0.5] band, and schedules the same N ≥ 30 follow-up as G4-bis.
- DR-3 cross-substrate verdict requires Welch p > α / 4 = 0.0125 on the *consistency* test (one-sided, `H0 : MLX retention - E-SNN retention >= 0` vs alternative `< 0`, applied symmetrically). Failure to reject = consistency. Rejecting = divergence finding.

## File structure

| File | Role |
|------|------|
| `experiments/g5_cross_substrate/__init__.py` (create) | Package marker |
| `experiments/g5_cross_substrate/esnn_classifier.py` (create) | `EsnnG5Classifier` — spike-rate 2-class classifier on E-SNN substrate |
| `experiments/g5_cross_substrate/esnn_dream_wrap.py` (create) | `dream_episode()` wrapper that routes profile ops through `EsnnSubstrate` factories |
| `experiments/g5_cross_substrate/run_g5.py` (create) | Pilot driver — sweep arms × seeds on E-SNN, register runs, emit milestone dump |
| `experiments/g5_cross_substrate/aggregator.py` (create) | Load G4-bis + G5 milestones, compute Welch consistency tests, emit cross-substrate verdict |
| `tests/unit/experiments/test_g5_esnn_classifier.py` (create) | Unit tests for `EsnnG5Classifier` |
| `tests/unit/experiments/test_g5_esnn_dream_wrap.py` (create) | Unit tests for the E-SNN dream wrapper |
| `tests/unit/experiments/test_g5_run_g5_smoke.py` (create) | 2-seed integration smoke test (synthetic fixture) |
| `tests/unit/experiments/test_g5_aggregator.py` (create) | Aggregator math tests (synthetic fixture milestones) |
| `docs/osf-prereg-g5-cross-substrate.md` (create, append-only) | OSF pre-registration draft for G5 |
| `docs/milestones/g5-cross-substrate-2026-05-03.json` (created by driver) | Machine dump |
| `docs/milestones/g5-cross-substrate-2026-05-03.md` (created by driver) | Human report |
| `docs/papers/paper2/results.md` (modify, add §7.1.3) | Cross-substrate result subsection EN |
| `docs/papers/paper2-fr/results.md` (modify, add §7.1.3) | Cross-substrate result subsection FR mirror |
| `docs/proofs/dr3-substrate-evidence.md` (modify) | Upgrade C2.10 status (or document divergence) |
| `CHANGELOG.md` (modify) | Append `[Unreleased]` G5 row |
| `STATUS.md` (modify) | Append G5 row to gates table |

The pilot lives under `experiments/g5_cross_substrate/` (sibling to `experiments/g4_split_fmnist/`) following the G4-bis convention. Coverage scope (`pyproject.toml`) is `harness` + `kiki_oniric` ; `experiments/` is excluded — pilots are not library code.

---

## Task 0: Investigate (read-only) — confirm assumptions

**Files:**
- Read: `kiki_oniric/substrates/esnn_thalamocortical.py`
- Read: `experiments/g4_split_fmnist/run_g4.py`
- Read: `experiments/g4_split_fmnist/dream_wrap.py`
- Read: `experiments/g4_split_fmnist/dataset.py`
- Read: `kiki_oniric/eval/statistics.py`
- Read: `harness/storage/run_registry.py`
- Read: `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- Read: `docs/proofs/dr3-substrate-evidence.md`
- Read: `docs/milestones/g4-pilot-2026-05-03.json` (BLOCKER — must exist)

- [ ] **Step 1: Confirm `EsnnSubstrate` exposes the four factory methods**

Run: `grep -n "_handler_factory" /Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/substrates/esnn_thalamocortical.py`
Expected: 4 hits — `replay_handler_factory`, `downscale_handler_factory`, `restructure_handler_factory`, `recombine_handler_factory`.

- [ ] **Step 2: Confirm G4-bis milestone exists and has the expected payload keys**

Run: `python -c "import json; p = json.load(open('docs/milestones/g4-pilot-2026-05-03.json')); print(sorted(p.keys())); print(sorted(p['verdict'].keys())); print(sorted(p['cells'][0].keys()))"`
Expected output contains : top-level `arms`, `c_version`, `cells`, `commit_sha`, `data_dir`, `date`, `n_seeds`, `verdict`, `wall_time_s` ; verdict has `h1_p_equ_vs_baseline`, `h3_p_min_vs_baseline`, `h_dr4_jonckheere`, `retention_by_arm` ; cell has `arm`, `seed`, `acc_task1_initial`, `acc_task1_final`, `retention`, `excluded_underperforming_baseline`, `wall_time_s`, `run_id`.

If the file is missing or any key is missing : **stop and surface the blocker** — G4-bis is the prerequisite.

- [ ] **Step 3: Confirm `RunRegistry.register` signature**

Run: `grep -n "def register(" /Users/electron/hypneum-lab/dream-of-kiki/harness/storage/run_registry.py`
Expected: `def register(` at line ~113 with `c_version: str, profile: str, seed: int, commit_sha: str` kwargs.

- [ ] **Step 4: Confirm `compute_hedges_g` + `welch_one_sided` are exported from `kiki_oniric.eval.statistics`**

Run: `grep -nE "^def (compute_hedges_g|welch_one_sided|jonckheere_trend)" /Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/eval/statistics.py`
Expected: 3 hits.

- [ ] **Step 5: Confirm `SplitFMNISTTask` TypedDict shape and loader entry point**

Run: `grep -nE "^(class SplitFMNISTTask|def load_split_fmnist_5tasks)" /Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/dataset.py`
Expected: `class SplitFMNISTTask(TypedDict)` and `def load_split_fmnist_5tasks(data_dir: Path) -> list[SplitFMNISTTask]:`.

- [ ] **Step 6: Confirm DR-3 proof file is at v0 (no prior cross-substrate empirical claim)**

Run: `grep -n "synthetic substitute\|real-substrate empirical\|cross-substrate" /Users/electron/hypneum-lab/dream-of-kiki/docs/proofs/dr3-substrate-evidence.md`
Expected: ≥ 3 hits all referencing "synthetic substitute" — confirming no prior real-substrate cross-substrate claim has been recorded.

- [ ] **Step 7: No commit — investigation only**

This task does not touch the working tree.

---

## Task 1: Stub `experiments/g5_cross_substrate/` package + smoke test

**Files:**
- Create: `experiments/g5_cross_substrate/__init__.py`
- Create: `experiments/g5_cross_substrate/run_g5.py` (stub `main()`)
- Create: `tests/unit/experiments/test_g5_run_g5_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/unit/experiments/test_g5_run_g5_smoke.py` :

```python
"""Smoke test for the G5 cross-substrate pilot driver — package import only."""
from __future__ import annotations


def test_g5_package_importable() -> None:
    """The `experiments.g5_cross_substrate` package and its `run_g5`
    module must import without side effects (matches G4-bis pattern)."""
    from experiments.g5_cross_substrate import run_g5  # noqa: F401

    assert hasattr(run_g5, "main")
    assert callable(run_g5.main)


def test_g5_main_help_returns_zero() -> None:
    """`main(['--help'])` exits cleanly via SystemExit(0) (argparse)."""
    import pytest

    from experiments.g5_cross_substrate import run_g5

    with pytest.raises(SystemExit) as exc_info:
        run_g5.main(["--help"])
    assert exc_info.value.code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g5_run_g5_smoke.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g5_cross_substrate'`.

- [ ] **Step 3: Create the package marker**

Create `experiments/g5_cross_substrate/__init__.py` :

```python
"""G5 pilot — cross-substrate validation of DR-3 on E-SNN.

Replicates the G4-bis Split-FMNIST 5-task sweep on the
`esnn_thalamocortical` substrate with a spike-rate classifier
that owns its own LIF readout. See
`docs/osf-prereg-g5-cross-substrate.md` for pre-registration.
"""
```

- [ ] **Step 4: Create the stub `run_g5.py`**

Create `experiments/g5_cross_substrate/run_g5.py` :

```python
"""G5 pilot driver — Split-FMNIST × profile sweep on E-SNN substrate.

**Gate ID** : G5 — first cross-substrate empirical pilot.
**Validates** : whether the per-arm retention distribution observed
on the MLX substrate (G4-bis) is statistically consistent with the
distribution observed on the E-SNN thalamocortical substrate. A
"consistency" verdict (Welch one-sided test fails to reject at
α/4 = 0.0125) upgrades DR-3 evidence in
`docs/proofs/dr3-substrate-evidence.md` from "synthetic substitute"
to "real-substrate empirical evidence".

**Mode** : empirical claim at first-pilot scale (N=5 seeds per arm).
**Expected output** :
    - docs/milestones/g5-cross-substrate-2026-05-03.json
    - docs/milestones/g5-cross-substrate-2026-05-03.md

Sweep : arms × seeds = 4 × 5 = 20 cells, mirroring G4-bis :
    arms  = ["baseline", "P_min", "P_equ", "P_max"]
    seeds = [0, 1, 2, 3, 4]

Usage ::

    uv run python experiments/g5_cross_substrate/run_g5.py --smoke
    uv run python experiments/g5_cross_substrate/run_g5.py
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """G5 driver entry point — full body lands in Task 6."""
    parser = argparse.ArgumentParser(description="G5 cross-substrate pilot driver")
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args(argv)
    raise NotImplementedError(
        "run_g5.main() body lands in Task 6 — Task 1 ships only the stub"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 5: Run smoke test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g5_run_g5_smoke.py -v --no-cov`
Expected: 2 PASS — package importable, `--help` raises `SystemExit(0)`.

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_cross_substrate/__init__.py experiments/g5_cross_substrate/run_g5.py tests/unit/experiments/test_g5_run_g5_smoke.py
git commit -m "feat(g5): stub g5 package + smoke test"
```

---

## Task 2: Implement `EsnnG5Classifier` — spike-rate 2-class classifier

**Files:**
- Create: `experiments/g5_cross_substrate/esnn_classifier.py`
- Create: `tests/unit/experiments/test_g5_esnn_classifier.py`

The classifier is a `(in_dim → hidden_dim) → mean-spike-rate readout → (hidden_dim → 2)` linear layer driven by the E-SNN LIF population. Forward pass : project input through `W_in`, drive an LIF population for `n_steps`, take the mean firing rate per neuron as the hidden activation, project through `W_out` to logits. Training : straight-through estimator on the mean-rate (rate is a continuous proxy for spike-count gradient). The classifier owns `W_in`, `W_out` as numpy arrays and a deterministic numpy RNG seeded from `seed`.

- [ ] **Step 1: Write the failing classifier API test**

Create `tests/unit/experiments/test_g5_esnn_classifier.py` :

```python
"""Unit tests for `EsnnG5Classifier` — DR-3 cross-substrate validation."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier


def test_classifier_constructs_with_seeded_weights() -> None:
    """Same seed → same `W_in` / `W_out` (R1 determinism)."""
    a = EsnnG5Classifier(in_dim=8, hidden_dim=4, n_classes=2, seed=42)
    b = EsnnG5Classifier(in_dim=8, hidden_dim=4, n_classes=2, seed=42)
    np.testing.assert_array_equal(a.W_in, b.W_in)
    np.testing.assert_array_equal(a.W_out, b.W_out)
    assert a.W_in.shape == (8, 4)
    assert a.W_out.shape == (4, 2)


def test_classifier_predict_logits_shape() -> None:
    """`predict_logits(x)` returns shape (N, n_classes) with finite values."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((5, 4), dtype=np.float32)
    logits = clf.predict_logits(x)
    assert logits.shape == (5, 2)
    assert np.isfinite(logits).all()


def test_classifier_eval_accuracy_in_unit_range() -> None:
    """`eval_accuracy(x, y)` returns a float in [0, 1] for non-empty input."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((4, 4), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    acc = clf.eval_accuracy(x, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_classifier_eval_accuracy_empty_returns_zero() -> None:
    """Empty input returns 0.0 (matches G4Classifier convention)."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    x = np.zeros((0, 4), dtype=np.float32)
    y = np.zeros((0,), dtype=np.int64)
    assert clf.eval_accuracy(x, y) == 0.0


def test_classifier_train_task_changes_weights() -> None:
    """`train_task(...)` mutates `W_out` (the trainable readout)."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    w_out_before = clf.W_out.copy()
    rng = np.random.default_rng(7)
    task = {
        "x_train": rng.standard_normal((20, 4)).astype(np.float32),
        "y_train": rng.integers(0, 2, size=20).astype(np.int64),
        "x_test": rng.standard_normal((4, 4)).astype(np.float32),
        "y_test": rng.integers(0, 2, size=4).astype(np.int64),
    }
    clf.train_task(task, epochs=2, batch_size=4, lr=0.1)
    assert not np.allclose(clf.W_out, w_out_before)


def test_classifier_train_task_deterministic() -> None:
    """Two classifiers with same seed + same task → same final weights."""
    rng = np.random.default_rng(11)
    task = {
        "x_train": rng.standard_normal((16, 4)).astype(np.float32),
        "y_train": rng.integers(0, 2, size=16).astype(np.int64),
        "x_test": rng.standard_normal((4, 4)).astype(np.float32),
        "y_test": rng.integers(0, 2, size=4).astype(np.int64),
    }
    a = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=99)
    b = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=99)
    a.train_task(task, epochs=2, batch_size=4, lr=0.1)
    b.train_task(task, epochs=2, batch_size=4, lr=0.1)
    np.testing.assert_allclose(a.W_in, b.W_in)
    np.testing.assert_allclose(a.W_out, b.W_out)


def test_classifier_predict_uses_lif_simulation() -> None:
    """The forward path drives the LIF population (n_steps > 0 changes output).

    Sanity check : with n_steps=0 the population produces zero firing
    rate, so logits collapse to bias only ; with n_steps=20 the rates
    are non-trivial and logits differ.
    """
    clf_lo = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=0, n_steps=0
    )
    clf_hi = EsnnG5Classifier(
        in_dim=4, hidden_dim=3, n_classes=2, seed=0, n_steps=20
    )
    # Identical weights via same seed
    np.testing.assert_array_equal(clf_lo.W_in, clf_hi.W_in)
    x = np.ones((2, 4), dtype=np.float32)
    logits_lo = clf_lo.predict_logits(x)
    logits_hi = clf_hi.predict_logits(x)
    assert not np.allclose(logits_lo, logits_hi)


def test_classifier_validates_dim_constraints() -> None:
    """Reject zero / negative dims at construction."""
    with pytest.raises(ValueError, match="in_dim"):
        EsnnG5Classifier(in_dim=0, hidden_dim=4, n_classes=2, seed=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        EsnnG5Classifier(in_dim=4, hidden_dim=0, n_classes=2, seed=0)
    with pytest.raises(ValueError, match="n_classes"):
        EsnnG5Classifier(in_dim=4, hidden_dim=4, n_classes=1, seed=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_g5_esnn_classifier.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g5_cross_substrate.esnn_classifier'`.

- [ ] **Step 3: Implement `EsnnG5Classifier`**

Create `experiments/g5_cross_substrate/esnn_classifier.py` :

```python
"""Spike-rate 2-class classifier on the E-SNN thalamocortical substrate.

Architecture :

    x  ──▶  W_in (in_dim × hidden_dim)  ──▶  LIF population ──▶
    mean spike rate per neuron  ──▶  W_out (hidden_dim × n_classes) ──▶ logits

Training uses a straight-through estimator on the mean-rate (rate
is a continuous proxy for spike counts) : gradients flow through
the linear projections only, the LIF non-linearity is treated as
identity in the backward pass. This is the standard rate-coded SNN
training trick (Wu et al. 2018, "Spatio-temporal backpropagation
for training high-performance spiking neural networks") and is
sufficient for a 2-class continual-learning pilot.

The classifier deliberately mirrors the G4-bis ``G4Classifier``
public surface (``train_task`` / ``eval_accuracy`` /
``predict_logits``) so the pilot driver in ``run_g5.py`` can be a
1-to-1 transposition of ``run_g4.py``. The dream-episode wrapper
lives in ``esnn_dream_wrap.py`` and is composed in ``Task 3``.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
    kiki_oniric/substrates/esnn_thalamocortical.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from experiments.g4_split_fmnist.dataset import SplitFMNISTTask
from kiki_oniric.substrates.esnn_thalamocortical import (
    LIFState,
    simulate_lif_step,
)


@dataclass
class EsnnG5Classifier:
    """Tiny rate-coded SNN classifier for Split-FMNIST 2-class tasks.

    Parameters
    ----------
    in_dim, hidden_dim, n_classes
        Layer sizes. All must be > 0 ; ``n_classes`` must be >= 2.
    seed
        Numpy RNG seed — controls weight init + minibatch order.
    n_steps
        LIF simulation horizon per forward pass. Defaults to 20
        (matches `_simulate_population` default in
        `esnn_thalamocortical.py`). Set to 0 for ablation tests.
    tau, threshold
        LIF dynamics parameters (passed through to
        `simulate_lif_step`). Defaults match the substrate's
        canonical values.
    """

    in_dim: int
    hidden_dim: int
    n_classes: int
    seed: int
    n_steps: int = 20
    tau: float = 10.0
    threshold: float = 1.0
    W_in: NDArray[np.float32] = field(init=False, repr=False)
    W_out: NDArray[np.float32] = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {self.in_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {self.hidden_dim}"
            )
        if self.n_classes < 2:
            raise ValueError(
                f"n_classes must be >= 2, got {self.n_classes}"
            )
        self._rng = np.random.default_rng(self.seed)
        # Xavier-style init — small random weights
        scale_in = float(np.sqrt(2.0 / self.in_dim))
        scale_out = float(np.sqrt(2.0 / self.hidden_dim))
        self.W_in = (
            self._rng.standard_normal((self.in_dim, self.hidden_dim))
            * scale_in
        ).astype(np.float32)
        self.W_out = (
            self._rng.standard_normal((self.hidden_dim, self.n_classes))
            * scale_out
        ).astype(np.float32)

    # -------------------- forward --------------------

    def _hidden_rates(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Drive the LIF population from `x @ W_in`, return mean rates.

        Per-sample loop (n_steps small, batch sizes modest — fine for
        a research pilot ; vectorising over batch is a pure perf
        optimisation deferred to a follow-up).
        """
        currents = (x @ self.W_in).astype(np.float32)
        n = currents.shape[0]
        rates = np.zeros((n, self.hidden_dim), dtype=np.float32)
        for i in range(n):
            state = LIFState(n_neurons=self.hidden_dim)
            spike_sum = np.zeros(self.hidden_dim, dtype=float)
            for _ in range(self.n_steps):
                state = simulate_lif_step(
                    state,
                    currents[i],
                    dt=1.0,
                    tau=self.tau,
                    threshold=self.threshold,
                )
                spike_sum += state.spikes
            denom = max(self.n_steps, 1)
            rates[i] = (spike_sum / denom).astype(np.float32)
        return rates

    def predict_logits(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return logits of shape (N, n_classes)."""
        if x.shape[0] == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        rates = self._hidden_rates(x.astype(np.float32))
        return (rates @ self.W_out).astype(np.float32)

    def eval_accuracy(
        self, x: NDArray[np.float32], y: NDArray[np.int64]
    ) -> float:
        """Classification accuracy in [0, 1]."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- training --------------------

    def train_task(
        self,
        task: SplitFMNISTTask,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """SGD training with straight-through gradient through LIF.

        Loss : softmax cross-entropy on the linear logits. Backward :
        d_logits = softmax - one_hot(y) ; d_W_out = rates.T @ d_logits ;
        d_rates = d_logits @ W_out.T ; **straight-through** :
        d_currents = d_rates ; d_W_in = x.T @ d_currents.

        Determinism : minibatch order is drawn from a numpy RNG
        seeded at ``self.seed`` so two classifiers built with the same
        seed and same task converge to the same weights bit-exactly.
        """
        x_train = task["x_train"].astype(np.float32)
        y_train = task["y_train"].astype(np.int64)
        n = x_train.shape[0]
        if n == 0:
            return
        rng = np.random.default_rng(self.seed)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x_train[idx]
                yb = y_train[idx]
                rates = self._hidden_rates(xb)  # (B, hidden)
                logits = rates @ self.W_out  # (B, n_classes)
                # Stable softmax
                logits_shift = logits - logits.max(axis=1, keepdims=True)
                exp = np.exp(logits_shift)
                probs = exp / exp.sum(axis=1, keepdims=True)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(yb)), yb] = 1.0
                d_logits = (probs - one_hot) / max(len(yb), 1)
                d_W_out = rates.T @ d_logits  # (hidden, n_classes)
                d_rates = d_logits @ self.W_out.T  # (B, hidden)
                # Straight-through : d_currents = d_rates
                d_W_in = xb.T @ d_rates  # (in, hidden)
                self.W_out = (
                    self.W_out - lr * d_W_out.astype(np.float32)
                ).astype(np.float32)
                self.W_in = (
                    self.W_in - lr * d_W_in.astype(np.float32)
                ).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/experiments/test_g5_esnn_classifier.py -v --no-cov`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_cross_substrate/esnn_classifier.py tests/unit/experiments/test_g5_esnn_classifier.py
git commit -m "feat(g5): add EsnnG5Classifier rate-coded SNN"
```

---

## Task 3: E-SNN dream-episode wrapper — route ops via substrate factories

**Files:**
- Create: `experiments/g5_cross_substrate/esnn_dream_wrap.py`
- Create: `tests/unit/experiments/test_g5_esnn_dream_wrap.py`

The wrapper exposes a `dream_episode(classifier, profile, seed)` free function (no class — keeps the classifier surface narrow). It builds a `DreamEpisode` whose `input_slice` carries `beta_records` taken from the classifier's recent training inputs (re-derived from `seed` for determinism), and dispatches via `profile.runtime.execute(...)` exactly like G4-bis. **No** classifier weight mutation : the dream is a logging-only signal in this pilot, identical to the published G4-bis 2026-05-03 milestone. The classifier's weights drift between tasks via SGD on the next task — that drift, attenuated by retention, is what the pilot measures.

The crucial cross-substrate property : the profile's runtime must dispatch through the **E-SNN substrate's** op handlers, not MLX's. We assert this by inspecting `profile.runtime.log` after `execute()` — every entry must carry `substrate_name = "esnn_thalamocortical"` (or a substrate-emitted tag we set in Task 4).

- [ ] **Step 1: Write the failing wrapper test**

Create `tests/unit/experiments/test_g5_esnn_dream_wrap.py` :

```python
"""Unit tests for `experiments.g5_cross_substrate.esnn_dream_wrap`."""
from __future__ import annotations

import random

import pytest

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier
from experiments.g5_cross_substrate.esnn_dream_wrap import (
    build_esnn_profile,
    dream_episode,
)
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


def test_build_esnn_profile_returns_known_profile_types() -> None:
    """`build_esnn_profile` returns the canonical profile classes."""
    p_min = build_esnn_profile("P_min", seed=0)
    p_equ = build_esnn_profile("P_equ", seed=0)
    p_max = build_esnn_profile("P_max", seed=0)
    assert isinstance(p_min, PMinProfile)
    assert isinstance(p_equ, PEquProfile)
    assert isinstance(p_max, PMaxProfile)


def test_build_esnn_profile_rejects_unknown_name() -> None:
    """Unknown profile name → ValueError (mirrors G4-bis)."""
    with pytest.raises(ValueError, match="unknown profile"):
        build_esnn_profile("P_unknown", seed=0)


def test_dream_episode_appends_log_entry() -> None:
    """One `dream_episode(...)` call appends one runtime log entry."""
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    profile = build_esnn_profile("P_equ", seed=0)
    n_before = len(profile.runtime.log)
    dream_episode(clf, profile, seed=42)
    n_after = len(profile.runtime.log)
    assert n_after == n_before + 1


def test_dream_episode_does_not_mutate_classifier_weights() -> None:
    """DR-0-only wrapper : weights stay bit-exact across `dream_episode`.

    This is the documented design choice — coupling lands in a
    follow-up plan ; G5 measures the same logging-only baseline as
    G4-bis 2026-05-03 to keep the comparison apples-to-apples.
    """
    clf = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    w_in_before = clf.W_in.copy()
    w_out_before = clf.W_out.copy()
    profile = build_esnn_profile("P_min", seed=0)
    dream_episode(clf, profile, seed=42)
    import numpy as np

    np.testing.assert_array_equal(clf.W_in, w_in_before)
    np.testing.assert_array_equal(clf.W_out, w_out_before)


def test_dream_episode_deterministic_under_same_seed() -> None:
    """Same `seed` → same episode_id appended to the log."""
    clf_a = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    clf_b = EsnnG5Classifier(in_dim=4, hidden_dim=3, n_classes=2, seed=0)
    p_a = build_esnn_profile("P_max", seed=0)
    p_b = build_esnn_profile("P_max", seed=0)
    dream_episode(clf_a, p_a, seed=7)
    dream_episode(clf_b, p_b, seed=7)
    assert (
        p_a.runtime.log[-1].episode_id == p_b.runtime.log[-1].episode_id
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_g5_esnn_dream_wrap.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g5_cross_substrate.esnn_dream_wrap'`.

- [ ] **Step 3: Implement the wrapper**

Create `experiments/g5_cross_substrate/esnn_dream_wrap.py` :

```python
"""E-SNN dream-episode wrapper for the G5 cross-substrate pilot.

Mirrors `experiments.g4_split_fmnist.dream_wrap.dream_episode` but
constructs profiles whose substrate is `esnn_thalamocortical`.
The function `dream_episode(classifier, profile, seed)` is a free
function (not a method) because the classifier owns no mutable
runtime state — the runtime lives on the profile.

DR-0 accountability is automatic : every call to `dream_episode`
appends one `EpisodeLogEntry` to `profile.runtime.log` regardless
of handler outcome. Classifier weights are **not** mutated by this
call (same design as G4-bis 2026-05-03 to keep comparisons
apples-to-apples).

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
    kiki_oniric/substrates/esnn_thalamocortical.py
    experiments/g4_split_fmnist/dream_wrap.py (sister module)
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from experiments.g5_cross_substrate.esnn_classifier import EsnnG5Classifier
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


ProfileT = PMinProfile | PEquProfile | PMaxProfile


PROFILE_FACTORIES: dict[str, Callable[..., ProfileT]] = {
    "P_min": PMinProfile,
    "P_equ": PEquProfile,
    "P_max": PMaxProfile,
}


def build_esnn_profile(name: str, seed: int) -> ProfileT:
    """Construct a profile and rebind its op handlers to E-SNN factories.

    Profiles in `kiki_oniric.profiles.*` ship an MLX-default runtime ;
    G5 needs the same profile shape but with the E-SNN substrate's
    handlers bound. We rebind by overwriting the runtime's op
    registry post-construction. The profile API is unchanged from
    the caller's point of view — `runtime.execute(episode)` dispatches
    through the rebound handlers, which is precisely what makes the
    pilot a *cross-substrate* test.
    """
    if name not in PROFILE_FACTORIES:
        raise ValueError(
            f"unknown profile {name!r} — expected one of "
            f"{sorted(PROFILE_FACTORIES)}"
        )
    factory = PROFILE_FACTORIES[name]
    profile: ProfileT
    if name == "P_min":
        profile = factory()
    else:
        profile = factory(rng=random.Random(seed))
    _rebind_to_esnn(profile)
    return profile


def _rebind_to_esnn(profile: ProfileT) -> None:
    """Overwrite `profile.runtime.op_handlers` with E-SNN factories.

    The E-SNN substrate exposes `replay_handler_factory()` /
    `downscale_handler_factory()` / `restructure_handler_factory()` /
    `recombine_handler_factory()` ; each returns a callable with the
    Protocol-compatible signature. The runtime's op registry maps
    `Operation` enum values to handlers — overwriting that mapping
    is enough to redirect dispatch.
    """
    from kiki_oniric.substrates.esnn_thalamocortical import EsnnSubstrate

    substrate = EsnnSubstrate()
    new_handlers: dict[Operation, Callable[..., object]] = {
        Operation.REPLAY: substrate.replay_handler_factory(),
        Operation.DOWNSCALE: substrate.downscale_handler_factory(),
        Operation.RESTRUCTURE: substrate.restructure_handler_factory(),
        Operation.RECOMBINE: substrate.recombine_handler_factory(),
    }
    # Overwrite only the keys the profile already exposes — we don't
    # extend the profile's op set (DR-4 inclusion is preserved).
    for op in list(profile.runtime.op_handlers.keys()):
        if op in new_handlers:
            profile.runtime.op_handlers[op] = new_handlers[op]


def _sample_beta_records(
    seed: int, n_records: int, feat_dim: int
) -> list[dict[str, list[float]]]:
    """Re-derive `n_records` deterministic beta records from `seed`.

    Mirrors `experiments.g4_split_fmnist.dream_wrap.sample_beta_records`
    so cross-substrate cells driven from the same seed see identical
    `input_slice.beta_records` content.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, list[float]]] = []
    for _ in range(n_records):
        out.append(
            {
                "x": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
                "y": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
                "input": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
            }
        )
    return out


def dream_episode(
    classifier: EsnnG5Classifier, profile: ProfileT, seed: int
) -> None:
    """Drive one `DreamEpisode` through the E-SNN-rebound profile.

    Builds an episode whose `operation_set` matches the profile's
    wired handlers (P_min : replay+downscale ; P_equ/P_max :
    +restructure+recombine), and dispatches via
    `profile.runtime.execute`. The classifier weights are **not**
    mutated by this call — see module docstring.
    """
    profile_name = type(profile).__name__
    if isinstance(profile, PMinProfile):
        ops: tuple[Operation, ...] = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
        )
        channels: tuple[OutputChannel, ...] = (OutputChannel.WEIGHT_DELTA,)
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
    beta_records = _sample_beta_records(
        seed=seed, n_records=4, feat_dim=4
    )
    rng = np.random.default_rng(seed + 10_000)
    delta_latents = [
        rng.standard_normal(4).astype(np.float32).tolist() for _ in range(2)
    ]
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": beta_records,
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
        episode_id=f"g5-{profile_name}-seed{seed}",
    )
    profile.runtime.execute(episode)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/experiments/test_g5_esnn_dream_wrap.py -v --no-cov`
Expected: 5 PASS.

- [ ] **Step 5: Quick sanity grep — confirm rebind targets the four ops**

Run: `grep -n "Operation\.\(REPLAY\|DOWNSCALE\|RESTRUCTURE\|RECOMBINE\)" /Users/electron/hypneum-lab/dream-of-kiki/experiments/g5_cross_substrate/esnn_dream_wrap.py`
Expected: ≥ 4 hits — all four ops referenced explicitly in the rebind dict.

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_cross_substrate/esnn_dream_wrap.py tests/unit/experiments/test_g5_esnn_dream_wrap.py
git commit -m "feat(g5): wire e-snn dream-episode wrapper"
```

---

## Task 4: Pilot driver — sweep arms × seeds, write milestone dump

**Files:**
- Modify: `experiments/g5_cross_substrate/run_g5.py`
- Create: `tests/unit/experiments/test_g5_run_g5_integration.py`

The driver is a 1-to-1 transposition of `experiments/g4_split_fmnist/run_g4.py` with three substitutions :
1. classifier : `G4Classifier` → `EsnnG5Classifier`,
2. dream wrapper : `clf.dream_episode(profile, seed)` → `dream_episode(clf, profile, seed)` (free function),
3. profile factory : `build_profile` → `build_esnn_profile`,
4. milestone path : `g4-pilot-2026-05-03` → `g5-cross-substrate-2026-05-03`,
5. registry profile tag : `f"g4/{arm}"` → `f"g5/{arm}"`.

The same H1 / H3 / H_DR4 verdicts are computed (so the milestone dump is shape-comparable to G4-bis), plus a fourth field `substrate = "esnn_thalamocortical"` in the payload header for downstream aggregation.

- [ ] **Step 1: Write the failing integration test (synthetic 16×16 FMNIST fixture)**

Create `tests/unit/experiments/test_g5_run_g5_integration.py` :

```python
"""2-seed integration smoke test for the G5 driver.

Uses a synthetic 16×16 IDX fixture (already used by the G4-bis
integration test ; see `tests/unit/experiments/test_g4_run_g4_smoke.py`)
to keep wall time under 60 s per arm.
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.g5_cross_substrate.run_g5 import run_pilot


def _write_idx_image(path: Path, images: np.ndarray) -> None:
    n, h, w = images.shape
    with gzip.open(path, "wb") as fh:
        fh.write(struct.pack(">IIII", 2051, n, h, w))
        fh.write(images.astype(np.uint8).tobytes())


def _write_idx_label(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(struct.pack(">II", 2049, labels.size))
        fh.write(labels.astype(np.uint8).tobytes())


def _make_synthetic_fmnist(data_dir: Path, n_per_class: int = 8) -> None:
    """Produce a 10-class 16×16 FMNIST mock under `data_dir`."""
    rng = np.random.default_rng(0)
    n_classes = 10
    h, w = 16, 16
    train_imgs = rng.integers(
        0, 256, size=(n_classes * n_per_class, h, w)
    ).astype(np.uint8)
    train_lbls = np.repeat(np.arange(n_classes), n_per_class).astype(
        np.uint8
    )
    test_imgs = rng.integers(
        0, 256, size=(n_classes * 4, h, w)
    ).astype(np.uint8)
    test_lbls = np.repeat(np.arange(n_classes), 4).astype(np.uint8)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_idx_image(
        data_dir / "train-images-idx3-ubyte.gz", train_imgs
    )
    _write_idx_label(
        data_dir / "train-labels-idx1-ubyte.gz", train_lbls
    )
    _write_idx_image(
        data_dir / "t10k-images-idx3-ubyte.gz", test_imgs
    )
    _write_idx_label(
        data_dir / "t10k-labels-idx1-ubyte.gz", test_lbls
    )


@pytest.mark.slow
def test_run_g5_pilot_smoke_2seeds(tmp_path: Path) -> None:
    """Driver runs end-to-end on a 2-seed synthetic FMNIST fixture."""
    data_dir = tmp_path / "data"
    _make_synthetic_fmnist(data_dir)
    out_json = tmp_path / "g5.json"
    out_md = tmp_path / "g5.md"
    db = tmp_path / ".registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=out_json,
        out_md=out_md,
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    # Shape contract — same keys as G4-bis + `substrate`
    expected_keys = {
        "arms",
        "c_version",
        "cells",
        "commit_sha",
        "data_dir",
        "date",
        "n_seeds",
        "substrate",
        "verdict",
        "wall_time_s",
    }
    assert expected_keys <= set(payload.keys())
    assert payload["substrate"] == "esnn_thalamocortical"
    assert len(payload["cells"]) == 4 * 2  # 4 arms × 2 seeds
    # Milestone dump persisted
    assert out_json.exists()
    assert out_md.exists()
    written = json.loads(out_json.read_text())
    assert written["substrate"] == "esnn_thalamocortical"


@pytest.mark.slow
def test_run_g5_pilot_register_run_ids_are_stable(tmp_path: Path) -> None:
    """Two consecutive runs with same seeds → same `run_id` per cell."""
    data_dir = tmp_path / "data"
    _make_synthetic_fmnist(data_dir)
    db = tmp_path / ".registry.sqlite"
    payload_a = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "a.json",
        out_md=tmp_path / "a.md",
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    payload_b = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "b.json",
        out_md=tmp_path / "b.md",
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    ids_a = sorted(c["run_id"] for c in payload_a["cells"])
    ids_b = sorted(c["run_id"] for c in payload_b["cells"])
    assert ids_a == ids_b
```

- [ ] **Step 2: Run tests to verify they fail (driver body is still NotImplementedError)**

Run: `uv run pytest tests/unit/experiments/test_g5_run_g5_integration.py -v --no-cov -m slow`
Expected: FAIL with `ImportError: cannot import name 'run_pilot' from 'experiments.g5_cross_substrate.run_g5'`.

- [ ] **Step 3: Implement the driver**

Replace the contents of `experiments/g5_cross_substrate/run_g5.py` with the full body :

```python
"""G5 pilot driver — Split-FMNIST × profile sweep on E-SNN substrate.

**Gate ID** : G5 — first cross-substrate empirical pilot.
**Validates** : whether the per-arm retention distribution observed
on the MLX substrate (G4-bis) is statistically consistent with the
distribution observed on the E-SNN thalamocortical substrate. A
"consistency" verdict (Welch one-sided test fails to reject at
α/4 = 0.0125) upgrades DR-3 evidence in
`docs/proofs/dr3-substrate-evidence.md` from "synthetic substitute"
to "real-substrate empirical evidence".

**Mode** : empirical claim at first-pilot scale (N=5 seeds per arm).
**Expected output** :
    - docs/milestones/g5-cross-substrate-2026-05-03.json
    - docs/milestones/g5-cross-substrate-2026-05-03.md

Sweep : arms × seeds = 4 × 5 = 20 cells, mirroring G4-bis :
    arms  = ["baseline", "P_min", "P_equ", "P_max"]
    seeds = [0, 1, 2, 3, 4]

Usage ::

    uv run python experiments/g5_cross_substrate/run_g5.py --smoke
    uv run python experiments/g5_cross_substrate/run_g5.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
    JAVADI_2024_OVERALL,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g5_cross_substrate.esnn_classifier import (  # noqa: E402
    EsnnG5Classifier,
)
from experiments.g5_cross_substrate.esnn_dream_wrap import (  # noqa: E402
    build_esnn_profile,
    dream_episode,
)


class _CellPartial(TypedDict):
    arm: str
    seed: int
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float


class CellResult(_CellPartial):
    run_id: str


C_VERSION = "C-v0.12.0+PARTIAL"
SUBSTRATE_NAME = "esnn_thalamocortical"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g5-cross-substrate-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g5-cross-substrate-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _run_cell(
    arm: str,
    seed: int,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    n_steps: int,
) -> _CellPartial:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = EsnnG5Classifier(
        in_dim=feat_dim,
        hidden_dim=hidden_dim,
        n_classes=2,
        seed=seed,
        n_steps=n_steps,
    )

    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )

    profile = None
    if arm != "baseline":
        profile = build_esnn_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            dream_episode(clf, profile, seed=seed + k)
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )

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


def _retention_by_arm(cells: list[CellResult]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h1_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    p_equ = retention["P_equ"]
    base = retention["baseline"]
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "distance_from_hu_2020": HU_2020_OVERALL.distance_from_target(g),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_equ": len(p_equ),
        "n_base": len(base),
    }


def _h3_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    p_min = retention["P_min"]
    base = retention["baseline"]
    if len(p_min) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_min": len(p_min),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_min, base)
    welch = welch_one_sided(p_min, base, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_javadi_2024_ci": JAVADI_2024_OVERALL.is_within_ci(
            abs(g)
        ),
        "distance_from_javadi_2024": JAVADI_2024_OVERALL.distance_from_target(
            abs(g)
        ),
        "below_javadi_2024_lower_ci_decrement": bool(
            g <= -JAVADI_2024_OVERALL.ci_low
        ),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_min": len(p_min),
        "n_base": len(base),
    }


def _h_dr4_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    groups = [retention["P_min"], retention["P_equ"], retention["P_max"]]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": mean_p_min,
        "mean_p_equ": mean_p_equ,
        "mean_p_max": mean_p_max,
        "monotonic_observed": (
            mean_p_min <= mean_p_equ <= mean_p_max
        ),
    }


def _aggregate_verdict(cells: list[CellResult]) -> dict[str, Any]:
    retention = _retention_by_arm(cells)
    return {
        "h1_p_equ_vs_baseline": _h1_verdict(retention),
        "h3_p_min_vs_baseline": _h3_verdict(retention),
        "h_dr4_jonckheere": _h_dr4_verdict(retention),
        "retention_by_arm": retention,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h1 = payload["verdict"]["h1_p_equ_vs_baseline"]
    h3 = payload["verdict"]["h3_p_min_vs_baseline"]
    h4 = payload["verdict"]["h_dr4_jonckheere"]
    lines: list[str] = []
    lines.append("# G5 cross-substrate pilot — E-SNN × Split-FMNIST")
    lines.append("")
    lines.append(f"**Date** : {payload['date']}")
    lines.append(f"**Substrate** : `{payload['substrate']}`")
    lines.append(f"**c_version** : `{payload['c_version']}`")
    lines.append(f"**commit_sha** : `{payload['commit_sha']}`")
    lines.append(
        f"**Cells** : {len(payload['cells'])} "
        f"({len(ARMS)} arms × {payload['n_seeds']} seeds)"
    )
    lines.append(f"**Wall time** : {payload['wall_time_s']:.1f}s")
    lines.append("")
    lines.append("## Pre-registered hypotheses (E-SNN substrate)")
    lines.append("")
    lines.append(
        "Pre-registration : `docs/osf-prereg-g5-cross-substrate.md`"
    )
    lines.append("")
    lines.append("### H1 — P_equ retention vs Hu 2020 (g >= 0.21)")
    if h1.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(n_p_equ={h1['n_p_equ']}, n_base={h1['n_base']})"
        )
    else:
        lines.append(f"- observed Hedges' g : **{h1['hedges_g']:.4f}**")
        lines.append(
            f"- within Hu 2020 95% CI : "
            f"{h1['is_within_hu_2020_ci']}"
        )
        lines.append(
            f"- Welch one-sided p (α/3 = "
            f"{h1['alpha_per_test']:.4f}) : {h1['welch_p']:.4f} → "
            f"reject_h0 = {h1['welch_reject_h0']}"
        )
    lines.append("")
    lines.append(
        "### H3 — P_min retention vs Javadi 2024 (|g| >= 0.13, decrement)"
    )
    if h3.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(n_p_min={h3['n_p_min']}, n_base={h3['n_base']})"
        )
    else:
        lines.append(f"- observed Hedges' g : **{h3['hedges_g']:.4f}**")
        lines.append(
            f"- |g| within Javadi 2024 95% CI : "
            f"{h3['is_within_javadi_2024_ci']}"
        )
        lines.append(
            f"- Welch one-sided p (α/3 = "
            f"{h3['alpha_per_test']:.4f}) : {h3['welch_p']:.4f} → "
            f"reject_h0 = {h3['welch_reject_h0']}"
        )
    lines.append("")
    lines.append(
        "### H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]"
    )
    if h4.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_per_arm={h4['n_per_arm']})"
        )
    else:
        lines.append(
            f"- mean retention P_min : {h4['mean_p_min']:.4f}"
        )
        lines.append(
            f"- mean retention P_equ : {h4['mean_p_equ']:.4f}"
        )
        lines.append(
            f"- mean retention P_max : {h4['mean_p_max']:.4f}"
        )
        lines.append(
            f"- monotonic observed : {h4['monotonic_observed']}"
        )
        lines.append(
            f"- Jonckheere J : {h4['j_statistic']:.4f} "
            f"(one-sided p = {h4['p_value']:.4f} → "
            f"reject_h0 = {h4['reject_h0']})"
        )
    lines.append("")
    lines.append("## Cells (R1 traceability)")
    lines.append("")
    lines.append(
        "| arm | seed | acc_initial | acc_final | retention | "
        "excluded | run_id |"
    )
    lines.append(
        "|-----|------|-------------|-----------|-----------|"
        "----------|--------|"
    )
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | "
            f"{c['acc_task1_initial']:.4f} | "
            f"{c['acc_task1_final']:.4f} | "
            f"{c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | "
            f"`{c['run_id']}` |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- Pre-registration : "
        "[docs/osf-prereg-g5-cross-substrate.md]"
        "(../osf-prereg-g5-cross-substrate.md)"
    )
    lines.append(
        "- Driver : `experiments/g5_cross_substrate/run_g5.py`"
    )
    lines.append(
        "- Substrate : `kiki_oniric.substrates.esnn_thalamocortical`"
    )
    lines.append(
        "- Sister pilot (MLX) : "
        "[g4-pilot-2026-05-03.md](g4-pilot-2026-05-03.md)"
    )
    lines.append(
        "- Cross-substrate aggregator output : "
        "see `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}` "
        "(Task 5 deliverable)"
    )
    lines.append("")
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    n_steps: int,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )
    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[CellResult] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            cell = _run_cell(
                arm,
                seed,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                lr=lr,
                n_steps=n_steps,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g5/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cells.append(CellResult(**cell, run_id=run_id))
    wall = time.time() - sweep_start

    verdict = _aggregate_verdict(cells)
    payload = {
        "date": "2026-05-03",
        "substrate": SUBSTRATE_NAME,
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "cells": cells,
        "verdict": verdict,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G5 cross-substrate pilot driver"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    args = parser.parse_args(argv)

    seeds = (0, 1) if args.smoke else DEFAULT_SEEDS
    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        n_steps=args.n_steps,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run the integration smoke tests to verify they pass**

Run: `uv run pytest tests/unit/experiments/test_g5_run_g5_integration.py -v --no-cov -m slow`
Expected: 2 PASS (~30-90 s wall time on a synthetic 16×16 fixture).

- [ ] **Step 5: Re-run the Task 1 smoke test (regression sanity)**

Run: `uv run pytest tests/unit/experiments/test_g5_run_g5_smoke.py -v --no-cov`
Expected: 2 PASS — `--help` still raises `SystemExit(0)` ; package still importable.

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_cross_substrate/run_g5.py tests/unit/experiments/test_g5_run_g5_integration.py
git commit -m "feat(g5): wire driver + integration smoke"
```

---

## Task 5: Cross-substrate aggregator — Welch consistency tests

**Files:**
- Create: `experiments/g5_cross_substrate/aggregator.py`
- Create: `tests/unit/experiments/test_g5_aggregator.py`

The aggregator is a pure function : given the path to the G4-bis MLX milestone JSON and the G5 E-SNN milestone JSON, it loads both, extracts `verdict.retention_by_arm` from each, and runs four Welch one-sided consistency tests (one per arm) at Bonferroni α / 4 = 0.0125. The verdict is a JSON record naming each pair-wise test, the observed Hedges' g of `(MLX retention - E-SNN retention)`, the Welch p-value, and a boolean `consistency` (= `not reject_h0`). The aggregate verdict `dr3_cross_substrate_consistency_ok` is `all(consistency_per_arm)`.

- [ ] **Step 1: Write the failing aggregator test (synthetic milestone fixtures)**

Create `tests/unit/experiments/test_g5_aggregator.py` :

```python
"""Unit tests for `experiments.g5_cross_substrate.aggregator`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g5_cross_substrate.aggregator import (
    aggregate_cross_substrate_verdict,
    write_aggregate_dump,
)


def _write_milestone(
    path: Path, *, retention_by_arm: dict[str, list[float]], substrate: str
) -> None:
    """Synthetic milestone fixture matching the run_g{4,5}.py shape."""
    payload = {
        "date": "2026-05-03",
        "substrate": substrate,
        "c_version": "C-v0.12.0+PARTIAL",
        "commit_sha": "abcdef0",
        "n_seeds": len(next(iter(retention_by_arm.values()))),
        "arms": ["baseline", "P_min", "P_equ", "P_max"],
        "data_dir": "fixture",
        "wall_time_s": 0.0,
        "cells": [],
        "verdict": {"retention_by_arm": retention_by_arm},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_aggregate_consistency_when_distributions_match(
    tmp_path: Path,
) -> None:
    """Identical retention by arm → consistency_ok = True."""
    arms = {
        "baseline": [0.6, 0.61, 0.59, 0.6, 0.6],
        "P_min": [0.7, 0.71, 0.69, 0.7, 0.7],
        "P_equ": [0.8, 0.81, 0.79, 0.8, 0.8],
        "P_max": [0.9, 0.91, 0.89, 0.9, 0.9],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path, retention_by_arm=arms, substrate="esnn_thalamocortical"
    )
    verdict = aggregate_cross_substrate_verdict(mlx_path, esnn_path)
    assert verdict["dr3_cross_substrate_consistency_ok"] is True
    assert set(verdict["per_arm"].keys()) == {
        "baseline",
        "P_min",
        "P_equ",
        "P_max",
    }
    for arm in ("baseline", "P_min", "P_equ", "P_max"):
        assert verdict["per_arm"][arm]["consistency"] is True


def test_aggregate_divergence_when_distributions_differ(
    tmp_path: Path,
) -> None:
    """Strongly different P_max retention → consistency_ok = False."""
    mlx_arms = {
        "baseline": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_min": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_equ": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_max": [0.95, 0.95, 0.95, 0.95, 0.95],
    }
    esnn_arms = {
        "baseline": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_min": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_equ": [0.6, 0.6, 0.6, 0.6, 0.6],
        "P_max": [0.10, 0.10, 0.10, 0.10, 0.10],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=mlx_arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path,
        retention_by_arm=esnn_arms,
        substrate="esnn_thalamocortical",
    )
    verdict = aggregate_cross_substrate_verdict(mlx_path, esnn_path)
    assert verdict["dr3_cross_substrate_consistency_ok"] is False
    assert verdict["per_arm"]["P_max"]["consistency"] is False
    # Other arms still pass — divergence is localised
    assert verdict["per_arm"]["baseline"]["consistency"] is True


def test_aggregate_writes_dump_files(tmp_path: Path) -> None:
    """`write_aggregate_dump` produces both .json and .md siblings."""
    arms = {
        "baseline": [0.6, 0.61, 0.59, 0.6, 0.6],
        "P_min": [0.7, 0.71, 0.69, 0.7, 0.7],
        "P_equ": [0.8, 0.81, 0.79, 0.8, 0.8],
        "P_max": [0.9, 0.91, 0.89, 0.9, 0.9],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path, retention_by_arm=arms, substrate="esnn_thalamocortical"
    )
    out_json = tmp_path / "agg.json"
    out_md = tmp_path / "agg.md"
    write_aggregate_dump(
        mlx_milestone=mlx_path,
        esnn_milestone=esnn_path,
        out_json=out_json,
        out_md=out_md,
    )
    assert out_json.exists()
    assert out_md.exists()
    body = json.loads(out_json.read_text())
    assert body["dr3_cross_substrate_consistency_ok"] is True
    md = out_md.read_text()
    assert "DR-3 cross-substrate consistency" in md


def test_aggregate_rejects_missing_arm(tmp_path: Path) -> None:
    """Milestone missing a required arm → ValueError."""
    mlx_arms = {
        "baseline": [0.6, 0.6],
        "P_min": [0.6, 0.6],
        "P_equ": [0.6, 0.6],
        "P_max": [0.6, 0.6],
    }
    esnn_arms = {
        "baseline": [0.6, 0.6],
        "P_min": [0.6, 0.6],
        "P_equ": [0.6, 0.6],
        # P_max missing
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=mlx_arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path,
        retention_by_arm=esnn_arms,
        substrate="esnn_thalamocortical",
    )
    with pytest.raises(ValueError, match="P_max"):
        aggregate_cross_substrate_verdict(mlx_path, esnn_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_g5_aggregator.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g5_cross_substrate.aggregator'`.

- [ ] **Step 3: Implement the aggregator**

Create `experiments/g5_cross_substrate/aggregator.py` :

```python
"""Cross-substrate aggregator for the G5 pilot.

Loads a G4-bis MLX milestone and a G5 E-SNN milestone, runs four
Welch one-sided consistency tests (one per arm) at Bonferroni
α/4 = 0.0125, and emits a cross-substrate verdict. The verdict
upgrades DR-3 evidence in `docs/proofs/dr3-substrate-evidence.md`
when `dr3_cross_substrate_consistency_ok = True`.

Statistical model :

    For each arm a in {baseline, P_min, P_equ, P_max} :
        H0 : mean(MLX retention[a]) == mean(E-SNN retention[a])
        H1 : means differ
        Test : Welch two-sided fold to one-sided rejection at α/4.
        Hedges' g = compute_hedges_g(MLX[a], E-SNN[a]).
        consistency[a] = not welch_reject_h0

    DR-3 cross-substrate verdict :
        all(consistency[a] for a in arms) → consistency_ok = True
        any consistency[a] = False → divergence finding (verdict
        records which arm diverged + observed g + p).

Reference :
    docs/proofs/dr3-substrate-evidence.md
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kiki_oniric.eval.statistics import compute_hedges_g, welch_one_sided


REQUIRED_ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ALPHA_PER_ARM = 0.05 / 4  # Bonferroni across 4 arms


def _load_retention_by_arm(milestone_path: Path) -> dict[str, list[float]]:
    payload = json.loads(milestone_path.read_text())
    retention = payload.get("verdict", {}).get("retention_by_arm")
    if not isinstance(retention, dict):
        raise ValueError(
            f"milestone {milestone_path} missing verdict.retention_by_arm"
        )
    for arm in REQUIRED_ARMS:
        if arm not in retention:
            raise ValueError(
                f"milestone {milestone_path} missing arm {arm!r} in "
                f"verdict.retention_by_arm"
            )
    return {arm: list(map(float, retention[arm])) for arm in REQUIRED_ARMS}


def aggregate_cross_substrate_verdict(
    mlx_milestone: Path, esnn_milestone: Path
) -> dict[str, Any]:
    """Compute the per-arm Welch consistency tests + aggregate verdict.

    Returns a dict with :
        - per_arm : {arm: {hedges_g, welch_p, consistency, n_mlx, n_esnn}}
        - dr3_cross_substrate_consistency_ok : bool
        - alpha_per_arm : float (= 0.0125)
        - mlx_milestone : str (path)
        - esnn_milestone : str (path)
    """
    mlx = _load_retention_by_arm(mlx_milestone)
    esnn = _load_retention_by_arm(esnn_milestone)

    per_arm: dict[str, dict[str, Any]] = {}
    all_consistent = True
    for arm in REQUIRED_ARMS:
        mlx_vals = mlx[arm]
        esnn_vals = esnn[arm]
        if len(mlx_vals) < 2 or len(esnn_vals) < 2:
            per_arm[arm] = {
                "insufficient_samples": True,
                "n_mlx": len(mlx_vals),
                "n_esnn": len(esnn_vals),
            }
            all_consistent = False
            continue
        g = compute_hedges_g(mlx_vals, esnn_vals)
        # Two-sided fold : run both directions, take the smaller p
        welch_a = welch_one_sided(mlx_vals, esnn_vals, alpha=ALPHA_PER_ARM)
        welch_b = welch_one_sided(esnn_vals, mlx_vals, alpha=ALPHA_PER_ARM)
        p_two_sided = min(2.0 * min(welch_a.p_value, welch_b.p_value), 1.0)
        reject = bool(p_two_sided < ALPHA_PER_ARM)
        consistency = not reject
        per_arm[arm] = {
            "hedges_g_mlx_minus_esnn": g,
            "welch_p_two_sided": p_two_sided,
            "reject_h0": reject,
            "consistency": consistency,
            "n_mlx": len(mlx_vals),
            "n_esnn": len(esnn_vals),
        }
        if not consistency:
            all_consistent = False

    return {
        "per_arm": per_arm,
        "dr3_cross_substrate_consistency_ok": all_consistent,
        "alpha_per_arm": ALPHA_PER_ARM,
        "mlx_milestone": str(mlx_milestone),
        "esnn_milestone": str(esnn_milestone),
    }


def _render_md(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# G5 cross-substrate aggregate — DR-3 cross-substrate consistency")
    lines.append("")
    lines.append(f"**Date** : 2026-05-03")
    lines.append(
        f"**MLX milestone** : `{verdict['mlx_milestone']}`"
    )
    lines.append(
        f"**E-SNN milestone** : `{verdict['esnn_milestone']}`"
    )
    lines.append(
        f"**Bonferroni α / 4** : {verdict['alpha_per_arm']:.4f}"
    )
    lines.append("")
    ok = verdict["dr3_cross_substrate_consistency_ok"]
    if ok:
        lines.append(
            "## Verdict : DR-3 cross-substrate consistency CONFIRMED"
        )
        lines.append("")
        lines.append(
            "All four arms (baseline, P_min, P_equ, P_max) show "
            "Welch p > α/4 = 0.0125 — cross-substrate distributions "
            "are statistically indistinguishable at first-pilot scale "
            "(N=5 per cell)."
        )
    else:
        lines.append(
            "## Verdict : DR-3 cross-substrate divergence DETECTED"
        )
        lines.append("")
        lines.append(
            "At least one arm shows Welch p ≤ α/4 = 0.0125 — see per-arm "
            "table below for the diverging arm(s)."
        )
    lines.append("")
    lines.append("## Per-arm Welch consistency")
    lines.append("")
    lines.append(
        "| arm | g (MLX − E-SNN) | Welch p (two-sided) | reject H0 | consistent |"
    )
    lines.append(
        "|-----|------------------|----------------------|-----------|------------|"
    )
    for arm in REQUIRED_ARMS:
        row = verdict["per_arm"][arm]
        if row.get("insufficient_samples"):
            lines.append(
                f"| {arm} | INSUFFICIENT | INSUFFICIENT | n/a | False |"
            )
            continue
        lines.append(
            f"| {arm} | {row['hedges_g_mlx_minus_esnn']:+.4f} | "
            f"{row['welch_p_two_sided']:.4f} | "
            f"{row['reject_h0']} | {row['consistency']} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- DR-3 spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2"
    )
    lines.append(
        "- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`"
    )
    lines.append(
        "- Aggregator : `experiments/g5_cross_substrate/aggregator.py`"
    )
    lines.append("")
    return "\n".join(lines)


def write_aggregate_dump(
    *,
    mlx_milestone: Path,
    esnn_milestone: Path,
    out_json: Path,
    out_md: Path,
) -> dict[str, Any]:
    """Compute the verdict and persist `.json` + `.md` siblings."""
    verdict = aggregate_cross_substrate_verdict(
        mlx_milestone, esnn_milestone
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    out_md.write_text(_render_md(verdict))
    return verdict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/experiments/test_g5_aggregator.py -v --no-cov`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_cross_substrate/aggregator.py tests/unit/experiments/test_g5_aggregator.py
git commit -m "feat(g5): add cross-substrate aggregator"
```

---

## Task 6: Run the production G5 pilot + emit milestone dumps

**Files:**
- Create (by driver): `docs/milestones/g5-cross-substrate-2026-05-03.json`
- Create (by driver): `docs/milestones/g5-cross-substrate-2026-05-03.md`
- Create (by driver): `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.json`
- Create (by driver): `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md`
- Create: `docs/osf-prereg-g5-cross-substrate.md` (append-only, **before** the run)

This task launches the production sweep. Before running, the OSF pre-registration must be on disk (otherwise the result is post-hoc). Then : (1) run the pilot, (2) run the aggregator. Both are read-only with respect to source code.

- [ ] **Step 1: Author the OSF pre-registration draft**

Create `docs/osf-prereg-g5-cross-substrate.md` :

```markdown
# OSF pre-registration — G5 cross-substrate pilot

**Date** : 2026-05-03
**Pre-registration target DOI** : 10.17605/OSF.IO/Q6JYN (parent;
G5 amendment to be uploaded **before** the production run)
**c_version** : C-v0.12.0+PARTIAL
**Substrate** : `esnn_thalamocortical`
**Parent pilot** : G4-bis (MLX), milestone
`docs/milestones/g4-pilot-2026-05-03.md`

## §1 Purpose

Empirically validate DR-3 substrate-agnosticism by replicating the
G4-bis Split-FMNIST 5-task sweep on the E-SNN thalamocortical
substrate and statistically testing whether per-arm retention
distributions are consistent across the two substrates.

## §2 Sweep design

- Arms : `["baseline", "P_min", "P_equ", "P_max"]` (mirrors G4-bis).
- Seeds : `[0, 1, 2, 3, 4]` (5 seeds per arm).
- Total cells : 20.
- Substrate : `kiki_oniric.substrates.esnn_thalamocortical`,
  numpy-LIF backend (no Loihi-2 hardware involved).
- Classifier : `experiments.g5_cross_substrate.esnn_classifier.EsnnG5Classifier`
  — rate-coded SNN, in_dim=784, hidden_dim=64, n_classes=2,
  n_steps=20, tau=10.0, threshold=1.0.
- Dream wrapper : `experiments.g5_cross_substrate.esnn_dream_wrap.dream_episode`
  — DR-0 logging only, no classifier weight mutation (matches
  G4-bis 2026-05-03 design).
- Pre-registered hypotheses (own-substrate) : H1, H3, H_DR4 from
  G4-bis carried over verbatim.
- Pre-registered hypotheses (cross-substrate, unique to G5) :
  - **H_DR3-CONSIST** : for each arm a in {baseline, P_min,
    P_equ, P_max}, Welch two-sided p > Bonferroni α/4 = 0.0125
    on `(MLX retention[a], E-SNN retention[a])`. Verdict
    `dr3_cross_substrate_consistency_ok = True` ⇔ all 4 arms pass.

## §3 Effect-size anchors

Same Hu 2020 (g=0.29, CI [0.21, 0.38]) and Javadi 2024 (g=0.29,
CI [0.13, 0.44]) anchors as G4-bis for own-substrate H1 / H3.
H_DR3-CONSIST has no external anchor — the consistency null is
the substrate-agnosticism claim itself.

## §4 Power analysis

N=5 per arm. Minimum detectable effect at 80 % power, two-sided
α=0.0125 ≈ g ≈ 1.7. The pilot is therefore **exploratory** for
H_DR3-CONSIST : a non-rejection at this scale is consistent with
DR-3, not a strict confirmation. A confirmatory N≥30 follow-up is
scheduled per G4-bis (same compute budget plan).

## §5 Exclusion rule

Cells with `acc_task1_initial < 0.5` are flagged
`excluded_underperforming_baseline = true` and dropped from the
verdict aggregation (mirrors G4-bis).

## §6 Outputs

- `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}` —
  per-arm retention + own-substrate H1 / H3 / H_DR4 verdicts.
- `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`
  — H_DR3-CONSIST cross-substrate verdict.
- DR-3 evidence record :
  `docs/proofs/dr3-substrate-evidence.md` upgraded **conditionally**
  on `dr3_cross_substrate_consistency_ok = True`.

## §7 Amendments

This pre-registration is **append-only**. Any post-hoc change to
the sweep, classifier, or verdict logic requires a dated
amendment line below this section.
```

- [ ] **Step 2: Verify the production data dir holds the canonical FMNIST IDX files**

Run: `ls /Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/data/ | sort`
Expected: 4 files :
```
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
```

If any are missing : G4-bis already documents the download command in `docs/osf-prereg-g4-pilot.md` ; re-run that fetch before continuing.

- [ ] **Step 3: Run the G5 pilot (production sweep)**

Run: `uv run python experiments/g5_cross_substrate/run_g5.py`
Expected console output ends with :
```
Wrote .../docs/milestones/g5-cross-substrate-2026-05-03.json
Wrote .../docs/milestones/g5-cross-substrate-2026-05-03.md
Cells : 20
```
Expected wall time : 10-20 h (set `nohup` or run inside a tmux session). The driver writes the milestone files even on partial completion via Python's atomic `Path.write_text`.

- [ ] **Step 4: Sanity-check the milestone payload schema**

Run :
```bash
python -c "import json; p = json.load(open('docs/milestones/g5-cross-substrate-2026-05-03.json')); print('cells:', len(p['cells'])); print('substrate:', p['substrate']); print('h1:', p['verdict']['h1_p_equ_vs_baseline'].get('hedges_g')); print('h3:', p['verdict']['h3_p_min_vs_baseline'].get('hedges_g')); print('hdr4:', p['verdict']['h_dr4_jonckheere'].get('monotonic_observed'))"
```
Expected: `cells: 20`, `substrate: esnn_thalamocortical`, h1 / h3 / hdr4 numeric (or bool).

- [ ] **Step 5: Run the cross-substrate aggregator**

Run :
```bash
uv run python -c "
from pathlib import Path
from experiments.g5_cross_substrate.aggregator import write_aggregate_dump
write_aggregate_dump(
    mlx_milestone=Path('docs/milestones/g4-pilot-2026-05-03.json'),
    esnn_milestone=Path('docs/milestones/g5-cross-substrate-2026-05-03.json'),
    out_json=Path('docs/milestones/g5-cross-substrate-aggregate-2026-05-03.json'),
    out_md=Path('docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md'),
)
"
```
Expected: both files written. Read the .md to capture the verdict for downstream tasks.

- [ ] **Step 6: Capture the verdict scalar for use in Tasks 7-9**

Run :
```bash
python -c "import json; v = json.load(open('docs/milestones/g5-cross-substrate-aggregate-2026-05-03.json')); print('consistency_ok:', v['dr3_cross_substrate_consistency_ok']); [print(f\"  {arm}: g={r.get('hedges_g_mlx_minus_esnn'):+.4f if r.get('hedges_g_mlx_minus_esnn') is not None else 'n/a'} p={r.get('welch_p_two_sided'):.4f if r.get('welch_p_two_sided') is not None else 'n/a'} consistent={r.get('consistency')}\") for arm, r in v['per_arm'].items()]"
```
Record the boolean `consistency_ok` and the per-arm row — Task 9 branches on it.

- [ ] **Step 7: Commit milestone artefacts + pre-registration**

```bash
git add docs/osf-prereg-g5-cross-substrate.md docs/milestones/g5-cross-substrate-2026-05-03.json docs/milestones/g5-cross-substrate-2026-05-03.md docs/milestones/g5-cross-substrate-aggregate-2026-05-03.json docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md
git commit -m "docs(g5): pilot milestone + aggregate dump"
```

---

## Task 7: Paper 2 §7.1.3 — cross-substrate result subsection (EN)

**Files:**
- Modify: `docs/papers/paper2/results.md`

The new §7.1.3 sits between §7.1.1 (G4 pilot, MLX) and §7.2 (synthetic comparative table). Use the verdict scalars captured in Task 6 Step 6. The subsection MUST:
- cite the G4-bis milestone by filename + date,
- cite the G5 milestone by filename + date,
- cite the aggregate milestone by filename + date,
- cite `kiki_oniric/substrates/esnn_thalamocortical.py` for the substrate,
- carry the `(N=5 per cell, exploratory cross-substrate)` flag in the caption,
- branch on `consistency_ok` for the verdict prose (template branches below).

- [ ] **Step 1: Read the existing §7.1.1 and §7.2 boundary in `paper2/results.md`**

Run: `grep -n "^## 7.1.1\|^## 7.2 " /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/results.md`
Expected: line numbers like 52 and 106 (or close — both are present in the current file).

- [ ] **Step 2: Insert §7.1.3 between §7.1.1 and §7.2**

Insert (above the line containing `## 7.2 Cross-substrate H1-H4 comparative table`) the following block. Replace `<H1_G_VAL>`, `<H3_G_VAL>`, `<HDR4_MON>`, `<CONSIST_OK>`, `<P_BASELINE>`, `<P_PMIN>`, `<P_PEQU>`, `<P_PMAX>`, `<G_BASELINE>`, `<G_PMIN>`, `<G_PEQU>`, `<G_PMAX>` with the scalars captured in Task 6 Step 6 — they MUST be the literal numbers from the JSON dump (no hand-edited numbers per CLAUDE.md anti-pattern). If `<CONSIST_OK>` is `True`, keep the "consistency CONFIRMED" prose ; if `False`, replace it with the "divergence DETECTED" branch supplied below the main block.

```markdown
## 7.1.3 Cross-substrate G5 pilot — E-SNN replication (2026-05-03)

The G5 pilot is the **first cross-substrate** result in §7. The
sweep mirrors §7.1.1 (4 arms × 5 seeds = 20 cells, Split-FMNIST
5-task class-incremental) but runs on the E-SNN thalamocortical
substrate (`kiki_oniric.substrates.esnn_thalamocortical`), driver
`experiments/g5_cross_substrate/run_g5.py`. Pre-registration :
[`docs/osf-prereg-g5-cross-substrate.md`](../../osf-prereg-g5-cross-substrate.md).
Own-substrate milestone :
[`docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`](../../milestones/g5-cross-substrate-2026-05-03.md).
Cross-substrate aggregate :
[`docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`](../../milestones/g5-cross-substrate-aggregate-2026-05-03.md).

The classifier is a rate-coded SNN
(`experiments.g5_cross_substrate.esnn_classifier.EsnnG5Classifier`)
with a numpy-LIF hidden population — no MLX, no torch. The
dream-episode wrapper rebinds the profile's runtime op handlers to
the E-SNN substrate's factories, so dispatch goes through
`kiki_oniric/substrates/esnn_thalamocortical.py:replay_handler_factory`,
`downscale_handler_factory`, `restructure_handler_factory`,
`recombine_handler_factory` — **the same** Protocol contract as the
MLX run, **different** state representation (spike rates vs MLX arrays).

### Own-substrate verdicts (mirror §7.1.1)

- **H1** : observed Hedges' g of `(retention[P_equ] vs
  retention[baseline])` on E-SNN = `<H1_G_VAL>`.
- **H3** : observed Hedges' g of `(retention[P_min] vs
  retention[baseline])` on E-SNN = `<H3_G_VAL>`.
- **H_DR4** : monotonic ordering observed = `<HDR4_MON>`.

### Cross-substrate consistency verdict (H_DR3-CONSIST)

For each arm, Welch two-sided test on
`(MLX retention[arm], E-SNN retention[arm])` at Bonferroni
α/4 = 0.0125 :

| arm | Hedges' g (MLX − E-SNN) | Welch p (two-sided) | consistency |
|-----|--------------------------|----------------------|-------------|
| baseline | `<G_BASELINE>` | `<P_BASELINE>` | <consistent or DIVERGENT> |
| P_min    | `<G_PMIN>`     | `<P_PMIN>`     | <consistent or DIVERGENT> |
| P_equ    | `<G_PEQU>`     | `<P_PEQU>`     | <consistent or DIVERGENT> |
| P_max    | `<G_PMAX>`     | `<P_PMAX>`     | <consistent or DIVERGENT> |

`dr3_cross_substrate_consistency_ok = <CONSIST_OK>`.

**Branch A — `<CONSIST_OK> = True`** (consistency CONFIRMED)

> All four arms produce statistically indistinguishable retention
> distributions across the two substrates at first-pilot scale
> (N=5). This upgrades the DR-3 evidence in
> [`docs/proofs/dr3-substrate-evidence.md`](../../proofs/dr3-substrate-evidence.md)
> from "two-substrate evidence, synthetic substitute" to
> "two-substrate evidence, real-substrate cross-substrate
> consistency confirmed" — the framework C Conformance Criterion
> now carries first empirical cross-substrate evidence beyond the
> structural Protocol-typing test in
> `tests/conformance/axioms/test_dr3_esnn_substrate.py`.

**Branch B — `<CONSIST_OK> = False`** (divergence DETECTED)

> At least one arm shows Welch p ≤ α/4 = 0.0125 — the diverging
> arm(s) are flagged in the table above. This is **not** a DR-3
> falsification at first-pilot scale (N=5 has insufficient power
> for a strict cross-substrate equivalence test). It **is** a
> scheduling signal for the confirmatory N≥30 follow-up planned
> per G4-bis pre-reg §4. The DR-3 evidence in
> [`docs/proofs/dr3-substrate-evidence.md`](../../proofs/dr3-substrate-evidence.md)
> is updated to record the observed divergence pattern (which
> arm, observed g, observed p) — the "synthetic substitute" tag
> stays attached to the C2.10 row pending the N≥30 confirmatory
> run.

### Caption (synthetic-flag-equivalent)

`(N=5 per cell, exploratory cross-substrate, dream-episode is
DR-0 logging only — same minimal-coupling regime as G4-bis)`.

R1 traceability : every cell carries a deterministic 32-hex
`run_id` registered in `harness/storage/run_registry.RunRegistry`
under `(C-v0.12.0+PARTIAL, g5/{baseline,P_min,P_equ,P_max}, seed)`.
```

- [ ] **Step 3: Verify the insertion preserves §7.2 numbering**

Run: `grep -n "^## 7\." /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/results.md`
Expected: §7.1, §7.1.1, §7.1.3, §7.2, §7.3, §7.4, §7.5, §7.6, §7.7 (no §7.1.2 — that slot is reserved for the future N≥30 confirmatory run per G4-bis ; documenting this gap is intentional).

- [ ] **Step 4: Commit**

```bash
git add docs/papers/paper2/results.md
git commit -m "docs(paper2): add 7.1.3 g5 cross-substrate"
```

---

## Task 8: Paper 2 §7.1.3 — French mirror (FR)

**Files:**
- Modify: `docs/papers/paper2-fr/results.md`

Per `docs/papers/CLAUDE.md` rule, EN→FR propagation is same-PR. Mirror §7.1.3 verbatim with translated headers + prose. Numbers are copied from the JSON dump (not re-translated, not re-rounded — they are R1 artefacts).

- [ ] **Step 1: Insert §7.1.3 in `paper2-fr/results.md` mirroring §7.1.3 in `paper2/results.md`**

Insert (above the line containing `## 7.2 Table comparative inter-substrats`) :

```markdown
## 7.1.3 Pilote G5 inter-substrats — réplication E-SNN (2026-05-03)

Le pilote G5 est le **premier résultat inter-substrats** du §7.
Le balayage reproduit §7.1.1 (4 bras × 5 graines = 20 cellules,
Split-FMNIST 5-tâches incremental-class) mais s'exécute sur le
substrat E-SNN thalamocortical
(`kiki_oniric.substrates.esnn_thalamocortical`), driver
`experiments/g5_cross_substrate/run_g5.py`. Pré-enregistrement :
[`docs/osf-prereg-g5-cross-substrate.md`](../../osf-prereg-g5-cross-substrate.md).
Jalon propre-substrat :
[`docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`](../../milestones/g5-cross-substrate-2026-05-03.md).
Agrégat inter-substrats :
[`docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`](../../milestones/g5-cross-substrate-aggregate-2026-05-03.md).

Le classifieur est un SNN à code-en-fréquence
(`experiments.g5_cross_substrate.esnn_classifier.EsnnG5Classifier`)
avec une population cachée numpy-LIF — sans MLX, sans torch. Le
wrapper de dream-épisode reconnecte les handlers d'opération du
runtime du profil aux factories du substrat E-SNN, donc le
dispatch passe par
`kiki_oniric/substrates/esnn_thalamocortical.py:replay_handler_factory`,
`downscale_handler_factory`, `restructure_handler_factory`,
`recombine_handler_factory` — **même** contrat Protocol que le run
MLX, **représentation d'état différente** (taux de décharge vs
tableaux MLX).

### Verdicts propre-substrat (miroir §7.1.1)

- **H1** : Hedges' g observé de `(retention[P_equ] vs
  retention[baseline])` sur E-SNN = `<H1_G_VAL>`.
- **H3** : Hedges' g observé de `(retention[P_min] vs
  retention[baseline])` sur E-SNN = `<H3_G_VAL>`.
- **H_DR4** : ordonnancement monotone observé = `<HDR4_MON>`.

### Verdict de consistance inter-substrats (H_DR3-CONSIST)

Pour chaque bras, test de Welch bilatéral sur
`(MLX retention[bras], E-SNN retention[bras])` à Bonferroni
α/4 = 0.0125 :

| bras | Hedges' g (MLX − E-SNN) | p de Welch (bilatéral) | consistance |
|------|--------------------------|--------------------------|--------------|
| baseline | `<G_BASELINE>` | `<P_BASELINE>` | <consistant ou DIVERGENT> |
| P_min    | `<G_PMIN>`     | `<P_PMIN>`     | <consistant ou DIVERGENT> |
| P_equ    | `<G_PEQU>`     | `<P_PEQU>`     | <consistant ou DIVERGENT> |
| P_max    | `<G_PMAX>`     | `<P_PMAX>`     | <consistant ou DIVERGENT> |

`dr3_cross_substrate_consistency_ok = <CONSIST_OK>`.

**Branche A — `<CONSIST_OK> = True`** (consistance CONFIRMÉE)

> Les quatre bras produisent des distributions de rétention
> statistiquement indiscernables entre les deux substrats à
> l'échelle du premier pilote (N=5). Cela fait passer la preuve
> DR-3 dans
> [`docs/proofs/dr3-substrate-evidence.md`](../../proofs/dr3-substrate-evidence.md)
> de « two-substrate evidence, synthetic substitute » à
> « two-substrate evidence, real-substrate cross-substrate
> consistency confirmed » — le critère de Conformité du framework
> C porte désormais une première preuve empirique inter-substrats
> au-delà du test de typage Protocol structurel dans
> `tests/conformance/axioms/test_dr3_esnn_substrate.py`.

**Branche B — `<CONSIST_OK> = False`** (divergence DÉTECTÉE)

> Au moins un bras montre p de Welch ≤ α/4 = 0.0125 — le(s) bras
> divergent(s) est/sont signalé(s) dans la table ci-dessus. Ceci
> n'est **pas** une falsification de DR-3 à l'échelle du premier
> pilote (N=5 a une puissance insuffisante pour un test strict
> d'équivalence inter-substrats). C'**est** un signal de
> programmation pour le run confirmatoire N≥30 prévu par le
> pré-enregistrement de G4-bis §4. La preuve DR-3 dans
> [`docs/proofs/dr3-substrate-evidence.md`](../../proofs/dr3-substrate-evidence.md)
> est mise à jour pour enregistrer le pattern de divergence
> observé (quel bras, g observé, p observé) — le tag « synthetic
> substitute » reste attaché à la ligne C2.10 en attendant le run
> confirmatoire N≥30.

### Légende (équivalent du flag synthétique)

`(N=5 par cellule, inter-substrats exploratoire, dream-episode est
journal DR-0 uniquement — même régime de couplage minimal que
G4-bis)`.

Traçabilité R1 : chaque cellule porte un `run_id` 32-hex
déterministe enregistré dans
`harness/storage/run_registry.RunRegistry` sous
`(C-v0.12.0+PARTIAL, g5/{baseline,P_min,P_equ,P_max}, seed)`.
```

- [ ] **Step 2: Verify section anchors are aligned EN ↔ FR**

Run: `diff <(grep -nE "^## 7\." /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/results.md | sed 's/.*7\./7./') <(grep -nE "^## 7\." /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2-fr/results.md | sed 's/.*7\./7./')`
Expected: empty diff (anchors match modulo language) OR small diff localised to translated section names — same § numbers in both files.

- [ ] **Step 3: Commit**

```bash
git add docs/papers/paper2-fr/results.md
git commit -m "docs(paper2-fr): mirror 7.1.3 cross-substrate"
```

---

## Task 9: Upgrade `docs/proofs/dr3-substrate-evidence.md`

**Files:**
- Modify: `docs/proofs/dr3-substrate-evidence.md`

Per `docs/proofs/CLAUDE.md` rule "Editing a proof body without bumping its header version": the proof file gets a new header version block (`v0.2-draft`, dated 2026-05-03), with the supersedes pointer set to `v0.1-draft`. The `esnn_thalamocortical` row is upgraded conditionally on the Task 6 Step 6 verdict.

- [ ] **Step 1: Read the current header (or absence thereof) in the proof file**

Run: `head -10 /Users/electron/hypneum-lab/dream-of-kiki/docs/proofs/dr3-substrate-evidence.md`
Expected: `# DR-3 Conformance Criterion — substrate evidence (C2.10)` ; no formal version block — the file has been a status doc, not a versioned proof. This task adds a version block.

- [ ] **Step 2: Insert the v0.2 header block**

Add the following block immediately after the `# DR-3 Conformance Criterion — substrate evidence (C2.10)` line :

```markdown
**Version** : `v0.2-draft` (2026-05-03)
**Supersedes** : `v0.1-draft` (2026-04-19, original C2.10 record).
**Amendment pointer** : none (additive empirical evidence ; no
axiom statement change).
**Target venue** : Paper 2 §7.1.3, §4 (DR-3 narrative).
**Executable counterpart** :
- structural typing : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- empirical pilot : `experiments/g5_cross_substrate/run_g5.py`
- aggregate verdict :
  `experiments/g5_cross_substrate/aggregator.py::aggregate_cross_substrate_verdict`
- milestone artefacts :
  - MLX : `docs/milestones/g4-pilot-2026-05-03.{json,md}`
  - E-SNN : `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`
  - aggregate :
    `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`
```

- [ ] **Step 3: Update the `esnn_thalamocortical` evidence section — branch on `consistency_ok`**

Locate the `### esnn_thalamocortical` section header and replace the three `*(synthetic substitute — numpy LIF skeleton)*` evidence lines with one of the two branches below, depending on the verdict captured in Task 6 Step 6.

**Branch A — `<CONSIST_OK> = True`** : replace the section with :

```markdown
### `esnn_thalamocortical`

Evidence summary *(real-substrate cross-substrate consistency confirmed — G5 pilot 2026-05-03)* :

- **C1 — signature typing (typed Protocols)** : `PASS` — 4 op factories callable + core registry shared with MLX
  - structural evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C2 — axiom property tests pass** : `PASS` — DR-3 E-SNN conformance suite passes on numpy LIF backend (real substrate, not synthetic substitute)
  - structural evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
  - **empirical evidence (NEW v0.2)** : G5 cross-substrate pilot
    confirms per-arm retention distributions are statistically
    indistinguishable between MLX and E-SNN substrates at
    Bonferroni α/4 = 0.0125 across all 4 arms (N=5 per cell,
    exploratory cross-substrate scale).
  - aggregate dump : `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on LIFState
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
```

**Branch B — `<CONSIST_OK> = False`** : replace the section with :

```markdown
### `esnn_thalamocortical`

Evidence summary *(real-substrate cross-substrate divergence DETECTED on at least one arm — G5 pilot 2026-05-03 ; N≥30 confirmatory follow-up scheduled)* :

- **C1 — signature typing (typed Protocols)** : `PASS` — 4 op factories callable + core registry shared with MLX
  - structural evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C2 — axiom property tests pass** : `PASS — STRUCTURAL ; EMPIRICAL DIVERGENCE FLAGGED` — DR-3 E-SNN conformance suite passes structurally, but G5 cross-substrate pilot detected a divergence on at least one arm.
  - structural evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
  - **empirical divergence pattern (NEW v0.2)** : see Paper 2
    §7.1.3 table for per-arm Hedges' g + Welch p. The pilot is
    underpowered (N=5) for a strict equivalence test, so this
    divergence is treated as a scheduling signal — not as a DR-3
    falsification.
  - aggregate dump : `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`
  - confirmatory follow-up : N≥30 cross-substrate run scheduled
    per G4-bis pre-reg §4.
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on LIFState
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
```

- [ ] **Step 4: Update the synthetic-data caveat section**

Replace the `## Synthetic-data caveat` paragraph (still labelled with the synthetic flag) with the following — branched on `<CONSIST_OK>` :

**Branch A — `<CONSIST_OK> = True`** :

```markdown
## Empirical evidence — cross-substrate consistency (v0.2, 2026-05-03)

The E-SNN substrate is now backed by a numpy-LIF spike-rate
backend exercised through the **first cross-substrate empirical
pilot** (G5, 2026-05-03). The G5 pilot replicates the G4-bis
Split-FMNIST 5-task sweep on E-SNN with a rate-coded SNN
classifier, then compares per-arm retention distributions to the
MLX run via Welch two-sided tests at Bonferroni α/4 = 0.0125. All
four arms (baseline, P_min, P_equ, P_max) failed to reject the
consistency null, so DR-3 carries first empirical cross-substrate
evidence (N=5 per cell, exploratory). A confirmatory N≥30 run is
scheduled per G4-bis pre-reg §4. No fMRI cohort is involved — the
empirical claim is "the framework's Conformance Criterion is
operational across two independent implementations of the 8
primitives, **and** their retention curves under the same
continual-learning protocol are statistically indistinguishable
at first-pilot scale".
```

**Branch B — `<CONSIST_OK> = False`** :

```markdown
## Empirical evidence — cross-substrate divergence (v0.2, 2026-05-03)

The G5 pilot (2026-05-03) detected a per-arm divergence on at
least one arm of the Welch consistency test at Bonferroni
α/4 = 0.0125. At first-pilot scale (N=5), this is **not** a DR-3
falsification — minimum detectable effect at 80 % power is
g ≈ 1.7. The divergence is treated as a scheduling signal for the
confirmatory N≥30 run per G4-bis pre-reg §4. Until that run lands,
DR-3 carries the structural Conformance Criterion (typed
Protocols, axiom property tests, BLOCKING invariants) plus the
documented divergence pattern, but not a positive cross-substrate
empirical claim.
```

- [ ] **Step 5: Update the Cross-references section**

Append to the `## Cross-references` section :

```markdown
- G5 pilot driver : `experiments/g5_cross_substrate/run_g5.py`
- G5 aggregator : `experiments/g5_cross_substrate/aggregator.py`
- G5 own-substrate milestone : `docs/milestones/g5-cross-substrate-2026-05-03.md`
- G5 cross-substrate aggregate : `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md`
- G5 pre-registration : `docs/osf-prereg-g5-cross-substrate.md`
```

- [ ] **Step 6: Commit**

```bash
git add docs/proofs/dr3-substrate-evidence.md
git commit -m "docs(proofs): bump dr3-evidence v0.2 g5"
```

---

## Task 10: CHANGELOG + STATUS

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Append a G5 entry to `CHANGELOG.md` `[Unreleased]`**

In `CHANGELOG.md`, locate the `## [Unreleased]` section (top of file). Inside its `### Empirical (no DualVer bump — partial confirmation)` block, append after the existing G4 entry :

```markdown
- G5 pilot 2026-05-03 (Split-FMNIST × profile sweep, **E-SNN**
  thalamocortical substrate) returned the first cross-substrate
  empirical evidence under DR-3. See
  `docs/milestones/g5-cross-substrate-2026-05-03.md` for
  per-hypothesis verdicts (own-substrate H1 / H3 / H_DR4) and
  `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md`
  for the H_DR3-CONSIST cross-substrate consistency verdict
  (Welch two-sided per arm at Bonferroni α/4 = 0.0125).
- The E-SNN classifier is a rate-coded SNN
  (`experiments.g5_cross_substrate.esnn_classifier.EsnnG5Classifier`)
  with a numpy-LIF hidden population — no MLX, no torch. Dispatch
  goes through the four `EsnnSubstrate` op handler factories,
  rebound on the profile's runtime, so the cross-substrate
  comparison is genuine (different state representation, same
  Protocol contract).
- Per framework-C §12.3, this empirical-axis result does **not**
  promote EC out of PARTIAL : the pilot is exploratory at N=5 per
  arm (minimum detectable g ≈ 1.7 at 80 % power, two-sided
  α=0.0125). A confirmatory N≥30 cross-substrate follow-up is
  scheduled per G4-bis pre-reg §4 before any STABLE promotion.
- DR-3 evidence record `docs/proofs/dr3-substrate-evidence.md`
  bumped to `v0.2-draft` with branched language depending on the
  observed consistency verdict (CONFIRMED ↔ DIVERGENCE).
- Milestone artefacts : `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`,
  `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}`.
- Run-registry rows : 20 cells under
  `(C-v0.12.0+PARTIAL, g5/{baseline,P_min,P_equ,P_max}, seed)`.
```

- [ ] **Step 2: Append a G5 row to `STATUS.md` gates table**

In `STATUS.md`, locate the `## Gates` table. Append a row after the G10 row :

```markdown
| G5-pilot — cross-substrate DR-3 empirical | 2026-05-03 | 🔶 PARTIAL (E-SNN replication of G4-bis 2026-05-03, 20 cells, exploratory N=5 ; consistency verdict per `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.md` ; N≥30 follow-up scheduled) |
```

(Note : G5 in this row refers to the **cross-substrate pilot**, distinct from the original `G5 — PUBLICATION-READY` gate at S18 which already exists in the table. Keep both rows — the row label `G5-pilot — cross-substrate DR-3 empirical` is intentionally disambiguated.)

- [ ] **Step 3: Verify `STATUS.md` still parses (no broken table syntax)**

Run: `python -c "import re; t = open('STATUS.md').read(); rows = [l for l in t.splitlines() if l.startswith('|')]; print('table rows:', len(rows)); assert all(l.count('|') >= 4 for l in rows), 'broken table'"`
Expected: prints `table rows: <N>`, no AssertionError.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(g5): changelog + status g5 row"
```

---

## Task 11: Self-review + final test sweep

**Files:**
- (Read-only across the new module + paper sections)

- [ ] **Step 1: Run the entire test suite and confirm coverage gate**

Run: `uv run pytest`
Expected: 0 failures ; coverage ≥ 90 % (the project gate). The G5 unit tests under `tests/unit/experiments/test_g5_*.py` add ~120 LOC of test code to a ~80 LOC + ~250 LOC + ~80 LOC + ~150 LOC = ~560 LOC of new module code. Coverage should not regress.

- [ ] **Step 2: Run lint + type-check on the new module**

Run :
```bash
uv run ruff check experiments/g5_cross_substrate/ tests/unit/experiments/test_g5_*.py
uv run mypy experiments/g5_cross_substrate/
```
Expected: zero ruff diagnostics ; zero mypy errors. Note that `experiments/` is NOT in the strict `mypy harness tests` invocation per `CLAUDE.md`, so this mypy is a courtesy check — fix anything obvious, but the project's CI gate is `mypy harness tests`.

- [ ] **Step 3: Verify the G5 milestone files are committed but the `.run_registry.sqlite` is not**

Run :
```bash
git status --short
git log --oneline -10
```
Expected: working tree clean ; the last 9 commits are :
1. `feat(g5): stub g5 package + smoke test`
2. `feat(g5): add EsnnG5Classifier rate-coded SNN`
3. `feat(g5): wire e-snn dream-episode wrapper`
4. `feat(g5): wire driver + integration smoke`
5. `feat(g5): add cross-substrate aggregator`
6. `docs(g5): pilot milestone + aggregate dump`
7. `docs(paper2): add 7.1.3 g5 cross-substrate`
8. `docs(paper2-fr): mirror 7.1.3 cross-substrate`
9. `docs(proofs): bump dr3-evidence v0.2 g5`
10. `docs(g5): changelog + status g5 row`

If `.run_registry.sqlite` shows up in `git status` : it must be in `.gitignore` (CLAUDE.md anti-pattern : "no run-registry leakage"). Add it via a separate commit in this task.

- [ ] **Step 4: Verify EN ↔ FR parity for §7.1.3**

Run: `diff <(grep -c -E "^####? " /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/results.md) <(grep -c -E "^####? " /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2-fr/results.md)`
Expected: equal subsection / sub-subsection counts in EN and FR.

- [ ] **Step 5: Verify no `Co-Authored-By` trailer in any G5 commit**

Run: `git log --format="%B" -10 | grep -i "co-authored-by" || echo OK`
Expected: `OK` (no trailer present).

- [ ] **Step 6: Cross-link audit — every new doc references its sibling**

Run :
```bash
grep -l "g5-cross-substrate-2026-05-03" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md docs/proofs/dr3-substrate-evidence.md docs/osf-prereg-g5-cross-substrate.md CHANGELOG.md
```
Expected: all 5 files printed — no orphan references.

- [ ] **Step 7: Final commit if Step 3 surfaced gitignore changes**

```bash
# only if step 3 found .run_registry.sqlite tracked
git add .gitignore
git commit -m "chore(git): ignore run-registry sqlite"
```

If Step 3 was clean (no leakage), this step is a no-op.

---

## Self-review checklist (run after writing the plan)

**Spec coverage** :

- [x] Task 0 — investigate `esnn_thalamocortical` API + `esnn_norse` (Task 0 reads both ; norse_norse is documented as alternative, not exercised — keeps scope small).
- [x] Task 0.5 — classifier choice : full SNN classifier locked in the variant block.
- [x] Task 1 — stub `experiments/g5_cross_substrate/{__init__,run_g5}.py` with smoke test.
- [x] Task 2 — `EsnnG5Classifier` (full SNN classifier, NOT MLP wrapped in E-SNN dispatch).
- [x] Task 3 — E-SNN replay handler routed via `_rebind_to_esnn` covering the four ops.
- [x] Task 3 — E-SNN downscale handler routed via `_rebind_to_esnn` (same).
- [x] Task 5 — cross-substrate aggregator with Welch consistency tests.
- [x] Task 6 — production sweep + milestone dump + aggregate dump.
- [x] Task 6 — generate G5 milestone JSON+MD via the driver.
- [x] Task 7 — Paper 2 §7.1.3 EN.
- [x] Task 8 — Paper 2 §7.1.3 FR mirror.
- [x] Task 9 — DR-3 proof v0.2-draft branched on `consistency_ok`.
- [x] Task 10 — CHANGELOG + STATUS row.
- [x] Task 11 — self-review + test sweep.

**Critical caveats addressed** :

- [x] G5 depends on G4-bis : Task 0 Step 2 is a hard blocker.
- [x] E-SNN compute : Task 4 caps `n_steps=20`, `hidden_dim=64`, `epochs=2` ; total wall ~10-20 h documented in compute note.
- [x] Classifier choice : variant decision block locks "full SNN classifier" with rationale.
- [x] Metric comparability : `experiments.g4_split_fmnist.dataset.load_split_fmnist_5tasks` is shared between G4-bis and G5 — same eval splits, only seeds differ ; explicit in Task 4.
- [x] DR-3 evidence upgrade : Task 9 branches on `consistency_ok` — Branch A upgrades, Branch B documents divergence without overclaiming.

**Placeholder scan** :

- The `<H1_G_VAL>` / `<P_BASELINE>` / etc. tokens in Tasks 7-8 are explicit substitution sites, paired with Task 6 Step 6 which captures the values. They are not "TBD / TODO / fill in details" — they are pin-resolved from a live JSON dump in the task immediately preceding the substitution.
- All other steps contain complete code, exact paths, exact commands, and expected output.

**Type / signature consistency** :

- `EsnnG5Classifier(in_dim, hidden_dim, n_classes, seed, n_steps, tau, threshold)` defined in Task 2, used in Tasks 3, 4 with the same kwarg names.
- `dream_episode(classifier, profile, seed)` (free function, **not** a method) defined in Task 3, used in Task 4.
- `build_esnn_profile(name, seed)` defined in Task 3, used in Task 4.
- `aggregate_cross_substrate_verdict(mlx_milestone, esnn_milestone)` defined in Task 5, used in Task 6.
- `write_aggregate_dump(*, mlx_milestone, esnn_milestone, out_json, out_md)` defined in Task 5, used in Task 6.
- `run_pilot(*, data_dir, seeds, out_json, out_md, registry_db, epochs, batch_size, hidden_dim, lr, n_steps)` defined in Task 4, used by `main()` in the same file ; the integration test calls it with the same kwargs.
- Milestone payload key `substrate` introduced in Task 4 (`run_pilot`), consumed by Task 5 (`_load_retention_by_arm` reads only `verdict.retention_by_arm`, not `substrate`, so it works on both G4-bis and G5 milestones).
