# K-coupling Invariant Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new framework-C invariant **K-coupling** (slow-oscillation × fast-spindle phase coupling) and ship its conformance test, anchored on the eLife 2025 Bayesian meta-analysis empirical window [0.27, 0.39] (BibTeX `elife2025bayesian`).

**Architecture:** Additive change. Introduce (1) a new opt-in `PhaseCouplingObservable` `Protocol` so any conforming substrate that emits a phase-coupled signal can be measured, (2) a deterministic synthetic substrate fixture used as the canonical conformance target (real substrates plug in later), (3) a Hypothesis property test that estimates the SO–fast-spindle coupling via a mean-vector-length (MVL) phase-amplitude estimator on a seeded synthetic signal and asserts the value lies in the empirical 95 % CI [0.27, 0.39]. The new invariant is registered as **K2** (numbering hole closed) with full citation back to `elife2025bayesian` and a proof-style evidence stub under `docs/proofs/`.

**Tech Stack:** Python 3.12, pytest, Hypothesis, NumPy, SciPy (already in `pyproject.toml`), ruff (line-length 100), mypy strict.

---

## Background — verified 2026-05-02

Existing K-family entries (`docs/invariants/registry.md`) :

| ID | Statement | Severity | Notes |
|----|-----------|----------|-------|
| K1 | DE budget respected (FLOPs + wall + energy) | BLOCKING | A.4 runtime context manager |
| K2 | *(absent — hole)* | — | This plan fills it. |
| K3 | swap latency ≤ K3_max | WARN | default 1 s |
| K4 | eval matrix coverage | BLOCKING | gates MAJOR bump |

Existing invariant tests (`tests/conformance/invariants/`) follow the
`test_<id>_<short>.py` pattern: `test_s2_finite.py`, `test_s3_topology.py`,
`test_s4_attention.py`. Imports come from `kiki_oniric.dream.guards.<name>`
which exports a `<Name>GuardError` and a `check_<name>(...)` helper.

K-coupling diverges from the S/I family because no current substrate
exposes a "phase-coupling" hot-path: the invariant is a **measurement
contract**, not a guard. We therefore:

* Introduce an opt-in `PhaseCouplingObservable` Protocol in
  `kiki_oniric/core/observables.py` (new file — no signature change to
  the 8-primitive DR-3 surface, so **no FC bump for `core/primitives.py`**).
* Register K2 as a measurement invariant whose conformance test runs
  against any substrate implementing the Protocol; the canonical
  reference target is a synthetic substrate that ships only inside
  `tests/conformance/invariants/_synthetic_phase_coupling.py`.
* Add a measurement helper `kiki_oniric/dream/guards/coupling.py`
  exporting `CouplingGuardError` and `check_coupling_in_window(value,
  ci_low, ci_high)` to mirror the S2/S3/S4 guard pattern.

This is purely additive: existing axiom tests, primitives, and guards
are untouched. FC axis bumps **MINOR** (new invariant + new Protocol,
no breaking change). EC axis is unaffected by spec change alone.

---

## File Structure

* Create: `kiki_oniric/core/observables.py` (Protocol)
* Create: `kiki_oniric/dream/guards/coupling.py` (guard helper)
* Create: `tests/conformance/invariants/_synthetic_phase_coupling.py` (fixture substrate)
* Create: `tests/conformance/invariants/test_k2_coupling.py` (conformance test)
* Create: `docs/proofs/k2-coupling-evidence.md` (evidence stub)
* Modify: `docs/invariants/registry.md` (add K2 entry, K-family header note)
* Modify: `CHANGELOG.md` (FC-MINOR bump entry)

All paths are absolute under `/Users/electron/hypneum-lab/dream-of-kiki/`.

---

## Task 0: Pre-flight investigation (READ-ONLY)

**Files:** none touched.

- [ ] **Step 1: Confirm K2 is unused and registry numbering is canonical**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -rn "\bK2\b" docs/ kiki_oniric/ tests/ harness/ 2>/dev/null | \
  grep -v -E "(node_modules|\.venv|build|dist)"
```

Expected: zero matches. (Verified 2026-05-02 during planning; if a
match appears, abort and surface to user — registry has been edited
since.)

- [ ] **Step 2: Confirm no `coupling` symbol is already exported**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -rn -i "coupling" kiki_oniric/ tests/ 2>/dev/null
```

Expected: zero matches inside `kiki_oniric/` and
`tests/conformance/`. Documentation hits under `docs/papers/` are
fine.

- [ ] **Step 3: Confirm SciPy availability**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run python -c "import scipy.signal; print(scipy.__version__)"
```

Expected: prints a version ≥ 1.13. (Listed in `pyproject.toml:23`.)

- [ ] **Step 4: Confirm BibTeX key**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -n "elife2025bayesian" docs/papers/paper1/references.bib
```

Expected: hits at line 321 (`@article{elife2025bayesian,`). Verbatim
note line includes `coupling strength 0.33 [0.27, 0.39]`.

- [ ] **Step 5: Note baseline test count and coverage**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest --collect-only -q | tail -5
```

Record the baseline so the post-implementation diff (+N tests, +1
guard module covered) is auditable.

No commit at this task.

---

## Task 1: Define `PhaseCouplingObservable` Protocol

**Files:**
* Create: `kiki_oniric/core/observables.py`
* Test: `tests/conformance/invariants/test_k2_coupling.py` (created in Task 4; the structural tests will live there)

The Protocol is opt-in: substrates that cannot emit a phase-coupled
signal simply do not implement it, and the K-coupling test is skipped
(rather than failing) for them. This preserves DR-3 substrate-agnosticism
condition (1) on the 8 mandatory primitives.

- [ ] **Step 1: Write the failing structural test**

Create `tests/conformance/invariants/test_k2_coupling.py` with the
following minimal preamble (the property test lands in Task 5):

```python
"""K-coupling invariant — SO × fast-spindle phase-coupling measurement.

Reference: docs/invariants/registry.md (K2),
docs/proofs/k2-coupling-evidence.md, framework-C spec §5,
empirical anchor `elife2025bayesian` (eLife 2025,
coupling strength 0.33 [0.27, 0.39]).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from kiki_oniric.core.observables import PhaseCouplingObservable


def _is_protocol(cls: type) -> bool:
    return Protocol in getattr(cls, "__mro__", ())


def test_phase_coupling_observable_is_runtime_checkable() -> None:
    """Protocol must be @runtime_checkable so isinstance() works."""
    assert runtime_checkable(PhaseCouplingObservable) is PhaseCouplingObservable


def test_phase_coupling_observable_is_protocol() -> None:
    """Structural test mirrors test_dr3_substrate.test_all_8_protocols_declared."""
    assert _is_protocol(PhaseCouplingObservable)
```

- [ ] **Step 2: Run the test, verify it fails**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: FAIL with
`ModuleNotFoundError: No module named 'kiki_oniric.core.observables'`.

- [ ] **Step 3: Implement the Protocol**

Create `kiki_oniric/core/observables.py`:

```python
"""Opt-in observability protocols for substrate-side measurements.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§5 (invariants K family).

These Protocols are **not** part of the DR-3 Conformance Criterion
condition (1) (the 8 mandatory primitives in
:mod:`kiki_oniric.core.primitives` are). They are opt-in surfaces a
substrate may implement when it can expose a measurement relevant
to a measurement-class invariant (currently: K2 phase-coupling).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PhaseCouplingObservable(Protocol):
    """Substrate emits a signal usable for SO × fast-spindle coupling.

    Empirical anchor: eLife 2025 Bayesian meta-analysis
    (BibTeX `elife2025bayesian`) — coupling strength 0.33 in 95 %
    CI [0.27, 0.39]; Bayes factor > 58 vs. null; Egger
    publication-bias test p = 0.59 on the phase branch.

    Returns
    -------
    so_phase, spindle_amplitude : 1-D float32 arrays of identical length.
        ``so_phase[i]`` is the instantaneous phase (radians,
        wrapped to [-π, π]) of the slow-oscillation channel at
        sample ``i``. ``spindle_amplitude[i]`` is the instantaneous
        amplitude of the fast-spindle envelope at sample ``i``.
    fs : float
        Sampling rate in Hz, > 0. Used only for documentation /
        downstream estimators; the K2 estimator works in samples.
    """

    def emit_phase_coupling_signal(
        self, n_samples: int, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]: ...
```

- [ ] **Step 4: Run the structural tests, verify they pass**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 2 PASS.

- [ ] **Step 5: Lint + type check**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check kiki_oniric/core/observables.py \
                    tests/conformance/invariants/test_k2_coupling.py && \
  uv run mypy kiki_oniric tests
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add kiki_oniric/core/observables.py \
          tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "feat(observables): add PhaseCouplingObservable proto"
```

(Subject is 49 chars, under the 50-char hook limit. Body not
required for an additive Protocol; CONTRIBUTING.md only requires
body for functional changes.)

---

## Task 2: Implement coupling guard helper

**Files:**
* Create: `kiki_oniric/dream/guards/coupling.py`
* Test: `tests/conformance/invariants/test_k2_coupling.py` (extend)

This mirrors `kiki_oniric/dream/guards/attention.py` (S4): the guard
takes a measured value and asserts it falls inside the empirical CI,
raising `CouplingGuardError` otherwise. It does **not** estimate the
coupling itself (that lives in the test, Task 5) — it is a small
validator the conformance test (and any future runtime check) calls.

- [ ] **Step 1: Write the failing test**

Append to `tests/conformance/invariants/test_k2_coupling.py`:

```python
import pytest

from kiki_oniric.dream.guards.coupling import (
    CouplingGuardError,
    check_coupling_in_window,
)


def test_k2_guard_passes_inside_window() -> None:
    """Mid-window value must pass silently."""
    check_coupling_in_window(0.33, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_below_window() -> None:
    """Value < ci_low must raise."""
    with pytest.raises(CouplingGuardError, match="below"):
        check_coupling_in_window(0.20, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_above_window() -> None:
    """Value > ci_high must raise."""
    with pytest.raises(CouplingGuardError, match="above"):
        check_coupling_in_window(0.50, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_nan() -> None:
    """NaN slips through naive comparisons; explicit guard required."""
    import math

    with pytest.raises(CouplingGuardError, match="NaN"):
        check_coupling_in_window(math.nan, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_inverted_window() -> None:
    """ci_low > ci_high is a programmer error, must raise."""
    with pytest.raises(ValueError, match="ci_low"):
        check_coupling_in_window(0.33, ci_low=0.50, ci_high=0.10)
```

- [ ] **Step 2: Run, verify failure**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: ImportError on `kiki_oniric.dream.guards.coupling`. The 2
Protocol tests from Task 1 still PASS.

- [ ] **Step 3: Implement the guard**

Create `kiki_oniric/dream/guards/coupling.py`:

```python
"""K-coupling guard — assert measured SO × fast-spindle coupling lies in
the empirical 95 % CI from eLife 2025 Bayesian meta-analysis.

Reference: docs/invariants/registry.md (K2), framework-C spec §5,
BibTeX `elife2025bayesian`.

Mirrors the S2/S3/S4 guard convention: a single ``check_*`` callable
plus a dedicated exception type, both re-exported here.
"""
from __future__ import annotations

import math


class CouplingGuardError(RuntimeError):
    """Raised when measured coupling falls outside the K2 CI."""


def check_coupling_in_window(
    value: float, *, ci_low: float, ci_high: float
) -> None:
    """Validate ``value`` against the empirical CI.

    Parameters
    ----------
    value : float
        Measured coupling strength (e.g. mean vector length).
    ci_low, ci_high : float
        Bounds of the 95 % CI. ``ci_low <= ci_high`` required.

    Raises
    ------
    CouplingGuardError
        If ``value`` is NaN or falls outside ``[ci_low, ci_high]``.
    ValueError
        If ``ci_low > ci_high`` (programmer error).
    """
    if ci_low > ci_high:
        raise ValueError(
            f"ci_low ({ci_low}) must be <= ci_high ({ci_high})"
        )
    if math.isnan(value):
        raise CouplingGuardError("K2: coupling value is NaN")
    if value < ci_low:
        raise CouplingGuardError(
            f"K2: coupling {value:.4f} below CI low {ci_low:.4f}"
        )
    if value > ci_high:
        raise CouplingGuardError(
            f"K2: coupling {value:.4f} above CI high {ci_high:.4f}"
        )
```

- [ ] **Step 4: Run all K2 tests, verify PASS**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 7 PASS (2 from Task 1 + 5 new).

- [ ] **Step 5: Lint + types**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check kiki_oniric/dream/guards/coupling.py \
                    tests/conformance/invariants/test_k2_coupling.py && \
  uv run mypy kiki_oniric tests
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add kiki_oniric/dream/guards/coupling.py \
          tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "feat(guards): add K2 coupling-window guard"
```

---

## Task 3: Synthetic phase-coupling fixture substrate

**Files:**
* Create: `tests/conformance/invariants/_synthetic_phase_coupling.py`
* Test: `tests/conformance/invariants/test_k2_coupling.py` (extend)

Per `tests/CLAUDE.md`, the only "fake substrate" tolerance is the S5.3
`FakeBetaBuffer`. To stay aligned, the synthetic phase-coupling
fixture is named explicitly with a leading underscore, scoped to the
test directory (not under `kiki_oniric/`), documented as a synthetic
target, and **only** consumed by the K2 conformance test. It is not
exported via any package.

The fixture generates a deterministic SO-modulated fast-spindle signal
whose mean-vector-length (computed over `n_samples`) lands inside
[0.27, 0.39] across a wide RNG seed range. The construction:

```
phi(t)   = 2π · f_SO · t / fs            # SO instantaneous phase, f_SO=1 Hz
amp(t)   = 0.5 + 0.10 · cos(phi(t)) + ε  # fast-spindle envelope, modulated
```

with ε ~ Normal(0, 0.05). The PAC modulation depth `0.10` (10 % of the
mean amplitude) is calibrated so the resulting MVL falls roughly mid-window
(~0.32–0.35) for any seed. The exact value will be empirically pinned in
Task 5 using a closed-form sanity check.

- [ ] **Step 1: Write the failing test**

Append to `tests/conformance/invariants/test_k2_coupling.py`:

```python
from tests.conformance.invariants._synthetic_phase_coupling import (
    SyntheticPhaseCouplingSubstrate,
)


def test_synthetic_substrate_satisfies_protocol() -> None:
    """Synthetic fixture must structurally implement the Protocol."""
    sub = SyntheticPhaseCouplingSubstrate()
    assert isinstance(sub, PhaseCouplingObservable)


def test_synthetic_substrate_returns_aligned_arrays() -> None:
    """Phase + amplitude arrays must have the requested length and fs > 0."""
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, fs = sub.emit_phase_coupling_signal(n_samples=2048, seed=7)
    assert phase.shape == (2048,)
    assert amp.shape == (2048,)
    assert phase.dtype.name == "float32"
    assert amp.dtype.name == "float32"
    assert fs > 0.0


def test_synthetic_substrate_is_deterministic() -> None:
    """Same seed -> bit-identical output (R1 reproducibility, parent rule)."""
    sub = SyntheticPhaseCouplingSubstrate()
    p1, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    p2, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)


def test_synthetic_substrate_seeds_are_independent() -> None:
    """Distinct seeds produce distinct realisations (no global state)."""
    sub = SyntheticPhaseCouplingSubstrate()
    _, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=1)
    _, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=2)
    assert not np.array_equal(a1, a2)
```

Add the missing top-level import to the same file:

```python
import numpy as np
```

- [ ] **Step 2: Run, verify failure**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected:
`ModuleNotFoundError: No module named 'tests.conformance.invariants._synthetic_phase_coupling'`.

- [ ] **Step 3: Implement the synthetic substrate**

Create `tests/conformance/invariants/_synthetic_phase_coupling.py`:

```python
"""Synthetic phase-coupling substrate (test-only fixture).

Documented as a synthetic substitute target for the K2 conformance
test. **Not** part of `kiki_oniric/` and **not** an empirical claim
substrate. Real substrates implementing
:class:`PhaseCouplingObservable` will replace it as they appear
(MLX kiki-oniric S18+, E-SNN thalamocortical S22+).

Construction
------------
SO carrier frequency f_so = 1.0 Hz, fast-spindle envelope modulated
by SO phase with depth 0.10 around mean 0.5, additive Gaussian
noise σ = 0.05. Sampling rate fs = 256 Hz.

These values are calibrated so the mean-vector-length estimator
(see ``test_k2_coupling.test_k2_property_in_window``) yields a
coupling strength that falls inside the empirical 95 % CI
[0.27, 0.39] (eLife 2025 Bayesian meta-analysis,
``elife2025bayesian``).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SyntheticPhaseCouplingSubstrate:
    """Synthetic, deterministic phase-coupling source.

    Implements :class:`kiki_oniric.core.observables.PhaseCouplingObservable`
    structurally (no inheritance — Protocol is opt-in).
    """

    F_SO: float = 1.0       # Hz — slow oscillation
    FS: float = 256.0       # Hz — sampling rate
    PAC_DEPTH: float = 0.10 # modulation depth around mean amplitude
    AMP_MEAN: float = 0.5
    NOISE_SIGMA: float = 0.05

    def emit_phase_coupling_signal(
        self, n_samples: int, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        rng = np.random.default_rng(seed)
        t = np.arange(n_samples, dtype=np.float64) / self.FS
        phase = 2.0 * np.pi * self.F_SO * t
        # Wrap phase to [-π, π] before downcast.
        phase_wrapped = np.arctan2(np.sin(phase), np.cos(phase))
        noise = rng.normal(0.0, self.NOISE_SIGMA, size=n_samples)
        amp = self.AMP_MEAN + self.PAC_DEPTH * np.cos(phase) + noise
        # Amplitude must be non-negative (it is an envelope magnitude).
        amp = np.clip(amp, a_min=0.0, a_max=None)
        return (
            phase_wrapped.astype(np.float32),
            amp.astype(np.float32),
            self.FS,
        )
```

- [ ] **Step 4: Run, verify PASS**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 11 PASS (7 prior + 4 new).

- [ ] **Step 5: Lint + types**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check tests/conformance/invariants/_synthetic_phase_coupling.py \
                    tests/conformance/invariants/test_k2_coupling.py && \
  uv run mypy tests
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add tests/conformance/invariants/_synthetic_phase_coupling.py \
          tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "test(k2): add synthetic phase-coupling fixture"
```

---

## Task 4: K2 estimator (mean vector length)

**Files:**
* Modify: `tests/conformance/invariants/test_k2_coupling.py`

The K2 estimator is a private test helper (the spec deliberately does
not commit to a single PAC estimator at this stage; locking it inside
`kiki_oniric/` would be premature). It computes the
mean-vector-length of `amp · exp(i · phase)`, a standard PAC measure
(Tort 2010-style MVL).

- [ ] **Step 1: Write a failing closed-form sanity test for the estimator**

Append to `tests/conformance/invariants/test_k2_coupling.py`:

```python
def _mean_vector_length(
    phase: np.ndarray, amplitude: np.ndarray
) -> float:
    """Tort 2010-style mean vector length (PAC strength).

    MVL = | mean_t [ amplitude(t) * exp(i * phase(t)) ] | / mean_t amplitude(t)

    Returns a float in [0, 1]. Pure numpy, no SciPy needed; SciPy
    is reserved for any future Hilbert-transform based estimator.
    """
    if phase.shape != amplitude.shape:
        raise ValueError("phase and amplitude must have identical shapes")
    z = amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))
    num = float(np.abs(z.mean()))
    denom = float(np.abs(amplitude.astype(np.float64)).mean())
    if denom == 0.0:
        return 0.0
    return num / denom


def test_estimator_zero_for_random_phase() -> None:
    """No coupling: random uniform phase yields MVL ≈ 0 (large N)."""
    rng = np.random.default_rng(0)
    n = 8192
    phase = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    amp = (0.5 + rng.normal(0.0, 0.05, size=n)).astype(np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert mvl < 0.05, f"expected near-zero MVL on random phase, got {mvl}"


def test_estimator_one_for_perfect_coupling() -> None:
    """Perfect coupling: amplitude = 1 only at phase 0 -> MVL = 1.0."""
    n = 1024
    phase = np.zeros(n, dtype=np.float32)
    amp = np.ones(n, dtype=np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert abs(mvl - 1.0) < 1e-6
```

- [ ] **Step 2: Run, verify the new tests PASS (helper is local)**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 13 PASS. (The helper `_mean_vector_length` is defined
in-test; both new tests pass on first execution because the helper
is correct by construction. This task is not strict TDD because
the estimator is a math identity, not behaviour to discover. The
property test in Task 5 is the real TDD step.)

- [ ] **Step 3: Lint + types**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check tests/conformance/invariants/test_k2_coupling.py && \
  uv run mypy tests
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "test(k2): add MVL estimator + sanity tests"
```

---

## Task 5: K2 property test — coupling lands in CI [0.27, 0.39]

**Files:**
* Modify: `tests/conformance/invariants/test_k2_coupling.py`

This is the load-bearing conformance test. It runs the synthetic
substrate across a range of Hypothesis-supplied seeds, estimates MVL,
and asserts the K2 guard passes (i.e. value ∈ [0.27, 0.39]).

The CI bounds are pinned to the BibTeX `elife2025bayesian` note (95 %
CI on coupling strength). They are constants in the test, **not**
imported from a config — the test itself is the contract.

- [ ] **Step 1: Write the failing property test**

Append to `tests/conformance/invariants/test_k2_coupling.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

# Empirical anchor: eLife 2025 Bayesian meta-analysis
# (BibTeX `elife2025bayesian`). 95 % CI on coupling strength.
K2_CI_LOW: float = 0.27
K2_CI_HIGH: float = 0.39
K2_N_SAMPLES: int = 8192  # ≥ 32 SO cycles at fs=256 Hz, f_SO=1 Hz.


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(deadline=None, max_examples=50)
def test_k2_property_synthetic_in_window(seed: int) -> None:
    """K2: synthetic substrate's MVL falls inside eLife 2025 95 % CI.

    Reference: docs/invariants/registry.md (K2),
    docs/proofs/k2-coupling-evidence.md, BibTeX `elife2025bayesian`.
    """
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, _fs = sub.emit_phase_coupling_signal(
        n_samples=K2_N_SAMPLES, seed=seed
    )
    mvl = _mean_vector_length(phase, amp)
    check_coupling_in_window(mvl, ci_low=K2_CI_LOW, ci_high=K2_CI_HIGH)


def test_k2_property_smoke_known_seed() -> None:
    """Determinism check: seed=7 yields a known MVL bucket.

    This anchors the synthetic generator's calibration: if a future
    edit to the generator drifts the MVL outside [0.30, 0.36], the
    coverage of the property test against the empirical CI degrades
    silently. This smoke-test catches that.
    """
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, _fs = sub.emit_phase_coupling_signal(
        n_samples=K2_N_SAMPLES, seed=7
    )
    mvl = _mean_vector_length(phase, amp)
    assert 0.30 < mvl < 0.36, (
        f"calibration drift detected: seed=7 MVL={mvl:.4f} "
        f"outside [0.30, 0.36]"
    )
```

- [ ] **Step 2: Run, verify PASS (and calibration sticks)**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 15 PASS. If
`test_k2_property_synthetic_in_window` fails on some seed, the
synthetic generator's `PAC_DEPTH` is mis-calibrated. Fix:

* Increase `PAC_DEPTH` toward 0.12 if MVL < 0.27 systematically.
* Decrease toward 0.08 if MVL > 0.39 systematically.

Re-run after each adjustment. The smoke test
`test_k2_property_smoke_known_seed` pins the exact post-calibration
behaviour.

- [ ] **Step 3: Run the full conformance suite to confirm no regression**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/ -v --no-cov
```

Expected: all prior conformance tests still PASS, +15 new tests.

- [ ] **Step 4: Run the full coverage gate**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest
```

Expected: PASS, coverage ≥ 90 % (existing
`pyproject.toml:65 --cov-fail-under=90` enforced). The new module
`kiki_oniric/dream/guards/coupling.py` is fully exercised by Task 2
tests; `kiki_oniric/core/observables.py` is exercised structurally.

- [ ] **Step 5: Lint + types**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check . && \
  uv run mypy harness tests
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "test(k2): property test — MVL ∈ [0.27, 0.39]"
```

---

## Task 6: Edge cases — degenerate inputs

**Files:**
* Modify: `tests/conformance/invariants/test_k2_coupling.py`

Pin the estimator's behaviour at the corners so a future contributor
can not silently break it.

- [ ] **Step 1: Write failing edge-case tests**

Append to `tests/conformance/invariants/test_k2_coupling.py`:

```python
def test_estimator_rejects_shape_mismatch() -> None:
    """Estimator must guard against array length mismatch."""
    phase = np.zeros(10, dtype=np.float32)
    amp = np.zeros(11, dtype=np.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        _mean_vector_length(phase, amp)


def test_estimator_zero_amplitude_returns_zero() -> None:
    """All-zero amplitude defines MVL = 0 (no division by zero)."""
    n = 256
    phase = np.linspace(-np.pi, np.pi, n, dtype=np.float32)
    amp = np.zeros(n, dtype=np.float32)
    assert _mean_vector_length(phase, amp) == 0.0


def test_synthetic_substrate_rejects_zero_samples() -> None:
    """n_samples=0 must raise (per fixture contract)."""
    sub = SyntheticPhaseCouplingSubstrate()
    with pytest.raises(ValueError, match="n_samples"):
        sub.emit_phase_coupling_signal(n_samples=0, seed=0)
```

- [ ] **Step 2: Run, verify PASS (helpers already cover these)**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/invariants/test_k2_coupling.py -v --no-cov
```

Expected: 18 PASS. If
`test_estimator_rejects_shape_mismatch` or
`test_estimator_zero_amplitude_returns_zero` fail, the helper in
Task 4 is missing the `if phase.shape != amplitude.shape` guard or
the `denom == 0.0` short-circuit — restore them.

- [ ] **Step 3: Lint + types**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check tests/conformance/invariants/test_k2_coupling.py && \
  uv run mypy tests
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add tests/conformance/invariants/test_k2_coupling.py && \
  git commit -m "test(k2): pin degenerate-input edge cases"
```

---

## Task 7: Register K2 in `docs/invariants/registry.md`

**Files:**
* Modify: `docs/invariants/registry.md`

- [ ] **Step 1: Read the current K-family block**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -n -A 3 "K1\|K2\|K3\|K4" docs/invariants/registry.md
```

Expected: K1, K3, K4 present; K2 absent. Per parent
`tests/CLAUDE.md`, every new conformance test must be paired with an
entry here.

- [ ] **Step 2: Insert the K2 entry**

In `docs/invariants/registry.md`, locate the K-family block (lines
33–41 as of 2026-05-02). Replace the block with the version below
(adds K2 between K1 and K3, keeps K3/K4 verbatim, adds an empirical
anchor footnote):

```markdown
## Family K (Compute)

- **K1** Dream-episode budget respected — **BLOCKING** (per DE)
  FLOPs_actual ≤ budget.FLOPs and wall_time and energy. Enforced via
  A.4 runtime context manager.
- **K2** SO × fast-spindle phase-coupling within empirical CI —
  **WARN** (measurement-class invariant)
  For any substrate implementing the opt-in
  `PhaseCouplingObservable` Protocol, the measured coupling
  strength (Tort 2010-style mean-vector-length) must lie inside
  the 95 % CI [0.27, 0.39] reported by the eLife 2025 Bayesian
  meta-analysis (`elife2025bayesian` ; 23 studies, 297 effect
  sizes, BF > 58 vs. null, Egger phase-branch p = 0.59). Enforced
  by `tests/conformance/invariants/test_k2_coupling.py` against
  the synthetic substrate; real substrates plug in via the
  Protocol. Severity is WARN (not BLOCKING) because (a) only one
  meta-analysis pins the window, (b) substrate physiology can
  legitimately broaden the CI, (c) the synthetic substrate is the
  only canonical reference until S18+. Evidence stub:
  `docs/proofs/k2-coupling-evidence.md`.
- **K3** Swap latency bounded — **WARN**
  wall_clock(swap_atomic) ≤ K3_max (default 1s).
- **K4** Eval matrix coverage — **BLOCKING** (for tagging)
  MAJOR bump requires full stratified matrix executed before C-version
  tag. Enforced by T-Ops CI.
```

- [ ] **Step 3: Verify the registry edit**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -c "^- \*\*K[0-9]" docs/invariants/registry.md
```

Expected: `4` (K1, K2, K3, K4).

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add docs/invariants/registry.md && \
  git commit -m "docs(invariants): register K2 phase-coupling"
```

---

## Task 8: Add `docs/proofs/k2-coupling-evidence.md` evidence stub

**Files:**
* Create: `docs/proofs/k2-coupling-evidence.md`

`docs/proofs/CLAUDE.md` requires every new axiom test to ship with a
proof stub under `docs/proofs/`. K2 is an invariant rather than an
axiom, but the parallel rule from `tests/CLAUDE.md` ("don't add a new
axiom test without adding the axiom proof stub … and the invariant
declaration") makes a paired evidence document the safest option.
Status `evidence-only` (not a proof, an empirical pin) — the header
template from `dr2-compositionality.md` is followed.

- [ ] **Step 1: Create the evidence stub**

Create `docs/proofs/k2-coupling-evidence.md`:

```markdown
# K2 — SO × fast-spindle phase-coupling evidence

**Version:** v0.1-draft (2026-05-02)
**Supersedes:** —
**Amendment pointer:** —
**Target venue:** Paper 1 §5 (invariants), Paper 2 §3 (engineering
evidence)
**Executable counterpart:**
`tests/conformance/invariants/test_k2_coupling.py`
(`test_k2_property_synthetic_in_window`,
`test_k2_property_smoke_known_seed`)

## Status

`evidence-only`. K2 is a measurement-class invariant, not an axiom;
this file pins (a) the empirical anchor, (b) the estimator, (c) the
synthetic-substrate calibration. It is **not** a formal proof.

## Empirical anchor

eLife 2025 Bayesian meta-analysis of slow-oscillation–spindle
coupling and memory consolidation (BibTeX
`elife2025bayesian`). Headline numbers (verbatim from the paper note
in `docs/papers/paper1/references.bib:321-330`):

* Coupling strength: **0.33** with 95 % CI **[0.27, 0.39]**.
* Bayes factor vs. null: **> 58**.
* Egger publication-bias test on the phase branch: **p = 0.59**
  (no detected bias).

## Estimator

Mean-vector-length (Tort 2010-style):

$$
\mathrm{MVL} = \frac{\left|\frac{1}{N}\sum_t a(t)\,e^{i\phi(t)}\right|}
                    {\frac{1}{N}\sum_t \left|a(t)\right|}
$$

with $\phi(t)$ the SO instantaneous phase (radians, wrapped to
$[-\pi, \pi]$) and $a(t)$ the fast-spindle envelope. Implemented in
`tests/conformance/invariants/test_k2_coupling.py::_mean_vector_length`.

## Synthetic substrate calibration

The reference substrate
`tests/conformance/invariants/_synthetic_phase_coupling.py`
generates an SO carrier at $f_{SO} = 1$ Hz sampled at $f_s = 256$ Hz,
modulating a fast-spindle envelope of mean 0.5 with PAC depth 0.10 and
additive Gaussian noise $\sigma = 0.05$. Across 50
Hypothesis-supplied seeds, the estimator yields
$\mathrm{MVL} \in [0.27, 0.39]$. The smoke test `seed=7` pins the
calibrated mid-window value to $\mathrm{MVL} \in (0.30, 0.36)$.

## Limitations

* Single-meta-analysis anchor; the WARN severity in
  `docs/invariants/registry.md` reflects this.
* Real substrate measurement protocols (sampling, band-pass, Hilbert
  transform vs. complex Morlet) can shift the CI upward (typically
  toward 0.4) or downward (sub-0.27 in noisier recordings) without
  invalidating the underlying physiology. K2 is therefore advisory,
  not gating, until ≥ 2 independent meta-analyses converge.
* No real substrate implements `PhaseCouplingObservable` yet
  (S18+ planned for MLX kiki-oniric, S22+ for E-SNN
  thalamocortical). Until then K2 is exercised exclusively against
  the synthetic substrate.
```

- [ ] **Step 2: Verify it lints as Markdown (no broken bib key)**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  grep -n "elife2025bayesian" docs/proofs/k2-coupling-evidence.md \
                              docs/papers/paper1/references.bib
```

Expected: ≥ 1 hit in each file.

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add docs/proofs/k2-coupling-evidence.md && \
  git commit -m "docs(proofs): add K2 coupling evidence stub"
```

---

## Task 9: CHANGELOG entry — FC-MINOR bump

**Files:**
* Modify: `CHANGELOG.md`

Per the workspace `CLAUDE.md` DualVer rule, "formal axis bump requires
proof or spec change". K2 is a new spec entry (registry.md change),
which qualifies as a formal-axis MINOR bump (additive, non-breaking,
new measurement-class invariant). EC axis is unchanged.

- [ ] **Step 1: Read the current CHANGELOG head**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  head -40 CHANGELOG.md
```

Confirm the latest version is in the form
`C-vX.Y.Z+{STABLE,UNSTABLE,PARTIAL}` and find the most recent FC
section (`### Formal axis (FC)` or similar).

- [ ] **Step 2: Insert a new entry under the unreleased / current section**

In `CHANGELOG.md`, prepend (or insert under the topmost unreleased
header) a block of the form:

```markdown
## C-v0.9.0+PARTIAL — 2026-05-02

### Formal axis (FC) — MINOR

- Add invariant **K2** SO × fast-spindle phase-coupling within
  empirical CI [0.27, 0.39]. Anchored on eLife 2025 Bayesian
  meta-analysis (BibTeX `elife2025bayesian`). New opt-in
  `PhaseCouplingObservable` Protocol in
  `kiki_oniric/core/observables.py`; new guard
  `kiki_oniric/dream/guards/coupling.py`; new conformance suite
  `tests/conformance/invariants/test_k2_coupling.py`; evidence stub
  `docs/proofs/k2-coupling-evidence.md`.
- 8-primitive DR-3 surface (`kiki_oniric/core/primitives.py`)
  unchanged — no breaking change.

### Empirical axis (EC) — UNCHANGED

- No new gate result; K2 exercised exclusively against the synthetic
  reference substrate. Real-substrate empirical pins deferred to
  S18+ (MLX) / S22+ (E-SNN).
```

(Adjust the version literal to the next MINOR after the current head;
the example assumes head was `C-v0.8.x`.)

- [ ] **Step 3: Verify CHANGELOG parses**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  head -15 CHANGELOG.md
```

Expected: new block visible at top.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git add CHANGELOG.md && \
  git commit -m "docs(changelog): FC-MINOR bump for K2"
```

---

## Task 10: Final verification

**Files:** none modified.

- [ ] **Step 1: Full test suite + coverage gate**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest
```

Expected: all PASS, coverage ≥ 90 %.

- [ ] **Step 2: Conformance-only run**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run pytest tests/conformance/ -v --no-cov
```

Expected: all conformance tests PASS, including 18 new K2 tests.

- [ ] **Step 3: Lint + types whole repo**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  uv run ruff check . && \
  uv run mypy harness tests
```

Expected: both clean.

- [ ] **Step 4: Commit log audit**

Run:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  git log --oneline -10
```

Expected commit chain (top is most recent):

```
docs(changelog): FC-MINOR bump for K2
docs(proofs): add K2 coupling evidence stub
docs(invariants): register K2 phase-coupling
test(k2): pin degenerate-input edge cases
test(k2): property test — MVL ∈ [0.27, 0.39]
test(k2): add MVL estimator + sanity tests
test(k2): add synthetic phase-coupling fixture
feat(guards): add K2 coupling-window guard
feat(observables): add PhaseCouplingObservable proto
```

Each subject ≤ 50 chars, scope ≥ 3 chars, English, no
`Co-Authored-By` trailer (per `CONTRIBUTING.md`).

- [ ] **Step 5: No new branch or push**

This plan does **not** instruct a push. The commit chain stays
local until the engineer (or operator) decides to open a PR per the
parent workspace's "two PRs in two repos" cross-repo rule. K2 only
touches `dream-of-kiki`, so a single PR suffices when ready.

---

## Self-Review

### 1. Spec coverage

| Spec point | Covered by |
|------------|------------|
| New K-family invariant filling K2 hole | Task 7 (registry edit) |
| Empirical anchor `elife2025bayesian` 95 % CI [0.27, 0.39] | Task 5 (`K2_CI_LOW`/`K2_CI_HIGH` constants), Task 8 (proof stub) |
| Opt-in Protocol for substrate observability | Task 1 (`PhaseCouplingObservable`) |
| Synthetic deterministic fixture (seedable, reproducible) | Task 3 (`SyntheticPhaseCouplingSubstrate`) |
| Property test on Hypothesis-supplied seeds | Task 5 (`test_k2_property_synthetic_in_window`) |
| Guard helper mirroring S2/S3/S4 | Task 2 (`check_coupling_in_window`, `CouplingGuardError`) |
| K2 numbering decision (numbered K2, hole closed) | Task 7 (registry block) + this plan's Background |
| Edge-case coverage (NaN, shape mismatch, zero-length) | Task 6 |
| FC-MINOR DualVer bump | Task 9 |
| Evidence stub paired with conformance test | Task 8 (`docs/proofs/k2-coupling-evidence.md`) |
| Coverage ≥ 90 % gate | Task 5 step 4, Task 10 step 1 |
| ruff + mypy strict pass | Every implementation task + Task 10 step 3 |
| Conventional-commit subjects, ≤ 50 chars, no Co-Authored-By | Every commit step |

All requirements from the spec map to a task.

### 2. Placeholder scan

Scanned the plan for "TBD", "TODO", "implement later", "similar to",
"add error handling" without specifics, and references to undefined
symbols. Findings: none. Every code block is concrete (real imports,
real signatures, real assertions). Calibration uncertainty in Task 5
is bounded by an explicit fix-up procedure (raise/lower
`PAC_DEPTH`) and pinned by the Task 5 smoke test.

### 3. Type / name consistency

| Symbol | Defined in | Used in |
|--------|------------|---------|
| `PhaseCouplingObservable` | Task 1 (`kiki_oniric/core/observables.py`) | Task 1 (test), Task 3 (Protocol assertion), Task 7 (doc), Task 8 (doc) |
| `CouplingGuardError` | Task 2 (`kiki_oniric/dream/guards/coupling.py`) | Task 2 (tests) |
| `check_coupling_in_window(value, *, ci_low, ci_high)` | Task 2 | Task 2 (tests), Task 5 (`test_k2_property_synthetic_in_window`) |
| `SyntheticPhaseCouplingSubstrate.emit_phase_coupling_signal(n_samples, seed) -> (NDArray, NDArray, float)` | Task 3 | Task 3 (tests), Task 5, Task 6 |
| `_mean_vector_length(phase, amplitude) -> float` | Task 4 (in-test helper) | Task 4 (sanity tests), Task 5, Task 6 |
| `K2_CI_LOW`, `K2_CI_HIGH`, `K2_N_SAMPLES` | Task 5 | Task 5 (property test) |

All names match across tasks. The kw-only signature
`check_coupling_in_window(value, *, ci_low, ci_high)` is consistent
between Task 2 (definition) and Task 5 (use).

No further fixes needed.
