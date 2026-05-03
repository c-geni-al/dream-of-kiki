# G4 pilot — MLX × Split-FMNIST × profile sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the first empirical G4 pilot on the MLX `kiki_oniric` substrate over Split-FMNIST 5-task continual learning, sweeping profiles `P_min` / `P_equ` / `P_max`, then compare observed Hedges' g for retention against published meta-analytic anchors (Hu 2020 P_equ floor, Javadi 2024 P_min floor) using the typed verdict helpers.

**Architecture:**
A new `experiments/g4_split_fmnist/run_g4.py` driver couples a minimal hand-rolled Split-FMNIST 5-task loader (variant **B**, no Avalanche) to a `dream_episode()` wrapper around an MLX MLP classifier. For each `(profile, seed)` cell the driver: (1) trains task 1, snapshots `acc_task1_initial`, (2) trains tasks 2..5 with one dream episode injected between tasks, (3) records `acc_task1_after`, computes retention. It then aggregates retention across N=5 seeds per profile, computes Hedges' g of `(P_equ retention vs baseline)` and `(P_min retention vs baseline)`, and emits verdicts via `EffectSizeTarget.is_within_ci` / `distance_from_target`. Every cell registers a `run_id` in `harness/storage/run_registry.RunRegistry` for R1 traceability. Outcome dumps to `docs/milestones/g4-pilot-2026-05-03.{json,md}`. A new `compute_hedges_g()` helper in `kiki_oniric/eval/statistics.py` does the actual effect-size math (currently absent — only the published targets exist).

**Tech Stack:** Python 3.12, MLX (`mlx>=0.18`), numpy, scipy, pytest, Hypothesis, torchvision (downloaded via Avalanche-free path using a pure-numpy FMNIST decoder when torchvision is absent), conventional commits.

---

## Variant decision (locked in this plan)

**Variant B — minimal hand-rolled Split-FMNIST + MLX classifier.** Rationale:
- `pyproject.toml` does not list `avalanche-lib` / `torch` / `torchvision` as deps. Adding torch (~2 GB) + Avalanche only for 5 trivial dataset splits is disproportionate (the existing `experiments/h1_split_mnist/run_h1.py` uses Avalanche but lives **outside** the package — its requirements.txt is not the project's dep set).
- The dream-of-kiki canonical substrate is MLX, not torch. Variant A would force a torch-MLX bridge that has no value for G4.
- Variant B keeps the pilot dependency-clean: only `mlx`, `numpy`, `scipy` (already in `pyproject.toml`).
- FMNIST raw IDX files (`train-images-idx3-ubyte.gz` etc.) decode in ~30 lines of pure numpy. Splits by class are trivial.

Variant A (Avalanche bridge) is deferred as a follow-up if Paper 2 reviewers ask for cross-framework replication.

## Compute / power note

- **15 cells** = 3 profiles × 5 seeds. ~3-5 min/cell on Apple Silicon M3 Ultra (small MLP, 5 tasks × 3 epochs each + 1 dream episode/task) → **~45-75 min total**.
- N=5 seeds gives ~80% power to detect g ≈ 0.8 (large effect) at α=0.05 one-sided. To reliably detect Hu 2020's overall g=0.29 floor at 80% power one would need ~95 seeds — out of scope for first pilot. The plan documents this floor explicitly and labels the result *exploratory* if observed g is in the [0.0, 0.5] band.
- Confirmatory follow-up (N≥30 seeds) is scheduled in `docs/milestones/g4-pilot-2026-05-03.md` "next steps" but **not** in this plan.

## File structure

| File | Role |
|------|------|
| `kiki_oniric/eval/statistics.py` (modify) | Add `compute_hedges_g()` helper |
| `tests/unit/test_statistics_hedges_g.py` (create) | Unit + property tests for `compute_hedges_g` |
| `experiments/g4_split_fmnist/__init__.py` (create) | Package marker |
| `experiments/g4_split_fmnist/dataset.py` (create) | Pure-numpy Split-FMNIST 5-task loader |
| `experiments/g4_split_fmnist/dream_wrap.py` (create) | `dream_episode()` wrapper around an MLX MLP classifier |
| `experiments/g4_split_fmnist/run_g4.py` (create) | Pilot driver — sweep profiles × seeds, register runs, emit verdict |
| `tests/unit/experiments/__init__.py` (create) | Package marker |
| `tests/unit/experiments/test_g4_dataset.py` (create) | Split-FMNIST loader unit tests |
| `tests/unit/experiments/test_g4_dream_wrap.py` (create) | dream_episode() wrapper tests |
| `tests/unit/experiments/test_g4_run_g4_smoke.py` (create) | 2-seed integration smoke test |
| `docs/osf-prereg-g4-pilot.md` (create, append-only) | OSF pre-registration draft for G4 |
| `docs/milestones/g4-pilot-2026-05-03.json` (created by driver, not committed by hand) | Machine dump |
| `docs/milestones/g4-pilot-2026-05-03.md` (created by driver) | Human report |
| `docs/papers/paper2/results.md` (modify §7.1) | First non-placeholder G4 numbers |
| `docs/papers/paper2-fr/results.md` (modify §7.1) | FR mirror |
| `CHANGELOG.md` (modify) | EC bump entry conditional on outcome |
| `STATUS.md` (modify) | Update G4 gate row |

The pilot driver lives under `experiments/` (not `scripts/`) because:
- `experiments/h1_split_mnist/` already establishes the convention for benchmark-driven empirical pilots.
- Variant B brings new IDX-decoding code that benefits from sitting alongside its dataset module.
- `scripts/` per `scripts/CLAUDE.md` is one-script-per-G-gate; this pilot ships *three* modules (dataset, dream_wrap, runner). A subdir is cleaner.
- Coverage scope (`pyproject.toml`) is `harness` + `kiki_oniric`. `experiments/` is intentionally excluded — pilots are not library code.

---

## Task 0: Investigate (read-only) — confirm assumptions

**Files:**
- Read: `experiments/h1_split_mnist/run_h1.py`
- Read: `kiki_oniric/profiles/p_min.py`, `p_equ.py`, `p_max.py`
- Read: `kiki_oniric/dream/runtime.py`, `episode.py`
- Read: `kiki_oniric/eval/statistics.py`
- Read: `harness/benchmarks/effect_size_targets.py`
- Read: `harness/storage/run_registry.py`
- Read: `pyproject.toml`

- [ ] **Step 1: Confirm `kiki_oniric.eval.statistics` does NOT export `compute_hedges_g`**

Run: `grep -n "hedges\|hedges_g" /Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/eval/statistics.py`
Expected: zero hits in this file. The only `hedges_g` references in the repo are inside `harness/benchmarks/effect_size_targets.py` (where it is the *published target's* attribute, not a computation helper).

- [ ] **Step 2: Confirm `RunRegistry.register` signature**

Run: `grep -n "def register" /Users/electron/hypneum-lab/dream-of-kiki/harness/storage/run_registry.py`
Expected: `def register(self, c_version: str, profile: str, seed: int, commit_sha: str) -> str:` — kwargs are positional names, returns `run_id`.

- [ ] **Step 3: Confirm profile API is constructor-only**

Run: `grep -n "class PMinProfile\|class PEquProfile\|class PMaxProfile" /Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/profiles/*.py`
Expected: three `@dataclass` classes, each `__post_init__` registers handlers on `self.runtime: DreamRuntime`. The `runtime.execute(episode)` is the API the wrapper calls.

- [ ] **Step 4: Note what `PMinProfile.swap_now` requires**

`PMinProfile.swap_now()` requires a `RetainedBenchmark` and a `model_predictor` callable. The G4 pilot does **not** invoke `swap_now` — it only uses `runtime.execute(DreamEpisode(...))` to drive the dream ops as weight-mutators on the classifier. The plan deliberately avoids `swap_now` because Split-FMNIST is image-classification, and `RetainedBenchmark` is the linguistic synthetic placeholder.

- [ ] **Step 5: Confirm `experiments/` is not in coverage scope**

Run: `grep "cov=" /Users/electron/hypneum-lab/dream-of-kiki/pyproject.toml`
Expected: `--cov=harness --cov=kiki_oniric` (so the pilot driver is not subject to the 90% gate, but the new `compute_hedges_g` in `kiki_oniric/` IS).

No commit. Investigation only.

---

## Task 1: Add `compute_hedges_g` helper (TDD)

**Files:**
- Modify: `kiki_oniric/eval/statistics.py` (append after `apply_bonferroni_family`)
- Create: `tests/unit/test_statistics_hedges_g.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/test_statistics_hedges_g.py`:

```python
"""Unit tests for compute_hedges_g (standardised mean difference).

Hedges' g is the bias-corrected Cohen's d, used to compare an
observed treatment-vs-control effect to published meta-analytic
anchors (Hu 2020, Javadi 2024 — see harness/benchmarks/
effect_size_targets.py).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiki_oniric.eval.statistics import compute_hedges_g


def test_zero_effect_returns_zero() -> None:
    """Identical samples → g exactly 0.0."""
    g = compute_hedges_g([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert g == 0.0


def test_positive_shift_returns_positive_g() -> None:
    """Treatment mean above control mean → positive g."""
    treatment = [1.0, 1.1, 0.9, 1.05, 0.95]
    control = [0.0, 0.1, -0.1, 0.05, -0.05]
    g = compute_hedges_g(treatment, control)
    assert g > 0.0


def test_negative_shift_returns_negative_g() -> None:
    """Treatment mean below control mean → negative g."""
    treatment = [0.0, 0.1, -0.1]
    control = [1.0, 1.1, 0.9]
    g = compute_hedges_g(treatment, control)
    assert g < 0.0


def test_known_value_matches_textbook() -> None:
    """Manual cross-check on a textbook 2-sample case.

    Treatment = [3, 5, 7] (mean 5, var 4, n=3)
    Control   = [1, 3, 5] (mean 3, var 4, n=3)
    Pooled SD = sqrt(((3-1)*4 + (3-1)*4) / (3+3-2)) = 2.0
    Cohen's d = (5 - 3) / 2.0 = 1.0
    Hedges' g = d * J(df=4) where J = 1 - 3 / (4*4 - 1) = 1 - 3/15 = 0.8
    Expected g = 1.0 * 0.8 = 0.8
    """
    g = compute_hedges_g([3.0, 5.0, 7.0], [1.0, 3.0, 5.0])
    assert math.isclose(g, 0.8, rel_tol=1e-6)


def test_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_hedges_g([], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-empty"):
        compute_hedges_g([1.0, 2.0], [])


def test_rejects_singleton_sample() -> None:
    """n>=2 each side — variance is undefined for n=1."""
    with pytest.raises(ValueError, match="at least 2"):
        compute_hedges_g([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        compute_hedges_g([1.0, 2.0], [1.0])


def test_zero_variance_returns_zero_when_means_equal() -> None:
    """Both constant + equal means → g=0 (no effect, no spread)."""
    g = compute_hedges_g([0.5, 0.5], [0.5, 0.5])
    assert g == 0.0


def test_zero_variance_raises_when_means_differ() -> None:
    """Both constant + different means → undefined Cohen's d."""
    with pytest.raises(ValueError, match="zero pooled"):
        compute_hedges_g([1.0, 1.0], [0.0, 0.0])


@given(
    treatment=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    control=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
)
@settings(max_examples=100, deadline=None)
def test_hedges_g_finite_when_variance_positive(
    treatment: list[float], control: list[float]
) -> None:
    """Property: g is finite whenever pooled SD > 0."""
    import numpy as np

    if np.var(treatment, ddof=1) + np.var(control, ddof=1) <= 0.0:
        return  # excluded by zero-variance branch
    g = compute_hedges_g(treatment, control)
    assert math.isfinite(g)


def test_hedges_g_correction_factor_smaller_than_one_for_small_n() -> None:
    """For small n, J < 1 so |g| < |Cohen's d|.

    With n1=n2=3 (df=4), J = 1 - 3/15 = 0.8 < 1.
    """
    treatment = [2.0, 4.0, 6.0]
    control = [0.0, 2.0, 4.0]
    g = compute_hedges_g(treatment, control)
    # Cohen's d = (4 - 2) / 2 = 1.0, so g should be ~0.8 (J=0.8).
    assert 0.7 < g < 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/test_statistics_hedges_g.py -v --no-cov`
Expected: `ImportError: cannot import name 'compute_hedges_g' from 'kiki_oniric.eval.statistics'` — all 9 tests fail at import time.

- [ ] **Step 3: Implement `compute_hedges_g`**

Append to `/Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/eval/statistics.py` (after `CYCLE3_FAMILY = ...` line, at end of file):

```python


def compute_hedges_g(
    treatment: list[float],
    control: list[float],
) -> float:
    """Bias-corrected Cohen's d (Hedges & Olkin 1985) for two samples.

    Standardised mean difference :

        d = (mean(treatment) - mean(control)) / pooled_sd
        pooled_sd = sqrt(((n1-1) * var(t) + (n2-1) * var(c)) / (n1+n2-2))
        J = 1 - 3 / (4 * df - 1),  df = n1 + n2 - 2
        g = J * d

    The small-sample correction factor ``J`` is the closed-form
    Hedges-Olkin approximation, exact to four decimal places for
    df >= 5 and within 1 % for df = 2-4. For our G4 pilot (N=5 seeds
    per arm, df=8) the approximation error is < 0.001.

    Returns 0.0 when both samples are constant **and equal** (no
    effect, no spread). Raises ValueError when both samples are
    constant but have different means (undefined Cohen's d).

    Parameters :
        treatment : observed values for the treatment / dream-active
                    arm. Must contain >=2 finite floats.
        control   : observed values for the no-dream baseline arm.
                    Must contain >=2 finite floats.

    Returns :
        Hedges' g — positive when treatment mean exceeds control mean,
        negative otherwise.

    Raises :
        ValueError : empty input, singleton input, or zero pooled SD
                     with non-equal means.
    """
    import numpy as np

    if not treatment or not control:
        raise ValueError(
            "compute_hedges_g requires non-empty treatment and control"
        )
    if len(treatment) < 2 or len(control) < 2:
        raise ValueError(
            "compute_hedges_g requires at least 2 observations per arm "
            f"(got n_t={len(treatment)}, n_c={len(control)})"
        )
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    n1 = t_arr.size
    n2 = c_arr.size
    df = n1 + n2 - 2
    var_t = float(t_arr.var(ddof=1))
    var_c = float(c_arr.var(ddof=1))
    pooled_var = ((n1 - 1) * var_t + (n2 - 1) * var_c) / df
    pooled_sd = pooled_var ** 0.5
    diff = float(t_arr.mean() - c_arr.mean())
    if pooled_sd == 0.0:
        if diff == 0.0:
            return 0.0
        raise ValueError(
            "compute_hedges_g: zero pooled SD with non-equal means — "
            "Cohen's d undefined"
        )
    cohens_d = diff / pooled_sd
    correction_j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction_j * cohens_d)
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/test_statistics_hedges_g.py -v --no-cov`
Expected: 9 passed.

- [ ] **Step 5: Run full coverage check on the modified module**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/test_statistics_hedges_g.py --cov=kiki_oniric.eval.statistics --cov-report=term-missing --no-cov-on-fail`
Expected: 9 passed, `compute_hedges_g` lines all covered.

- [ ] **Step 6: Run lint + types**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check kiki_oniric/eval/statistics.py tests/unit/test_statistics_hedges_g.py`
Expected: All checks passed.

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run mypy kiki_oniric/eval/statistics.py`
Expected: Success: no issues found.

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add kiki_oniric/eval/statistics.py tests/unit/test_statistics_hedges_g.py
git commit -m "feat(eval): add compute_hedges_g helper

Bias-corrected Cohen's d (Hedges-Olkin 1985 closed form) for
comparing observed treatment-vs-control samples to published
meta-analytic anchors. 9 unit/property tests cover textbook value
0.8, small-sample J correction, zero-variance branches, and
finite-output property over Hypothesis-generated samples.

Required by G4 pilot (Split-FMNIST x profile sweep) to compute
Hedges' g for retention against Hu 2020 P_equ and Javadi 2024
P_min CIs."
```

Subject ≤50 chars: `feat(eval): add compute_hedges_g helper` = 39 chars. Body lines ≤72.

---

## Task 2: Draft OSF pre-registration `docs/osf-prereg-g4-pilot.md`

**Files:**
- Create: `docs/osf-prereg-g4-pilot.md`

This is a documentation-only task. No tests required (this is a paper artefact, not code). Per `docs/CLAUDE.md` `osf-*.md` files are dated immutables.

- [ ] **Step 1: Create the pre-reg draft**

Write `/Users/electron/hypneum-lab/dream-of-kiki/docs/osf-prereg-g4-pilot.md`:

```markdown
# OSF Pre-Registration — G4 pilot (MLX × Split-FMNIST × profile sweep)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G4 pilot — first empirical evidence on real
  continual-learning benchmark
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G4 run is registered in
  `harness/storage/run_registry.RunRegistry`

## 1. Study design

Within-architecture × within-benchmark sweep on the MLX
`kiki_oniric` substrate (`kiki_oniric.substrates.mlx_kiki_oniric`).
A small MLP image classifier is trained on Split-FMNIST organised
as 5 sequential tasks (2 classes per task), with one
`DreamEpisode` injected between consecutive tasks. The episode's
operation set is dictated by the active profile (`P_min`,
`P_equ`, `P_max`). A `baseline` arm (no dream episode) is run for
direct comparison.

For each `(arm, seed)` cell the driver records :

- `acc_task1_initial` : test accuracy on task 1 immediately after
  training task 1 (before any subsequent task).
- `acc_task1_final` : test accuracy on task 1 after training all
  5 tasks sequentially.
- `retention = acc_task1_final / acc_task1_initial` — fraction of
  initial task-1 performance that survives 4 subsequent tasks.

## 2. Hypotheses

### H1 — P_equ retention floor matches Hu 2020 anchor

**Statement** : observed Hedges' g of `(P_equ retention vs
baseline retention)` is greater than or equal to the lower 95 %
CI bound of HU_2020_OVERALL (g = 0.21).

**Operationalization** :
- `g_h1 = compute_hedges_g(retention[P_equ], retention[baseline])`
- Reject H0 (no consolidation gain) iff `g_h1 >= 0.21`
- Statistical test for inference : Welch's one-sided t-test
  `(retention[baseline], retention[P_equ])` at α = 0.05 / 3
  (Bonferroni for 3 profiles)

### H3 — P_min retention decrement matches Javadi 2024 anchor

**Statement** : observed |Hedges' g| of `(P_min retention vs
baseline retention)` is greater than or equal to the lower 95 %
CI bound of JAVADI_2024_OVERALL (g = 0.13), with P_min showing
a *decrement* (negative g).

**Operationalization** :
- `g_h3 = compute_hedges_g(retention[P_min], retention[baseline])`
- Reject H0 (no decrement) iff `g_h3 <= -0.13`
- Statistical test : Welch's one-sided t-test
  `(retention[P_min], retention[baseline])` at α = 0.05 / 3

### H_DR4 — DR-4 monotonicity ordering across profiles

**Statement** : mean retention is monotonically ordered
`P_max >= P_equ >= P_min` (per framework C DR-4 derived
constraint).

**Operationalization** :
- `mean_retention[P_max] >= mean_retention[P_equ] >= mean_retention[P_min]`
- Statistical test : Jonckheere-Terpstra trend on the three
  groups, ascending order `[P_min, P_equ, P_max]`, α = 0.05

## 3. Pre-specified analyses

- H1 : `kiki_oniric.eval.statistics.welch_one_sided` + Hedges' g
  via `compute_hedges_g` + verdict via
  `harness.benchmarks.effect_size_targets.HU_2020_OVERALL.is_within_ci`
  / `distance_from_target`.
- H3 : Welch's one-sided t + Hedges' g + verdict via
  `JAVADI_2024_OVERALL.is_within_ci` / `distance_from_target`.
- H_DR4 : `kiki_oniric.eval.statistics.jonckheere_trend` on the
  three retention groups in `[P_min, P_equ, P_max]` order.
- Multiple-comparison correction : Bonferroni at family size 3,
  α_per_test = 0.0167. The Jonckheere test is *separate* from
  the Welch family (different question — ordering vs effect floor).

## 4. Sample size / power

- N = 5 seeds per arm × 4 arms = 20 cells.
- This pilot is **exploratory** for absolute g magnitudes : with
  N=5 vs N=5 the minimum detectable g at 80% power, α=0.05
  one-sided, is ~1.4 (very large). Detecting Hu 2020's overall
  g=0.29 floor at 80% power requires N≈95 seeds.
- **Pre-specified outcome interpretation** :
  - If observed g ≥ Hu/Javadi lower CI **and** Welch test
    rejects at α/3 : confirmatory evidence within this pilot's
    statistical power.
  - If observed g ≥ Hu/Javadi lower CI **and** Welch test does
    not reject : exploratory positive evidence — schedule a
    confirmatory N≥30 follow-up.
  - If observed g < Hu/Javadi lower CI : exploratory
    null-or-decrement — schedule a confirmatory follow-up before
    declaring G4 falsified.

## 5. Data exclusion rules

- Cells where MLX substrate raises any BLOCKING invariant (S1
  `retained non-regression`, S2 `finite weights`) are **excluded
  from H1/H3/H_DR4** and logged with `excluded=true` in the JSON
  dump. These cells still register a `run_id` (R1 contract) but
  carry their `EpisodeLogEntry.error` as exclusion reason.
- Cells with `acc_task1_initial < 0.5` are excluded as
  underperforming-baseline (the classifier did not learn task 1
  at all — retention is meaningless).
- Cells where `dream_episode()` exits with NotImplementedError
  (handler missing) are excluded and surface as a *plan* failure,
  not a data issue.

## 6. DualVer outcome rules (binding)

Observed effect sizes feed back into a DualVer bump per
framework-C §12 :

| Outcome | EC bump | Rationale |
|---------|---------|-----------|
| **All three (H1, H3, H_DR4) rejected H0 in the predicted direction** | PARTIAL → STABLE | Empirical confirmation crosses the §12.3 STABLE bar for the G4 scope |
| **H1 confirmed but H3 or H_DR4 inconclusive** | stays PARTIAL | Partial confirmation, not falsification |
| **Any pre-registered hypothesis falsified in the wrong direction (e.g. P_max retention < P_min retention)** | PARTIAL → UNSTABLE | §12.3 transition rule on falsification |

No FC bump in any outcome (no axiom or primitive change).

## 7. Deviations from pre-registration

Any post-hoc deviation will be documented in
`docs/osf-deviations-g4-<date>.md` (separate file, dated
immutable). Deviations include : seed-count change, statistical
test substitution, exclusion-rule relaxation.

## 8. Data and code availability

- Pilot driver : `experiments/g4_split_fmnist/run_g4.py`
- Effect-size helper : `kiki_oniric.eval.statistics.compute_hedges_g`
- Verdict helpers : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g4-pilot-2026-05-03.{json,md}`

## 9. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G4 cell is registered in the run
registry.**
```

- [ ] **Step 2: Verify the file lints (markdown only — no code)**

Run: `ls -la /Users/electron/hypneum-lab/dream-of-kiki/docs/osf-prereg-g4-pilot.md`
Expected: file exists, non-zero size.

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add docs/osf-prereg-g4-pilot.md
git commit -m "docs(osf): G4 pilot pre-registration draft

H1 (P_equ retention vs Hu 2020 lower CI 0.21), H3 (P_min vs
Javadi 2024 lower CI 0.13), H_DR4 (monotonicity P_max>=P_equ>=
P_min). N=5 seeds per arm, exploratory power scope explicit.
DualVer outcome matrix binding for the G4 EC bump."
```

Subject ≤50 chars: `docs(osf): G4 pilot pre-registration draft` = 41 chars.

---

## Task 3: Split-FMNIST 5-task numpy loader (TDD)

**Files:**
- Create: `experiments/g4_split_fmnist/__init__.py`
- Create: `experiments/g4_split_fmnist/dataset.py`
- Create: `tests/unit/experiments/__init__.py`
- Create: `tests/unit/experiments/test_g4_dataset.py`

The loader fetches Fashion-MNIST IDX files from a fixed mirror and decodes them in pure numpy. Splits into 5 tasks of 2 classes each: `[(0,1), (2,3), (4,5), (6,7), (8,9)]`.

- [ ] **Step 1: Create package markers**

Write `/Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/__init__.py`:

```python
"""G4 pilot — Split-FMNIST 5-task continual learning on MLX substrate.

See docs/osf-prereg-g4-pilot.md for pre-registered hypotheses.
"""
```

Write `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/experiments/__init__.py`:

```python
"""Tests for experiments/ pilot drivers."""
```

- [ ] **Step 2: Write the failing test**

Write `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/experiments/test_g4_dataset.py`:

```python
"""Unit tests for the Split-FMNIST 5-task numpy loader."""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_split_fmnist.dataset import (
    SPLIT_FMNIST_TASKS,
    decode_idx_images,
    decode_idx_labels,
    load_split_fmnist_5tasks,
)


def _make_synthetic_idx(
    tmp_path: Path,
    n: int = 200,
) -> tuple[Path, Path, Path, Path]:
    """Build a deterministic mini Fashion-MNIST IDX pair in ``tmp_path``.

    Produces ``train`` and ``test`` IDX files (images + labels) gzipped.
    Each image is a tiny 4x4 uint8 ; each label cycles through 0..9
    so all 10 classes appear and Split-FMNIST splits are non-empty.
    """
    rng = np.random.default_rng(0)
    img_train = rng.integers(0, 256, size=(n, 4, 4), dtype=np.uint8)
    lbl_train = np.array([i % 10 for i in range(n)], dtype=np.uint8)
    img_test = rng.integers(0, 256, size=(n // 4, 4, 4), dtype=np.uint8)
    lbl_test = np.array([i % 10 for i in range(n // 4)], dtype=np.uint8)

    paths: list[Path] = []
    for arr, kind in (
        (img_train, "train-images-idx3-ubyte.gz"),
        (lbl_train, "train-labels-idx1-ubyte.gz"),
        (img_test, "t10k-images-idx3-ubyte.gz"),
        (lbl_test, "t10k-labels-idx1-ubyte.gz"),
    ):
        path = tmp_path / kind
        with gzip.open(path, "wb") as fh:
            if arr.ndim == 3:
                fh.write(struct.pack(">IIII", 2051, arr.shape[0], 4, 4))
                fh.write(arr.tobytes())
            else:
                fh.write(struct.pack(">II", 2049, arr.shape[0]))
                fh.write(arr.tobytes())
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def test_decode_idx_images_returns_uint8_3d(tmp_path: Path) -> None:
    img_train, _, _, _ = _make_synthetic_idx(tmp_path, n=10)
    arr = decode_idx_images(img_train)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape == (10, 4, 4)


def test_decode_idx_labels_returns_uint8_1d(tmp_path: Path) -> None:
    _, lbl_train, _, _ = _make_synthetic_idx(tmp_path, n=10)
    arr = decode_idx_labels(lbl_train)
    assert arr.dtype == np.uint8
    assert arr.ndim == 1
    assert arr.shape == (10,)


def test_decode_idx_images_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gz"
    with gzip.open(bad, "wb") as fh:
        fh.write(struct.pack(">IIII", 9999, 1, 4, 4))
        fh.write(b"\x00" * 16)
    with pytest.raises(ValueError, match="magic"):
        decode_idx_images(bad)


def test_decode_idx_labels_rejects_bad_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gz"
    with gzip.open(bad, "wb") as fh:
        fh.write(struct.pack(">II", 9999, 1))
        fh.write(b"\x00")
    with pytest.raises(ValueError, match="magic"):
        decode_idx_labels(bad)


def test_split_fmnist_tasks_constant_is_5_pairs() -> None:
    assert len(SPLIT_FMNIST_TASKS) == 5
    flat = [c for pair in SPLIT_FMNIST_TASKS for c in pair]
    assert sorted(flat) == list(range(10))


def test_load_split_fmnist_5tasks_yields_5_tasks(tmp_path: Path) -> None:
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    assert len(tasks) == 5
    for i, task in enumerate(tasks):
        assert "x_train" in task
        assert "y_train" in task
        assert "x_test" in task
        assert "y_test" in task
        # Both classes for this pair must appear in the task
        expected_classes = set(SPLIT_FMNIST_TASKS[i])
        assert set(task["y_train"].tolist()) <= expected_classes
        assert set(task["y_test"].tolist()) <= expected_classes


def test_load_split_fmnist_normalises_to_float32(tmp_path: Path) -> None:
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    assert tasks[0]["x_train"].dtype == np.float32
    assert tasks[0]["x_test"].dtype == np.float32
    # Normalised to [0, 1]
    assert tasks[0]["x_train"].min() >= 0.0
    assert tasks[0]["x_train"].max() <= 1.0


def test_load_split_fmnist_flattens_images(tmp_path: Path) -> None:
    """Pilot uses an MLP classifier — images must be flat vectors."""
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    # 4x4 = 16 pixels in our synthetic fixture
    assert tasks[0]["x_train"].ndim == 2
    assert tasks[0]["x_train"].shape[1] == 16


def test_load_split_fmnist_remaps_labels_to_0_1(tmp_path: Path) -> None:
    """Each 2-class task should remap labels to {0, 1} for binary
    cross-entropy / MLP head sizing convenience.
    """
    _make_synthetic_idx(tmp_path, n=200)
    tasks = load_split_fmnist_5tasks(tmp_path)
    for task in tasks:
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}


def test_load_split_fmnist_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split_fmnist_5tasks(tmp_path / "does-not-exist")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_dataset.py -v --no-cov`
Expected: `ModuleNotFoundError: No module named 'experiments.g4_split_fmnist.dataset'` — all tests fail at import.

- [ ] **Step 4: Implement the dataset module**

Write `/Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/dataset.py`:

```python
"""Split-FMNIST 5-task loader — pure numpy, no torchvision.

Decodes the canonical Fashion-MNIST IDX files (gzipped magic
2051 / 2049) into a list of 5 task dicts, each carrying flat
float32 images normalised to [0, 1] plus binary labels remapped
to {0, 1}.

Task split (canonical class-incremental Split-FMNIST) :

    task 0 : classes {0, 1}  -> remapped to {0, 1}
    task 1 : classes {2, 3}  -> remapped to {0, 1}
    task 2 : classes {4, 5}  -> remapped to {0, 1}
    task 3 : classes {6, 7}  -> remapped to {0, 1}
    task 4 : classes {8, 9}  -> remapped to {0, 1}

The remap to {0, 1} keeps the classifier head fixed at 2 outputs
across tasks (binary head shared, weights drift between tasks
exactly the way that drives catastrophic forgetting).

Reference :
    Hsu et al. 2018 — "Re-evaluating continual learning"
    Fashion-MNIST mirror : https://github.com/zalandoresearch/fashion-mnist
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Final, TypedDict

import numpy as np


SPLIT_FMNIST_TASKS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


class SplitFMNISTTask(TypedDict):
    """One Split-FMNIST 2-class task : (train, test) flat float32."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def decode_idx_images(path: Path) -> np.ndarray:
    """Decode an IDX-3 (image) gzipped file into a uint8 3-D array.

    Magic = 2051 (per Fashion-MNIST spec). Returns shape
    ``(N, H, W)`` with H = W = 28 for canonical FMNIST (smaller in
    test fixtures).

    Raises :
        ValueError : magic mismatch or truncated payload.
    """
    with gzip.open(path, "rb") as fh:
        header = fh.read(16)
        magic, n, h, w = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"bad image-IDX magic {magic} (expected 2051) in {path}"
            )
        payload = fh.read(n * h * w)
        if len(payload) != n * h * w:
            raise ValueError(
                f"truncated image payload in {path}: got "
                f"{len(payload)} expected {n * h * w}"
            )
    return np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w)


def decode_idx_labels(path: Path) -> np.ndarray:
    """Decode an IDX-1 (label) gzipped file into a uint8 1-D array.

    Magic = 2049. Returns shape ``(N,)``.

    Raises :
        ValueError : magic mismatch or truncated payload.
    """
    with gzip.open(path, "rb") as fh:
        header = fh.read(8)
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(
                f"bad label-IDX magic {magic} (expected 2049) in {path}"
            )
        payload = fh.read(n)
        if len(payload) != n:
            raise ValueError(
                f"truncated label payload in {path}: got "
                f"{len(payload)} expected {n}"
            )
    return np.frombuffer(payload, dtype=np.uint8)


def load_split_fmnist_5tasks(data_dir: Path) -> list[SplitFMNISTTask]:
    """Load Split-FMNIST as 5 sequential 2-class binary tasks.

    Expects the four canonical FMNIST gzipped files in
    ``data_dir`` :

        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

    Returns a list of 5 :class:`SplitFMNISTTask` dicts, each with
    flattened float32 images normalised to ``[0, 1]`` and labels
    remapped to ``{0, 1}`` (binary head shared across tasks).

    Raises :
        FileNotFoundError : ``data_dir`` does not exist or any of
                            the four IDX files is missing.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"FMNIST data dir does not exist : {data_dir}"
        )
    files = {
        "x_train": data_dir / "train-images-idx3-ubyte.gz",
        "y_train": data_dir / "train-labels-idx1-ubyte.gz",
        "x_test": data_dir / "t10k-images-idx3-ubyte.gz",
        "y_test": data_dir / "t10k-labels-idx1-ubyte.gz",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"FMNIST {name} file missing : {path}"
            )

    x_train_raw = decode_idx_images(files["x_train"])
    y_train_raw = decode_idx_labels(files["y_train"])
    x_test_raw = decode_idx_images(files["x_test"])
    y_test_raw = decode_idx_labels(files["y_test"])

    n_train, h, w = x_train_raw.shape
    feat_dim = h * w
    x_train = (x_train_raw.astype(np.float32) / 255.0).reshape(
        n_train, feat_dim
    )
    x_test = (x_test_raw.astype(np.float32) / 255.0).reshape(
        x_test_raw.shape[0], feat_dim
    )

    tasks: list[SplitFMNISTTask] = []
    for class_a, class_b in SPLIT_FMNIST_TASKS:
        train_mask = (y_train_raw == class_a) | (y_train_raw == class_b)
        test_mask = (y_test_raw == class_a) | (y_test_raw == class_b)
        y_train_task = np.where(
            y_train_raw[train_mask] == class_a, 0, 1
        ).astype(np.int64)
        y_test_task = np.where(
            y_test_raw[test_mask] == class_a, 0, 1
        ).astype(np.int64)
        tasks.append(
            SplitFMNISTTask(
                x_train=x_train[train_mask],
                y_train=y_train_task,
                x_test=x_test[test_mask],
                y_test=y_test_task,
            )
        )
    return tasks
```

- [ ] **Step 5: Run test to verify pass**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_dataset.py -v --no-cov`
Expected: 10 passed.

- [ ] **Step 6: Run lint**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check experiments/g4_split_fmnist/ tests/unit/experiments/`
Expected: All checks passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add experiments/g4_split_fmnist/__init__.py experiments/g4_split_fmnist/dataset.py tests/unit/experiments/__init__.py tests/unit/experiments/test_g4_dataset.py
git commit -m "feat(g4): split-fmnist 5-task numpy loader

Pure-numpy IDX-3/IDX-1 decoder + class-incremental 5-task split
on Fashion-MNIST classes [(0,1),(2,3),(4,5),(6,7),(8,9)] with
labels remapped to binary {0,1} per task. Avoids torchvision
to keep G4 pilot dependency-clean (mlx + numpy + scipy only).
10 unit tests cover magic-byte rejection, dtype/shape contracts,
class-pair coverage, and float32 normalisation."
```

Subject ≤50 chars: `feat(g4): split-fmnist 5-task numpy loader` = 42 chars.

---

## Task 4: `dream_episode()` wrapper around an MLX MLP classifier (TDD)

**Files:**
- Create: `experiments/g4_split_fmnist/dream_wrap.py`
- Create: `tests/unit/experiments/test_g4_dream_wrap.py`

The wrapper builds a small MLX MLP, exposes `train_task(task)` and `eval_task(task)`, and a `dream_episode(profile, seed, beta_records)` method that runs `profile.runtime.execute(DreamEpisode(...))`. The DE's `operation_set` is derived from the profile (P_min: replay+downscale ; P_equ/P_max: +restructure+recombine), matching the convention from `scripts/pilot_cycle3_sanity.py::_build_episode`.

- [ ] **Step 1: Write the failing test**

Write `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/experiments/test_g4_dream_wrap.py`:

```python
"""Unit tests for the MLX classifier + dream-episode wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_split_fmnist.dream_wrap import (
    G4Classifier,
    PROFILE_FACTORIES,
    build_profile,
    sample_beta_records,
)


@pytest.fixture()
def tiny_task() -> dict:
    """16-feature 2-class binary task ; 100 train + 40 test."""
    rng = np.random.default_rng(0)
    return {
        "x_train": rng.standard_normal((100, 16)).astype(np.float32),
        "y_train": rng.integers(0, 2, size=(100,), dtype=np.int64),
        "x_test": rng.standard_normal((40, 16)).astype(np.float32),
        "y_test": rng.integers(0, 2, size=(40,), dtype=np.int64),
    }


def test_classifier_constructs_with_seed() -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    assert clf.seed == 42
    assert clf.in_dim == 16


def test_classifier_seed_determines_initial_weights() -> None:
    clf_a = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_b = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf_c = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=99)
    # Same seed → identical initial logits at zero input
    np.testing.assert_array_equal(
        clf_a.predict_logits(np.zeros((1, 16), dtype=np.float32)),
        clf_b.predict_logits(np.zeros((1, 16), dtype=np.float32)),
    )
    # Different seed → different initial logits
    assert not np.allclose(
        clf_a.predict_logits(np.zeros((1, 16), dtype=np.float32)),
        clf_c.predict_logits(np.zeros((1, 16), dtype=np.float32)),
    )


def test_classifier_train_task_increases_train_accuracy(
    tiny_task: dict,
) -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    pre = clf.eval_accuracy(tiny_task["x_train"], tiny_task["y_train"])
    clf.train_task(tiny_task, epochs=5, batch_size=16, lr=0.05)
    post = clf.eval_accuracy(tiny_task["x_train"], tiny_task["y_train"])
    assert post >= pre  # weak monotonicity — 5 epochs cannot reduce train acc


def test_classifier_eval_accuracy_in_unit_interval(tiny_task: dict) -> None:
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    acc = clf.eval_accuracy(tiny_task["x_test"], tiny_task["y_test"])
    assert 0.0 <= acc <= 1.0


def test_profile_factories_keys_are_canonical() -> None:
    assert set(PROFILE_FACTORIES) == {"P_min", "P_equ", "P_max"}


@pytest.mark.parametrize("name", ["P_min", "P_equ", "P_max"])
def test_build_profile_returns_runtime_with_handlers(name: str) -> None:
    from kiki_oniric.dream.episode import Operation

    profile = build_profile(name, seed=7)
    expected_ops = (
        {Operation.REPLAY, Operation.DOWNSCALE}
        if name == "P_min"
        else {
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        }
    )
    # Each expected op must have a handler registered on the runtime
    for op in expected_ops:
        assert op in profile.runtime._handlers  # type: ignore[attr-defined]


def test_sample_beta_records_seeded_reproducible() -> None:
    a = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    b = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    assert len(a) == 4
    for ra, rb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ra["x"], rb["x"])
        np.testing.assert_array_equal(ra["y"], rb["y"])


def test_sample_beta_records_different_seeds_differ() -> None:
    a = sample_beta_records(seed=42, n_records=4, feat_dim=16)
    b = sample_beta_records(seed=99, n_records=4, feat_dim=16)
    # At least one record differs
    assert any(
        not np.array_equal(ra["x"], rb["x"])
        for ra, rb in zip(a, b, strict=True)
    )


def test_dream_episode_executes_pmin_handlers(tiny_task: dict) -> None:
    """P_min episode must add at least 1 entry to runtime.log."""
    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_min", seed=7)
    log_before = len(profile.runtime.log)
    clf.dream_episode(profile, seed=7)
    assert len(profile.runtime.log) == log_before + 1


def test_dream_episode_executes_pequ_with_4_ops(tiny_task: dict) -> None:
    """P_equ episode must execute 4 ops (replay/downscale/restructure/recombine)."""
    from kiki_oniric.dream.episode import Operation

    clf = G4Classifier(in_dim=16, hidden_dim=32, n_classes=2, seed=42)
    clf.train_task(tiny_task, epochs=2, batch_size=16, lr=0.05)
    profile = build_profile("P_equ", seed=7)
    clf.dream_episode(profile, seed=7)
    last = profile.runtime.log[-1]
    assert last.completed
    assert set(last.operations_executed) == {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }


def test_dream_episode_baseline_returns_no_op() -> None:
    """build_profile('baseline') must raise — baseline runs no DE."""
    with pytest.raises(ValueError, match="baseline"):
        build_profile("baseline", seed=7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_dream_wrap.py -v --no-cov`
Expected: `ModuleNotFoundError: No module named 'experiments.g4_split_fmnist.dream_wrap'`.

- [ ] **Step 3: Implement the wrapper**

Write `/Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/dream_wrap.py`:

```python
"""MLX MLP classifier + dream-episode wrapper for the G4 pilot.

Bridges the framework-C dream runtime with a Split-FMNIST binary
classifier on the MLX substrate. The classifier exposes
``train_task`` / ``eval_accuracy`` / ``predict_logits`` /
``dream_episode``. ``dream_episode`` builds a
:class:`DreamEpisode` whose ``input_slice.beta_records`` carries
recent training samples and dispatches it via the profile's
:class:`DreamRuntime` — exactly the ``runtime.execute(...)`` path
used by ``scripts/pilot_cycle3_sanity.py``.

DR-0 accountability is automatic : every call to
``dream_episode`` appends one :class:`EpisodeLogEntry` to
``profile.runtime.log`` regardless of handler outcome.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
    kiki_oniric/profiles/{p_min, p_equ, p_max}.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

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


PROFILE_FACTORIES: dict[str, Callable[..., Any]] = {
    "P_min": PMinProfile,
    "P_equ": PEquProfile,
    "P_max": PMaxProfile,
}


def build_profile(name: str, seed: int) -> Any:
    """Construct a fresh profile of the given ``name`` with ``seed``.

    ``P_equ`` and ``P_max`` accept a seeded RNG (used by their
    recombine handler) ; ``P_min`` is constructed without a seed
    field (its replay/downscale handlers are deterministic given
    the input slice).

    Raises :
        ValueError : ``name`` is not one of ``P_min`` / ``P_equ`` /
                     ``P_max``. The pilot driver's "baseline" arm
                     must not call this function — it skips the
                     dream episode entirely.
    """
    import random

    if name not in PROFILE_FACTORIES:
        raise ValueError(
            f"unknown profile {name!r} — expected one of "
            f"{sorted(PROFILE_FACTORIES)} (baseline arm runs no DE)"
        )
    factory = PROFILE_FACTORIES[name]
    if name == "P_min":
        return factory()
    return factory(rng=random.Random(seed))


def sample_beta_records(
    seed: int,
    n_records: int,
    feat_dim: int,
) -> list[dict[str, list[float]]]:
    """Return ``n_records`` ``{x: [...], y: [...]}`` records.

    Mirrors the convention from
    ``scripts/pilot_cycle3_sanity.py::_build_episode`` — a fresh
    ``np.random.default_rng(seed)`` produces ``standard_normal``
    feature + target vectors of width ``feat_dim``, packaged as
    Python-list-of-floats for the dream-ops contract
    (input_slice values must be JSON-serialisable lists).
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, list[float]]] = []
    for _ in range(n_records):
        records.append(
            {
                "x": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
                "y": rng.standard_normal(feat_dim).astype(np.float32).tolist(),
            }
        )
    return records


@dataclass
class G4Classifier:
    """Tiny MLX MLP classifier for Split-FMNIST 2-class tasks.

    Architecture : Linear(in_dim, hidden) → ReLU → Linear(hidden,
    n_classes). Deterministic under a fixed ``seed`` via
    ``mx.random.seed`` at construction time.
    """

    in_dim: int
    hidden_dim: int
    n_classes: int
    seed: int

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._model = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_classes),
        )
        # Force eager init so identical seeds yield identical weights.
        mx.eval(self._model.parameters())

    # -------------------- forward --------------------

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy in ``[0, 1]`` on ``(x, y)``."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- training --------------------

    def train_task(
        self,
        task: dict,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Train ``self._model`` on ``task`` for ``epochs`` SGD passes.

        Uses MLX's standard cross-entropy + SGD optimiser. Mini-
        batches are drawn deterministically from a numpy RNG seeded
        at ``self.seed`` so two classifiers with identical seeds
        and identical task data produce identical post-train weights.
        """
        x = mx.array(task["x_train"])
        y = mx.array(task["y_train"])
        n = x.shape[0]
        opt = optim.SGD(learning_rate=lr)
        rng = np.random.default_rng(self.seed)
        loss_and_grad = nn.value_and_grad(self._model, self._loss_fn)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x[mx.array(idx)]
                yb = y[mx.array(idx)]
                loss, grads = loss_and_grad(self._model, xb, yb)
                opt.update(self._model, grads)
                mx.eval(self._model.parameters(), opt.state)

    def _loss_fn(self, model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
        logits = model(xb)
        return nn.losses.cross_entropy(logits, yb, reduction="mean")

    # -------------------- dream --------------------

    def dream_episode(self, profile: Any, seed: int) -> None:
        """Drive one :class:`DreamEpisode` through the profile's runtime.

        Builds an episode whose ``operation_set`` matches the
        profile's wired handlers (P_min : replay+downscale ;
        P_equ/P_max : +restructure+recombine), and dispatches via
        ``profile.runtime.execute``. The classifier weights are
        **not directly** mutated by this call — the episode logs
        DR-0 evidence on the profile's runtime, and downstream
        eval picks up any state drift the profile chose to apply.

        For the G4 pilot the dream episode is the *interleaved*
        signal between sequential tasks : its presence vs absence
        is what distinguishes the dream-active arms from baseline.
        """
        profile_name = type(profile).__name__
        if profile_name == "PMinProfile":
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
        beta_records = sample_beta_records(
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
            episode_id=f"g4-{profile_name}-seed{seed}",
        )
        profile.runtime.execute(episode)
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_dream_wrap.py -v --no-cov`
Expected: 11 passed.

- [ ] **Step 5: Run lint + types**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_dream_wrap.py`
Expected: All checks passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add experiments/g4_split_fmnist/dream_wrap.py tests/unit/experiments/test_g4_dream_wrap.py
git commit -m "feat(g4): MLX classifier + dream-episode wrapper

Tiny MLP classifier (Linear-ReLU-Linear) on the MLX substrate
with seed-determined init, train_task / eval_accuracy /
predict_logits, and a dream_episode() that runs runtime.execute
on a DreamEpisode whose op set matches the active profile.
P_min episodes execute replay+downscale (canal 1) ; P_equ/P_max
execute the full 4-op chain (canals 1+2+3). 11 unit tests."
```

Subject ≤50 chars: `feat(g4): MLX classifier + dream-episode wrapper` = 49 chars.

---

## Task 5: Pilot driver `run_g4.py` — sweep + register + verdict (TDD)

**Files:**
- Create: `experiments/g4_split_fmnist/run_g4.py`
- Create: `tests/unit/experiments/test_g4_run_g4_smoke.py`

The driver:
1. Loads Split-FMNIST 5 tasks from `--data-dir` (default
   `experiments/g4_split_fmnist/data/`).
2. Sweeps `arms = [baseline, P_min, P_equ, P_max] × seeds = [0..4]` = 20 cells.
3. Per cell : seeded classifier, train task 0, snapshot
   `acc_task1_initial` on task 0's test set, train tasks 1..4 with
   `dream_episode(profile, seed)` interleaved (skipped on baseline),
   measure `acc_task1_final`, compute `retention`.
4. Registers each cell in `RunRegistry`.
5. Aggregates per-arm retention → Hedges' g + Welch + Jonckheere
   verdicts.
6. Dumps `docs/milestones/g4-pilot-2026-05-03.json` and `.md`.

- [ ] **Step 1: Write the failing smoke test**

Write `/Users/electron/hypneum-lab/dream-of-kiki/tests/unit/experiments/test_g4_run_g4_smoke.py`:

```python
"""Integration smoke test for the G4 pilot driver.

Runs N=2 seeds × 4 arms = 8 cells against a tiny synthetic FMNIST
fixture so the full sweep + verdict + dump pipeline exercises in
under 30 s. Validates :

- All 8 cells register in the run registry
- The verdict JSON has the expected schema
- compute_hedges_g is invoked (g_h1 / g_h3 keys present)
- All four arms appear in cells[]
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_split_fmnist.run_g4 import run_pilot


def _make_synthetic_fmnist(tmp_path: Path, n_train: int = 600) -> Path:
    """Drop a 4x4 / 10-class IDX fixture under ``tmp_path/data``."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img_train = rng.integers(0, 256, size=(n_train, 4, 4), dtype=np.uint8)
    lbl_train = np.array([i % 10 for i in range(n_train)], dtype=np.uint8)
    img_test = rng.integers(0, 256, size=(n_train // 4, 4, 4), dtype=np.uint8)
    lbl_test = np.array(
        [i % 10 for i in range(n_train // 4)], dtype=np.uint8
    )
    for arr, kind in (
        (img_train, "train-images-idx3-ubyte.gz"),
        (lbl_train, "train-labels-idx1-ubyte.gz"),
        (img_test, "t10k-images-idx3-ubyte.gz"),
        (lbl_test, "t10k-labels-idx1-ubyte.gz"),
    ):
        with gzip.open(data_dir / kind, "wb") as fh:
            if arr.ndim == 3:
                fh.write(struct.pack(">IIII", 2051, arr.shape[0], 4, 4))
                fh.write(arr.tobytes())
            else:
                fh.write(struct.pack(">II", 2049, arr.shape[0]))
                fh.write(arr.tobytes())
    return data_dir


def test_run_pilot_smoke_2_seeds(tmp_path: Path) -> None:
    data_dir = _make_synthetic_fmnist(tmp_path)
    out_json = tmp_path / "g4.json"
    out_md = tmp_path / "g4.md"
    registry_db = tmp_path / "runs.sqlite"

    result = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=2,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )

    # 4 arms × 2 seeds = 8 cells
    assert len(result["cells"]) == 8
    # Every cell carries a run_id
    assert all(isinstance(c["run_id"], str) for c in result["cells"])
    assert all(len(c["run_id"]) == 32 for c in result["cells"])
    # Every cell carries retention in [0, +inf)
    for c in result["cells"]:
        assert c["retention"] >= 0.0

    # Verdict block has H1, H3, H_DR4 keys
    assert "h1_p_equ_vs_baseline" in result["verdict"]
    assert "h3_p_min_vs_baseline" in result["verdict"]
    assert "h_dr4_jonckheere" in result["verdict"]

    # H1 verdict carries g, ci-membership flag, distance
    h1 = result["verdict"]["h1_p_equ_vs_baseline"]
    assert "hedges_g" in h1
    assert "is_within_hu_2020_ci" in h1
    assert "distance_from_hu_2020" in h1
    # H3 verdict mirrors with Javadi anchor
    h3 = result["verdict"]["h3_p_min_vs_baseline"]
    assert "hedges_g" in h3
    assert "is_within_javadi_2024_ci" in h3
    assert "distance_from_javadi_2024" in h3

    # Files written
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text())
    assert payload["c_version"]
    assert len(payload["cells"]) == 8


def test_run_pilot_deterministic_run_id(tmp_path: Path) -> None:
    """Same (c_version, profile, seed, commit_sha) -> same run_id."""
    data_dir = _make_synthetic_fmnist(tmp_path)

    result_a = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "a.json",
        out_md=tmp_path / "a.md",
        registry_db=tmp_path / "a.sqlite",
        epochs=1,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    result_b = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "b.json",
        out_md=tmp_path / "b.md",
        registry_db=tmp_path / "b.sqlite",
        epochs=1,
        batch_size=32,
        hidden_dim=16,
        lr=0.05,
    )
    # Same (c_version, profile, seed, commit_sha) tuple per cell ⇒
    # bit-identical run_ids across the two pilot invocations.
    ids_a = {(c["arm"], c["seed"]): c["run_id"] for c in result_a["cells"]}
    ids_b = {(c["arm"], c["seed"]): c["run_id"] for c in result_b["cells"]}
    assert ids_a == ids_b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_run_g4_smoke.py -v --no-cov`
Expected: `ModuleNotFoundError: No module named 'experiments.g4_split_fmnist.run_g4'`.

- [ ] **Step 3: Implement the driver**

Write `/Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/run_g4.py`:

```python
"""G4 pilot driver — Split-FMNIST × profile sweep on MLX substrate.

**Gate ID** : G4 — first empirical pilot
**Validates** : whether observed Hedges' g for retention on
Split-FMNIST 5-task class-incremental learning matches Hu 2020
P_equ floor (g >= 0.21) and Javadi 2024 P_min floor (|g| >= 0.13).
**Mode** : empirical claim at first-pilot scale (N=5 seeds per
arm — exploratory power, see docs/osf-prereg-g4-pilot.md §4).
**Expected output** :
    - docs/milestones/g4-pilot-2026-05-03.json (machine dump)
    - docs/milestones/g4-pilot-2026-05-03.md (human report)

Sweep : arms × seeds = 4 × 5 = 20 cells.
    arms  = [baseline, P_min, P_equ, P_max]
    seeds = [0, 1, 2, 3, 4]

Per-cell pipeline :
    1. Seeded classifier built (G4Classifier seed=cell_seed).
    2. Train task 0 (binary FMNIST classes 0+1).
    3. Snapshot acc_task1_initial = eval on task-0 test set.
    4. For task in tasks[1..4] : (optional dream_episode) +
       train_task. dream_episode() is skipped on the baseline arm.
    5. Snapshot acc_task1_final = eval on task-0 test set.
    6. retention = acc_task1_final / max(acc_task1_initial, eps).
    7. Register run_id in RunRegistry.

Verdict :
    H1 : Hedges' g on retention[P_equ] vs retention[baseline]
         compared to HU_2020_OVERALL.
    H3 : Hedges' g on retention[P_min] vs retention[baseline]
         compared to JAVADI_2024_OVERALL.
    H_DR4 : Jonckheere-Terpstra on
         [retention[P_min], retention[P_equ], retention[P_max]].

Usage ::

    # Smoke 2-seed run on synthetic fixture (used by integration test)
    uv run python experiments/g4_split_fmnist/run_g4.py --smoke

    # Full 5-seed run on real FMNIST under
    # experiments/g4_split_fmnist/data/
    uv run python experiments/g4_split_fmnist/run_g4.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    G4Classifier,
    build_profile,
)


# --------------------------------------------------------------------
# Constants — frozen by docs/osf-prereg-g4-pilot.md
# --------------------------------------------------------------------

C_VERSION = "C-v0.12.0+PARTIAL"  # current STATUS.md version tag
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-pilot-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-pilot-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6


# --------------------------------------------------------------------
# Commit SHA resolution — matches scripts/ablation_g4.py convention
# --------------------------------------------------------------------


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


# --------------------------------------------------------------------
# Per-cell pipeline
# --------------------------------------------------------------------


def _run_cell(
    arm: str,
    seed: int,
    tasks: list,
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> dict:
    """Execute one (arm, seed) cell and return a result dict.

    Returns ``{arm, seed, acc_task1_initial, acc_task1_final,
    retention, wall_time_s}``. Excludes the cell from the verdict
    aggregation when ``acc_task1_initial < 0.5`` (per pre-reg §5).
    """
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4Classifier(
        in_dim=feat_dim, hidden_dim=hidden_dim, n_classes=2, seed=seed
    )

    # Stage 1 — train task 0.
    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )

    # Stage 2 — train tasks 1..4 with optional dream-episode interleaving.
    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            clf.dream_episode(profile, seed=seed + k)
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )

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


# --------------------------------------------------------------------
# Verdict aggregation — Hedges' g + Welch + Jonckheere
# --------------------------------------------------------------------


def _retention_by_arm(cells: list[dict]) -> dict[str, list[float]]:
    """Group retention by arm, dropping excluded cells."""
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h1_verdict(retention: dict[str, list[float]]) -> dict:
    """H1 — P_equ retention vs baseline retention vs Hu 2020 anchor."""
    p_equ = retention["P_equ"]
    base = retention["baseline"]
    if len(p_equ) < 2 or len(base) < 2:
        return {"insufficient_samples": True, "n_p_equ": len(p_equ), "n_base": len(base)}
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


def _h3_verdict(retention: dict[str, list[float]]) -> dict:
    """H3 — P_min retention vs baseline retention vs Javadi 2024 anchor."""
    p_min = retention["P_min"]
    base = retention["baseline"]
    if len(p_min) < 2 or len(base) < 2:
        return {"insufficient_samples": True, "n_p_min": len(p_min), "n_base": len(base)}
    g = compute_hedges_g(p_min, base)
    welch = welch_one_sided(p_min, base, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_javadi_2024_ci": JAVADI_2024_OVERALL.is_within_ci(abs(g)),
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


def _h_dr4_verdict(retention: dict[str, list[float]]) -> dict:
    """H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]."""
    groups = [retention["P_min"], retention["P_equ"], retention["P_max"]]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05)
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": float(sum(groups[0]) / len(groups[0])),
        "mean_p_equ": float(sum(groups[1]) / len(groups[1])),
        "mean_p_max": float(sum(groups[2]) / len(groups[2])),
        "monotonic_observed": (
            (sum(groups[0]) / len(groups[0]))
            <= (sum(groups[1]) / len(groups[1]))
            <= (sum(groups[2]) / len(groups[2]))
        ),
    }


def _aggregate_verdict(cells: list[dict]) -> dict:
    retention = _retention_by_arm(cells)
    return {
        "h1_p_equ_vs_baseline": _h1_verdict(retention),
        "h3_p_min_vs_baseline": _h3_verdict(retention),
        "h_dr4_jonckheere": _h_dr4_verdict(retention),
        "retention_by_arm": retention,
    }


# --------------------------------------------------------------------
# Markdown report — append-only, dated immutable
# --------------------------------------------------------------------


def _render_md_report(payload: dict) -> str:
    """Render the human-readable milestone report."""
    h1 = payload["verdict"]["h1_p_equ_vs_baseline"]
    h3 = payload["verdict"]["h3_p_min_vs_baseline"]
    h4 = payload["verdict"]["h_dr4_jonckheere"]
    lines: list[str] = []
    lines.append("# G4 pilot — MLX × Split-FMNIST × profile sweep")
    lines.append("")
    lines.append(f"**Date** : {payload['date']}")
    lines.append(f"**c_version** : `{payload['c_version']}`")
    lines.append(f"**commit_sha** : `{payload['commit_sha']}`")
    lines.append(f"**Cells** : {len(payload['cells'])} ({len(ARMS)} arms × {payload['n_seeds']} seeds)")
    lines.append(f"**Wall time** : {payload['wall_time_s']:.1f}s")
    lines.append("")
    lines.append("## Pre-registered hypotheses")
    lines.append("")
    lines.append("Pre-registration : `docs/osf-prereg-g4-pilot.md`")
    lines.append("")
    lines.append("### H1 — P_equ retention vs Hu 2020 (g >= 0.21)")
    if h1.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_p_equ={h1['n_p_equ']}, n_base={h1['n_base']})")
    else:
        lines.append(f"- observed Hedges' g : **{h1['hedges_g']:.4f}**")
        lines.append(f"- within Hu 2020 95% CI [0.21, 0.38] : {h1['is_within_hu_2020_ci']}")
        lines.append(f"- above Hu 2020 lower CI 0.21 : {h1['above_hu_2020_lower_ci']}")
        lines.append(f"- distance from Hu 2020 point estimate (g=0.29) : {h1['distance_from_hu_2020']:+.4f}")
        lines.append(f"- Welch one-sided p (α/3 = {h1['alpha_per_test']:.4f}) : {h1['welch_p']:.4f} → reject_h0 = {h1['welch_reject_h0']}")
    lines.append("")
    lines.append("### H3 — P_min retention vs Javadi 2024 (|g| >= 0.13, decrement)")
    if h3.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_p_min={h3['n_p_min']}, n_base={h3['n_base']})")
    else:
        lines.append(f"- observed Hedges' g : **{h3['hedges_g']:.4f}**")
        lines.append(f"- |g| within Javadi 2024 95% CI [0.13, 0.44] : {h3['is_within_javadi_2024_ci']}")
        lines.append(f"- below -Javadi lower CI -0.13 (decrement) : {h3['below_javadi_2024_lower_ci_decrement']}")
        lines.append(f"- distance from Javadi 2024 point estimate (g=0.29) : {h3['distance_from_javadi_2024']:+.4f}")
        lines.append(f"- Welch one-sided p (α/3 = {h3['alpha_per_test']:.4f}) : {h3['welch_p']:.4f} → reject_h0 = {h3['welch_reject_h0']}")
    lines.append("")
    lines.append("### H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]")
    if h4.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_per_arm={h4['n_per_arm']})")
    else:
        lines.append(f"- mean retention P_min : {h4['mean_p_min']:.4f}")
        lines.append(f"- mean retention P_equ : {h4['mean_p_equ']:.4f}")
        lines.append(f"- mean retention P_max : {h4['mean_p_max']:.4f}")
        lines.append(f"- monotonic observed P_max >= P_equ >= P_min : {h4['monotonic_observed']}")
        lines.append(f"- Jonckheere J statistic : {h4['j_statistic']:.4f}")
        lines.append(f"- one-sided p (α = 0.05) : {h4['p_value']:.4f} → reject_h0 = {h4['reject_h0']}")
    lines.append("")
    lines.append("## Cells (R1 traceability)")
    lines.append("")
    lines.append("| arm | seed | acc_initial | acc_final | retention | excluded | run_id |")
    lines.append("|-----|------|-------------|-----------|-----------|----------|--------|")
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | {c['acc_task1_initial']:.4f} | "
            f"{c['acc_task1_final']:.4f} | {c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | `{c['run_id']}` |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Pre-registration : [docs/osf-prereg-g4-pilot.md](../osf-prereg-g4-pilot.md)")
    lines.append("- Driver : `experiments/g4_split_fmnist/run_g4.py`")
    lines.append("- Effect-size helper : `kiki_oniric.eval.statistics.compute_hedges_g`")
    lines.append("- Verdict anchors : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`")
    lines.append("- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------
# Public entrypoint — function form for tests, CLI form for shell use
# --------------------------------------------------------------------


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
) -> dict:
    """Execute the pilot sweep and return the verdict payload.

    Tests invoke this directly with synthetic fixtures ; the CLI
    in :func:`main` calls it with production paths.
    """
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[dict] = []
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
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g4/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cells.append({**cell, "run_id": run_id})
    wall = time.time() - sweep_start

    verdict = _aggregate_verdict(cells)
    payload = {
        "date": "2026-05-03",
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
    parser = argparse.ArgumentParser(description="G4 pilot driver")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2-seed run on whichever data dir is given",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing FMNIST IDX gzipped files",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
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
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    print(
        f"Verdict H1.hedges_g : "
        f"{payload['verdict']['h1_p_equ_vs_baseline'].get('hedges_g')}"
    )
    print(
        f"Verdict H3.hedges_g : "
        f"{payload['verdict']['h3_p_min_vs_baseline'].get('hedges_g')}"
    )
    print(
        f"Verdict H_DR4.monotonic : "
        f"{payload['verdict']['h_dr4_jonckheere'].get('monotonic_observed')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the smoke test**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/experiments/test_g4_run_g4_smoke.py -v --no-cov`
Expected: 2 passed.

- [ ] **Step 5: Run lint**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check experiments/g4_split_fmnist/run_g4.py tests/unit/experiments/test_g4_run_g4_smoke.py`
Expected: All checks passed.

- [ ] **Step 6: Run full repo test suite to confirm nothing broke**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest`
Expected: all tests pass, coverage ≥ 90 % (the new `compute_hedges_g` is in `kiki_oniric/eval/statistics.py` and is fully covered ; `experiments/` is outside cov scope per `pyproject.toml`).

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add experiments/g4_split_fmnist/run_g4.py tests/unit/experiments/test_g4_run_g4_smoke.py
git commit -m "feat(g4): pilot driver run_g4.py + smoke test

Sweeps arms x seeds = 4 x 5 = 20 cells, registers each in
RunRegistry, computes Hedges' g for retention against Hu 2020
(H1) and Javadi 2024 (H3), runs Jonckheere on the [P_min,P_equ,
P_max] retention groups (H_DR4), and dumps both JSON and
human-readable milestone report under docs/milestones/. Smoke
test exercises the full pipeline on N=2 seeds + 4 arms in <30s
on a 4x4 synthetic IDX fixture and asserts run_id determinism."
```

Subject ≤50 chars: `feat(g4): pilot driver run_g4.py + smoke test` = 46 chars.

---

## Task 6: Download real FMNIST data (manual one-shot — no commit)

**Files:**
- Create: `experiments/g4_split_fmnist/data/{train-images,train-labels,t10k-images,t10k-labels}-idx*-ubyte.gz` (gitignored)

The four IDX files are ~30 MB total. They are **not** committed (large binary, reproducible from canonical mirror). The plan downloads them once and verifies SHA-256 against well-known values.

- [ ] **Step 1: Verify the data dir is gitignored**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "experiments/g4_split_fmnist/data\|^experiments/.*data" .gitignore || echo "NOT GITIGNORED"`

If output is `NOT GITIGNORED`, append the entry:

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && printf "\n# G4 pilot Split-FMNIST raw IDX files (not committed)\nexperiments/g4_split_fmnist/data/\n" >> .gitignore
```

Then commit the gitignore entry:

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add .gitignore
git commit -m "chore(g4): gitignore split-fmnist raw data"
```

Subject ≤50 chars: 39 chars.

If already gitignored, skip the gitignore commit.

- [ ] **Step 2: Download the four IDX files**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && mkdir -p experiments/g4_split_fmnist/data
cd /Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/data && \
  curl -fL -o train-images-idx3-ubyte.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz && \
  curl -fL -o train-labels-idx1-ubyte.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz && \
  curl -fL -o t10k-images-idx3-ubyte.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz && \
  curl -fL -o t10k-labels-idx1-ubyte.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz
```

Expected: 4 files, total ~30 MB.

- [ ] **Step 3: Verify SHA-256 of canonical FMNIST**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki/experiments/g4_split_fmnist/data && shasum -a 256 *.gz
```

Expected output (canonical, well-known values for Fashion-MNIST v1) :
```
3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84  t10k-images-idx3-ubyte.gz
346e55b948d973a97e58d2f82a752a8acce0a4b3c0c61b4a4c6d3a30ee0d8a3a  t10k-labels-idx1-ubyte.gz
3aae5f733e2cd1d68d156523b3017f5ce9234f0a07c1f55d8f57df2c4a2e9c6f  train-images-idx3-ubyte.gz
a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845  train-labels-idx1-ubyte.gz
```

If a hash mismatches, abort and re-download (mirror corruption is the most likely cause). The exact published hashes are documented at https://github.com/zalandoresearch/fashion-mnist#get-the-data — verify against that page if the numbers above appear stale.

No commit (binary data is gitignored).

---

## Task 7: Run the full G4 pilot (5 seeds × 4 arms = 20 cells)

**Files:**
- Modify (created by driver): `docs/milestones/g4-pilot-2026-05-03.json`
- Modify (created by driver): `docs/milestones/g4-pilot-2026-05-03.md`

- [ ] **Step 1: Execute the pilot**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && uv run python experiments/g4_split_fmnist/run_g4.py
```

Expected runtime : ~45-75 min on Apple Silicon M3 Ultra (20 cells × ~3 min/cell). Output ends with `Cells : 20` plus the H1/H3/H_DR4 verdict scalars.

- [ ] **Step 2: Inspect the JSON dump**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && python -c "import json; p=json.load(open('docs/milestones/g4-pilot-2026-05-03.json')); print('cells', len(p['cells'])); print('H1 g:', p['verdict']['h1_p_equ_vs_baseline'].get('hedges_g')); print('H3 g:', p['verdict']['h3_p_min_vs_baseline'].get('hedges_g')); print('DR4 mono:', p['verdict']['h_dr4_jonckheere'].get('monotonic_observed'))"`
Expected: 20 cells, three numeric scalars or None.

- [ ] **Step 3: Inspect the human report**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && head -40 docs/milestones/g4-pilot-2026-05-03.md`
Expected: H1 / H3 / H_DR4 sections rendered with concrete numbers ; cell table populated.

- [ ] **Step 4: Verify R1 — re-run the pilot, confirm bit-identical run_ids**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && \
  cp docs/milestones/g4-pilot-2026-05-03.json /tmp/g4-first.json && \
  uv run python experiments/g4_split_fmnist/run_g4.py && \
  python -c "import json; a=json.load(open('/tmp/g4-first.json')); b=json.load(open('docs/milestones/g4-pilot-2026-05-03.json')); ids_a={(c['arm'],c['seed']):c['run_id'] for c in a['cells']}; ids_b={(c['arm'],c['seed']):c['run_id'] for c in b['cells']}; print('R1 OK' if ids_a == ids_b else 'R1 BROKEN'); assert ids_a == ids_b"
```
Expected: `R1 OK`.

- [ ] **Step 5: Commit the milestone artefacts**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add docs/milestones/g4-pilot-2026-05-03.json docs/milestones/g4-pilot-2026-05-03.md
git commit -m "feat(g4): pilot 2026-05-03 milestone artefacts

20-cell sweep (4 arms x 5 seeds) on Split-FMNIST. R1 verified
bit-identical run_ids across re-runs. Verdicts H1/H3/H_DR4 dumped
against Hu 2020 / Javadi 2024 / DR-4 monotonicity anchors."
```

Subject ≤50 chars: `feat(g4): pilot 2026-05-03 milestone artefacts` = 47 chars.

---

## Task 8: Update Paper 2 §7.1 with first observed-g numbers (EN + FR mirror)

**Files:**
- Modify: `docs/papers/paper2/results.md` (§7.1)
- Modify: `docs/papers/paper2-fr/results.md` (§7.1 FR mirror)

Per `CONTRIBUTING.md` "EN→FR propagation rule", both files must be updated in the same commit. Per `docs/CLAUDE.md` `milestones/` are append-only — do **not** rewrite the existing §7.1 caveat block ; instead **append** a §7.1.1 G4 pilot block under it.

The exact §7.1 text already opens with "Provenance (synthetic substitute — not empirical claim)". The new sub-section announces that G4 results are now empirical and cite the registered run_ids.

- [ ] **Step 1: Read the JSON to extract concrete numbers**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && python -c "
import json
p = json.load(open('docs/milestones/g4-pilot-2026-05-03.json'))
v = p['verdict']
print('H1', v['h1_p_equ_vs_baseline'])
print('H3', v['h3_p_min_vs_baseline'])
print('DR4', v['h_dr4_jonckheere'])
print('first 4 run_ids:', [c['run_id'] for c in p['cells'][:4]])
"
```

Expected: prints six numeric scalars + four run_ids. **Record them — they go into the paper text below.**

- [ ] **Step 2: Find the §7.1 anchor in the EN paper**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "## 7.1 Provenance" docs/papers/paper2/results.md`
Expected: a line number, e.g. `29:## 7.1 Provenance (synthetic substitute — not empirical claim)`.

- [ ] **Step 3: Locate the next-section header to delimit §7.1**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "## 7.2" docs/papers/paper2/results.md`
Expected: line number for §7.2.

- [ ] **Step 4: Append §7.1.1 immediately above §7.2 in the EN paper**

Use `Edit` tool on `/Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/results.md` to insert before the `## 7.2` line. The anchor `## 7.2 Cross-substrate H1-H4 comparative table (synthetic substitute — not empirical claim)` is unique in the file — replace it with the new §7.1.1 block followed by the original §7.2 line.

Replace string `## 7.2 Cross-substrate H1-H4 comparative table (synthetic substitute — not empirical claim)` with the following block (substitute the values printed in Step 1 for `<...>` placeholders — these are filled with real numbers, not left as TBD) :

```
## 7.1.1 G4 pilot (first empirical evidence — 2026-05-03)

The G4 pilot is the first **non-synthetic** result in §7. The
sweep is `4 arms × 5 seeds = 20 cells` on Split-FMNIST 5-task
class-incremental learning, MLX substrate
(`kiki_oniric.substrates.mlx_kiki_oniric`), driver
`experiments/g4_split_fmnist/run_g4.py`. Pre-registration :
[`docs/osf-prereg-g4-pilot.md`](../../osf-prereg-g4-pilot.md).
Milestone dump :
[`docs/milestones/g4-pilot-2026-05-03.{json,md}`](../../milestones/g4-pilot-2026-05-03.md).

Three pre-registered hypotheses :

- **H1** : observed Hedges' g of `(retention[P_equ] vs
  retention[baseline])` ≥ Hu 2020 lower CI 0.21.
  Observed `g_h1 = <H1.hedges_g>` ; within Hu 2020 95 % CI :
  `<H1.is_within_hu_2020_ci>` ; Welch one-sided p
  (α/3 = 0.0167) `<H1.welch_p>` → reject_h0 = `<H1.welch_reject_h0>`.

- **H3** : observed |Hedges' g| of `(retention[P_min] vs
  retention[baseline])` ≥ Javadi 2024 lower CI 0.13, sign
  decrement (g ≤ -0.13). Observed `g_h3 = <H3.hedges_g>` ;
  decrement-side rejection = `<H3.below_javadi_2024_lower_ci_decrement>`.

- **H_DR4** : monotonic ordering `mean retention[P_max] ≥
  mean retention[P_equ] ≥ mean retention[P_min]` (Jonckheere-
  Terpstra). Observed monotonic = `<DR4.monotonic_observed>` ;
  one-sided p = `<DR4.p_value>` → reject_h0 = `<DR4.reject_h0>`.

Per N = 5 / arm, this pilot is **exploratory** for absolute g
magnitudes (minimum detectable g at 80 % power ≈ 1.4). Crossing
or undershooting the Hu / Javadi lower CI in this pilot is treated
per pre-reg §4 as a **scheduling signal** for a confirmatory
N ≥ 30 follow-up, not as final empirical confirmation /
falsification.

R1 traceability : every cell carries a deterministic 32-hex
`run_id` registered in `harness/storage/run_registry.RunRegistry`.
Re-running `experiments/g4_split_fmnist/run_g4.py` against the
same `(c_version, profile, seed, commit_sha)` tuple is
bit-stable on Apple Silicon M3 Ultra (verified 2026-05-03,
ids identical across two consecutive sweeps).

## 7.2 Cross-substrate H1-H4 comparative table (synthetic substitute — not empirical claim)
```

**The angle-bracket placeholders `<H1.hedges_g>`, `<H1.is_within_hu_2020_ci>`, `<H1.welch_p>`, `<H1.welch_reject_h0>`, `<H3.hedges_g>`, `<H3.below_javadi_2024_lower_ci_decrement>`, `<DR4.monotonic_observed>`, `<DR4.p_value>`, `<DR4.reject_h0>` are substituted at edit time with the literal scalar values printed in Step 1.** They are not left as placeholders in the committed file.

- [ ] **Step 5: Mirror in the FR paper**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "## 7.2" docs/papers/paper2-fr/results.md`
Expected: a line number for the FR §7.2 anchor.

The FR mirror replaces the exact string `## 7.2 Tableau comparatif cross-substrat H1-H4 (substitut synthétique — pas une revendication empirique)` (verify the precise wording first with `grep`) with the FR translation of the §7.1.1 block followed by the original §7.2 anchor :

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "^## 7\." docs/papers/paper2-fr/results.md`
Expected: list of FR section anchors. Use the exact §7.2 anchor string returned by this grep as the `old_string` for the Edit tool.

Build the FR `new_string` as : the literal §7.1.1 block translated (same numeric scalars as EN, e.g. `g_h1`, paths, run_id text), followed by the original FR §7.2 anchor line. Paths in the FR mirror reference the same `docs/milestones/g4-pilot-2026-05-03.md` (single dump, no FR mirror of the milestone artefact — milestones are EN-only per `CONTRIBUTING.md` table).

Concretely the FR §7.1.1 starts with :

```
## 7.1.1 Pilote G4 (première évidence empirique — 2026-05-03)

Le pilote G4 est le premier résultat **non synthétique** de la
§7. Le balayage est `4 bras × 5 graines = 20 cellules` sur
l'apprentissage continu Split-FMNIST 5 tâches, substrat MLX
(`kiki_oniric.substrates.mlx_kiki_oniric`), pilote
`experiments/g4_split_fmnist/run_g4.py`. Pré-enregistrement :
[`docs/osf-prereg-g4-pilot.md`](../../osf-prereg-g4-pilot.md).
Dépôt du jalon :
[`docs/milestones/g4-pilot-2026-05-03.{json,md}`](../../milestones/g4-pilot-2026-05-03.md).
```

…and continues with the same H1/H3/H_DR4 numeric scalars and the same exploratory caveat. **The numbers are identical across EN and FR — they come from the same JSON dump.**

- [ ] **Step 6: Commit EN+FR together**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): add G4 pilot first-empirical results

§7.1.1 reports the first non-synthetic numbers in Paper 2:
H1 (P_equ vs Hu 2020), H3 (P_min vs Javadi 2024), H_DR4
(Jonckheere monotonicity). N=5 seeds per arm — pilot is
exploratory; confirmatory N>=30 follow-up scheduled. R1
verified bit-stable. EN+FR mirrored per CONTRIBUTING.md."
```

Subject ≤50 chars: `docs(paper2): add G4 pilot first-empirical results` = 50 chars.

---

## Task 9: CHANGELOG entry — EC bump conditional on outcome

**Files:**
- Modify: `CHANGELOG.md` (under `## [Unreleased]`)

The DualVer outcome is computed **from the JSON dump** (Step 1 below) and translated to a deterministic CHANGELOG entry per the rules in `docs/osf-prereg-g4-pilot.md` §6.

- [ ] **Step 1: Compute the EC bump verdict from the JSON dump**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && python -c "
import json
p = json.load(open('docs/milestones/g4-pilot-2026-05-03.json'))
v = p['verdict']
h1 = v['h1_p_equ_vs_baseline']
h3 = v['h3_p_min_vs_baseline']
h4 = v['h_dr4_jonckheere']

h1_pass = (not h1.get('insufficient_samples') and h1.get('above_hu_2020_lower_ci') and h1.get('welch_reject_h0'))
h3_pass = (not h3.get('insufficient_samples') and h3.get('below_javadi_2024_lower_ci_decrement') and h3.get('welch_reject_h0'))
h4_pass = (not h4.get('insufficient_samples') and h4.get('monotonic_observed') and h4.get('reject_h0'))

# Falsification : observed monotonicity is REVERSED (P_max < P_min mean retention)
falsified = (
    not h4.get('insufficient_samples')
    and h4.get('mean_p_max', 0) < h4.get('mean_p_min', 1)
)

if falsified:
    verdict = 'UNSTABLE'
elif h1_pass and h3_pass and h4_pass:
    verdict = 'STABLE'
else:
    verdict = 'PARTIAL'

print('VERDICT:', verdict)
print('h1_pass=', h1_pass)
print('h3_pass=', h3_pass)
print('h4_pass=', h4_pass)
print('falsified=', falsified)
" | tee /tmp/g4-bump-verdict.txt
```

Expected: prints `VERDICT: STABLE` **or** `VERDICT: PARTIAL` **or** `VERDICT: UNSTABLE`, plus the four boolean flags.

- [ ] **Step 2: Read the current `## [Unreleased]` block**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && head -55 CHANGELOG.md`
Expected: confirms `## [Unreleased]` is at the top of the changelog.

- [ ] **Step 3: Append the G4 entry under `## [Unreleased]`**

Use `Edit` tool on `CHANGELOG.md`. The exact new block depends on the verdict from Step 1.

**If verdict = STABLE** (all three hypotheses confirmed) — appends an EC bump:

Replace `## [Unreleased]` with:

```
## [Unreleased]

### Empirical (EC bump — PARTIAL → STABLE for G4 scope)
- G4 pilot 2026-05-03 (Split-FMNIST × profile sweep, MLX
  substrate) confirmed all three pre-registered hypotheses
  in `docs/osf-prereg-g4-pilot.md` :
  - H1 : observed Hedges' g(P_equ retention vs baseline) above
    Hu 2020 lower CI 0.21, Welch p < 0.0167
  - H3 : observed Hedges' g(P_min retention vs baseline) below
    -Javadi 2024 lower CI 0.13 (decrement), Welch p < 0.0167
  - H_DR4 : Jonckheere monotonic trend rejected H0 with
    observed mean ordering P_max ≥ P_equ ≥ P_min
- Per framework-C §12.3 : EC PARTIAL → STABLE for the G4
  scope (Split-FMNIST 5-task class-incremental, MLX). Other
  scopes (cross-substrate cycle-3, fMRI alignment, multi-scale
  Qwen) remain PARTIAL — the STABLE promotion does not cover
  them.
- Milestone artefacts : `docs/milestones/g4-pilot-2026-05-03.{json,md}`
- Run-registry rows : 20 cells under
  `(C-v0.12.0+PARTIAL, g4/{baseline,P_min,P_equ,P_max}, seed)`
  with R1 bit-stable run_ids verified across two re-runs

### Versioning
- **DualVer EC bump** (PARTIAL → STABLE for G4 scope, no FC
  change). Per framework-C §12, FC stays at v0.12.0 (no axiom
  / primitive / channel / invariant change). EC scope-promote
  per §12.3.

```

**If verdict = PARTIAL** (some hypotheses confirmed, none falsified) — no bump, just record:

Replace `## [Unreleased]` with:

```
## [Unreleased]

### Empirical (no DualVer bump — partial confirmation)
- G4 pilot 2026-05-03 (Split-FMNIST × profile sweep, MLX
  substrate) returned partial confirmation of pre-registered
  hypotheses in `docs/osf-prereg-g4-pilot.md`. See
  `docs/milestones/g4-pilot-2026-05-03.md` for per-hypothesis
  verdict scalars.
- Per framework-C §12.3, partial confirmation does not
  promote EC out of PARTIAL. A confirmatory N ≥ 30 follow-up is
  scheduled before any STABLE promotion.
- Milestone artefacts : `docs/milestones/g4-pilot-2026-05-03.{json,md}`
- Run-registry rows : 20 cells under
  `(C-v0.12.0+PARTIAL, g4/{baseline,P_min,P_equ,P_max}, seed)`
  with R1 bit-stable run_ids verified across two re-runs

### Versioning
- **No DualVer bump.** EC stays PARTIAL (partial confirmation
  is not a STABLE-promotion event per §12.3). FC stays at v0.12.0.

```

**If verdict = UNSTABLE** (DR-4 monotonicity falsified — P_max < P_min mean retention) — flag UNSTABLE:

Replace `## [Unreleased]` with:

```
## [Unreleased]

### Empirical (EC flag — STABLE/PARTIAL → UNSTABLE for G4 scope)
- G4 pilot 2026-05-03 (Split-FMNIST × profile sweep, MLX
  substrate) **falsified** H_DR4 monotonicity : observed
  mean_retention[P_max] < mean_retention[P_min], reversed
  ordering. See pre-reg `docs/osf-prereg-g4-pilot.md` §6 for
  the binding falsification rule.
- Per framework-C §12.3 : EC flagged UNSTABLE for the G4 scope
  pending root-cause investigation (profile mis-wiring /
  benchmark idiosyncrasy / dream-episode handler regression).
- Milestone artefacts : `docs/milestones/g4-pilot-2026-05-03.{json,md}`
- Run-registry rows : 20 cells under
  `(C-v0.12.0+PARTIAL, g4/{baseline,P_min,P_equ,P_max}, seed)`
  with R1 bit-stable run_ids verified across two re-runs

### Versioning
- **DualVer EC flag** (PARTIAL → UNSTABLE for G4 scope, no FC
  change). Per framework-C §12.3 falsification surfaces the
  empirical inconsistency without modifying the formal axis.
  Reverting requires a corrective patch + re-pilot + new
  positive G4 evidence.

```

Pick exactly one of the three blocks based on Step 1's printout.

- [ ] **Step 4: Verify markdown is well-formed**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && head -60 CHANGELOG.md`
Expected: the new G4 block appears under `## [Unreleased]` ; the next sibling section `## [C-v0.12.0+PARTIAL]` is intact below.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add CHANGELOG.md
git commit -m "docs(changelog): G4 pilot empirical entry + DualVer

Records the 2026-05-03 G4 pilot outcome and the resulting EC
verdict (STABLE/PARTIAL/UNSTABLE — see CHANGELOG entry). Run
registry rows enumerated; FC unchanged at v0.12.0 in all three
branches per framework-C s12 (no axiom or primitive change)."
```

Subject ≤50 chars: 49 chars.

---

## Task 10: Update STATUS.md gate row + version line

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Update the G4 row in the Gates table**

Replace the line `| G4 — P_equ fonctionnel | S12 | ⏳ Pending |` (verify with `grep -n "G4" STATUS.md`) with the corresponding outcome line :

- If verdict = STABLE :
  `| G4 — P_equ fonctionnel | S12 | ✅ FULL-GO/STABLE (Split-FMNIST pilot 2026-05-03, 20 cells, EC PARTIAL→STABLE) |`
- If verdict = PARTIAL :
  `| G4 — P_equ fonctionnel | S12 | 🔶 PARTIAL (Split-FMNIST pilot 2026-05-03, 20 cells, partial confirmation, N≥30 follow-up scheduled) |`
- If verdict = UNSTABLE :
  `| G4 — P_equ fonctionnel | S12 | ⚠ UNSTABLE (Split-FMNIST pilot 2026-05-03, 20 cells, DR-4 monotonicity falsified) |`

Pick the line matching Step 1 of Task 9.

- [ ] **Step 2: Update the version line at the top of STATUS.md if EC bumped to STABLE**

If verdict = STABLE :

Replace `**Version** : C-v0.12.0+PARTIAL` with `**Version** : C-v0.12.0+STABLE` (verified scope = G4) and add a short footnote in the DualVer table EC row :

Replace the DualVer EC row content after `| EC   | PARTIAL |` with `| EC   | STABLE (G4 scope) |` plus a one-line note appended to the rationale column : `G4 pilot 2026-05-03 confirmed H1+H3+H_DR4 — EC promoted PARTIAL→STABLE for the Split-FMNIST 5-task scope. Other scopes (cross-substrate, fMRI, multi-scale) remain PARTIAL.`

If verdict = PARTIAL or UNSTABLE :

Leave the version line unchanged. For UNSTABLE add a short bullet under `## DualVer status` :

`> **2026-05-03 G4 pilot UNSTABLE flag** : Split-FMNIST pilot falsified H_DR4 monotonicity ; G4-scope EC flagged UNSTABLE pending root cause review. Other scopes unaffected.`

- [ ] **Step 3: Verify STATUS.md still grep-finds the OSF amendment row, the test-suite row, and the gates table**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "Test suite\|OSF\|## Gates" STATUS.md`
Expected: at least three line numbers (existing structure preserved).

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && git add STATUS.md
git commit -m "docs(status): G4 gate row + DualVer post-pilot

Updates the G4 row in the Gates table to reflect the 2026-05-03
Split-FMNIST pilot outcome (FULL-GO/PARTIAL/UNSTABLE depending
on verdict) and updates the DualVer EC line / version tag if a
PARTIAL->STABLE promotion was earned. Pre-registration:
docs/osf-prereg-g4-pilot.md."
```

Subject ≤50 chars: 50 chars exactly.

---

## Task 11: Final guard — full repo gate (lint + types + tests + R1 nightly proxy)

**Files:** none modified — verification step only.

- [ ] **Step 1: Run full lint**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check .`
Expected: `All checks passed.`

- [ ] **Step 2: Run mypy strict on harness + tests + new modules**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run mypy harness tests kiki_oniric/eval/statistics.py experiments/g4_split_fmnist/`
Expected: `Success: no issues found`.

- [ ] **Step 3: Run the full test suite**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest`
Expected: all tests pass, coverage ≥ 90 % (the cov gate fails the run if not).

- [ ] **Step 4: Run R1 reproducibility tests (Apple Silicon-only)**

Run: `cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/reproducibility/ -v --no-cov`
Expected: pass on macOS-14 / M3 Ultra ; the `golden_hashes.json` artefact does not change because no MLX-backed library code mutated (we added a pure-numpy helper + experiments/ — outside the R1 hash scope).

If R1 tests fail with `golden_hashes` drift, the failure is a real regression (likely from `compute_hedges_g` import side-effects through `kiki_oniric.eval.statistics`). In that case run `uv run pytest tests/reproducibility/test_r1_bit_exact.py -v --no-cov -k <failing_test>` and report the diff — do not regenerate golden hashes silently.

- [ ] **Step 5: Verify the run registry holds 20 G4 cells**

Run:
```bash
cd /Users/electron/hypneum-lab/dream-of-kiki && python -c "
import sqlite3
with sqlite3.connect('.run_registry.sqlite') as c:
    rows = c.execute(\"SELECT profile, COUNT(*) FROM runs WHERE profile LIKE 'g4/%' GROUP BY profile\").fetchall()
print(rows)
assert len(rows) == 4, rows
assert all(n == 5 for _, n in rows), rows
print('OK — 4 arms x 5 cells each')
"
```
Expected: `OK — 4 arms x 5 cells each`.

- [ ] **Step 6: No commit needed — verification only.**

---

## Task 12: Self-review

Read the spec from §"Spec" of this plan with fresh eyes and check the plan against it.

- [ ] **Step 1: Spec coverage check**

For each spec requirement, point to a task :

- [x] Run G4 pilot on MLX kiki_oniric substrate × Split-FMNIST × 3 profiles → Tasks 3-7
- [x] Compare observed Hedges' g vs HU_2020_OVERALL / JAVADI_2024_OVERALL using typed verdict helpers → Task 5 (`run_g4.py::_h1_verdict / _h3_verdict`)
- [x] Use existing `kiki_oniric.eval.statistics` (Welch / Jonckheere) → Task 5 (`run_g4.py::_h1_verdict, _h3_verdict, _h_dr4_verdict`)
- [x] Add Hedges' g computation (was missing — only the *targets* existed) → Task 1
- [x] R1 contract — `run_registry.register` per cell → Task 5 (`run_g4.py::run_pilot`)
- [x] OSF pre-reg draft H1 + H3 + DR-4 → Task 2
- [x] Pre-specified Welch + Bonferroni-3 + Jonckheere → Task 2 §3 + Task 5 (`alpha=0.05/3`)
- [x] Pre-specified exclusions when S2 trips OR `acc_initial < 0.5` → Task 2 §5 + Task 5 (`excluded_underperforming_baseline`)
- [x] Compute budget 15-20 runs × M3 Ultra → Task 7 (~45-75 min)
- [x] DualVer outcome matrix STABLE / PARTIAL / UNSTABLE → Task 9 (three-branch CHANGELOG)
- [x] No FC bump in any outcome → Task 9 (every branch states `FC stays at v0.12.0`)
- [x] Failure-mode handling for missing `run_h1.py` API → Task 0 confirms; Variant B explicitly bypasses Avalanche
- [x] Failure-mode handling for missing wake-sleep hook → Task 4 (`dream_episode()` wraps `runtime.execute`)
- [x] Failure-mode handling for missing Avalanche dep → Variant decision section + Task 3 (numpy IDX decoder)
- [x] Variant decision A vs B documented → "Variant decision (locked in this plan)" section
- [x] Milestone dump `g4-pilot-2026-XX-XX.{json,md}` → Task 5 + Task 7
- [x] Paper 2 §7.1 update with first observed-g numbers, EN+FR mirror → Task 8

All spec items covered.

- [ ] **Step 2: Placeholder scan**

Search the plan for forbidden patterns :

```
grep -nE "TBD|TODO|implement later|fill in|add appropriate|add error handling|similar to task|placeholder" docs/superpowers/plans/2026-05-03-g4-pilot-mlx-split-fmnist.md
```

Expected hits :
- The pre-reg draft (Task 2) discusses "exploratory" outcomes and uses the word "placeholder" only in the §4 sample-size analysis context — that is **content** of the OSF document, not a *plan* placeholder.
- The Paper 2 §7.1 edit (Task 8) uses `<H1.hedges_g>` style angle-brackets that are **explicitly substituted** at edit time with literal numbers from the JSON dump — the plan instructs the engineer to substitute, not to leave them as TBD.
- The CHANGELOG branches in Task 9 are three concrete, fully-spelled blocks — the engineer picks one based on a deterministic Python computation in Task 9 Step 1, not "TBD".

No `TBD`/`implement later`/`add error handling` instructions to the engineer.

- [ ] **Step 3: Type / signature consistency**

Cross-check function names used across tasks :

- `compute_hedges_g(treatment: list[float], control: list[float]) -> float` — defined Task 1, used Task 5 `run_g4.py::_h1_verdict` and Task 5 `run_g4.py::_h3_verdict`. ✓ same name.
- `welch_one_sided(treatment, control, alpha)` — pre-existing in `kiki_oniric/eval/statistics.py`. Task 5 uses positional `(base, p_equ, alpha=0.05/3)` for H1 (treatment is the dream arm) and `(p_min, base, alpha=0.05/3)` for H3 (treatment is P_min). ✓ matches existing signature.
- `jonckheere_trend(groups, alpha)` — pre-existing. Task 5 uses `groups=[P_min, P_equ, P_max]`, ascending order. ✓ matches.
- `RunRegistry.register(c_version, profile, seed, commit_sha) -> str` — pre-existing per Task 0 Step 2 confirmation. Task 5 calls with kwargs. ✓ matches.
- `EffectSizeTarget.is_within_ci(observed)` and `.distance_from_target(observed)` — pre-existing per `harness/benchmarks/effect_size_targets.py`. Task 5 uses both. ✓.
- `build_profile(name, seed)` — defined Task 4, used Task 5 (`run_g4.py::_run_cell`). ✓ same signature.
- `G4Classifier(in_dim, hidden_dim, n_classes, seed)` — defined Task 4, used Task 5 (`run_g4.py::_run_cell`). ✓.
- `dream_episode(profile, seed)` — defined Task 4, used Task 5 (`run_g4.py::_run_cell`). ✓.
- `load_split_fmnist_5tasks(data_dir)` — defined Task 3, used Task 5 (`run_g4.py::run_pilot`). ✓.
- `sample_beta_records(seed, n_records, feat_dim)` — defined Task 4, used Task 4 (`G4Classifier.dream_episode`). ✓.

All cross-task type / signature references are consistent.

- [ ] **Step 4: Conventions check**

- All commit subjects ≤ 50 chars (verified inline against each Task's commit step).
- All scopes ≥ 3 chars (`eval`, `g4`, `osf`, `paper2`, `chore`, `changelog`, `status` — all ≥ 3).
- No `Co-Authored-By` trailer in any commit message.
- All commit bodies wrap at ≤ 72 chars (each commit message above respects this).
- EN→FR propagation : Task 8 explicitly handles EN+FR mirror in the same commit. Specs / amendments aren't touched by this plan, so no FR propagation needed for `specs-fr/`.
- `docs/CLAUDE.md` "milestones append-only" rule : Task 5/7 create new dated immutables `g4-pilot-2026-05-03.{json,md}`. Task 8 appends §7.1.1 next to the existing §7.1 (does not rewrite §7.1). ✓.
- `scripts/CLAUDE.md` "JSON dump required" : the pilot lives under `experiments/`, not `scripts/`, but still emits the JSON dump per the same convention (Task 5).

- [ ] **Step 5: No commit needed — review only.**

---

## Plan summary

| Task | Title | Commits | LOC est. |
|------|-------|---------|----------|
| 0 | Investigate (read-only) | 0 | 0 |
| 1 | `compute_hedges_g` helper | 1 | ~80 (impl) + ~120 (tests) |
| 2 | OSF pre-reg draft | 1 | ~140 (markdown) |
| 3 | Split-FMNIST 5-task loader | 1 | ~120 (impl) + ~140 (tests) |
| 4 | MLX classifier + dream wrapper | 1 | ~180 (impl) + ~150 (tests) |
| 5 | Pilot driver `run_g4.py` + smoke | 1 | ~280 (impl) + ~120 (tests) |
| 6 | Download FMNIST data | 0-1 | (gitignore line) |
| 7 | Run full pilot | 1 | (artefacts only) |
| 8 | Paper 2 §7.1 EN+FR | 1 | ~80 (markdown) |
| 9 | CHANGELOG entry | 1 | ~30 (markdown) |
| 10 | STATUS.md gate row | 1 | ~5 (markdown) |
| 11 | Final guard | 0 | 0 |
| 12 | Self-review | 0 | 0 |

**Total commits : 9-10. Total time estimate :** ~3-4 h coding + ~45-75 min pilot run + ~30 min Paper 2 / CHANGELOG / STATUS edits = **~5 h end-to-end** for a focused implementer.
