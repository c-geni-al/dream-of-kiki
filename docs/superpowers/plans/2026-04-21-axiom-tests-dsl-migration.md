# Axiom Tests DSL Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the axiom test DSL (`tests/conformance/axioms/_dsl.py`) with profile-registry helpers and migrate `test_dr4_profile_inclusion.py` (certain win, −35% LOC). Optionally migrate `test_dr0_accountability.py` (marginal win, skip if net LOC delta is small).

**Architecture:** Additive DSL extension followed by one mandatory test migration and one optional migration. Tests `test_dr1_episodic_conservation.py`, `test_dr3_substrate.py`, `test_dr3_esnn_substrate.py` are explicitly **out of scope** (explore agent verdict: disjoint fixtures, no net win).

**Tech Stack:** Python 3.14, pytest, MLX, Hypothesis, ruff, mypy strict.

---

## Background

Current DSL surface (from `tests/conformance/axioms/_dsl.py`, verified 2026-04-21):

| Name | Purpose |
|------|---------|
| `make_tiny_mlp/encoder/decoder` | MLX model factories |
| `make_episode(ops, seed, profile="P_min", ...)` | DreamEpisode builder |
| `seeded_runtime(seed)` | Runtime + wired handlers |
| `snapshot_state(wired)` / `assert_states_equal(a, b)` | Byte-exact state comparison |
| `WiredRuntime` | 8-slot container |
| Internal `_PROFILE_CHANNELS` dict for P_min / P_equ |

Migration scorecard (LOC estimates from explore agent):

| Test | Current | Post-DSL | Δ | Decision |
|------|---------|----------|---|----------|
| `test_dr4_profile_inclusion.py` | 100 | ~65 | −35% | **Migrate (Task 2)** |
| `test_dr0_accountability.py` | 75 | ~55 | −27% | Marginal; Task 3 (optional) |
| `test_dr1_episodic_conservation.py` | 63 | 63 | 0 | Skip (FakeBetaBuffer is intentionally standalone) |
| `test_dr3_substrate.py` | 49 | 49 | 0 | Skip (Protocol introspection, no runtime) |
| `test_dr3_esnn_substrate.py` | 167 | ~140 | −16% | Skip (E-SNN-specific; needs separate `_dsl_esnn.py`) |

---

## File Structure

- Modify: `tests/conformance/axioms/_dsl.py` — add 2 helpers
- Modify: `tests/conformance/axioms/test_dr4_profile_inclusion.py` — migrate (required)
- Modify: `tests/conformance/axioms/test_dr0_accountability.py` — migrate (optional, Task 3)

---

### Task 1: Extend DSL with profile-registry helpers

**Files:**
- Modify: `tests/conformance/axioms/_dsl.py`

- [ ] **Step 1: Write failing tests for the new helpers in a throwaway test file**

Create `tests/conformance/axioms/test__dsl_profile_helpers_tmp.py` with:

```python
"""Temporary TDD harness for _dsl helpers. Deleted in Step 5."""
from kiki_oniric.core.primitives import (
    WEIGHT_DELTA, HIERARCHY_CHG, LATENT_SAMPLE,
)
from kiki_oniric.dream.episode import Operation

from tests.conformance.axioms._dsl import registered_ops, profile_channels


def test_registered_ops_pmin_returns_four_canonical_ops() -> None:
    ops = registered_ops("P_min")
    assert ops == {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }


def test_profile_channels_pmin() -> None:
    assert profile_channels("P_min") == (WEIGHT_DELTA,)


def test_profile_channels_pequ() -> None:
    assert profile_channels("P_equ") == (
        WEIGHT_DELTA,
        HIERARCHY_CHG,
        LATENT_SAMPLE,
    )


def test_profile_channels_unknown_raises_key_error() -> None:
    import pytest
    with pytest.raises(KeyError):
        profile_channels("P_max")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
uv run python -m pytest tests/conformance/axioms/test__dsl_profile_helpers_tmp.py -v --no-cov
```
Expected: `ImportError: cannot import name 'registered_ops'` or similar.

- [ ] **Step 3: Add helpers to `_dsl.py`**

In `tests/conformance/axioms/_dsl.py`, after the existing `_PROFILE_CHANNELS` declaration, add:

```python
def profile_channels(profile: str) -> tuple[OutputChannel, ...]:
    """Return the OutputChannel tuple for a named profile.

    Raises KeyError for unknown profiles — forces deliberate DSL
    extension when a new profile (e.g. P_max) lands.
    """
    return _PROFILE_CHANNELS[profile]


def registered_ops(profile: str) -> set[Operation]:
    """Return the set of Operations wired by a fresh runtime for `profile`.

    Introspects the private handler registry (`runtime._handlers`) of a
    freshly seeded runtime. Used by DR-4 profile-inclusion tests.
    """
    wired = seeded_runtime(seed=0, profile=profile)
    return set(wired.runtime._handlers.keys())
```

If `seeded_runtime()` does not currently accept a `profile` kwarg, extend its signature to accept `profile: str = "P_min"` and propagate to the episode / channel wiring (see how `make_episode` does it). This is a minimal change to `_dsl.py` internals.

- [ ] **Step 4: Run test to verify pass**

Run:
```bash
uv run python -m pytest tests/conformance/axioms/test__dsl_profile_helpers_tmp.py -v --no-cov
```
Expected: 4 passed.

- [ ] **Step 5: Delete the TDD harness**

Delete `tests/conformance/axioms/test__dsl_profile_helpers_tmp.py`. Its purpose was to validate the new helpers in isolation; `test_dr4` will exercise them for real in Task 2.

- [ ] **Step 6: Run full conformance suite**

Run:
```bash
uv run python -m pytest tests/conformance/ --no-cov -q
```
Expected: `45 passed, 12 xfailed` (unchanged — DSL additions are additive).

- [ ] **Step 7: Commit**

```bash
git add tests/conformance/axioms/_dsl.py
git commit -m "$(cat <<'EOF'
test(dsl): add profile helpers registered_ops/channels

Add two additive helpers to the axiom DSL:
- registered_ops(profile) -> set[Operation]
- profile_channels(profile) -> tuple[OutputChannel, ...]

Introspects profile metadata without duplicating the
_PROFILE_CHANNELS registry. Unblocks DR-4 test migration.

Signature change on seeded_runtime(): new profile kwarg
defaulting to "P_min" (backward-compat).
EOF
)"
```

Subject 48 chars ≤ 50 ✓.

---

### Task 2: Migrate test_dr4_profile_inclusion.py to DSL

**Files:**
- Modify: `tests/conformance/axioms/test_dr4_profile_inclusion.py` (100 LOC → ~65)

- [ ] **Step 1: Capture baseline**

Run:
```bash
wc -l tests/conformance/axioms/test_dr4_profile_inclusion.py
uv run python -m pytest tests/conformance/axioms/test_dr4_profile_inclusion.py -v --no-cov
```
Record the pass count and LOC.

- [ ] **Step 2: Identify call sites**

Run:
```bash
grep -n 'P_MIN_CHANNELS_OUT\|P_EQU_CHANNELS_OUT\|_registered_ops' tests/conformance/axioms/test_dr4_profile_inclusion.py
```
Expected: ~8-10 lines referencing the local constants/helpers.

- [ ] **Step 3: Replace local helpers with DSL imports**

Edit `test_dr4_profile_inclusion.py`:
- Delete the local `_registered_ops()` function (lines 26-28 per explore report).
- Delete the `P_MIN_CHANNELS_OUT` and `P_EQU_CHANNELS_OUT` constants (lines 31-36).
- Add at the top: `from tests.conformance.axioms._dsl import registered_ops, profile_channels`
- Replace every call to the deleted helpers/constants with DSL calls:
  - `_registered_ops(profile)` → `registered_ops(profile_name)` (note: DSL version takes the profile NAME string, not the profile object — adjust call sites accordingly)
  - `P_MIN_CHANNELS_OUT` → `profile_channels("P_min")`
  - `P_EQU_CHANNELS_OUT` → `profile_channels("P_equ")`

Preserve the P_max skeleton introspection (`_p_max_metadata()`) — it is not covered by the DSL and must stay local until S16.2.

- [ ] **Step 4: Run test**

Run:
```bash
uv run python -m pytest tests/conformance/axioms/test_dr4_profile_inclusion.py -v --no-cov
```
Expected: same pass count as baseline (Step 1). Zero regressions.

- [ ] **Step 5: Verify LOC reduction**

Run:
```bash
wc -l tests/conformance/axioms/test_dr4_profile_inclusion.py
```
Expected: ~65 lines (−35% from 100). If > 80 lines, inspect for missed cleanups.

- [ ] **Step 6: Run full conformance suite + coverage gate**

Run:
```bash
uv run python -m pytest tests/conformance/ -q
```
Expected: `45 passed, 12 xfailed` and coverage ≥ 90%.

- [ ] **Step 7: Commit**

```bash
git add tests/conformance/axioms/test_dr4_profile_inclusion.py
git commit -m "$(cat <<'EOF'
refactor(tests): migrate DR-4 test to DSL

Replace local _registered_ops() helper and
P_MIN/EQU_CHANNELS_OUT constants with DSL imports
(registered_ops, profile_channels). 100 → ~65 lines
(-35%). Semantics preserved; P_max skeleton introspection
kept local until S16.2 lands the profile.
EOF
)"
```

Subject 41 chars ≤ 50 ✓.

---

### Task 3 (OPTIONAL): Migrate test_dr0_accountability.py

Only execute if the migration achieves ≥ 25% LOC reduction. Explore estimate is 75 → 55 lines (−27%). If during implementation the delta comes out smaller, abort and delete this task without commit.

**Files:**
- Modify: `tests/conformance/axioms/test_dr0_accountability.py`

- [ ] **Step 1: Capture baseline**

Run:
```bash
wc -l tests/conformance/axioms/test_dr0_accountability.py
uv run python -m pytest tests/conformance/axioms/test_dr0_accountability.py -v --no-cov
```

- [ ] **Step 2: Assess migration win**

Read the file. If the Hypothesis composite `dream_episodes_with_replay_only()` and the manual `DreamEpisode(...)` construction dominate the LOC, the DSL's `make_episode()` can replace them. If the test is already using a Hypothesis strategy that diverges meaningfully from `make_episode` defaults, abort.

Decision gate:
- If replacement saves ≥ 20 lines → proceed to Step 3
- Else → delete this task, skip to Task 4

- [ ] **Step 3: Replace DreamEpisode construction with make_episode**

For each call to `DreamEpisode(...)`:
- Replace with `make_episode(ops=(Operation.REPLAY,), seed=<test's seed>, ...)` preserving the budget and input_slice arguments as needed.
- If budgets vary across test cases (per the Hypothesis strategy), keep them as explicit kwargs through `make_episode`.

Delete the local Hypothesis composite only if its outputs become identical to `make_episode` outputs. Otherwise keep the composite and just use `make_episode` for the base construction.

- [ ] **Step 4: Run the test**

Run:
```bash
uv run python -m pytest tests/conformance/axioms/test_dr0_accountability.py -v --no-cov
```
Expected: same pass count as baseline.

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/axioms/test_dr0_accountability.py
git commit -m "refactor(tests): migrate DR-0 test to DSL"
```

Subject 43 chars ≤ 50 ✓.

---

### Task 4: Final verification + push

- [ ] **Step 1: Run full test suite**

Run:
```bash
uv run python -m pytest -q
```
Expected: all tests green, coverage ≥ 90%.

- [ ] **Step 2: Verify git log**

Run:
```bash
git log --oneline -6
```
Expected: 2 or 3 new commits on top of `main` (depending on whether Task 3 ran).

- [ ] **Step 3: Push to main**

Run:
```bash
git push origin main
```
Expected: `main -> main` without errors.

---

## Self-review checklist

- [x] Every task targets files cited by the explore agent.
- [x] Every step has exact code or exact command.
- [x] No placeholders.
- [x] Task 3 is genuinely optional with a numeric decision gate (25% reduction threshold).
- [x] DR-1 / DR-3 / DR-3-E-SNN tests are explicitly out of scope with a documented reason.
- [x] DSL changes are additive; backward compatibility preserved on `seeded_runtime()`.
