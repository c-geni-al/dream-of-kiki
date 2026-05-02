# P_min Calibration to Sharon 2025 SO-trough Biomarker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the `PMinProfile` substrate-degraded profile against the slow-oscillation (SO) trough amplitude / frontocentral synchronisation gradient documented qualitatively in Sharon et al., *Alzheimer's & Dementia* 2025 (hd-EEG, N=55: 21 healthy older / 28 aMCI / 6 AD). Introduce a measurable, seed-reproducible `so_trough_amplitude_factor` parameter on the three profiles `P_min` / `P_equ` / `P_max`, expose a proxy reader, prove monotonicity `P_max ≥ P_equ ≥ P_min`, and pin §6.6 of `docs/papers/paper1/methodology.md` to the new calibration. The exercise is qualitative (Sharon 2025 reports a gradient, not absolute SO-trough amplitude in µV); the calibrated values are an *informed placeholder* whose final empirical value lands at G2 / G4 pilots — the plan documents that explicitly in docstrings.

**Non-goals:**
- We do **not** edit handlers, channels, or operation sets — DR-4 chain inclusion stays byte-identical.
- We do **not** introduce a new axiom, invariant, or primitive — this is a calibration parameter on the existing dataclasses, not a framework C extension.
- We do **not** trigger an FC bump (no axiom statement, primitive signature, channel set, or invariant ID changes). EC axis stays `PARTIAL`.
- We do **not** weaken or rewrite `tests/conformance/axioms/test_dr4_profile_inclusion.py` — calibration is additive and must coexist with existing DR-4 ops/channels assertions.

**Architecture:**
- Add a single float field `so_trough_amplitude_factor: float` to each profile dataclass. Default semantics:
  - `P_max`: `1.0` (intact substrate, healthy-young anchor)
  - `P_equ`: `1.0` (intact substrate, healthy-older anchor — matches Sharon's healthy-older arm baseline)
  - `P_min`: `0.45` (degraded substrate, aMCI-equivalent midpoint informed by the qualitative cognitive-performance / SO-coherence gradient in Sharon 2025; AD value `~0.20` is **not** baked in — `P_min` represents the aMCI midpoint per current spec §3.1)
- Introduce `kiki_oniric/profiles/so_calibration.py` with:
  - `compute_so_amplitude_proxy(profile) -> float` — reads `so_trough_amplitude_factor` (or raises `TypeError` for non-profiles).
  - Module-level constants `SHARON_2025_HEALTHY_OLDER_ANCHOR = 1.0`, `SHARON_2025_AMCI_MIDPOINT = 0.45`, `SHARON_2025_AD_FLOOR = 0.20` — citation-pinned, used by tests.
  - Bibliography pin at module-docstring: `Reference: Sharon et al. 2025 (sharon2025alzdementia in references.bib)`.
- New unit-test module `tests/unit/profiles/test_p_min_sharon_calibration.py` covers: (a) calibrated default value, (b) monotonicity `P_max ≥ P_equ ≥ P_min` via the proxy, (c) seeded reproducibility (calibration is deterministic across seeds — proxy is a constant, not RNG-driven), (d) docstring-citation presence (Sharon key string appears in the profile docstring).
- Methodology §6.6 already references `[@sharon2025alzdementia]` — the update converts the prose anchor into a code pointer (`kiki_oniric/profiles/p_min.py::so_trough_amplitude_factor`).

**Tech Stack:** Python 3.12+, uv, pytest 8, MLX (untouched here — calibration is substrate-side metadata only). Lint: `ruff` line-length 100, `mypy --strict`. Coverage gate ≥90% (already enforced in `pyproject.toml`).

**Versioning impact:**
- **FC axis:** *no bump* — no axiom statement edit, no primitive signature change, no channel set change, no invariant ID added. The calibration field is profile *metadata*, not a public API contract surface (it does not appear in `kiki_oniric/core/primitives.py` Protocols). Per `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12, FC bumps require a "proof or spec change". Calibration adds neither.
- **EC axis:** stays `PARTIAL` (no new gate result; the calibration is an informed placeholder, not a G2/G4 closure).
- **CHANGELOG.md:** a single entry under "Unreleased" documenting the calibration field + the Sharon anchor string. No version tag bump.

**Dependencies already DONE:**
- `kiki_oniric/profiles/p_min.py` (84 L), `p_equ.py` (73 L), `p_max.py` (112 L) — wired profiles with handlers + states (S9.4, S11.2, C2.8).
- `tests/conformance/axioms/test_dr4_profile_inclusion.py` — 6 tests covering ops/channels chain inclusion via `_dsl.profile_channels` / `_dsl.registered_ops`.
- `tests/conformance/axioms/_dsl.py` — DSL helpers including `_PROFILE_CLASSES = {"P_min": PMinProfile, "P_equ": PEquProfile}` (P_max instantiation lives inline in the test file).
- `docs/papers/paper1/references.bib:632` — `@article{sharon2025alzdementia, ...}` BibTeX entry confirmed present.
- `docs/papers/paper1/methodology.md` §6.6 already cites Sharon qualitatively (lines 119-123).

**Risks pre-flagged:**
1. **DR-4 monotonicity vs Lemma DR-4.L conflict.** Lemma DR-4.L (`docs/proofs/dr4-profile-inclusion.md` lines 62-91) states "if P_min satisfies all invariants on substrate S, then P_equ does not strictly worsen metrics monotone in capacity". Our `so_trough_amplitude_factor` is monotone in *capacity-equivalent biological substrate health* (more SO-trough amplitude = more replay-relevant slow-wave coherence = more consolidation capacity). A calibration where `P_min < P_equ < P_max` (numerically `0.45 < 1.0 = 1.0`) is **consistent** with DR-4.L (capacity-monotone metric, P_min lower → P_equ ≥ P_min holds with equality permissible). Task 0 *verifies* this by reading `test_dr4_profile_inclusion.py` to confirm no test currently asserts strict numeric ordering on a calibration field.
2. **Sharon 2025 has no extractable absolute number.** The bibliography note (`docs/papers/paper1/references.bib:632-641`) records the qualitative result only ("cognitive performance decreases with slow-wave trough amplitude"). The plan therefore commits to a **defensible informed-placeholder** value (`0.45` ≈ aMCI midpoint between healthy-older anchor `1.0` and AD-floor `0.2` *as a unit-arbitrary ratio*), with a docstring stating: "Calibrated qualitatively against the Sharon 2025 healthy-older / aMCI / AD gradient; absolute SO-trough µV not extractable from publication; final empirical value pending G2 P_min pilot (`scripts/pilot_g2.py`)."
3. **Existing P_max test (`test_dr4_ops_inclusion_p_equ_subset_p_max`) instantiates `PMaxProfile()` inline.** Adding a new field with a default keeps backward-compat (no constructor breakage). Tests pass without edits if the default is `1.0`.
4. **The `_dsl._PROFILE_CLASSES` dict only knows P_min and P_equ.** Our monotonicity test must instantiate `PMaxProfile` directly (mirroring the inline pattern of `_p_max_metadata()` in the DR-4 test file) rather than extending `_PROFILE_CLASSES` — the DSL extension is out-of-scope and would touch a load-bearing test helper.

**Conventions enforced (per `CONTRIBUTING.md` + repo CLAUDE.md):**
- Conventional commits, subject ≤ 50 chars, scope ≥ 3 chars (`p_min`, `profiles`, `paper1`).
- No `Co-Authored-By` trailer (hypneum-lab repo rule).
- English only in code/comments; methodology §6.6 has an FR mirror under `docs/papers/paper1-fr/methodology.md` — must be updated in same PR (Task 6 covers this).
- `--no-verify` forbidden.

---

## File Structure

```
dream-of-kiki/
├── kiki_oniric/profiles/
│   ├── p_min.py                              [MODIFY ~+12 LOC] add so_trough_amplitude_factor field + docstring
│   ├── p_equ.py                              [MODIFY ~+8 LOC] add so_trough_amplitude_factor field (default 1.0)
│   ├── p_max.py                              [MODIFY ~+8 LOC] add so_trough_amplitude_factor field (default 1.0)
│   └── so_calibration.py                     [NEW ~50 LOC] proxy reader + Sharon-2025 anchor constants
├── tests/unit/profiles/
│   ├── __init__.py                           [NEW empty file]
│   └── test_p_min_sharon_calibration.py      [NEW ~120 LOC] 6 tests covering calibration + monotonicity + reproducibility
├── docs/papers/paper1/
│   └── methodology.md                        [MODIFY §6.6] add code-pointer to so_trough_amplitude_factor
├── docs/papers/paper1-fr/
│   └── methodology.md                        [MODIFY §6.6] FR mirror update (lockstep)
└── CHANGELOG.md                              [MODIFY] Unreleased entry for calibration
```

Total estimated diff: **~210 LOC** (~78 source + ~120 test + ~12 doc).

---

## Task 0: Investigation gate (no code) — verify DR-4 invariants and current state

**Files (read-only):**
- Read: `kiki_oniric/profiles/p_min.py`, `p_equ.py`, `p_max.py`
- Read: `tests/conformance/axioms/test_dr4_profile_inclusion.py`
- Read: `docs/proofs/dr4-profile-inclusion.md` (lines 62-91 = Lemma DR-4.L)
- Read: `tests/conformance/axioms/_dsl.py` lines 143-190 (helpers)

- [ ] **Step 0.1: Confirm DR-4 tests reference only `ops` and `channels`, not numeric calibration fields.**
  Run:
  ```bash
  grep -n "so_trough\|amplitude\|calibration" /Users/electron/hypneum-lab/dream-of-kiki/tests/conformance/axioms/test_dr4_profile_inclusion.py
  ```
  Expected: zero matches. If any match, **STOP** and re-enter the brainstorm phase — this plan would conflict with an existing calibration assertion.

- [ ] **Step 0.2: Confirm no existing `so_*` attribute on any profile.**
  Run:
  ```bash
  grep -rn "so_trough\|amplitude_factor" /Users/electron/hypneum-lab/dream-of-kiki/kiki_oniric/profiles/
  ```
  Expected: zero matches. If non-zero, **STOP** — extend the existing field instead of duplicating.

- [ ] **Step 0.3: Confirm Sharon 2025 BibTeX entry is reachable from the paper1 build.**
  Run:
  ```bash
  grep -n "sharon2025alzdementia" /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper1/references.bib
  grep -n "sharon2025alzdementia" /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper1-fr/references.bib
  grep -n "sharon2025alzdementia" /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper2/references.bib
  ```
  Expected: at least one match in `paper1/references.bib` line ~632. Note FR mirror status — if absent there, add as part of Task 6.

- [ ] **Step 0.4: Confirm baseline test suite passes BEFORE any change.**
  Run:
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/conformance/axioms/test_dr4_profile_inclusion.py tests/unit/test_p_min.py tests/unit/test_p_equ.py tests/unit/test_p_max.py -v --no-cov
  ```
  Expected: all PASS. Record pass count + commit SHA in this checkbox before proceeding.
  If any failure: **STOP** and triage independently — calibration must not pile onto a red baseline.

**Exit criterion:** all 4 grep / pytest checks confirm a clean canvas. Proceed to Task 1.

---

## Task 1: Add `so_trough_amplitude_factor` field to all three profiles (TDD test first)

**Files:**
- Test: `tests/unit/profiles/test_p_min_sharon_calibration.py` (NEW)
- Test: `tests/unit/profiles/__init__.py` (NEW, empty)
- Modify: `kiki_oniric/profiles/p_min.py`
- Modify: `kiki_oniric/profiles/p_equ.py`
- Modify: `kiki_oniric/profiles/p_max.py`

- [ ] **Step 1.1: Create empty `tests/unit/profiles/__init__.py`** so pytest discovers the new sub-package.
  ```bash
  mkdir -p /Users/electron/hypneum-lab/dream-of-kiki/tests/unit/profiles
  touch /Users/electron/hypneum-lab/dream-of-kiki/tests/unit/profiles/__init__.py
  ```

- [ ] **Step 1.2: Write the failing test for default field values** in `tests/unit/profiles/test_p_min_sharon_calibration.py`:
  ```python
  """Sharon 2025 SO-trough biomarker calibration tests for P_min / P_equ / P_max.

  Reference: Sharon et al., Alzheimer's & Dementia 2025 (sharon2025alzdementia
  in docs/papers/paper1/references.bib). hd-EEG, N=55 (21 healthy older /
  28 aMCI / 6 AD). Cognitive performance decreases monotonically with
  slow-wave trough amplitude and frontocentral synchronization.

  These tests verify the qualitative calibration of the
  ``so_trough_amplitude_factor`` field on each profile: an informed
  placeholder whose final empirical value lands at G2 / G4 pilots
  (cf. ``scripts/pilot_g2.py``).
  """
  from __future__ import annotations

  import math

  from kiki_oniric.profiles.p_equ import PEquProfile
  from kiki_oniric.profiles.p_max import PMaxProfile
  from kiki_oniric.profiles.p_min import PMinProfile


  def test_p_min_so_trough_amplitude_factor_default_value() -> None:
      """P_min default factor = 0.45 (aMCI midpoint, Sharon 2025)."""
      profile = PMinProfile()
      assert math.isclose(profile.so_trough_amplitude_factor, 0.45)


  def test_p_equ_so_trough_amplitude_factor_default_value() -> None:
      """P_equ default factor = 1.0 (healthy-older anchor, Sharon 2025)."""
      profile = PEquProfile()
      assert math.isclose(profile.so_trough_amplitude_factor, 1.0)


  def test_p_max_so_trough_amplitude_factor_default_value() -> None:
      """P_max default factor = 1.0 (intact substrate, healthy-young anchor)."""
      profile = PMaxProfile()
      assert math.isclose(profile.so_trough_amplitude_factor, 1.0)
  ```

- [ ] **Step 1.3: Run the failing test, verify it fails for the right reason.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py -v --no-cov
  ```
  Expected: 3 FAIL with `AttributeError: 'PMinProfile' object has no attribute 'so_trough_amplitude_factor'` (and equivalents for P_equ, P_max). If any other error type, fix the test scaffolding first.

- [ ] **Step 1.4: Add the field to `kiki_oniric/profiles/p_min.py`.**
  Insert after the `downscale_state` field (line ~45, before `__post_init__`):
  ```python
      # Sharon et al. 2025 (sharon2025alzdementia, hd-EEG N=55) — qualitative
      # SO-trough amplitude / frontocentral synchronization gradient across
      # healthy-older / aMCI / AD groups. P_min is anchored on the aMCI
      # midpoint as an informed placeholder; absolute µV not extractable from
      # publication. Final empirical value lands at G2 P_min pilot
      # (scripts/pilot_g2.py). See docs/papers/paper1/methodology.md §6.6.
      so_trough_amplitude_factor: float = 0.45
  ```
  Also append one line to the module docstring (after the existing `Reference:` line):
  ```
  Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
  Calibration: docs/papers/paper1/methodology.md §6.6 (Sharon 2025 SO-trough)
  ```

- [ ] **Step 1.5: Add the field to `kiki_oniric/profiles/p_equ.py`.**
  Insert after `rng: random.Random = field(default_factory=random.Random)` (line ~57, before `__post_init__`):
  ```python
      # Sharon et al. 2025 healthy-older anchor (factor = 1.0 = intact SO
      # coherence baseline). See docs/papers/paper1/methodology.md §6.6.
      so_trough_amplitude_factor: float = 1.0
  ```

- [ ] **Step 1.6: Add the field to `kiki_oniric/profiles/p_max.py`.**
  Insert after `target_channels_out: set[OutputChannel] = field(...)` block ending (line ~96, before `__post_init__`):
  ```python
      # Sharon et al. 2025 — intact-substrate anchor (factor = 1.0). P_max
      # is healthy-young by spec §3.1; numerically equal to P_equ on this
      # axis (the SO-trough gradient saturates at "healthy"; richer
      # channels/ops differentiate P_max above the SO biomarker).
      so_trough_amplitude_factor: float = 1.0
  ```

- [ ] **Step 1.7: Run the test, verify all 3 PASS.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py -v --no-cov
  ```
  Expected: 3 PASS.

- [ ] **Step 1.8: Run the DR-4 conformance suite, verify nothing regressed.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/conformance/axioms/test_dr4_profile_inclusion.py -v --no-cov
  ```
  Expected: 6 PASS (matches Task 0.4 baseline). If any FAIL: **STOP** — read the failure, the field default must coexist with DR-4 (it should: DR-4 only inspects `runtime._handlers` and `_PROFILE_CHANNELS`).

- [ ] **Step 1.9: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add kiki_oniric/profiles/p_min.py kiki_oniric/profiles/p_equ.py kiki_oniric/profiles/p_max.py tests/unit/profiles/__init__.py tests/unit/profiles/test_p_min_sharon_calibration.py
  git commit -m "feat(profiles): add so_trough_amplitude_factor field

  Add Sharon 2025 SO-trough amplitude calibration field to P_min, P_equ,
  P_max profiles. P_min defaults to 0.45 (aMCI midpoint, qualitative
  placeholder); P_equ and P_max default to 1.0 (healthy anchor). Field
  is metadata-only; DR-4 chain inclusion unaffected.

  Ref: docs/papers/paper1/methodology.md §6.6 ; docs/papers/paper1/references.bib:632"
  ```
  Subject = 49 chars, scope = `profiles` (8 chars). Commit-validator-clean.

---

## Task 2: Implement `compute_so_amplitude_proxy` reader (TDD)

**Files:**
- Test: `tests/unit/profiles/test_p_min_sharon_calibration.py` (extend)
- Create: `kiki_oniric/profiles/so_calibration.py`

- [ ] **Step 2.1: Append the proxy-reader test** to `tests/unit/profiles/test_p_min_sharon_calibration.py`:
  ```python
  from kiki_oniric.profiles.so_calibration import (
      SHARON_2025_AD_FLOOR,
      SHARON_2025_AMCI_MIDPOINT,
      SHARON_2025_HEALTHY_OLDER_ANCHOR,
      compute_so_amplitude_proxy,
  )


  def test_compute_so_amplitude_proxy_reads_p_min() -> None:
      """compute_so_amplitude_proxy returns the field value on P_min."""
      profile = PMinProfile()
      assert math.isclose(compute_so_amplitude_proxy(profile), 0.45)


  def test_compute_so_amplitude_proxy_reads_p_equ_and_p_max() -> None:
      """compute_so_amplitude_proxy returns 1.0 on healthy anchors."""
      assert math.isclose(compute_so_amplitude_proxy(PEquProfile()), 1.0)
      assert math.isclose(compute_so_amplitude_proxy(PMaxProfile()), 1.0)


  def test_compute_so_amplitude_proxy_rejects_non_profile() -> None:
      """compute_so_amplitude_proxy raises TypeError on missing attribute."""
      import pytest

      class _NotAProfile:
          pass

      with pytest.raises(TypeError, match="so_trough_amplitude_factor"):
          compute_so_amplitude_proxy(_NotAProfile())  # type: ignore[arg-type]


  def test_sharon_2025_anchor_constants() -> None:
      """Module-level constants pin the Sharon 2025 anchor values."""
      assert math.isclose(SHARON_2025_HEALTHY_OLDER_ANCHOR, 1.0)
      assert math.isclose(SHARON_2025_AMCI_MIDPOINT, 0.45)
      assert math.isclose(SHARON_2025_AD_FLOOR, 0.20)
  ```

- [ ] **Step 2.2: Run failing tests.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py::test_compute_so_amplitude_proxy_reads_p_min tests/unit/profiles/test_p_min_sharon_calibration.py::test_sharon_2025_anchor_constants -v --no-cov
  ```
  Expected: FAIL with `ModuleNotFoundError: No module named 'kiki_oniric.profiles.so_calibration'`.

- [ ] **Step 2.3: Create `kiki_oniric/profiles/so_calibration.py`:**
  ```python
  """Sharon 2025 SO-trough biomarker calibration utilities.

  Provides a substrate-agnostic proxy reader for the
  ``so_trough_amplitude_factor`` field on dream profiles, plus the
  three anchor constants extracted (qualitatively) from the
  Sharon et al. 2025 hd-EEG dataset (N=55: 21 healthy older /
  28 aMCI / 6 AD).

  The publication does not report absolute SO-trough amplitudes in µV
  — only a monotone gradient of cognitive performance with slow-wave
  coherence across the three groups. The constants below are therefore
  unit-arbitrary ratios anchored on the healthy-older arm = 1.0:

  * ``SHARON_2025_HEALTHY_OLDER_ANCHOR = 1.0`` — intact slow-wave coherence.
  * ``SHARON_2025_AMCI_MIDPOINT = 0.45`` — informed placeholder for the
    aMCI cohort (midpoint between healthy anchor and AD floor; final
    empirical value lands at G2 P_min pilot, ``scripts/pilot_g2.py``).
  * ``SHARON_2025_AD_FLOOR = 0.20`` — informed placeholder for the AD
    cohort. Not currently consumed by any profile; reserved for a
    future P_pathological extension or sensitivity analysis.

  Reference: Sharon et al., Alzheimer's & Dementia 2025
  (sharon2025alzdementia in docs/papers/paper1/references.bib).
  Calibration narrative: docs/papers/paper1/methodology.md §6.6.
  """
  from __future__ import annotations

  from typing import Any

  SHARON_2025_HEALTHY_OLDER_ANCHOR: float = 1.0
  SHARON_2025_AMCI_MIDPOINT: float = 0.45
  SHARON_2025_AD_FLOOR: float = 0.20


  def compute_so_amplitude_proxy(profile: Any) -> float:
      """Read ``so_trough_amplitude_factor`` from a profile instance.

      Substrate-agnostic accessor used by DR-4-adjacent monotonicity
      tests and by the harness when reporting per-profile biomarker
      proxies.

      Parameters
      ----------
      profile :
          A dream profile instance — typically ``PMinProfile``,
          ``PEquProfile``, or ``PMaxProfile``. Any object exposing a
          float ``so_trough_amplitude_factor`` attribute is accepted.

      Returns
      -------
      float
          The calibration factor in arbitrary ratio units (anchor 1.0
          = healthy-older Sharon 2025 baseline).

      Raises
      ------
      TypeError
          If ``profile`` does not expose a numeric
          ``so_trough_amplitude_factor`` attribute.
      """
      factor = getattr(profile, "so_trough_amplitude_factor", None)
      if factor is None or not isinstance(factor, (int, float)):
          raise TypeError(
              f"{type(profile).__name__} does not expose a numeric "
              "so_trough_amplitude_factor attribute"
          )
      return float(factor)
  ```

- [ ] **Step 2.4: Run all calibration tests, verify PASS.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py -v --no-cov
  ```
  Expected: 7 PASS (3 from Task 1 + 4 new).

- [ ] **Step 2.5: Run mypy strict on the new module.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run mypy kiki_oniric/profiles/so_calibration.py
  ```
  Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 2.6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add kiki_oniric/profiles/so_calibration.py tests/unit/profiles/test_p_min_sharon_calibration.py
  git commit -m "feat(profiles): add SO-amplitude proxy reader

  Add compute_so_amplitude_proxy() and Sharon 2025 anchor constants
  (HEALTHY_OLDER=1.0, AMCI_MIDPOINT=0.45, AD_FLOOR=0.20) in new module
  kiki_oniric/profiles/so_calibration.py. Type-strict, raises TypeError
  on non-profile inputs."
  ```

---

## Task 3: Test monotonicity `P_max ≥ P_equ ≥ P_min` (TDD)

**Files:**
- Test: `tests/unit/profiles/test_p_min_sharon_calibration.py` (extend)

- [ ] **Step 3.1: Append the monotonicity test:**
  ```python
  def test_so_amplitude_proxy_monotonic_p_max_p_equ_p_min() -> None:
      """Monotone ordering: proxy(P_max) >= proxy(P_equ) >= proxy(P_min).

      Aligns with DR-4 Lemma DR-4.L (capacity-monotone metric across the
      profile chain): SO-trough amplitude is a substrate-health proxy
      that is monotone in capacity. The healthy anchors P_max and P_equ
      tie at 1.0 (Sharon 2025 healthy-older arm); P_min sits below at
      0.45 (aMCI midpoint placeholder).
      """
      proxy_max = compute_so_amplitude_proxy(PMaxProfile())
      proxy_equ = compute_so_amplitude_proxy(PEquProfile())
      proxy_min = compute_so_amplitude_proxy(PMinProfile())
      assert proxy_max >= proxy_equ, (
          f"SO-trough monotonicity broken: P_max={proxy_max} < P_equ={proxy_equ}"
      )
      assert proxy_equ >= proxy_min, (
          f"SO-trough monotonicity broken: P_equ={proxy_equ} < P_min={proxy_min}"
      )


  def test_so_amplitude_proxy_p_min_strictly_below_p_equ() -> None:
      """P_min sits *strictly* below the healthy anchor (degraded substrate).

      A strict-inequality assertion guards against accidental upward
      drift of the P_min default (e.g. someone copy-pasting from P_equ).
      The 1e-6 margin is paranoid — values are explicit floats not RNG.
      """
      proxy_equ = compute_so_amplitude_proxy(PEquProfile())
      proxy_min = compute_so_amplitude_proxy(PMinProfile())
      assert proxy_min < proxy_equ - 1e-6, (
          "P_min must be strictly degraded vs P_equ on SO-trough proxy"
      )
  ```

- [ ] **Step 3.2: Run, verify PASS** (no implementation needed — Task 1 + Task 2 already define the values).
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py -v --no-cov
  ```
  Expected: 9 PASS.

- [ ] **Step 3.3: Sanity-check DR-4 chain inclusion still passes.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/conformance/axioms/test_dr4_profile_inclusion.py -v --no-cov
  ```
  Expected: 6 PASS (matches Task 0.4 baseline).

- [ ] **Step 3.4: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add tests/unit/profiles/test_p_min_sharon_calibration.py
  git commit -m "test(profiles): SO-trough monotonicity P_max>=P_equ>=P_min

  Add 2 tests asserting compute_so_amplitude_proxy is monotone in
  capacity along the DR-4 profile chain, and that P_min is strictly
  degraded vs P_equ. Consistent with Lemma DR-4.L (no regression on
  capacity-monotone metrics)."
  ```
  Subject = 50 chars exactly. Re-count if uncertain.

---

## Task 4: Test seeded reproducibility (TDD)

**Goal:** Verify the calibration is deterministic — instantiating the same profile twice (with or without different `random.Random` seeds) yields the same proxy value. This guards against a future maintainer wiring SO-trough into an RNG-driven sampler.

**Files:**
- Test: `tests/unit/profiles/test_p_min_sharon_calibration.py` (extend)

- [ ] **Step 4.1: Append the reproducibility test:**
  ```python
  def test_so_amplitude_proxy_deterministic_across_instances() -> None:
      """Two independent P_min instances yield identical proxy values.

      Calibration must be a constant of the profile class, not RNG-driven.
      R1 reproducibility contract: same (c_version, profile, seed) ->
      same proxy value.
      """
      assert (
          compute_so_amplitude_proxy(PMinProfile())
          == compute_so_amplitude_proxy(PMinProfile())
      )


  def test_so_amplitude_proxy_independent_of_p_equ_rng_seed() -> None:
      """Seeding the P_equ rng field does not perturb the SO proxy.

      Guards against accidental future coupling between the recombine_light
      RNG (P_equ.rng) and the calibration field.
      """
      import random

      rng_a = random.Random(0)
      rng_b = random.Random(424242)
      profile_a = PEquProfile(rng=rng_a)
      profile_b = PEquProfile(rng=rng_b)
      assert (
          compute_so_amplitude_proxy(profile_a)
          == compute_so_amplitude_proxy(profile_b)
      )


  def test_so_amplitude_proxy_independent_of_p_max_rng_seed() -> None:
      """Seeding the P_max rng field does not perturb the SO proxy."""
      import random

      profile_a = PMaxProfile(rng=random.Random(0))
      profile_b = PMaxProfile(rng=random.Random(424242))
      assert (
          compute_so_amplitude_proxy(profile_a)
          == compute_so_amplitude_proxy(profile_b)
      )
  ```

- [ ] **Step 4.2: Run, verify PASS.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/unit/profiles/test_p_min_sharon_calibration.py -v --no-cov
  ```
  Expected: 12 PASS.

- [ ] **Step 4.3: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add tests/unit/profiles/test_p_min_sharon_calibration.py
  git commit -m "test(profiles): SO-trough proxy deterministic on seeds

  Verify compute_so_amplitude_proxy is independent of profile rng
  seeds (P_equ.rng, P_max.rng) and yields identical values across
  fresh instances. R1 contract guard."
  ```

---

## Task 5: Verify DR-4 + full conformance suite still green; lock with explicit guard test

**Files:**
- Test: `tests/conformance/axioms/test_dr4_profile_inclusion.py` (read-only)
- Test: `tests/unit/profiles/test_p_min_sharon_calibration.py` (extend)

- [ ] **Step 5.1: Append a guard test that ties the SO calibration ordering to the DR-4.L lemma narrative.**
  This is an *extra* assertion that lives in the unit module (not the conformance dir, per `tests/CLAUDE.md` anti-pattern "Don't put conformance assertions inside `tests/unit/`" — but **the underlying axiom is DR-4 and the strict ordering of the calibration is a derived corollary, not the axiom itself**, so unit-suite placement is appropriate). The test simultaneously runs `registered_ops` to confirm DR-4 chain is preserved.
  ```python
  def test_so_calibration_coexists_with_dr4_chain_inclusion() -> None:
      """SO calibration field must not perturb DR-4 ops/channels chain.

      Cross-check: instantiate all three profiles, verify their op-handler
      registries still satisfy DR-4 chain inclusion in the presence of
      the new calibration field. This is a regression guard, not the
      axiom test (cf. tests/conformance/axioms/test_dr4_profile_inclusion.py).
      """
      p_min = PMinProfile()
      p_equ = PEquProfile()
      p_max = PMaxProfile()
      ops_min = set(p_min.runtime._handlers.keys())
      ops_equ = set(p_equ.runtime._handlers.keys())
      assert ops_min <= ops_equ, "DR-4 ops chain regressed under SO calibration"
      assert ops_equ <= p_max.target_ops, "DR-4 P_equ subset P_max regressed"
      # And the SO proxy still evaluates on each — no AttributeError.
      _ = compute_so_amplitude_proxy(p_min)
      _ = compute_so_amplitude_proxy(p_equ)
      _ = compute_so_amplitude_proxy(p_max)
  ```

- [ ] **Step 5.2: Run the conformance suite + the new guard.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/conformance/axioms/ tests/unit/profiles/ -v --no-cov
  ```
  Expected: 6 (DR-4) + 13 (calibration) = 19 PASS at minimum (more if other axiom tests run; counts may grow with DR-0, DR-1, DR-2, DR-3 — they should all stay PASS).

- [ ] **Step 5.3: Run the FULL test suite with coverage gate.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest
  ```
  Expected: 277+ tests passing, coverage ≥ 90%. The new module adds ~50 LOC, the new test module covers them; coverage should not drop. If gate fails: read the missing-coverage report — likely `compute_so_amplitude_proxy` else-branch (TypeError raise) is exercised by `test_compute_so_amplitude_proxy_rejects_non_profile` already, so the only risk is the import of `SHARON_2025_AD_FLOOR` if unused — confirm the constant is referenced in `test_sharon_2025_anchor_constants`.

- [ ] **Step 5.4: Run lint.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run ruff check kiki_oniric/profiles/ tests/unit/profiles/
  ```
  Expected: zero issues.

- [ ] **Step 5.5: Run mypy.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run mypy harness tests
  ```
  Expected: `Success`. (Note: `mypy --strict` is configured globally; `kiki_oniric/` is included implicitly via test imports.)

- [ ] **Step 5.6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add tests/unit/profiles/test_p_min_sharon_calibration.py
  git commit -m "test(profiles): guard DR-4 chain under SO calibration

  Regression guard verifying DR-4 ops chain inclusion is preserved
  when so_trough_amplitude_factor field is present on all profiles.
  Calibration field is metadata-only; conformance suite stays green."
  ```

---

## Task 6: Update `docs/papers/paper1/methodology.md` §6.6 + FR mirror

**Files:**
- Modify: `docs/papers/paper1/methodology.md` (lines 115-129)
- Modify: `docs/papers/paper1-fr/methodology.md` (corresponding §6.6 block)

- [ ] **Step 6.1: Read current §6.6 in EN and FR.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && sed -n '115,130p' docs/papers/paper1/methodology.md
  cd /Users/electron/hypneum-lab/dream-of-kiki && grep -n "6\.6\|Biological grounding\|sharon2025" docs/papers/paper1-fr/methodology.md
  ```
  Identify the FR mirror line range; if §6.6 does not exist in the FR mirror, **STOP** and add a separate task to backfill (out-of-scope for this plan; flag as a Gap).

- [ ] **Step 6.2: In EN methodology.md §6.6, replace the existing P_min sentence (lines 119-123)** with the calibration-pinned version. Original:
  > `P_min` (degraded substrate) is anchored on the slow-oscillation trough amplitude and frontocentral synchronization gradient documented across healthy older / aMCI / AD groups (N = 55 hd-EEG) [@sharon2025alzdementia], where cognitive performance decreases monotonically with the loss of slow-wave coherence.

  Replace with:
  > `P_min` (degraded substrate) is anchored on the slow-oscillation trough amplitude and frontocentral synchronization gradient documented across healthy older / aMCI / AD groups (N = 55 hd-EEG) [@sharon2025alzdementia], where cognitive performance decreases monotonically with the loss of slow-wave coherence. The calibration is realised in code as a `so_trough_amplitude_factor` field on the profile dataclass (`kiki_oniric/profiles/p_min.py`), with the unit-arbitrary ratio `0.45` for `P_min` (aMCI midpoint placeholder) and `1.0` for the healthy anchors `P_equ` / `P_max`. Sharon 2025 reports the gradient qualitatively (no absolute SO-trough amplitude in µV); the placeholder ratio will be refined to an empirical value at the G2 P_min pilot (`scripts/pilot_g2.py`).

  Use `Edit` tool with the exact original text (no trailing-whitespace ambiguity).

- [ ] **Step 6.3: Mirror the change in `docs/papers/paper1-fr/methodology.md` §6.6.**
  Apply the equivalent FR translation (translator's note: keep file paths and code identifiers in English — `kiki_oniric/profiles/p_min.py`, `so_trough_amplitude_factor`, `scripts/pilot_g2.py` are not translated).
  French version of the appended sentence:
  > La calibration est implémentée comme un champ `so_trough_amplitude_factor` sur la dataclass de profil (`kiki_oniric/profiles/p_min.py`), avec la valeur ratio sans-unité `0,45` pour `P_min` (placeholder médian aMCI) et `1,0` pour les ancres saines `P_equ` / `P_max`. Sharon 2025 rapporte le gradient qualitativement (pas d'amplitude SO-trough absolue en µV) ; la valeur placeholder sera affinée empiriquement lors du pilote G2 P_min (`scripts/pilot_g2.py`).

- [ ] **Step 6.4: Verify `sharon2025alzdementia` BibTeX entry exists in `docs/papers/paper1-fr/references.bib`.**
  ```bash
  grep -n "sharon2025alzdementia" /Users/electron/hypneum-lab/dream-of-kiki/docs/papers/paper1-fr/references.bib
  ```
  If absent, copy the entry from `docs/papers/paper1/references.bib:632-641` to the FR mirror in the same PR (FR mirror discipline). If present, no action.

- [ ] **Step 6.5: Sanity-rebuild the paper1 PDF (optional, only if the build toolchain is available locally).**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && ls docs/papers/paper1/build/Makefile 2>&1
  ```
  If a Makefile exists, run its `methodology` target. If not, skip — the prose change does not break compilation since the bibkey is unchanged.

- [ ] **Step 6.6: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add docs/papers/paper1/methodology.md docs/papers/paper1-fr/methodology.md
  # Only stage docs/papers/paper1-fr/references.bib if Step 6.4 needed it
  git commit -m "docs(paper1): pin §6.6 to so_trough calibration field

  Add code-pointer in methodology §6.6 from the qualitative Sharon
  2025 anchor to the new so_trough_amplitude_factor field on the
  profile dataclasses. EN + FR mirror lockstep. Placeholder ratio
  0.45 (P_min) / 1.0 (healthy anchors); empirical refinement
  deferred to G2 pilot."
  ```

---

## Task 7: STATUS.md / CHANGELOG.md — confirm no FC bump, log calibration

**Files:**
- Modify: `CHANGELOG.md`
- Read-only check: `STATUS.md`

- [ ] **Step 7.1: Confirm STATUS.md needs no edit.**
  Re-read `STATUS.md` lines 56-63 (DualVer table). The calibration is:
  - Not an axiom statement edit → no FC bump.
  - Not a primitive signature change → no FC bump.
  - Not a channel set / invariant ID change → no FC bump.
  - Not a gate result → EC stays `PARTIAL`.

  No STATUS.md change. Document the rationale inline in the next CHANGELOG entry.

- [ ] **Step 7.2: Append to `CHANGELOG.md` under the `## [Unreleased]` heading** (or create that heading at the top if absent — check first):
  ```bash
  grep -n "^## \[Unreleased\]\|^# Changelog" /Users/electron/hypneum-lab/dream-of-kiki/CHANGELOG.md | head -5
  ```
  Add this entry under `[Unreleased]`:
  ```markdown
  ### Added
  - `kiki_oniric.profiles.so_calibration` module with
    `compute_so_amplitude_proxy()` reader and Sharon 2025 anchor
    constants (HEALTHY_OLDER=1.0, AMCI_MIDPOINT=0.45, AD_FLOOR=0.20).
    Reference: docs/papers/paper1/methodology.md §6.6.
  - `so_trough_amplitude_factor: float` field on `PMinProfile` (default
    `0.45`, aMCI midpoint placeholder), `PEquProfile` (default `1.0`),
    `PMaxProfile` (default `1.0`). Sharon et al. 2025
    (sharon2025alzdementia) qualitative calibration; final empirical
    value deferred to G2 P_min pilot.
  - `tests/unit/profiles/test_p_min_sharon_calibration.py` (12 tests):
    default values, proxy reader, monotonicity P_max ≥ P_equ ≥ P_min,
    seed-independence, DR-4 chain coexistence guard.

  ### Versioning
  - **No DualVer bump.** Calibration is profile metadata, not an axiom
    statement, primitive signature, channel set, or invariant ID
    change. FC stays at v0.10.0; EC stays PARTIAL. Per framework-C
    spec §12 (only spec/proof changes trigger FC; only gate results
    trigger EC).
  ```

- [ ] **Step 7.3: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git add CHANGELOG.md
  git commit -m "docs(changelog): SO-trough calibration unreleased entry

  Log the so_trough_amplitude_factor calibration field + proxy reader
  + Sharon 2025 anchor constants. Confirm no DualVer bump (calibration
  is metadata, not an axiom or primitive signature change)."
  ```

---

## Task 8: Self-review

- [ ] **Step 8.1: Spec coverage check.**
  Re-read the goal at the top of this plan. Verify each line maps to a completed task:
  - "Add `so_trough_amplitude_factor` field" → Task 1 ✓
  - "Compute proxy" → Task 2 ✓
  - "Monotone P_max ≥ P_equ ≥ P_min" → Task 3 ✓
  - "Seed-reproducible" → Task 4 ✓
  - "DR-4 still green" → Task 5 ✓
  - "Methodology §6.6 update" → Task 6 ✓
  - "STATUS / CHANGELOG" → Task 7 ✓
  Confirm all 7 are checked.

- [ ] **Step 8.2: Placeholder scan.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && grep -rn "TBD\|TODO\|FIXME\|implement later\|XXX" kiki_oniric/profiles/p_min.py kiki_oniric/profiles/p_equ.py kiki_oniric/profiles/p_max.py kiki_oniric/profiles/so_calibration.py tests/unit/profiles/test_p_min_sharon_calibration.py
  ```
  Expected: zero hits. The "informed placeholder" language in docstrings is intentional (it documents that G2 will refine the value), but the code itself contains no `TODO`/`TBD`. If hits found: rewrite the docstring to be declarative ("calibrated against ...") rather than a TODO.

- [ ] **Step 8.3: Type-consistency check.**
  - `so_trough_amplitude_factor: float` declared on all three profiles ✓
  - `compute_so_amplitude_proxy(profile: Any) -> float` ✓ (`Any` is acceptable here because the function is duck-typed across the three concrete profile classes; tightening to `Union[PMinProfile, PEquProfile, PMaxProfile]` would create an import cycle).
  - All test assertions use `math.isclose` or `==` on floats ✓.

- [ ] **Step 8.4: Repro-contract check.**
  - No new RNG was introduced; the calibration is a constant of the class. R1 contract preserved.
  - No filesystem state added; no run-registry write in tests. Tests use no `tmp_path` because no I/O is performed.

- [ ] **Step 8.5: EN→FR propagation check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && diff <(grep -c "so_trough_amplitude_factor" docs/papers/paper1/methodology.md) <(grep -c "so_trough_amplitude_factor" docs/papers/paper1-fr/methodology.md)
  ```
  Expected: identical counts. If diff non-empty: re-do Task 6.3.

- [ ] **Step 8.6: Conformance-test final pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest tests/conformance/ tests/unit/profiles/ tests/unit/test_p_min.py tests/unit/test_p_equ.py tests/unit/test_p_max.py -v --no-cov
  ```
  Expected: full green.

- [ ] **Step 8.7: Coverage final pass.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && uv run pytest
  ```
  Expected: 289+ tests passing (277 baseline + 12 new), coverage ≥ 90%.

- [ ] **Step 8.8: Commit history sanity-check.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && git log --oneline -10
  ```
  Expected: 6 new commits with conventional-commit prefixes (`feat(profiles)`, `feat(profiles)`, `test(profiles)`, `test(profiles)`, `test(profiles)`, `docs(paper1)`, `docs(changelog)` = 7 commits actually). Each subject ≤ 50 chars. No `Co-Authored-By` trailers.

---

## Implementation summary

| Task | Files touched | LOC delta | Test count delta |
|------|---------------|-----------|------------------|
| 0 — Investigation | (read-only) | 0 | 0 |
| 1 — Field on 3 profiles | `p_min.py`, `p_equ.py`, `p_max.py`, `tests/unit/profiles/` (new) | +28 src / +35 test | +3 |
| 2 — Proxy reader | `so_calibration.py` (new), test | +50 src / +30 test | +4 |
| 3 — Monotonicity test | test | +25 test | +2 |
| 4 — Seed-reproducibility test | test | +30 test | +3 |
| 5 — DR-4 guard test + lint/mypy | test | +18 test | +1 |
| 6 — methodology.md EN+FR | docs | +6 doc | 0 |
| 7 — CHANGELOG | docs | +18 doc | 0 |
| 8 — Self-review | (read-only) | 0 | 0 |
| **Total** | 9 files | **~78 src + ~138 test + ~24 doc** | **+13** |

Final test suite: 277 → 290 tests, coverage ≥ 90%, DR-4 conformance green, no FC/EC bump.

---

## Open questions / Gaps (to resolve before execution)

1. **Does `docs/papers/paper1-fr/methodology.md` already contain a §6.6 mirror?** Task 6.1 verifies; if absent, the FR backfill is out-of-scope and must be tracked as a follow-up issue.
2. **Does `CHANGELOG.md` use a `## [Unreleased]` heading convention?** Task 7.2 verifies; if not, adapt to the existing convention (e.g. dated section header).
3. **Should `SHARON_2025_AD_FLOOR = 0.20` ship if no profile consumes it?** The constant is defended as "reserved for future P_pathological extension or sensitivity analysis"; alternative is to drop it and revisit at G4. Decision deferred to executor — both options pass tests if `test_sharon_2025_anchor_constants` is adapted.
4. **Should the field name be `so_trough_amplitude_factor` or something shorter?** Long but explicit. Synonyms (`so_factor`, `sw_amplitude`) would force a glossary entry per `docs/glossary.md` discipline. The plan commits to the explicit name; rename is cheap if a reviewer pushes back.
5. **MLX / Apple-Silicon nightly R1 tests:** the calibration field is pure-Python metadata, untouched by MLX hot paths. The R1 nightly should remain bit-stable. If it doesn't (unexpected), re-investigate via `tests/reproducibility/` — but no plan task covers this proactively.
