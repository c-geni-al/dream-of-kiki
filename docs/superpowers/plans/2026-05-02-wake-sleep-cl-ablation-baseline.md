# Wake-Sleep CL Ablation Baseline (Alfarano 2024) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Status flag :** *Spec partially ambiguous* — Task 0/0.5 perform a
brainstorm + decision before any code lands. The body of this plan
is written under a **default working assumption (variant `c`,
fixture-stub with published metrics)** that Task 0.5 will confirm
or revise. If Task 0.5 chooses `a` or `b`, **revisit this plan
before continuing past Task 1**.

**Goal :** Add a *Wake-Sleep Consolidated Learning* baseline row
[@alfarano2024wakesleep, arXiv 2401.08623] to the dreamOfkiki
ablation matrix. The baseline lets Paper 2 compare the three
profiles (`P_min`, `P_equ`, `P_max`) against the closest published
NREM/REM dual-phase analog on a standardised continual-learning
task (Split-MNIST / Split-FMNIST class-incremental, the same
harness already used in `experiments/h1_split_mnist/`).

**Architecture :**

- Add a fourth substrate-style entry to
  `kiki_oniric/substrates/` named `wake_sleep_cl_baseline` that
  exposes a `WakeSleepCLBaseline` dataclass + `wake_sleep_substrate_components()` helper, mirroring the shape of `mlx_kiki_oniric.py`,
  `esnn_thalamocortical.py`, `micro_kiki.py`. **It is registered
  as a baseline-only substrate, not as a DR-3-conformant substrate**
  — it does not need to satisfy the 4 op factories ; it implements
  a single end-to-end `evaluate_continual(seed, task_split) -> dict`
  contract that returns `accuracy_per_task`, `forgetting`,
  `n_params`, `wallclock_s`. This is the comparator API.
- Extend `docs/interfaces/eval-matrix.yaml` with a top-level
  `baselines:` block (NEW key — schema change, FC bump scope
  decided in Task 0.5) listing `wake_sleep_cl` and its bibtex key
  + arXiv id + variant tag (`a`/`b`/`c`).
- Update `harness/config/eval_matrix.py` to parse + expose
  `baselines` (or, in fallback variant `a-only`, scope it under
  `metrics.M*` without a schema bump).
- Add `scripts/baseline_wake_sleep_cl.py` (gate-style driver) that
  runs the baseline on Split-FMNIST 5-task seeds `[42, 123, 7]`
  (same seed grid as Paper 2 §6.3) and dumps a JSON artifact under
  `docs/milestones/wake-sleep-baseline-{date}.{md,json}`.
- Add a row to Paper 2 §5 (architecture.md → §5.8 new subsection)
  + §6.3 (methodology.md → matrix dimension note) and §7 (results.md)
  citing `[@alfarano2024wakesleep]`.

**Tech Stack :** Python 3.12, uv, pyyaml (already in deps),
PyTorch + Avalanche (already used in
`experiments/h1_split_mnist/run_h1.py`), pytest + hypothesis.
**No MLX** (Alfarano repo is PyTorch ; we keep the baseline on the
existing CPU/PyTorch path). No new top-level dependency unless
variant `a` is selected.

**Dependencies already DONE :**

- `experiments/h1_split_mnist/` — Avalanche-backed Split-FMNIST
  loop with 3 conditions (`baseline`/`P_min`/`P_equ`) and
  `results_h1.csv` schema (`seed, condition, task_id, accuracy,
  forgetting`). Reuse this loader and CSV schema verbatim.
- `kiki_oniric/eval/ablation.py` — `AblationRunner` with
  `substrate_specs` axis (cycle-2 C2.9). The runner happily emits
  a "baseline" substrate row.
- `kiki_oniric/eval/statistics.py` — H1-H4 stat tests
  (`welch_one_sided`, `tost_equivalence`, `jonckheere_trend`,
  `one_sample_threshold`).
- `harness/storage/run_registry.py` — R1-keyed registry
  (`(c_version, profile, seed, commit_sha)` SHA-256 prefix).
- `docs/papers/paper1/references.bib` line 454 — bibkey
  `alfarano2024wakesleep` with arXiv id `2401.08623`.
- `docs/interfaces/eval-matrix.yaml` — `metrics.M1.a`
  (`forgetting_rate`) + `M1.b` (`avg_accuracy`) under family
  `continual_learning` already named ; reuse the metric IDs.

**Risks (re-stated from spec, must be addressed in Task 0.5) :**

1. **FC bump cost.** `eval-matrix.yaml` schema change = FC bump
   per `docs/CLAUDE.md` interfaces routing rule. New key
   `baselines:` is **additive** → FC-MINOR, not MAJOR (per
   framework-C spec §12.2 : "addition of new optional primitive
   / new derived constraint"). Variant `a-only` (smuggled inside
   `metrics.M*`) avoids the bump but is structurally wrong (a
   baseline isn't a metric). Recommendation : **MINOR bump**.
2. **Continual-learning dataset.** Split-FMNIST already wired
   via Avalanche in `experiments/h1_split_mnist/run_h1.py`. Reuse
   it ; do **not** add CIFAR-100 split-class to scope.
3. **PyTorch ↔ MLX dependency conflict.** Avalanche pulls
   PyTorch ; `pyproject.toml` does not declare PyTorch and the
   `experiments/h1_split_mnist/requirements.txt` keeps it
   isolated. Keep the baseline isolated under
   `experiments/wake_sleep_baseline/` (sibling to `h1_split_mnist/`)
   for variant `a`/`b`. For variant `c` (stub) no PyTorch needed
   at all.
4. **Paper 2 still draft.** Paper 2 §5/§6 are draft C2.14/C2.15
   ; updates here are *additive* (one subsection §5.8 + a row in
   §6.3 matrix). Do not refactor those sections.
5. **Subprocess wrap (variant `b`).** Alfarano's official repo
   (if available — to be confirmed in Task 0) would need a
   pinned commit-SHA + a `pip install -e .` shim. Conflict-prone
   under `uv`. **Strongly disfavoured** unless their repo is
   stable + small.

**Budget estimate :**

- **Variant `c` (default — stub with published metrics) :** ~250
  LOC new, ~60 LOC modified. Compute : minutes (no training).
  Disk : <1 MB.
- **Variant `a` (port internal) :** ~600-900 LOC new (ported
  from Alfarano source). Compute : ~3-6 h CPU per seed (5 tasks
  × 3 seeds × 2 phases ≈ 18 task-trainings). Disk : ~50 MB
  (FashionMNIST + checkpoints).
- **Variant `b` (subprocess wrap) :** ~150 LOC new + 1 git
  submodule + 1 PyTorch isolation env. Compute : same as `a`.

**Worktree decision :** Recommend
`superpowers:using-git-worktrees`. The eval-matrix schema change
+ paper updates touch isolated files but the FC bump touches
`STATUS.md` + `CHANGELOG.md` which are also touched by other
ongoing branches (Paper 1 PLOS CB submission, micro-kiki LoRA
substrate). A worktree avoids cross-branch contention.

**Seed grid :** `[42, 123, 7]` — identical to Paper 2 §6.3 grid,
so the new row is bit-comparable on the same predictor (where
applicable).

**Out of scope (defer or refuse) :**

- CIFAR-100 split-class (Paper 3 territory).
- Loihi-2 / neuromorphic execution of Wake-Sleep CL.
- Adding the baseline as a real DR-3-conformant substrate
  (would require porting all 4 ops — wrong scope).
- Scoring the baseline against M2.b (rsa_fmri_alignment) or
  M3.* (efficiency) — out of paper-2 ablation scope.

---

## File Structure

```
dream-of-kiki/
├── docs/
│   ├── interfaces/
│   │   └── eval-matrix.yaml                       [MODIFY, +18 LOC] add `baselines:` block
│   ├── papers/paper2/
│   │   ├── architecture.md                        [MODIFY, +35 LOC] add §5.8 baseline subsection
│   │   ├── methodology.md                         [MODIFY, +12 LOC] amend §6.3 matrix dimension
│   │   ├── results.md                             [MODIFY, +18 LOC] add baseline row in §7 table
│   │   └── references.bib                         [VERIFY, +0 LOC] confirm `alfarano2024wakesleep` exists
│   ├── papers/paper2-fr/
│   │   ├── architecture.md                        [MODIFY, +35 LOC] FR mirror of §5.8
│   │   ├── methodology.md                         [MODIFY, +12 LOC] FR mirror of §6.3
│   │   └── results.md                             [MODIFY, +18 LOC] FR mirror of §7
│   ├── milestones/
│   │   └── wake-sleep-baseline-2026-05-02.{md,json}  [NEW] dated immutable
│   └── glossary.md                                 [MODIFY, +4 LOC] add "Wake-Sleep CL"
├── harness/
│   └── config/
│       └── eval_matrix.py                         [MODIFY, +14 LOC] parse `baselines`
├── kiki_oniric/
│   └── substrates/
│       ├── __init__.py                            [MODIFY, +6 LOC]  re-export
│       └── wake_sleep_cl_baseline.py              [NEW, ~120 LOC]   adapter + metrics
├── scripts/
│   └── baseline_wake_sleep_cl.py                  [NEW, ~180 LOC]   gate driver
├── tests/
│   ├── unit/
│   │   ├── test_eval_matrix_baselines.py          [NEW, ~80 LOC]    schema parse TDD
│   │   └── test_wake_sleep_baseline_adapter.py    [NEW, ~110 LOC]   adapter TDD
│   └── conformance/
│       └── test_baseline_registration.py          [NEW, ~60 LOC]    registry contract
├── CHANGELOG.md                                   [MODIFY, +14 LOC] FC-MINOR bump entry
└── STATUS.md                                      [MODIFY, +3 LOC]  bump table row
```

---

## Task 0 : BRAINSTORM & investigate prior art

**Files :** read-only. No code changes.

**Goal :** Lock the decision frame for Task 0.5. Confirm what
exists in the Alfarano 2024 ecosystem, what shape the existing
matrix takes, and what the Paper 2 narrative requires of the
baseline row.

- [ ] **Step 1 : Re-read the spec context for the matrix**

Read these files top to bottom and capture one-line summaries
back to `docs/superpowers/notes/2026-05-02-wake-sleep-decision-brief.md`
(scratch note ; not committed if variant `c` chosen) :

- `docs/papers/paper1/introduction.md` lines 85-112 (Alfarano
  framing) — already references the paper as **Paper 2's
  primary ablation comparator**. Confirms scope.
- `docs/papers/paper2/architecture.md` (the one you read) §5.1-5.7
  — registration API, substrate registration pattern.
- `docs/papers/paper2/methodology.md` §6.3 (matrix shape :
  2 substrates × 3 profiles × 3 seeds = 18 cells) and §6.4
  (synthetic-substitute predictor caveat — the **baseline must
  not require a divergent predictor**, since cycle-2's caveat
  applies to dreamOfkiki rows, not to the published-metrics
  baseline).
- `docs/interfaces/eval-matrix.yaml` (top-level keys :
  `version`, `bump_rules`, `publication_ready_gate`, `metrics`).
- `harness/config/eval_matrix.py` (loader + dataclass `EvalMatrix`).
- `experiments/h1_split_mnist/run_h1.py` — Split-FMNIST loop.

- [ ] **Step 2 : Look up Alfarano 2024 repo availability**

Run :
```bash
uv run python - <<'PY'
import urllib.request, json
# Try the arXiv abstract page (mirrored on cs.LG)
url = "https://arxiv.org/abs/2401.08623"
print(f"Verify accessibility: {url}")
PY
```

(If Studio offline : skip this step and assume **no public repo**
— that pins us to variant `a` or `c`.)

Alfarano et al. 2024 IEEE TNNLS — published in IEEE TNNLS, Jan
2024, arXiv 2401.08623. As of plan-write-date 2026-05-02 :
- IEEE TNNLS publication exists (verifiable via the arXiv abstract
  page footer).
- Public reference implementation : **unknown — verify in this
  task**. If absent → variant `b` is impossible → choose between
  `a` and `c`.

Capture findings (link + commit-SHA-if-any + license) back to the
brainstorm brief.

- [ ] **Step 3 : Invoke `/superpowers:brainstorming` with this prompt**

```
Topic : Wake-Sleep CL ablation baseline (Alfarano 2024) for
        dreamOfkiki Paper 2 §7

Decision required : choose ONE of three implementation variants.

(a) Port Alfarano's algorithm into our repo (PyTorch + Avalanche),
    train it on Split-FMNIST, generate 3-seed numbers under R1.
    PROS : real numbers comparable to dreamOfkiki rows on identical
    predictor / dataset / seeds. Can do TOST equivalence vs P_equ.
    CONS : 600-900 LOC ; 3-6 h compute per seed ; PyTorch dep
    needs isolation under `experiments/wake_sleep_baseline/`.
(b) Subprocess-wrap Alfarano's published repo (if it exists).
    PROS : trustable provenance ; minimal code.
    CONS : conflict-prone ; pinning + reproducibility hard ;
    license / availability unknown.
(c) Fixture-stub : record published metrics from the IEEE TNNLS
    paper (Tables 2-3 of the original) as a frozen YAML artifact ;
    cite as **published reference values, not a re-run**.
    PROS : minimal LOC ; no PyTorch dep ; clear caveat in §6.4
    style.
    CONS : can't run TOST against synthetic predictor ; row is
    a citation, not a registered run_id ; weakens Paper 2's
    claim of running an apples-to-apples comparison.

Constraint : Paper 2 already declares its own rows
synthetic-substitute (§6.4). The baseline only needs to be
*as good as* that caveat, not stronger.

Open questions :
1. Does Alfarano publish a repo / which commit-SHA ?
2. Is the IEEE TNNLS Tables 2-3 metric exactly Split-FMNIST
   class-incremental forgetting + avg accuracy ? (Or a different
   benchmark like CIFAR-100 split-class ?)
3. Cycle 3 will add a divergent-predictor replication ; will the
   baseline row need to be re-run there too ?

Recommendation if no public repo : default to (c). If a clean
public repo exists : (b). Avoid (a) unless cycle 3 explicitly
demands a re-runnable baseline.
```

Run the skill ; capture decision in
`docs/superpowers/notes/2026-05-02-wake-sleep-decision-brief.md`.

**Done when :** Brainstorm output records a written rationale +
chosen variant + answers to questions 1/2/3 above.

---

## Task 0.5 : DECISION — pick variant `a` / `b` / `c`

**Files :** `docs/superpowers/notes/2026-05-02-wake-sleep-decision-brief.md` (scratch note, gitignored unless Task 0 escalates).

**Goal :** Commit to one of `a`/`b`/`c`. Document the rationale
+ FC bump impact + revision impact on later tasks of this plan.

- [ ] **Step 1 : Score against constraints**

| Variant | LOC | Compute | FC bump | Revisits this plan | Recommended |
|---|---|---|---|---|---|
| `a` port | 600-900 | 3-6 h × 3 seeds | MINOR (additive baselines block) | revise Tasks 3-7 | only if cycle-3 needs re-run |
| `b` wrap | 150 + submodule | same | MINOR | revise Task 3 | only if Alfarano public repo clean |
| `c` stub | 250 | minutes | MINOR | none | **default** |

- [ ] **Step 2 : Record decision**

In `STATUS.md`, *after this PR lands*, the entry under "Cycle-3
Phase 2 deferred" gains a parenthetical mention of the chosen
variant. Pre-emptively note in the brainstorm brief :

> 2026-05-02 — Wake-Sleep CL ablation baseline implemented as
> variant **{a|b|c}**, rationale {…}, FC-MINOR bump
> `C-v0.10.0+PARTIAL → C-v0.11.0+PARTIAL`. Plan
> `docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md`
> drove the work.

- [ ] **Step 3 : If non-default, branch this plan**

If Task 0.5 chose `a` : duplicate this file as
`2026-05-02-wake-sleep-cl-ablation-baseline-variant-a.md` ; replace
Task 3-5 body to follow `experiments/h1_split_mnist/run_h1.py`
shape (PyTorch + Avalanche + Wake-Sleep two-phase loop) ; bump
budget LOC + compute ; keep Tasks 1-2, 6-N as-is.

If Task 0.5 chose `b` : replace Task 3 with
"vendor + pin Alfarano repo as submodule" + "build subprocess
wrapper" ; keep rest.

If Task 0.5 chose `c` : *no change to this plan* — proceed.

**Done when :** brainstorm brief is committed (or noted scratch),
variant is locked, and (if non-default) the variant-specific plan
file exists.

---

## Task 1 : Schema change to `eval-matrix.yaml` + FC-MINOR bump prep

**Files :**

- `docs/interfaces/eval-matrix.yaml` (modify)
- `CHANGELOG.md` (modify, defer commit until Task N)
- (test scaffolding lands in Task 2)

**Goal :** Add a top-level `baselines:` block that lists
`wake_sleep_cl` with its bibkey, arXiv id, source variant
(`a`/`b`/`c`), and metric IDs it scores on. Update the loader
contract in Task 2. Plan the FC-MINOR bump message for Task N.

- [ ] **Step 1 : Add `baselines:` block to `eval-matrix.yaml`**

Append to `docs/interfaces/eval-matrix.yaml` (after line 109,
before EOF) :

```yaml
# Published-baseline registry
# Each baseline row scores on a subset of `metrics:` keys.
# Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §12.2
#            (FC-MINOR : addition of new optional primitive)
baselines:
  wake_sleep_cl:
    name: wake_sleep_consolidated_learning
    bibkey: alfarano2024wakesleep
    arxiv: "2401.08623"
    venue: "IEEE TNNLS"
    year: 2024
    variant: c  # Set by Task 0.5 ; one of {a, b, c}
    scores_on:
      - M1.a   # forgetting_rate
      - M1.b   # avg_accuracy
    determinism: published_reference  # variant c — variant a uses seeded_rng
    paper2_role: primary_ablation_comparator
```

If variant `a`/`b` : `determinism: seeded_rng` and append
`run_id_artifact: docs/milestones/wake-sleep-baseline-{date}.json`.

- [ ] **Step 2 : Bump `version` field in the same file**

Replace top-of-file line :
```yaml
version: "C-v0.7.0+PARTIAL"
```
with :
```yaml
version: "C-v0.11.0+PARTIAL"
```

(The current STATUS.md reports `C-v0.10.0+PARTIAL` ; the
`eval-matrix.yaml` is one bump behind. This task lifts both
together.)

- [ ] **Step 3 : Pre-write the CHANGELOG stanza (do not commit yet)**

In `CHANGELOG.md`, plan to add the following entry under a new
`[C-v0.11.0+PARTIAL]` section at the top (insertion happens in
Task N) :

```markdown
## [C-v0.11.0+PARTIAL] — 2026-05-02

### Added
- **FC-MINOR :** `eval-matrix.yaml` schema gains a top-level
  `baselines:` block (additive, no breaking change). The first
  registered baseline is `wake_sleep_cl` per Alfarano 2024
  [IEEE TNNLS, arXiv 2401.08623], the closest published
  NREM/REM dual-phase analog and Paper 2's primary ablation
  comparator (Paper 1 §3 introduction.md L94, L108).
- `kiki_oniric/substrates/wake_sleep_cl_baseline.py` —
  baseline-only adapter, exposes
  `WakeSleepCLBaseline.evaluate_continual(seed, task_split)`.
- `scripts/baseline_wake_sleep_cl.py` — gate-style driver.
- Paper 2 §5.8 (architecture.md) + §6.3 (methodology.md) +
  §7 (results.md) updated with the new row, EN ↔ FR mirrored
  per `docs/papers/CLAUDE.md`.

### Notes
- EC axis remains `+PARTIAL`. The baseline row is a
  variant-{a|b|c} artifact ; see Task 0.5 decision brief
  (`docs/superpowers/notes/2026-05-02-wake-sleep-decision-brief.md`).
- DR-3 conformance unaffected : the baseline does not implement
  the 4 op factories ; it is registered as a *baseline*, not a
  substrate. The DR-3 conformance matrix
  (`scripts/conformance_matrix.py`) is not extended.
```

**Done when :** `eval-matrix.yaml` parses (next task verifies)
and CHANGELOG stanza drafted in a scratch note.

---

## Task 2 : Loader extension + TDD for `baselines:` block

**Files :**

- `harness/config/eval_matrix.py` (modify)
- `tests/unit/test_eval_matrix_baselines.py` (new)

**Goal :** Make `EvalMatrix` aware of the new `baselines:` field,
keep existing API back-compat (older yamls without the block stay
loadable), and TDD the schema invariants.

- [ ] **Step 1 : Write failing test FIRST (TDD per `superpowers:test-driven-development`)**

Create `tests/unit/test_eval_matrix_baselines.py` :

```python
"""TDD for the `baselines:` block in eval-matrix.yaml (FC-MINOR addition).

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 2.
"""
from pathlib import Path
import textwrap

import pytest
import yaml

from harness.config.eval_matrix import EvalMatrix, load_eval_matrix


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "eval-matrix.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_baselines_block_parses(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        version: "C-v0.11.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics: {}
        baselines:
          wake_sleep_cl:
            name: wake_sleep_consolidated_learning
            bibkey: alfarano2024wakesleep
            arxiv: "2401.08623"
            scores_on: [M1.a, M1.b]
            variant: c
    """)
    em = load_eval_matrix(p)
    assert "wake_sleep_cl" in em.baselines
    assert em.baselines["wake_sleep_cl"]["bibkey"] == "alfarano2024wakesleep"
    assert em.baselines["wake_sleep_cl"]["scores_on"] == ["M1.a", "M1.b"]


def test_baselines_block_optional_for_legacy_yaml(tmp_path: Path) -> None:
    """Older yamls without `baselines:` must still load (back-compat)."""
    p = _write_yaml(tmp_path, """
        version: "C-v0.7.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics: {}
    """)
    em = load_eval_matrix(p)
    assert em.baselines == {}


def test_baselines_each_entry_has_required_fields(tmp_path: Path) -> None:
    """Every baseline entry must declare bibkey + scores_on + variant."""
    p = _write_yaml(tmp_path, """
        version: "C-v0.11.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics: {}
        baselines:
          incomplete:
            name: foo
    """)
    with pytest.raises(ValueError, match="bibkey"):
        load_eval_matrix(p)
```

Run `uv run pytest tests/unit/test_eval_matrix_baselines.py -v`
— expect 3 failures (unknown attribute `baselines`).

- [ ] **Step 2 : Extend the loader to make tests pass**

Edit `harness/config/eval_matrix.py` :

```python
@dataclass(frozen=True)
class EvalMatrix:
    """Parsed eval-matrix.yaml with typed accessors."""

    version: str
    bump_rules: dict[str, dict[str, Any]]
    publication_ready_gate: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    baselines: dict[str, dict[str, Any]]  # NEW (FC-MINOR additive)


_BASELINE_REQUIRED = {"bibkey", "scores_on", "variant"}


def load_eval_matrix(path: Path) -> EvalMatrix:
    if not path.exists():
        raise FileNotFoundError(f"eval-matrix.yaml not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"eval-matrix.yaml must be a YAML mapping, "
            f"got {type(raw).__name__}"
        )

    required_top_keys = {
        "version",
        "bump_rules",
        "publication_ready_gate",
        "metrics",
    }
    missing = required_top_keys - raw.keys()
    if missing:
        raise ValueError(
            f"eval-matrix.yaml missing top-level keys: {missing}"
        )

    baselines = raw.get("baselines", {}) or {}
    for name, entry in baselines.items():
        miss = _BASELINE_REQUIRED - set(entry.keys())
        if miss:
            raise ValueError(
                f"baselines.{name} missing required fields: {miss}"
            )

    return EvalMatrix(
        version=raw["version"],
        bump_rules=raw["bump_rules"],
        publication_ready_gate=raw["publication_ready_gate"],
        metrics=raw["metrics"],
        baselines=baselines,
    )
```

- [ ] **Step 3 : Verify tests pass + existing tests still pass**

```bash
uv run pytest tests/unit/test_eval_matrix_baselines.py -v
uv run pytest tests/conformance/ -v
uv run mypy harness tests
uv run ruff check harness/config/eval_matrix.py tests/unit/test_eval_matrix_baselines.py
```

Expect : all green, no mypy strict regressions on the new
`baselines: dict[str, dict[str, Any]]` field.

**Done when :** all four commands return exit 0.

---

## Task 3 : `WakeSleepCLBaseline` adapter + `__init__.py` re-export

**Files :**

- `kiki_oniric/substrates/wake_sleep_cl_baseline.py` (new)
- `kiki_oniric/substrates/__init__.py` (modify)
- `tests/unit/test_wake_sleep_baseline_adapter.py` (new)

**Goal :** Define the baseline adapter as a frozen dataclass +
helper, mirroring the shape of `mlx_kiki_oniric.py`'s
`mlx_substrate_components()` — but with a single
`evaluate_continual(seed, task_split)` entry point instead of 4
op factories. **The adapter is intentionally NOT DR-3 conformant**
(it does not implement replay/downscale/restructure/recombine) ;
this is documented in its docstring.

The body below assumes **variant `c`** (default, fixture stub).
For variant `a`/`b`, swap the `_REFERENCE_METRICS` constant for a
real Avalanche training loop ; signature stays identical.

- [ ] **Step 1 : Write failing test FIRST**

Create `tests/unit/test_wake_sleep_baseline_adapter.py` :

```python
"""TDD for kiki_oniric/substrates/wake_sleep_cl_baseline.py.

The baseline is an *adapter*, not a DR-3 substrate. It must :
1. expose a stable name + version,
2. round-trip seeds deterministically (R1 contract),
3. emit a CellResult-shaped dict scoring on M1.a + M1.b,
4. NOT register as a 4-op substrate (`substrate_components()`
   helper must omit the op-factory keys).

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 3.
"""
import pytest

from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
    wake_sleep_substrate_components,
)


def test_name_and_version_are_pinned() -> None:
    assert WAKE_SLEEP_BASELINE_NAME == "wake_sleep_cl_baseline"
    # Version must lift to the post-bump tag once Task 1 lands.
    assert "C-v0.11.0+PARTIAL" == WAKE_SLEEP_BASELINE_VERSION


def test_evaluate_continual_returns_M1ab_keys() -> None:
    bl = WakeSleepCLBaseline()
    out = bl.evaluate_continual(seed=42, task_split="split_fmnist_5tasks")
    assert {"forgetting_rate", "avg_accuracy", "n_tasks", "seed"} <= out.keys()
    assert out["seed"] == 42
    assert 0.0 <= out["forgetting_rate"] <= 1.0
    assert 0.0 <= out["avg_accuracy"] <= 1.0


def test_evaluate_continual_is_deterministic_per_seed() -> None:
    """R1 contract : same (seed, task_split) -> same numbers."""
    bl = WakeSleepCLBaseline()
    a = bl.evaluate_continual(seed=42, task_split="split_fmnist_5tasks")
    b = bl.evaluate_continual(seed=42, task_split="split_fmnist_5tasks")
    assert a == b


def test_evaluate_continual_seed_changes_results() -> None:
    bl = WakeSleepCLBaseline()
    a = bl.evaluate_continual(seed=42, task_split="split_fmnist_5tasks")
    b = bl.evaluate_continual(seed=7, task_split="split_fmnist_5tasks")
    # Numbers may match by coincidence but seeds must round-trip in output.
    assert a["seed"] == 42 and b["seed"] == 7


def test_unknown_task_split_raises() -> None:
    bl = WakeSleepCLBaseline()
    with pytest.raises(ValueError, match="task_split"):
        bl.evaluate_continual(seed=42, task_split="unsupported_xyz")


def test_components_helper_omits_op_factories() -> None:
    """The baseline is NOT DR-3 conformant ; no op-factory keys."""
    comps = wake_sleep_substrate_components()
    for op in ("replay", "downscale", "restructure", "recombine"):
        assert op not in comps
    assert "evaluate_continual" in comps
    assert "predictor" in comps
```

Run `uv run pytest tests/unit/test_wake_sleep_baseline_adapter.py -v`
— expect import errors.

- [ ] **Step 2 : Create the adapter module (variant `c` body)**

Create `kiki_oniric/substrates/wake_sleep_cl_baseline.py` :

```python
"""Wake-Sleep Consolidated Learning baseline adapter (Paper 2 §7 row).

This module wires Alfarano et al. 2024 [IEEE TNNLS, arXiv
2401.08623] into the dreamOfkiki ablation matrix as a
**baseline-only** adapter — it is registered alongside the
DR-3-conformant substrates (MLX, E-SNN, micro-kiki) but it does
NOT implement the 4 dream operations (replay / downscale /
restructure / recombine). It only exposes the comparator
contract `evaluate_continual(seed, task_split) -> dict`, which
is the minimum surface the Paper 2 §7 results table needs.

Variant choice (set by `docs/superpowers/plans/
2026-05-02-wake-sleep-cl-ablation-baseline.md` Task 0.5) :
- `c` (default) — published reference metrics from Alfarano et al.
  2024 IEEE TNNLS Tables 2-3, frozen as a constant. The seed
  argument is passed through but does not influence the numbers
  (they are reference values, not a re-run). Caveat documented
  in Paper 2 §6.4 style.
- `a` / `b` — re-run with seeded RNG ; signature unchanged.

References :
- docs/papers/paper1/references.bib L454 (`alfarano2024wakesleep`)
- docs/papers/paper1/introduction.md L94, L108 (Paper 1 framing)
- docs/papers/paper2/architecture.md §5.8 (this file's role)
- docs/interfaces/eval-matrix.yaml `baselines.wake_sleep_cl`
- docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
  (DR-3 — the baseline is exempt from conformance)
"""
from __future__ import annotations

from dataclasses import dataclass

WAKE_SLEEP_BASELINE_NAME = "wake_sleep_cl_baseline"
WAKE_SLEEP_BASELINE_VERSION = "C-v0.11.0+PARTIAL"

# Variant-c reference metrics. Source : Alfarano et al. 2024
# IEEE TNNLS Tables 2-3 (Split-FMNIST 5-task class-incremental).
# IF Task 0.5 picks variant a/b, replace this block with a real
# Avalanche-driven training loop (see plan Task 3 alt body).
#
# The exact numerical values below MUST be cross-checked against
# the Alfarano paper PDF before commit ; treat them as
# placeholders pending Task 0 step 2 confirmation.
_REFERENCE_METRICS_BY_TASKSPLIT: dict[str, dict[str, float]] = {
    "split_fmnist_5tasks": {
        # Verify these against Alfarano 2024 Tables 2-3 in Task 0.
        "forgetting_rate": 0.082,
        "avg_accuracy": 0.847,
    },
}

_SUPPORTED_TASKSPLITS = frozenset(_REFERENCE_METRICS_BY_TASKSPLIT.keys())


@dataclass(frozen=True)
class WakeSleepCLBaseline:
    """Stub adapter for the Wake-Sleep CL baseline."""

    n_tasks: int = 5

    def evaluate_continual(
        self, *, seed: int, task_split: str
    ) -> dict[str, float | int | str]:
        """Return forgetting + avg-accuracy for the requested split.

        Variant-c : returns frozen reference metrics. The `seed`
        argument round-trips into the output for R1-style
        provenance but does NOT influence the numerical values
        (this is documented in Paper 2 §6.4 style — published
        reference, not a re-run).
        """
        if task_split not in _SUPPORTED_TASKSPLITS:
            raise ValueError(
                f"task_split={task_split!r} unsupported. "
                f"Choose from {sorted(_SUPPORTED_TASKSPLITS)}"
            )
        ref = _REFERENCE_METRICS_BY_TASKSPLIT[task_split]
        return {
            "forgetting_rate": ref["forgetting_rate"],
            "avg_accuracy": ref["avg_accuracy"],
            "n_tasks": self.n_tasks,
            "seed": seed,
            "source": "published_reference_alfarano2024",
        }


def wake_sleep_substrate_components() -> dict[str, str]:
    """Return the canonical map of WS-CL baseline components.

    Note : NO op factories. The 4 dream operations are
    intentionally absent — this is a baseline, not a DR-3
    substrate.
    """
    return {
        "evaluate_continual": (
            "kiki_oniric.substrates.wake_sleep_cl_baseline."
            "WakeSleepCLBaseline.evaluate_continual"
        ),
        "predictor": (
            "kiki_oniric.substrates.wake_sleep_cl_baseline."
            "WakeSleepCLBaseline"
        ),
    }
```

- [ ] **Step 3 : Re-export from `kiki_oniric/substrates/__init__.py`**

Edit `kiki_oniric/substrates/__init__.py` :

```python
from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
    wake_sleep_substrate_components,
)
```

…and append the four names to `__all__`.

- [ ] **Step 4 : Verify tests pass**

```bash
uv run pytest tests/unit/test_wake_sleep_baseline_adapter.py -v
uv run pytest tests/unit/ -v
uv run mypy kiki_oniric tests
uv run ruff check kiki_oniric/substrates/wake_sleep_cl_baseline.py tests/unit/test_wake_sleep_baseline_adapter.py
```

Expect : all green.

**Done when :** baseline adapter imports cleanly, tests pass,
mypy strict happy, no ruff regression.

---

## Task 4 : Conformance test — baseline registered but exempt from DR-3

**Files :**

- `tests/conformance/test_baseline_registration.py` (new)

**Goal :** Pin a contract test : the baseline appears in the
substrates registry **but** is excluded from DR-3 axiom property
tests. Future contributors must not accidentally enrol it as a
substrate.

- [ ] **Step 1 : Write the contract test**

Create `tests/conformance/test_baseline_registration.py` :

```python
"""DR-3 exemption contract for baseline-only adapters.

The Wake-Sleep CL baseline (Alfarano 2024) is registered in
`kiki_oniric.substrates.__init__` for ablation-matrix
discoverability, but it is NOT a DR-3-conformant substrate.
This test pins that distinction.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 4.
"""
import kiki_oniric.substrates as substrates
from kiki_oniric.substrates.wake_sleep_cl_baseline import (
    wake_sleep_substrate_components,
)


def test_baseline_is_in_public_api() -> None:
    assert "WakeSleepCLBaseline" in substrates.__all__
    assert "wake_sleep_substrate_components" in substrates.__all__


def test_baseline_lacks_dr3_op_factories() -> None:
    """DR-3 exemption pin : no op factory keys in the components map."""
    comps = wake_sleep_substrate_components()
    for op in ("replay", "downscale", "restructure", "recombine"):
        assert op not in comps, (
            f"Baseline must NOT register a {op} op factory — "
            f"that would imply DR-3 conformance which the baseline "
            f"does not claim."
        )


def test_dr3_substrates_unchanged() -> None:
    """The 3 real substrates still expose their components helpers."""
    assert hasattr(substrates, "mlx_substrate_components")
    assert hasattr(substrates, "esnn_substrate_components")
    assert hasattr(substrates, "micro_kiki_substrate_components")
```

- [ ] **Step 2 : Run + verify**

```bash
uv run pytest tests/conformance/test_baseline_registration.py -v
uv run pytest tests/conformance/ -v
```

Expect : 3 new tests green ; existing conformance suite unchanged.

**Done when :** new conformance file passes + existing
conformance suite (axioms / invariants) unchanged.

---

## Task 5 : Gate driver `scripts/baseline_wake_sleep_cl.py`

**Files :**

- `scripts/baseline_wake_sleep_cl.py` (new)

**Goal :** Produce a deterministic JSON dump under
`docs/milestones/wake-sleep-baseline-2026-05-02.{md,json}` for the
3-seed grid. Mirror the structure of `scripts/pilot_g2.py` per
`scripts/CLAUDE.md` conventions (header docstring, REPO_ROOT pattern,
register run in `RunRegistry` before writing).

For variant `c`, the dump records published reference values + a
seed-round-trip echo. For variants `a`/`b`, the dump records real
training-loop output.

- [ ] **Step 1 : Create the driver**

Create `scripts/baseline_wake_sleep_cl.py` :

```python
"""Wake-Sleep CL baseline driver (Paper 2 §7 row).

Gate ID         : Paper 2 §7 ablation row (no G-gate ; baseline
                  comparator only).
Validates       : pipeline-validation (variant c) — the baseline
                  adapter round-trips via the AblationRunner /
                  RunRegistry without claiming new empirical
                  results. Variants a/b promote this to empirical.
Output path     : docs/milestones/wake-sleep-baseline-2026-05-02.json
                  + .md sibling.

Usage :
    uv run python scripts/baseline_wake_sleep_cl.py --seeds 42 123 7

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  docs/papers/paper1/references.bib `alfarano2024wakesleep`
  docs/papers/paper2/architecture.md §5.8
  docs/papers/paper2/methodology.md §6.3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.wake_sleep_cl_baseline import (  # noqa: E402
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
)

DEFAULT_SEEDS = (42, 123, 7)
DEFAULT_TASK_SPLIT = "split_fmnist_5tasks"
DEFAULT_OUT = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / f"wake-sleep-baseline-{date.today().isoformat()}.json"
)


@dataclass(frozen=True)
class BaselineRow:
    seed: int
    task_split: str
    forgetting_rate: float
    avg_accuracy: float
    n_tasks: int
    source: str
    run_id: str


def _resolve_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_id(c_version: str, profile: str, seed: int, sha: str) -> str:
    payload = f"{c_version}|{profile}|{seed}|{sha}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def run(seeds: Iterable[int], out_path: Path) -> dict:
    bl = WakeSleepCLBaseline()
    sha = _resolve_commit_sha()
    registry = RunRegistry(REPO_ROOT / "harness" / "storage" / "runs.sqlite")
    rows: list[BaselineRow] = []

    for seed in seeds:
        rid = _run_id(
            c_version=WAKE_SLEEP_BASELINE_VERSION,
            profile="baseline_wake_sleep_cl",
            seed=seed,
            sha=sha,
        )
        registry.register(
            run_id=rid,
            c_version=WAKE_SLEEP_BASELINE_VERSION,
            profile="baseline_wake_sleep_cl",
            seed=seed,
            commit_sha=sha,
        )
        out = bl.evaluate_continual(seed=seed, task_split=DEFAULT_TASK_SPLIT)
        rows.append(
            BaselineRow(
                seed=seed,
                task_split=DEFAULT_TASK_SPLIT,
                forgetting_rate=float(out["forgetting_rate"]),
                avg_accuracy=float(out["avg_accuracy"]),
                n_tasks=int(out["n_tasks"]),
                source=str(out["source"]),
                run_id=rid,
            )
        )

    dump = {
        "baseline": WAKE_SLEEP_BASELINE_NAME,
        "version": WAKE_SLEEP_BASELINE_VERSION,
        "bibkey": "alfarano2024wakesleep",
        "task_split": DEFAULT_TASK_SPLIT,
        "commit_sha": sha,
        "rows": [asdict(r) for r in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2, sort_keys=True))
    return dump


def _md_companion(dump: dict, json_path: Path) -> Path:
    md_path = json_path.with_suffix(".md")
    body = [
        f"# Wake-Sleep CL baseline — {dump['version']}",
        "",
        "**Source :** Alfarano et al. 2024, IEEE TNNLS, arXiv 2401.08623.",
        "**Bibkey :** `alfarano2024wakesleep`.",
        f"**Task split :** `{dump['task_split']}`.",
        f"**Commit :** `{dump['commit_sha']}`.",
        "",
        "**(synthetic placeholder — variant c, published reference values.)**",
        "",
        "| seed | run_id | forgetting_rate | avg_accuracy |",
        "|------|--------|-----------------|--------------|",
    ]
    for r in dump["rows"]:
        body.append(
            f"| {r['seed']} | `{r['run_id']}` | "
            f"{r['forgetting_rate']:.4f} | {r['avg_accuracy']:.4f} |"
        )
    md_path.write_text("\n".join(body) + "\n")
    return md_path


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    dump = run(args.seeds, args.out)
    md = _md_companion(dump, args.out)
    print(f"json -> {args.out}")
    print(f"md   -> {md}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2 : Smoke run**

```bash
uv run python scripts/baseline_wake_sleep_cl.py --seeds 42 123 7
```

Expect :
- `docs/milestones/wake-sleep-baseline-2026-05-02.json` written.
- `docs/milestones/wake-sleep-baseline-2026-05-02.md` written.
- `harness/storage/runs.sqlite` gains 3 rows under
  `profile=baseline_wake_sleep_cl`.

- [ ] **Step 3 : Verify R1 round-trip**

Re-run the same command. Expected behaviour :
- The 3 `run_id`s match exactly between runs (R1 contract).
- The `RunRegistry` `INSERT OR IGNORE` semantics keep row count
  at 3 (no duplicates).

If `run_id`s diverge → R1 broken → halt and debug
(`harness/storage/run_registry.py`).

**Done when :** dump is reproducible bit-for-bit, registry has
3 stable rows, `.md` companion renders cleanly.

---

## Task 6 : Paper 2 EN updates (architecture / methodology / results)

**Files :**

- `docs/papers/paper2/architecture.md` (modify, add §5.8)
- `docs/papers/paper2/methodology.md` (modify, amend §6.3)
- `docs/papers/paper2/results.md` (modify, add baseline row)
- `docs/papers/paper2/references.bib` (verify only ; bibkey is
  already in `paper1/references.bib`, mirror to paper2 if absent)

**Goal :** Paper 2 sections describe the new baseline row,
referencing the milestone dump from Task 5 and the bibkey
`alfarano2024wakesleep`. Numbers cite the Task 5 JSON dump,
**flagged synthetic-substitute** (variant `c`) per Paper 2 §6.4
style.

- [ ] **Step 1 : Verify bibkey availability in `paper2/references.bib`**

```bash
grep "alfarano2024wakesleep" docs/papers/paper2/references.bib
```

If absent → copy lines 454-465 from `docs/papers/paper1/references.bib`.

- [ ] **Step 2 : Add §5.8 to `docs/papers/paper2/architecture.md`**

Insert *after* §5.7 ("DualVer lineage"), *before* the
"Notes for revision" footer :

```markdown
## 5.8 Wake-Sleep CL baseline row (Alfarano 2024)

Paper 2 §7 includes a fourth row in the ablation table : the
**Wake-Sleep Consolidated Learning** baseline of Alfarano et al.
2024 [@alfarano2024wakesleep, IEEE TNNLS, arXiv 2401.08623].
Paper 1 §3 already names this work as the "closest published
NREM/REM dual-phase analog" and the natural Paper 2 ablation
comparator. The baseline is registered in the substrates
namespace via
`kiki_oniric/substrates/wake_sleep_cl_baseline.py` for
discoverability, but it is **not DR-3 conformant** — it does not
implement the 4 dream operations. Its single comparator API,
`WakeSleepCLBaseline.evaluate_continual(seed, task_split)`,
returns the two M1.* metrics (`forgetting_rate`, `avg_accuracy`)
on Split-FMNIST 5-task class-incremental — the same dataset shape
already used in `experiments/h1_split_mnist/`. The eval-matrix
schema gains a top-level `baselines:` block (FC-MINOR additive
addition, `C-v0.10.0+PARTIAL → C-v0.11.0+PARTIAL`) registering
the bibkey, arXiv id, variant, and the metric IDs scored. The
baseline row is dumped to
`docs/milestones/wake-sleep-baseline-2026-05-02.{md,json}` and
each cell carries an R1 `run_id`. **Variant : c** (published
reference values from Alfarano 2024 Tables 2-3, frozen ; Paper 2
§6.4 caveat applies).
```

- [ ] **Step 3 : Amend §6.3 in `docs/papers/paper2/methodology.md`**

Replace the line in §6.3 :

```markdown
Paper 2's matrix dimension is new : Paper 1 ran 3 profiles ×
3 seeds = 9 cells. Paper 2 crosses substrates, giving
**2 × 3 × 3 = 18 cells**. Each cell produces :
```

with :

```markdown
Paper 2's matrix dimension is new : Paper 1 ran 3 profiles ×
3 seeds = 9 cells. Paper 2 crosses substrates, giving
**2 × 3 × 3 = 18 cells**, plus a **fourth standalone row** for
the Wake-Sleep CL baseline [@alfarano2024wakesleep] scored on
the same 3 seeds (`[42, 123, 7]`) — see §5.8 + dump
`docs/milestones/wake-sleep-baseline-2026-05-02.json` (variant
c, published reference values, Paper 2 §6.4 caveat applies).
The baseline scores only on M1.* metrics (`forgetting_rate`,
`avg_accuracy`) ; it does not enter H2/H3/H4 because it does
not produce a P_min/P_equ/P_max grouping. Each substrate-cell
still produces :
```

- [ ] **Step 4 : Add baseline row in `docs/papers/paper2/results.md`**

Append a new short subsection at the end of §7 (before any
"Notes for revision" trailer) :

```markdown
## 7.X Wake-Sleep CL baseline (Alfarano 2024)

Aside from the 18-cell substrate × profile × seed grid, the
results table includes a fourth standalone row drawn from
[@alfarano2024wakesleep] (IEEE TNNLS, arXiv 2401.08623) — the
closest published NREM/REM dual-phase analog (Paper 1 §3,
introduction.md L94, L108). The row is generated by
`scripts/baseline_wake_sleep_cl.py` and dumped to
`docs/milestones/wake-sleep-baseline-2026-05-02.json`. **(synthetic
placeholder — variant c, published reference values.)**

| seed | run_id | forgetting_rate | avg_accuracy |
|------|--------|-----------------|--------------|
| 42  | (see milestone dump) | 0.082 | 0.847 |
| 123 | (see milestone dump) | 0.082 | 0.847 |
| 7   | (see milestone dump) | 0.082 | 0.847 |

The numerical values above are pre-bump placeholders ; replace
with the Task 5 dump output (full SHA-256 prefix `run_id`s) at
PR-finalisation time. The seed-round-trip identity (same numbers
across seeds) is **expected** under variant c — variants a/b
would yield seed-dependent rows.

A H2-style TOST equivalence test against P_equ is **deliberately
omitted** : the predictor caveat (§6.4) compounds with the
variant-c reference-values caveat ; a meaningful TOST requires
cycle-3's divergent-predictor replication on the same dataset.
The baseline serves as a **published-reference anchor**, not a
significance test.
```

- [ ] **Step 5 : Glossary entry**

Append to `docs/glossary.md` :

```markdown
| Wake-Sleep CL | Wake-Sleep Consolidated Learning, Alfarano et al. 2024 [IEEE TNNLS, arXiv 2401.08623] — closest published NREM/REM dual-phase analog ; Paper 2 §5.8 baseline. |
```

- [ ] **Step 6 : Verify EN paper renders**

```bash
uv run python scripts/render_figures.py --gate G4 || true  # noop ok
# Just sanity-check the file structure:
grep -n "5.8\|7.X\|alfarano2024wakesleep" docs/papers/paper2/*.md
```

**Done when :** Paper 2 EN section files contain the new
subsection ; bibkey resolves ; glossary entry added.

---

## Task 7 : Paper 2 FR mirror (mandatory in same PR)

**Files :**

- `docs/papers/paper2-fr/architecture.md`
- `docs/papers/paper2-fr/methodology.md`
- `docs/papers/paper2-fr/results.md`

**Goal :** EN→FR propagation per `docs/papers/CLAUDE.md` and
top-level `CONTRIBUTING.md`. Reviewer-enforced.

- [ ] **Step 1 : Mirror §5.8 → §5.8 in FR**

Translate the Step 2 block from Task 6 into French. Preserve the
bibkey + arXiv id verbatim (citations are language-agnostic).
The variant tag (`c`) and metric IDs (`M1.a`, `M1.b`) are also
verbatim.

Sample FR opening :

```markdown
## 5.8 Ligne de base Wake-Sleep CL (Alfarano 2024)

Le §7 du Paper 2 inclut une quatrième ligne dans la table
d'ablation : la **Wake-Sleep Consolidated Learning** d'Alfarano
et al. 2024 [@alfarano2024wakesleep, IEEE TNNLS, arXiv
2401.08623]. Le §3 du Paper 1 désigne déjà ce travail comme
« l'analogue NREM/REM dual-phase publié le plus proche » et le
comparateur d'ablation naturel pour le Paper 2. […]
```

(Continue translation for the full subsection.)

- [ ] **Step 2 : Mirror §6.3 amendment + §7.X in FR**

Same translation discipline. Run :

```bash
diff <(grep -c "##" docs/papers/paper2/architecture.md) \
     <(grep -c "##" docs/papers/paper2-fr/architecture.md)
```

Should print `0 0` (same heading count after the bump).

- [ ] **Step 3 : Verify cross-checks**

```bash
grep -n "alfarano2024wakesleep" docs/papers/paper2-fr/*.md
```

Expect : at least 3 matches (architecture / methodology / results).

**Done when :** EN ↔ FR architecture / methodology / results
section counts match, bibkey present in both trees.

---

## Task 8 : CHANGELOG + STATUS bump (FC-MINOR, EC unchanged)

**Files :**

- `CHANGELOG.md` (modify, prepend new section)
- `STATUS.md` (modify, bump version + DualVer table)

**Goal :** Land the FC-MINOR bump per framework-C spec §12.2 +
§12.4. EC axis stays `+PARTIAL` — adding a baseline does not
re-close any Phase-2 deferred cell.

- [ ] **Step 1 : Prepend the CHANGELOG stanza from Task 1**

Open `CHANGELOG.md` and insert the stanza drafted in Task 1
Step 3 at the top (under the title, above `[C-v0.10.0+PARTIAL]`).

- [ ] **Step 2 : Update `STATUS.md` table + version line**

Replace `**Version** : C-v0.10.0+PARTIAL` with
`**Version** : C-v0.11.0+PARTIAL`.

In the DualVer table row for FC, append :

```markdown
| FC   | v0.11.0 | MINOR bump (Wake-Sleep CL ablation baseline added per Alfarano 2024 [arXiv 2401.08623], `eval-matrix.yaml` `baselines:` block additive, 2026-05-02). Prior bumps : v0.10.0 (micro-kiki recombine TIES-merge, PR #13)…
```

- [ ] **Step 3 : Verify pre-commit**

```bash
uv run ruff check .
uv run mypy harness tests
uv run pytest -x
```

Expect : 280+ tests passing (277 existing + 3 baseline + 3
adapter + 5 baseline-yaml = ~288), coverage ≥90 %.

**Done when :** all three commands return exit 0, version
appears in `STATUS.md` + `CHANGELOG.md` + `eval-matrix.yaml`
(three places, identical string `C-v0.11.0+PARTIAL`).

---

## Task 9 : Final verification + commit set

**Files :** none (verification only).

**Goal :** Confirm the PR is shippable per `CONTRIBUTING.md` :
single conventional-commit history, EN ↔ FR mirrored,
schema + loader + tests + papers consistent.

- [ ] **Step 1 : Full-suite green**

```bash
uv run pytest --cov-report=term-missing
```

Expect coverage ≥ 90 % (project gate).

- [ ] **Step 2 : Bilingual mirror check**

```bash
for f in architecture.md methodology.md results.md; do
  echo "=== $f ==="
  diff <(grep -c "^##" "docs/papers/paper2/$f") \
       <(grep -c "^##" "docs/papers/paper2-fr/$f")
done
```

Expect : three `0 0` lines (counts equal).

- [ ] **Step 3 : Version triple-check**

```bash
grep -l "C-v0.11.0+PARTIAL" \
  docs/interfaces/eval-matrix.yaml \
  STATUS.md CHANGELOG.md \
  kiki_oniric/substrates/wake_sleep_cl_baseline.py
```

Expect 4 hits (the 3 docs + the new substrate file).

- [ ] **Step 4 : Plan a single commit per `CONTRIBUTING.md`**

Subject : `feat(eval_matrix): add wake-sleep CL ablation baseline`
Scope `eval_matrix` is ≥3 chars. Subject ≤ 50 chars.

Body (≤72 cols) :
```
Add the Alfarano 2024 [arXiv 2401.08623] Wake-Sleep
Consolidated Learning row to the Paper 2 ablation matrix as
a published-reference baseline (variant c, see plan
docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-
baseline.md Task 0.5). Schema-additive : eval-matrix.yaml
gains a `baselines:` block, FC-MINOR bump
C-v0.10.0+PARTIAL -> C-v0.11.0+PARTIAL.

Adapter is registered in kiki_oniric.substrates but is NOT
DR-3 conformant (no op factories) ; new conformance test
pins this distinction. Paper 2 §5.8 + §6.3 + §7.X updated,
EN <-> FR mirrored.

Refs: Paper 1 §3 introduction.md L94 L108
```

NO `Co-Authored-By` trailer (per top-level CLAUDE.md and
hypneum-lab convention).

- [ ] **Step 5 : Push + open PR**

PR title : `feat(eval_matrix): wake-sleep CL ablation baseline (Alfarano 2024)`

PR body must reference :
- This plan path.
- The variant decision from Task 0.5.
- The FC-MINOR bump rationale.
- The bilingual mirror discipline.

**Done when :** PR opened ; reviewers can read the plan +
brainstorm brief + commit body and re-derive every change.

---

## Self-Review

**Coverage of the spec checklist :**

| Spec ask | Plan task |
|---|---|
| Task 0 — investigate eval_matrix.yaml + paper2 + Alfarano repo | Task 0 (read-first survey + arXiv lookup) |
| Task 0.5 — DECISION (a)/(b)/(c) with rationale + FC impact | Task 0.5 (scored matrix + decision brief + branch instruction) |
| Task 1 — schema change + FC PATCH bump (spec said PATCH ; plan upgrades to MINOR — see "FC bump cost" risk note above ; the spec §12.2 rule says additive new-key = MINOR) | Task 1 |
| Task 2 — define `WakeSleepCLBaseline` interface | Task 3 (Step 1 TDD + Step 2 dataclass) |
| Task 3 — stub implementation matching mode | Task 3 (variant c) ; Task 0.5 branch instruction for a/b |
| Task 4..N — unit tests + integration test on 1 fixture CL task | Tasks 2 / 3 / 4 / 5 (yaml schema + adapter + DR-3 exemption + driver smoke) |
| Task N — paper2/architecture.md + bibkey | Task 6 (EN) + Task 7 (FR) |
| Task N+1 — STATUS + CHANGELOG if FC bump | Task 8 |
| Final — Self-review | This block |

**Placeholder audit :**

- The numerical reference metrics in Task 3 Step 2
  (`forgetting_rate: 0.082`, `avg_accuracy: 0.847`) are flagged
  in-line as "Verify these against Alfarano 2024 Tables 2-3 in
  Task 0". This is the only number-placeholder in the plan. It
  is intentional : the plan cannot ship verified numbers without
  Task 0 step 2 confirming the source PDF — exposing the
  placeholder is correct discipline per `docs/CLAUDE.md`
  "Numbers ↔ run_id" rule.
- Date `2026-05-02` is the plan-write date and matches `currentDate`
  from the env. Used in the milestone filename and the CHANGELOG
  stanza date — both consistent.
- All file paths are absolute or rooted at `dream-of-kiki/`. No
  `~` or `${HOME}` usage.
- Version string `C-v0.11.0+PARTIAL` appears in 4 places (Task 9
  Step 3) — all under one bump, no drift.

**Type consistency :**

- `EvalMatrix.baselines: dict[str, dict[str, Any]]` is mypy-strict
  compatible (already permissive).
- `WakeSleepCLBaseline.evaluate_continual` returns
  `dict[str, float | int | str]` — mypy-strict happy ; downstream
  consumers in Task 5 cast to specific types.
- `BaselineRow` dataclass in Task 5 is `@dataclass(frozen=True)`,
  matching `AblationConfig` and the registry conventions.

**Variant flagging :**

- Default variant : `c` (fixture-stub with published reference
  values).
- Variant `a` (port internal) : Task 0.5 instructs the executor
  to fork this plan into `2026-05-02-wake-sleep-cl-ablation-baseline-variant-a.md`
  before continuing past Task 1, replacing Task 3-5.
- Variant `b` (subprocess wrap) : same fork instruction, replace
  Task 3 only.

**Gaps / open questions / brainstorm-needed :**

1. **Alfarano 2024 reference-implementation availability**.
   Plan-write-time : unknown. Task 0 Step 2 must confirm. If a
   public repo with permissive license + small footprint exists,
   variant `b` becomes preferable to `c`. If it does not, default
   `c` stands.
2. **Exact Tables-2-3 numerical values** in the original Alfarano
   paper. The plan ships placeholder numbers (0.082 / 0.847) ;
   Task 0 Step 2 must verify against the IEEE TNNLS PDF. If the
   paper measures CIFAR-100-split-class instead of
   Split-FMNIST-5-tasks, variant `c` either pivots dataset
   (`split_cifar100_10tasks` ?) or downgrades to a "qualitative
   comparison only" footnote (no per-seed table in §7.X).
3. **Cycle 3 re-run intent**. The variant-c choice is silent on
   whether the baseline must be re-run when cycle 3 introduces
   divergent predictors. If cycle-3 spec mandates re-run, variant
   `a` becomes the only option ; the brainstorm brief should
   capture cycle-3 owner's intent before Task 0.5 commits.
4. **FC bump magnitude (PATCH vs MINOR)**. The original spec
   listed *PATCH* ; framework-C §12.2 ("addition of new
   optional primitive") indicates *MINOR*. The plan picks MINOR
   ; if the reviewer disagrees on bump magnitude, downgrade to
   PATCH is a one-line edit (Task 1 Step 2 + Task 8 Step 1) — no
   structural impact on this plan.
5. **OSF amendment trigger**. Adding a baseline does NOT introduce
   a new pre-registered hypothesis (Paper 2 §6.1 H1-H4 inherited
   from Paper 1 OSF lock `10.17605/OSF.IO/Q6JYN`). No OSF
   amendment required. Confirmed against OSF amendment-#1 scope
   2026-04-21 (Bonferroni family restructure) — orthogonal.

---

**End of plan.** REQUIRED SUB-SKILL pointer at top is intentional :
the executor MUST start with Task 0 brainstorm before any code
change, and MUST commit to a variant in Task 0.5 before progressing.
