# G6 micro-kiki Qwen-35B + MMLU Subdomain CL Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First validation of framework C on a real LLM (Qwen3.6-35B-A3B + LoRA adapters) against an MMLU subdomain continual-learning stream, measuring catastrophic forgetting under four arms (`baseline`, `P_min`, `P_equ`, `P_max`).

**Architecture:** Drives the existing `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` over a 5-subdomain MMLU sequence. Each cell adapts (or, in inference-only path, swaps) LoRA adapter tensors per subdomain, runs an optional dream episode (profile-dependent), then evaluates accuracy on every subdomain seen so far. Forgetting = mean drop on prior subdomains after training the current one. Per-cell wall time and acc matrix are dumped to a milestone JSON; verdict aggregates Hedges' g vs Hu 2020 / Javadi 2024 floors plus a Jonckheere monotonicity test, mirroring the G4 driver pattern.

**Tech Stack:** Python 3.12, MLX 0.18+, `mlx_lm` (LoRA training, optional / Studio-only), numpy, scipy, pytest, hypothesis. Driver lives under `experiments/g6_mmlu_stream/`. Milestone outputs land in `docs/milestones/g6-pilot-*.{json,md}`.

---

## Critical context the executing agent must read first

1. `STATUS.md` — current version `C-v0.12.0+PARTIAL`. G4 partial confirmation noted; G6 must declare its dependency on G4-bis (replay coupling MLX) confirming non-zero coupling effect before launching the full sweep. If G4-bis verdict is null/decrement, G6 is **BLOCKED** (see Task 0.5 decision matrix below).
2. `kiki_oniric/substrates/micro_kiki.py` (1188 lines) — exposes 4 numpy-only handlers (`replay_handler_factory`, `downscale_handler_factory`, `restructure_handler_factory` (OPLoRA), `recombine_handler_factory` (TIES-Merge)). **It does NOT expose a LoRA fine-tune loop.** The `mlx_lm` lora training path lives in the sibling `~/KIKI-Mac_tunner/` workspace, which is **NOT** present on `localhost` (`/Users/electron/`). The plan below splits into two paths accordingly.
3. `harness/real_benchmarks/mmlu.py` — `MMLULoader` (subdomain-aware via `MMLURecord.subject`), `evaluate_mmlu(model, tokenizer, *, n_samples, seed, fixture_path)` (letter-argmax proxy), and `_load_mmlu_records(...)` with R1-pinned SHA-256 enforcement. The fallback `tests/fixtures/mmlu_sanity.jsonl` is a 30-row toy mix across ~9 subjects — **not enough for a real subdomain split**. Task 3 must verify the full `cais/mmlu` HF cache is materialised on the run host.
4. `harness/real_models/base_model_registry.py` — Qwen pins. Use `qwen3p6-35b-bf16-local` (Studio-local, no SHA, exploratory pilot tag) for the full-pilot path; fall back to `qwen3p5-1p5b-fp16` for the smoke test.
5. `experiments/g4_split_fmnist/run_g4.py` — driver pattern to mirror (CLI, `_run_cell`, `_aggregate_verdict`, `_render_md_report`, `run_pilot`, `RunRegistry.register` per cell).
6. `docs/osf-prereg-g4-pilot.md` — pre-reg template to mirror.
7. `scripts/pilot_cycle3_real.py` — production-grade fresh-wrapper pattern for FP16 Qwen + dream-op SGD. The `_replay_qwen_handler` (CE-loss SGD over Qwen on (input, target) token pairs) is the canonical micro-pattern for the **full-pilot** training path. **Do not copy it verbatim** into G6 — G6 must invoke it through the substrate's `replay_handler_factory` so DR-0 / DR-1 stamps land on `MicroKikiSubstrate._restructure_state` / `_recombine_state`.
8. `docs/CLAUDE.md` — Milestone files (`docs/milestones/`, `docs/osf-*.md`) are **append-only dated immutables**. Never edit a previous G-pilot dump.
9. `kiki_oniric/CLAUDE.md` — substrate package; do not loosen primitive types or rename methods (DR-3 conformance).

## Workspace dependency status (verified 2026-05-03)

```
/Users/electron/hypneum-lab/dream-of-kiki     <- this repo
/Users/electron/hypneum-lab/micro-kiki        <- runtime + configs (no weights)
/Users/electron/KIKI-Mac_tunner               <- MISSING (training pipeline + datasets + mlx-lm fork)
/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B  <- Studio-only (referenced by qwen3p6-35b-bf16-local pin)
```

The plan offers **two paths**, selected at Task 0.5:

- **Path A — Full pilot** (Studio-only, requires `KIKI-Mac_tunner` accessible + Qwen3.6-35B weights on disk + `mlx_lm.lora` training callable). Genuine LoRA fine-tune per subdomain; dream episode mutates LoRA adapter tensors via the four handlers; this is the publishable G6 result.
- **Path B — Inference-only pilot** (any host, fallback). No fine-tuning; subdomain "training" is replaced by adapter-tensor manipulation via handler factories on a synthetic LoRA stack matching the production rank/shape. Measures whether dream episodes alone produce a forgetting signature on the existing pre-trained adapter; **flagged as exploratory only**, not a publishable G6 confirmation.

**Compute budget** :
- Path A : 5 subdomains × 4 arms × 3 seeds = **60 cells**. Per-cell wall time on Studio M3 Ultra = ~10–25 min (LoRA train 100 examples × 5 inner steps × 35B-A3B + 5×eval@100 samples). **Total : 10–25 h on Studio.** Smoke (Task 9) is 1 cell at ~15 min.
- Path B : 60 cells × ~30 s/cell on any Apple Silicon (no fine-tune; handler ops + 5×eval@100 samples on the 1.5B-fp16 fallback). **Total : ~30 min** but the result is exploratory.

---

## File structure

Files **created** by this plan :

```
experiments/g6_mmlu_stream/
├── __init__.py                             # package marker
├── run_g6.py                               # CLI driver (mirrors run_g4.py)
├── stream.py                               # MMLU subdomain stream loader
├── micro_kiki_train.py                     # Path A LoRA train shim (no-op stub on Path B hosts)
├── micro_kiki_inference.py                 # Path B adapter-tensor mutation shim
└── dream_wrap.py                           # dream-episode wrapper using the 4 handler factories

tests/unit/experiments/
├── test_g6_stream.py                       # subdomain stream loader tests
├── test_g6_dream_wrap.py                   # dream-episode dispatch + DR-0 stamp tests
├── test_g6_micro_kiki_train.py             # LoRA train shim contract tests (mocked)
├── test_g6_micro_kiki_inference.py         # Path B adapter-tensor mutation tests
└── test_g6_run_pilot.py                    # end-to-end pilot smoke (synthetic fixture)

docs/
├── osf-prereg-g6-pilot.md                  # pre-reg (locked before any cell registers)
└── milestones/                             # output target (created at run time)
    ├── g6-pilot-2026-05-03.json            # machine dump
    └── g6-pilot-2026-05-03.md              # human report
```

Files **modified** by this plan :

```
docs/papers/paper2/results.md               # add §7.1.4 G6 results subsection
docs/papers/paper2-fr/results.md            # FR mirror (same edit; EN→FR rule)
CHANGELOG.md                                # [Unreleased] empirical row + DualVer entry
STATUS.md                                   # G6 row in Gates table; version bump if STABLE promo
```

---

## Task 0: Read the codebase and lock the path decision

**Files (read-only):**
- `kiki_oniric/substrates/micro_kiki.py:425-1188` — handler factories signatures, OPLoRA + TIES-Merge algebra, `_current_delta` accumulator semantics
- `harness/real_benchmarks/mmlu.py:1-435` — `MMLULoader`, `MMLURecord`, `evaluate_mmlu`, `_load_mmlu_records`
- `harness/real_models/qwen_mlx_fp16.py` — wrapper API (`.model`, `.tokenizer`, `.parameters()`, `.update_parameters(...)`)
- `harness/real_models/base_model_registry.py:91-275` — pin entries; note `qwen3p6-35b-bf16-local` is exploratory (no SHA), `qwen3p5-1p5b-fp16` is the smoke fallback
- `experiments/g4_split_fmnist/run_g4.py:1-609` — driver pattern (CLI, _run_cell, _aggregate_verdict, _render_md_report, run_pilot, registry.register per cell)
- `experiments/g4_split_fmnist/dream_wrap.py:1-254` — `DreamEpisode` construction with `operation_set`, `output_channels`, `BudgetCap` per profile
- `scripts/pilot_cycle3_real.py:300-450` — `_replay_qwen_handler` CE-SGD pattern for Path A
- `docs/osf-prereg-g4-pilot.md:1-153` — pre-reg template

- [ ] **Step 1: Verify workspace state on the run host**

Run :
```bash
ls -ld /Users/electron/KIKI-Mac_tunner /Users/clems/KIKI-Mac_tunner 2>&1
test -d ~/KIKI-Mac_tunner && echo "PATH_A_AVAILABLE" || echo "PATH_A_MISSING"
uv run python -c "import mlx_lm; print(mlx_lm.__version__)" 2>&1
uv run python -c "from datasets import load_dataset; \
  ds=load_dataset('cais/mmlu','all',split='test'); print(len(ds), sorted({r['subject'] for r in ds.select(range(100))}))" 2>&1
```
Expected: PATH_A_AVAILABLE iff `~/KIKI-Mac_tunner` exists; mlx_lm version printed; MMLU subjects listed (proves cache is offline-materialised).

Record the output verbatim in the Task 0.5 decision log below.

- [ ] **Step 2: Inspect the current substrate for any LoRA train hook**

Run :
```bash
grep -n "def train\|def fit\|lora_train\|train_lora\|train_loop" \
  kiki_oniric/substrates/micro_kiki.py
```
Expected: zero matches. The substrate has no train method — confirms the executing agent must implement one in `experiments/g6_mmlu_stream/micro_kiki_train.py`.

- [ ] **Step 3: No commit. This is a context-gathering task.**

---

## Task 0.5: Lock decisions (compute budget + path + parameters)

**Files:**
- Create: `docs/milestones/g6-pilot-decisions-2026-05-03.md` (decision log; immutable once committed)

- [ ] **Step 1: Pick the path based on Task 0 evidence**

Decision matrix (resolve **before** writing any code) :

| Evidence from Task 0 | Decision |
|---|---|
| `~/KIKI-Mac_tunner` exists AND `mlx_lm` importable AND HF cache for `cais/mmlu` materialised AND ≥ 64 GB free RAM AND Apple Silicon detected | **Path A — full pilot** |
| Any of the above missing | **Path B — inference-only exploratory pilot**, mark all G6 results as exploratory, do NOT trigger STABLE promotion regardless of effect size |
| `~/KIKI-Mac_tunner` exists but `mlx_lm.lora` API is gone or signature drifted | **Path A degraded** — substitute `mlx_lm.lora` with a hand-rolled SGD over `wrapper.parameters()` (mirrors `scripts/pilot_cycle3_real.py:_replay_qwen_handler`); document the deviation in the decisions log and update the pre-reg before locking |

- [ ] **Step 2: Lock the per-pilot parameters**

Write `docs/milestones/g6-pilot-decisions-2026-05-03.md` with these exact values :

```markdown
# G6 pilot decisions (locked 2026-05-03)

## Subdomain selection (5 subdomains)
S1 = anatomy
S2 = astronomy
S3 = business_ethics
S4 = clinical_knowledge
S5 = college_biology

Rationale: 5 distinct MMLU subjects spanning life-science, hard-science,
humanities, applied medicine; alphabetical-prefix order chosen for
neutrality (no curriculum bias). Each subject in cais/mmlu has ≥100 test
records, sufficient for `n_train=100 / n_eval=100` per cell.

## Per-cell volumes
n_train_per_subdomain = 100  # train split, capped
n_eval_per_subdomain = 100   # held-out eval split per subject (subset of test)
seeds_per_arm = 3            # {0, 1, 2}
arms = (baseline, P_min, P_equ, P_max)
n_cells = 4 arms × 3 seeds = 12 cell sequences (each sequence touches 5 subdomains)
n_eval_calls_total = 12 sequences × 5 subdomains × 5 ckpt steps = 300

## LoRA training hyperparams (Path A only)
lr = 5e-5
inner_steps_per_subdomain = 50  # mlx_lm.lora "iters"
batch_size = 4                  # M3 Ultra holds it for 35B-A3B + r=16 LoRA
rank = 16                       # matches micro-kiki production
alpha = 16                      # 1:1 ratio per micro-kiki invariant

## Compute budget acceptance
Path A budget: 60 cells × ~15 min = 15 h on Studio M3 Ultra (overnight run).
Path B budget: 60 cells × ~30 s = 30 min on any Apple Silicon.

## Path selected (fill in based on Task 0.5 step 1 evidence)
PATH = ?  # A or B
RATIONALE = ?  # one sentence
```

- [ ] **Step 3: Verify dependence on G4-bis verdict**

Add to the decisions doc :
```markdown
## G4-bis dependency

G6 is BLOCKED until G4-bis confirms non-zero coupling effect on the MLX
substrate. If G4-bis verdict (g_h1 < Hu 2020 lower CI 0.21) is null,
G6 should not launch — the synthetic-to-real generalisation hypothesis
H_NEW becomes vacuous if no synthetic effect exists in the first place.

G4-bis verdict status (fill before unblocking G6):
- g4_bis_milestone_path = ?
- g4_bis_g_h1 = ?
- g4_bis_above_hu_lower_ci = ?
- g6_unblocked = ?  # true | false
```

- [ ] **Step 4: Commit**

```bash
git add docs/milestones/g6-pilot-decisions-2026-05-03.md
git commit -m "docs(g6): lock pilot decisions + path"
```

---

## Task 1: Write OSF pre-registration draft

**Files:**
- Create: `docs/osf-prereg-g6-pilot.md`

- [ ] **Step 1: Write the failing test for pre-reg presence + content**

Create `tests/unit/test_osf_prereg_g6_present.py` :
```python
"""Smoke check that the G6 OSF pre-reg landed and pins the right hypotheses."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG = REPO_ROOT / "docs" / "osf-prereg-g6-pilot.md"


def test_g6_prereg_exists() -> None:
    assert PREREG.is_file(), f"missing {PREREG}"


def test_g6_prereg_pins_hypotheses() -> None:
    text = PREREG.read_text(encoding="utf-8")
    for token in (
        "H1'", "H3'", "H_DR4'", "H_NEW",
        "Hu 2020", "Javadi 2024",
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
        "Bonferroni",
    ):
        assert token in text, f"pre-reg missing token {token!r}"


def test_g6_prereg_pins_path_branches() -> None:
    text = PREREG.read_text(encoding="utf-8")
    assert "Path A" in text and "Path B" in text, (
        "pre-reg must enumerate both Path A (full pilot) and "
        "Path B (inference-only) branches"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_osf_prereg_g6_present.py -v --no-cov`
Expected: 3 FAIL with `missing docs/osf-prereg-g6-pilot.md`.

- [ ] **Step 3: Write the pre-reg document**

Create `docs/osf-prereg-g6-pilot.md` :
```markdown
# OSF Pre-Registration — G6 pilot (micro-kiki Qwen-35B × MMLU subdomain CL stream)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G6 pilot — first empirical evidence on a real production
  LLM (Qwen3.6-35B-A3B + LoRA adapters) using MMLU subdomains as a
  continual-learning task stream.
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G6 cell is registered in
  `harness/storage/run_registry.RunRegistry`.

## 1. Study design

Within-architecture × within-benchmark sweep on the
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` (Qwen3.6-35B-A3B
+ LoRA, rank 16, alpha 16). Five MMLU subdomains are presented as a
sequential continual-learning stream :

S1 = anatomy → S2 = astronomy → S3 = business_ethics →
S4 = clinical_knowledge → S5 = college_biology

For each (arm, seed) cell the driver :

1. Loads a fresh wrapper (Path A) or fresh adapter buffer (Path B).
2. For i in 1..5 :
   a. Adapt to subdomain S_i (Path A: LoRA fine-tune on 100 train
      examples, lr 5e-5, 50 inner steps; Path B: deterministic
      perturbation of the LoRA delta tensor seeded by (subdomain, seed)).
   b. (Optional, profile-dependent) Inject one DreamEpisode whose
      operation set is dictated by the active profile.
   c. Evaluate on S_1..S_i held-out eval splits (100 records each,
      letter-argmax proxy via `harness.real_benchmarks.mmlu.evaluate_mmlu`).
3. Compute per-subdomain forgetting :
   `forgetting[S_j] = acc[S_j after S_j] - acc[S_j after S_5]` for j < 5.
4. Compute retention metric :
   `retention = mean(acc[S_j after S_5] / max(acc[S_j after S_j], eps))`
   over j in 1..4.

## 2. Hypotheses

### H1' — P_equ retention floor on real LLM matches Hu 2020 anchor
**Statement** : observed Hedges' g of (P_equ retention vs baseline retention)
on the real Qwen-35B + MMLU stream is ≥ HU_2020_OVERALL.ci_low (0.21).
**Operationalisation** :
- g_h1' = compute_hedges_g(retention[P_equ], retention[baseline])
- Reject H0 iff g_h1' >= 0.21
- Welch one-sided (baseline, P_equ) at α = 0.05 / 4 (Bonferroni for 4 hypotheses)

### H3' — P_min retention decrement on real LLM matches Javadi 2024
**Statement** : observed |Hedges' g| of (P_min vs baseline) ≥ JAVADI_2024_OVERALL.ci_low (0.13), with negative sign (decrement).
**Operationalisation** :
- g_h3' = compute_hedges_g(retention[P_min], retention[baseline])
- Reject H0 iff g_h3' <= -0.13
- Welch one-sided (P_min, baseline) at α = 0.05 / 4

### H_DR4' — DR-4 monotonicity on real LLM
**Statement** : mean retention is monotonically ordered
P_max >= P_equ >= P_min on the real Qwen-35B stream.
**Operationalisation** :
- Jonckheere-Terpstra trend on [retention[P_min], retention[P_equ], retention[P_max]]
- α = 0.05 (separate family from Welch tests)

### H_NEW — Synthetic-to-real generalisation
**Statement** : the effect-size for retention on the real Qwen-35B stream
(g_h1') is not smaller than the corresponding G4-bis synthetic effect
(g_h1 from the MLX coupling pilot) by more than 0.10 (one-sided
non-inferiority margin).
**Operationalisation** :
- g_synthetic = G4-bis g_h1 (read from `docs/milestones/g4-bis-pilot-*.json`)
- non_inferiority_observed = (g_h1' >= g_synthetic - 0.10)
- Bootstrap-CI on the difference (B = 10_000 resamples) at α = 0.05 / 4
- Reject H0 (failure of generalisation) iff lower bootstrap CI of (g_h1' - g_synthetic) > -0.10

## 3. Pre-specified analyses

- H1', H3' : `kiki_oniric.eval.statistics.welch_one_sided` + `compute_hedges_g`
  + `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- H_DR4' : `kiki_oniric.eval.statistics.jonckheere_trend`
- H_NEW : new helper `experiments.g6_mmlu_stream.run_g6.h_new_bootstrap_ci`
  (B = 10_000, seeded; deterministic given (g_h1', g_synthetic, B, seed=0))
- Bonferroni family size = 4 (H1', H3', H_DR4', H_NEW), α_per_test = 0.0125

## 4. Sample size / power

- Path A : N = 3 seeds per arm × 4 arms = 12 cell sequences (60 forgetting measurements).
- Path B : same N (compute budget irrelevant for Path B).
- Power note : N = 3 vs N = 3 is severely underpowered for absolute g
  magnitudes (minimum detectable g at 80% power ≈ 2.4). The pilot is
  **exploratory** for absolute effects; pre-registered floors (Hu, Javadi)
  serve only to anchor the *direction* of the verdict.
- A confirmatory N ≥ 30 follow-up is scheduled iff this pilot returns
  exploratory positive evidence (g_h1' ≥ 0.21 with Welch p < α/4).

## 5. Data exclusion rules

- Cells where MicroKikiSubstrate raises any BLOCKING invariant (S1
  retained-non-regression, S2 finite weights) are excluded; their
  EpisodeLogEntry.error is recorded as exclusion reason.
- Cells where `acc[S_1 after S_1] < 0.30` (random-baseline + 5%) are
  excluded as underperforming-baseline (Qwen failed to learn S_1 at all,
  retention is meaningless).
- Cells where Path A LoRA fine-tune raises an MLX OOM are excluded;
  the run continues with the remaining seeds. If > 50% of cells OOM,
  the pilot is aborted and the decision log records the failure.

## 6. DualVer outcome rules (binding)

| Outcome | EC bump | Rationale |
|---|---|---|
| Path A: H1', H3', H_DR4' all reject H0 in predicted direction AND H_NEW non-inferiority holds | PARTIAL → STABLE (scope: G4 + G5 + G6) | Three confirmatory pilots cross §12.3 STABLE bar |
| Path A: H1' confirmed, others mixed | stays PARTIAL | Partial confirmation only |
| Path A: any predicted direction violated (e.g. P_max < P_min mean) | PARTIAL → UNSTABLE | §12.3 falsification rule |
| Path B (any outcome) | stays PARTIAL | Path B is exploratory only — never triggers STABLE/UNSTABLE |

No FC bump in any outcome (no axiom or primitive change).

## 7. Deviations from pre-registration

Any post-hoc deviation logged in
`docs/osf-deviations-g6-<date>.md` (separate dated immutable).

## 8. Data and code availability

- Pilot driver : `experiments/g6_mmlu_stream/run_g6.py`
- Stream loader : `experiments/g6_mmlu_stream/stream.py`
- Train shim (Path A) : `experiments/g6_mmlu_stream/micro_kiki_train.py`
- Inference shim (Path B) : `experiments/g6_mmlu_stream/micro_kiki_inference.py`
- Verdict anchors : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`, SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g6-pilot-2026-05-03.{json,md}`
- Decisions log : `docs/milestones/g6-pilot-decisions-2026-05-03.md`

## 9. Path A vs Path B disclosure

The pilot host availability check (Task 0.5 step 1) determines which
branch executes. The path selection is **locked before any cell registers**
and recorded in the decisions log. Switching paths post-hoc requires a
deviation document + a new dated pre-reg.

## 10. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G6 cell is registered in the run registry.**
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_osf_prereg_g6_present.py -v --no-cov`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/osf-prereg-g6-pilot.md tests/unit/test_osf_prereg_g6_present.py
git commit -m "docs(g6): pre-reg + presence smoke test"
```

---

## Task 2: Stub experiments/g6_mmlu_stream package

**Files:**
- Create: `experiments/g6_mmlu_stream/__init__.py`
- Create: `experiments/g6_mmlu_stream/run_g6.py` (stub only — full driver lands in later tasks)
- Test: `tests/unit/experiments/__init__.py`, `tests/unit/experiments/test_g6_run_pilot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/__init__.py` (empty file) and `tests/unit/experiments/test_g6_run_pilot.py` :
```python
"""Import-and-CLI smoke for the G6 pilot driver."""
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_g6_module_importable() -> None:
    mod = importlib.import_module("experiments.g6_mmlu_stream.run_g6")
    assert hasattr(mod, "run_pilot"), "run_g6 must export run_pilot"
    assert hasattr(mod, "main"), "run_g6 must export main"
    assert hasattr(mod, "ARMS"), "run_g6 must export ARMS"
    assert tuple(mod.ARMS) == ("baseline", "P_min", "P_equ", "P_max")


def test_g6_help_smokes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.g6_mmlu_stream.run_g6", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help failed: stderr={result.stderr!r}"
    )
    assert "G6" in result.stdout or "g6" in result.stdout
    assert "--smoke" in result.stdout
    assert "--path" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g6_run_pilot.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: experiments.g6_mmlu_stream`.

- [ ] **Step 3: Write the stub**

Create `experiments/g6_mmlu_stream/__init__.py` (single-line docstring) :
```python
"""G6 pilot — micro-kiki Qwen-35B × MMLU subdomain CL stream."""
```

Create `experiments/g6_mmlu_stream/run_g6.py` :
```python
"""G6 pilot driver — micro-kiki Qwen-35B × MMLU subdomain CL stream.

**Gate ID** : G6 — first empirical evidence on a real production LLM.
**Validates** : whether observed Hedges' g for retention on a 5-subdomain
MMLU continual-learning stream matches Hu 2020 / Javadi 2024 floors when
the substrate is `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate`
(Qwen3.6-35B-A3B + r=16 LoRA).

**Path branches** (locked at Task 0.5) :
- Path A — full pilot (Studio + KIKI-Mac_tunner + mlx_lm.lora).
- Path B — inference-only exploratory (any host, no fine-tune).

**Mode** : empirical claim at first-pilot scale (3 seeds × 4 arms = 12
sequences). Pre-registered as exploratory for absolute g magnitudes.
**Expected output** :
    - docs/milestones/g6-pilot-2026-05-03.json
    - docs/milestones/g6-pilot-2026-05-03.md

Reference :
    docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md
    docs/osf-prereg-g6-pilot.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
)
C_VERSION = "C-v0.12.0+PARTIAL"  # synced with STATUS.md at plan-write time


def run_pilot(**kwargs: Any) -> dict[str, Any]:
    """Run the G6 pilot. Stub — full implementation lands in Task 5+."""
    raise NotImplementedError(
        "G6 run_pilot is implemented incrementally — see plan tasks 3..7"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G6 pilot driver — micro-kiki Qwen-35B × MMLU CL stream",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 cell (P_min, seed 0, S1 only) to validate the pipeline.",
    )
    parser.add_argument(
        "--path",
        choices=("A", "B"),
        default="B",
        help="Path A (full LoRA pilot) or B (inference-only). Default B.",
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.parse_args(argv)
    print("G6 pilot stub — implementation lands in plan tasks 3..7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g6_run_pilot.py -v --no-cov`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/__init__.py \
        experiments/g6_mmlu_stream/run_g6.py \
        tests/unit/experiments/__init__.py \
        tests/unit/experiments/test_g6_run_pilot.py
git commit -m "feat(g6): stub pilot driver + CLI"
```

---

## Task 3: Implement MMLU subdomain stream loader

**Files:**
- Create: `experiments/g6_mmlu_stream/stream.py`
- Test: `tests/unit/experiments/test_g6_stream.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_stream.py` :
```python
"""Tests for the G6 MMLU subdomain stream loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g6_mmlu_stream.stream import (
    SubdomainSplit,
    build_subdomain_stream,
)
from harness.real_benchmarks.mmlu import MMLURecord


def _write_fixture(tmp_path: Path) -> Path:
    """Write 8 records each across 5 target subdomains (40 rows total)."""
    rows: list[dict] = []
    subjects = (
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    )
    for subj in subjects:
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4,
                "subject": subj,
            })
    # Plus 3 distractor records that should never make it into a split.
    for i in range(3):
        rows.append({
            "question": f"distractor-{i}",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "subject": "elementary_mathematics",
        })
    path = tmp_path / "mmlu_subset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_build_subdomain_stream_returns_5_splits(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    splits = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
        n_train=4,
        n_eval=4,
        seed=0,
    )
    assert len(splits) == 5
    for split in splits:
        assert isinstance(split, SubdomainSplit)
        assert split.subject in {
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        }
        assert len(split.train) == 4
        assert len(split.eval_) == 4
        # No leakage between train and eval inside a single subject.
        train_qs = {r.question for r in split.train}
        eval_qs = {r.question for r in split.eval_}
        assert train_qs.isdisjoint(eval_qs), (
            f"train/eval overlap in {split.subject}: "
            f"{train_qs & eval_qs}"
        )
        # No cross-subject contamination.
        for r in split.train + split.eval_:
            assert isinstance(r, MMLURecord)
            assert r.subject == split.subject


def test_build_subdomain_stream_is_deterministic(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    a = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=("anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology"),
        n_train=4, n_eval=4, seed=42,
    )
    b = build_subdomain_stream(
        fixture_path=fixture,
        subdomains=("anatomy", "astronomy", "business_ethics",
                    "clinical_knowledge", "college_biology"),
        n_train=4, n_eval=4, seed=42,
    )
    assert [s.subject for s in a] == [s.subject for s in b]
    for sa, sb in zip(a, b):
        assert [r.question for r in sa.train] == [r.question for r in sb.train]
        assert [r.question for r in sa.eval_] == [r.question for r in sb.eval_]


def test_build_subdomain_stream_raises_on_insufficient_rows(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="not enough records"):
        build_subdomain_stream(
            fixture_path=fixture,
            subdomains=("anatomy",),
            n_train=20, n_eval=20, seed=0,  # only 8 rows for anatomy
        )


def test_build_subdomain_stream_raises_on_unknown_subject(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    with pytest.raises(KeyError, match="no MMLU records for subject"):
        build_subdomain_stream(
            fixture_path=fixture,
            subdomains=("not_a_real_subject",),
            n_train=2, n_eval=2, seed=0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g6_stream.py -v --no-cov`
Expected: 4 FAIL with `ModuleNotFoundError: experiments.g6_mmlu_stream.stream`.

- [ ] **Step 3: Write the implementation**

Create `experiments/g6_mmlu_stream/stream.py` :
```python
"""MMLU subdomain stream loader for the G6 pilot.

Builds a sequence of SubdomainSplit (train, eval_) pairs, one per
target subject, drawn deterministically from a JSONL fixture matching
the HF cais/mmlu schema. Used by the G6 pilot driver to construct a
continual-learning task stream.

The loader is network-free : it consumes a pre-materialised JSONL
fixture (see `tests/fixtures/mmlu_sanity.jsonl` for shape, but note
that the production pilot requires the full cais/mmlu export — the
sanity fixture has < 5 records per subject).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from harness.real_benchmarks.mmlu import MMLURecord


@dataclass(frozen=True)
class SubdomainSplit:
    """Frozen (train, eval) split for a single MMLU subject.

    Fields
    ------
    subject
        MMLU subject (one of the 57 cais/mmlu subjects).
    train
        Training records (input features for the per-subdomain
        adaptation step). Ordered as drawn — driver re-shuffles
        if it wants epoch-style mini-batches.
    eval_
        Held-out evaluation records (disjoint from ``train``).
        Used by `evaluate_mmlu` after each subdomain step.
    """

    subject: str
    train: list[MMLURecord]
    eval_: list[MMLURecord]


def _record_from_raw(row: dict) -> MMLURecord:
    """Validate a raw JSON row and lift it to a frozen MMLURecord.

    Mirrors `MMLULoader._record_from_raw` (private) so this loader
    stays decoupled from MMLULoader's R1-pin lifecycle.
    """
    choices = row["choices"]
    if len(choices) != 4:
        raise ValueError(
            f"MMLU row has {len(choices)} choices, expected 4: {row!r}"
        )
    answer = int(row["answer"])
    if not 0 <= answer <= 3:
        raise ValueError(
            f"MMLU row has answer index {answer} outside [0,3]: {row!r}"
        )
    return MMLURecord(
        question=str(row["question"]),
        choices=(
            str(choices[0]),
            str(choices[1]),
            str(choices[2]),
            str(choices[3]),
        ),
        answer=answer,
        subject=str(row.get("subject", "unknown")),
    )


def build_subdomain_stream(
    *,
    fixture_path: Path,
    subdomains: Sequence[str],
    n_train: int,
    n_eval: int,
    seed: int,
) -> list[SubdomainSplit]:
    """Build a stream of (train, eval) splits, one per subdomain.

    Parameters
    ----------
    fixture_path
        JSONL file matching the HF cais/mmlu schema. The full cais/mmlu
        ``test`` split exported as JSONL is the production target ; the
        sanity fixture under ``tests/fixtures/mmlu_sanity.jsonl`` has
        < 5 records per subject and only validates pipeline shape.
    subdomains
        Ordered tuple of MMLU subjects forming the CL stream. Order
        matters — it dictates the curriculum.
    n_train
        Number of training records per subdomain. Drawn from the
        per-subject pool via seeded shuffle.
    n_eval
        Number of held-out eval records per subdomain. Drawn from
        the per-subject pool *after* removing the train slice
        (no leakage).
    seed
        Pins the per-subject shuffle so the same (fixture, seed) pair
        always yields identical splits.

    Returns
    -------
    list[SubdomainSplit]
        Length matches ``len(subdomains)``, in the same order.

    Raises
    ------
    KeyError
        A subject in ``subdomains`` has no rows in the fixture.
    ValueError
        Per-subject pool has fewer than ``n_train + n_eval`` rows.
    """
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"MMLU stream fixture not found at {fixture_path!s}"
        )
    by_subject: dict[str, list[dict]] = {}
    with fixture_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_subject.setdefault(
                str(row.get("subject", "unknown")), [],
            ).append(row)

    splits: list[SubdomainSplit] = []
    for subject in subdomains:
        pool = by_subject.get(subject)
        if not pool:
            raise KeyError(
                f"no MMLU records for subject {subject!r} in "
                f"{fixture_path!s} (subjects present : "
                f"{sorted(by_subject)})"
            )
        if len(pool) < n_train + n_eval:
            raise ValueError(
                f"not enough records for subject {subject!r}: "
                f"need {n_train + n_eval}, got {len(pool)}"
            )
        rng = random.Random((seed, subject))  # tuple seed → stable per-subject
        order = list(range(len(pool)))
        rng.shuffle(order)
        train_idx = order[:n_train]
        eval_idx = order[n_train : n_train + n_eval]
        train = [_record_from_raw(pool[i]) for i in train_idx]
        eval_ = [_record_from_raw(pool[i]) for i in eval_idx]
        splits.append(SubdomainSplit(subject=subject, train=train, eval_=eval_))
    return splits


__all__ = ["SubdomainSplit", "build_subdomain_stream"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g6_stream.py -v --no-cov`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/stream.py \
        tests/unit/experiments/test_g6_stream.py
git commit -m "feat(g6): MMLU subdomain stream loader"
```

---

## Task 4: Implement micro-kiki dream-episode coupling wrapper

**Files:**
- Create: `experiments/g6_mmlu_stream/dream_wrap.py`
- Test: `tests/unit/experiments/test_g6_dream_wrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_dream_wrap.py` :
```python
"""Tests for the G6 dream-episode wrapper over MicroKikiSubstrate."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g6_mmlu_stream.dream_wrap import (
    G6DreamRunner,
    build_episode_payload,
)
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


def _seeded_substrate() -> MicroKikiSubstrate:
    return MicroKikiSubstrate(num_layers=4, rank=8, seed=0)


def test_build_episode_payload_shape() -> None:
    payload = build_episode_payload(
        seed=0,
        adapter_keys=("layer_0_lora_B",),
        out_dim=8,
        rank=4,
    )
    assert "beta_records" in payload
    assert "deltas" in payload
    assert "prior_deltas" in payload
    assert "shrink_factor" in payload
    assert payload["shrink_factor"] == pytest.approx(0.99)
    assert all("input" in r for r in payload["beta_records"])
    assert all("episode_id" not in r for r in payload["beta_records"])


def test_g6_dream_runner_baseline_is_no_op() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="baseline")
    before_recombine = sub.recombine_state.total_episodes_handled
    before_restructure = sub.restructure_state.total_episodes_handled
    runner.run_episode(seed=0, subdomain="anatomy")
    # baseline arm runs no DE — substrate state unchanged.
    assert sub.recombine_state.total_episodes_handled == before_recombine
    assert sub.restructure_state.total_episodes_handled == before_restructure


def test_g6_dream_runner_p_min_invokes_replay_downscale_only() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_min")
    runner.run_episode(seed=0, subdomain="anatomy")
    # P_min must NOT call restructure / recombine — they are not in
    # the P_min op set.
    assert sub.recombine_state.total_episodes_handled == 0
    assert sub.restructure_state.total_episodes_handled == 0
    # Replay + downscale handlers do not bump state on micro-kiki
    # substrate (they return tensors, not mutate _state). The wrapper
    # must therefore expose its own counter.
    assert runner.episodes_run == 1
    assert "P_min" in runner.last_episode_id


def test_g6_dream_runner_p_equ_invokes_all_four_handlers() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_equ")
    runner.run_episode(seed=0, subdomain="astronomy")
    # P_equ wires restructure + recombine — substrate state bumps.
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.total_episodes_handled == 1
    assert sub.recombine_state.last_episode_id is not None
    assert "astronomy" in sub.recombine_state.last_episode_id


def test_g6_dream_runner_p_max_invokes_all_four_handlers() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_max")
    runner.run_episode(seed=0, subdomain="business_ethics")
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.total_episodes_handled == 1


def test_g6_dream_runner_rejects_unknown_profile() -> None:
    sub = _seeded_substrate()
    with pytest.raises(ValueError, match="unknown profile"):
        G6DreamRunner(substrate=sub, profile_name="P_quantum")


def test_g6_dream_runner_episode_ids_are_unique_per_call() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_equ")
    runner.run_episode(seed=0, subdomain="anatomy")
    runner.run_episode(seed=0, subdomain="astronomy")
    # Subdomain enters the episode_id, so two distinct subjects with
    # the same seed must yield distinct ids.
    assert sub.recombine_state.episode_ids[0] != sub.recombine_state.episode_ids[1]
    assert "anatomy" in sub.recombine_state.episode_ids[0]
    assert "astronomy" in sub.recombine_state.episode_ids[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g6_dream_wrap.py -v --no-cov`
Expected: 7 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `experiments/g6_mmlu_stream/dream_wrap.py` :
```python
"""Dream-episode coupling for the G6 pilot.

Wires the four MicroKikiSubstrate handler factories (replay, downscale,
restructure (OPLoRA), recombine (TIES-Merge)) into a profile-aware
runner that fires one episode per subdomain transition. baseline arm
runs no episode; P_min runs replay + downscale only; P_equ / P_max run
all four.

DR-0 / DR-1 stamps land on substrate._restructure_state and
substrate._recombine_state via the handler closures. The wrapper holds
its own `episodes_run` + `last_episode_id` counters because the replay
and downscale handlers on MicroKikiSubstrate are *pure* (they return
tensors and do not mutate accumulator state — see micro_kiki.py
docstrings).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


PROFILE_OPS: dict[str, tuple[str, ...]] = {
    "baseline": (),  # baseline = no episode
    "P_min": ("replay", "downscale"),
    "P_equ": ("replay", "downscale", "restructure", "recombine"),
    "P_max": ("replay", "downscale", "restructure", "recombine"),
}


def build_episode_payload(
    *,
    seed: int,
    adapter_keys: tuple[str, ...],
    out_dim: int,
    rank: int,
) -> dict[str, object]:
    """Build a synthetic dream-episode payload for the four handlers.

    The payload's shape mirrors the contract documented in
    ``MicroKikiSubstrate.{replay, downscale, restructure, recombine}_handler_factory``.

    Parameters
    ----------
    seed
        Pins the RNG so the same seed always produces the same payload.
    adapter_keys
        LoRA tensor keys to include in the synthetic adapter dict
        (used by restructure_handler to know which key to project).
    out_dim
        Output dimension of the LoRA B matrix (matches Qwen-35B
        hidden size at runtime — synthesised here).
    rank
        LoRA rank (matches r=16 in production; smaller in tests).

    Returns
    -------
    dict
        Keys ::
            beta_records   : list[{"input": list[float]}]  (replay)
            shrink_factor  : float (downscale)
            prior_deltas   : list[ndarray] (restructure / OPLoRA priors)
            deltas         : list[ndarray] (recombine / TIES-Merge inputs)
            adapter_keys   : tuple[str, ...]
            <each adapter_key> : ndarray of shape (out_dim, rank)
    """
    rng = np.random.default_rng(seed)
    beta_records = [
        {"input": rng.standard_normal(out_dim).astype(np.float32).tolist()}
        for _ in range(4)
    ]
    prior_deltas = [
        rng.standard_normal((out_dim, rank)).astype(np.float32)
        for _ in range(2)
    ]
    deltas = [
        rng.standard_normal((out_dim, rank)).astype(np.float32)
        for _ in range(3)
    ]
    payload: dict[str, object] = {
        "beta_records": beta_records,
        "shrink_factor": 0.99,
        "prior_deltas": prior_deltas,
        "deltas": deltas,
        "adapter_keys": adapter_keys,
    }
    for key in adapter_keys:
        payload[key] = rng.standard_normal(
            (out_dim, rank),
        ).astype(np.float32)
    return payload


@dataclass
class G6DreamRunner:
    """Profile-aware episode runner over a MicroKikiSubstrate.

    Attributes
    ----------
    substrate
        The MicroKikiSubstrate instance under test.
    profile_name
        One of {"baseline", "P_min", "P_equ", "P_max"}.
    out_dim
        Synthetic LoRA out_dim used for replay / restructure / recombine
        payloads. 8 in tests; 4096 (Qwen-35B hidden) in production.
    rank
        Synthetic LoRA rank. 4 in tests; 16 in production.
    episodes_run
        Bumped on every successful run_episode call (DR-0 wrapper-level).
    last_episode_id
        Set on every run_episode call to ``f"g6-{profile}-{subdomain}-seed{seed}"``.
    """

    substrate: MicroKikiSubstrate
    profile_name: str
    out_dim: int = 8
    rank: int = 4
    episodes_run: int = field(default=0, init=False)
    last_episode_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.profile_name not in PROFILE_OPS:
            raise ValueError(
                f"unknown profile {self.profile_name!r}; expected one "
                f"of {sorted(PROFILE_OPS)}"
            )

    def run_episode(self, *, seed: int, subdomain: str) -> dict[str, object]:
        """Run one dream episode for the given (seed, subdomain).

        Returns the payload that was constructed (useful for tests +
        debugging). For the baseline arm returns an empty dict and
        does NOT touch the substrate.
        """
        ops = PROFILE_OPS[self.profile_name]
        if not ops:
            return {}

        adapter_key = "layer_0_lora_B"
        payload = build_episode_payload(
            seed=seed,
            adapter_keys=(adapter_key,),
            out_dim=self.out_dim,
            rank=self.rank,
        )
        episode_id = (
            f"g6-{self.profile_name}-{subdomain}-seed{seed}"
        )
        payload["episode_id"] = episode_id

        # 1. replay → vector aggregate (stub; production: feeds replay LoRA SGD)
        if "replay" in ops:
            replay = self.substrate.replay_handler_factory()
            replay(payload["beta_records"], 20)  # type: ignore[arg-type]

        # 2. downscale → multiplicative shrink (stub; production: shrinks LoRA B)
        if "downscale" in ops:
            downscale = self.substrate.downscale_handler_factory()
            arr = np.asarray(payload[adapter_key])
            downscale(arr, float(payload["shrink_factor"]))  # type: ignore[arg-type]

        # 3. restructure → OPLoRA project (writes substrate._restructure_state).
        if "restructure" in ops:
            restructure = self.substrate.restructure_handler_factory()
            adapter: dict = {
                adapter_key: payload[adapter_key],
                "prior_deltas": payload["prior_deltas"],
                "episode_id": episode_id,
            }
            restructure(adapter, "oplora", adapter_key)

        # 4. recombine → TIES-Merge (writes substrate._recombine_state).
        if "recombine" in ops:
            recombine = self.substrate.recombine_handler_factory()
            recombine_payload = {
                "deltas": payload["deltas"],
                "episode_id": episode_id,
            }
            recombine(recombine_payload, "ties")

        self.episodes_run += 1
        self.last_episode_id = episode_id
        return payload


__all__ = ["G6DreamRunner", "PROFILE_OPS", "build_episode_payload"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g6_dream_wrap.py -v --no-cov`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/dream_wrap.py \
        tests/unit/experiments/test_g6_dream_wrap.py
git commit -m "feat(g6): dream-episode coupling wrapper"
```

---

## Task 5: Implement Path B inference-only adaptation shim

**Files:**
- Create: `experiments/g6_mmlu_stream/micro_kiki_inference.py`
- Test: `tests/unit/experiments/test_g6_micro_kiki_inference.py`

> **Path B is the always-runnable fallback.** It must work on any
> Apple Silicon host (no `KIKI-Mac_tunner` required). The shim synthesises
> a LoRA delta tensor per subdomain, applies the dream-episode handlers
> to it (when the arm is non-baseline), and uses `evaluate_mmlu` against
> the registered Qwen wrapper for accuracy measurement. Any
> "forgetting" signal observed under Path B is exploratory — it
> reflects the handlers' impact on a synthetic adapter, not a true
> trained adapter.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_micro_kiki_inference.py` :
```python
"""Tests for the Path B inference-only adaptation shim."""
from __future__ import annotations

import numpy as np

from experiments.g6_mmlu_stream.micro_kiki_inference import (
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from harness.real_benchmarks.mmlu import MMLURecord


def _record(subject: str, idx: int, gold: int = 0) -> MMLURecord:
    return MMLURecord(
        question=f"{subject}-Q{idx}?",
        choices=("A", "B", "C", "D"),
        answer=gold,
        subject=subject,
    )


def test_inference_adapter_starts_with_zero_delta() -> None:
    adapter = InferenceOnlyAdapter(
        out_dim=8, rank=4, seed=0,
    )
    delta = adapter.current_delta("layer_0_lora_B")
    assert delta.shape == (8, 4)
    assert np.allclose(delta, 0.0)


def test_adapt_subdomain_mutates_delta() -> None:
    adapter = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train_records = [_record("anatomy", i) for i in range(4)]
    before = adapter.current_delta("layer_0_lora_B").copy()
    adapt_subdomain(
        adapter=adapter,
        subdomain="anatomy",
        train=train_records,
        seed=0,
    )
    after = adapter.current_delta("layer_0_lora_B")
    # Delta must change by a non-trivial amount to model "training".
    assert not np.allclose(before, after)
    # Magnitude must be bounded so dream-episode downscale can still
    # detect a change (sanity bound; not load-bearing).
    assert np.max(np.abs(after)) < 10.0


def test_adapt_subdomain_is_deterministic() -> None:
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train = [_record("anatomy", i) for i in range(4)]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train, seed=42)
    adapt_subdomain(adapter=a2, subdomain="anatomy", train=train, seed=42)
    np.testing.assert_array_equal(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )


def test_adapt_subdomain_different_subjects_yield_different_deltas() -> None:
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=0)
    train_a = [_record("anatomy", i) for i in range(4)]
    train_b = [_record("astronomy", i) for i in range(4)]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train_a, seed=0)
    adapt_subdomain(adapter=a2, subdomain="astronomy", train=train_b, seed=0)
    assert not np.allclose(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g6_micro_kiki_inference.py -v --no-cov`
Expected: 4 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `experiments/g6_mmlu_stream/micro_kiki_inference.py` :
```python
"""Path B inference-only adaptation shim for the G6 pilot.

When the run host lacks ``KIKI-Mac_tunner`` (the training pipeline +
``mlx_lm.lora`` fork), the G6 driver falls back to this Path B shim.
Per-subdomain "training" is replaced by a deterministic perturbation
of a synthetic LoRA delta tensor seeded by ``(subdomain, seed)``. The
shim does NOT actually fine-tune Qwen; it provides a state surface
on which the four MicroKikiSubstrate handlers can operate so the
dream-episode signature is observable end-to-end.

Any forgetting effect reported under Path B is **exploratory only** —
it reflects the dream-episode handlers' impact on a synthetic adapter,
not a true CL signal. The pre-reg explicitly forbids STABLE/UNSTABLE
promotion under Path B (see osf-prereg-g6-pilot.md §6).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from harness.real_benchmarks.mmlu import MMLURecord


def _stable_seed(*tokens: object) -> int:
    """Deterministic positive int seed from a tuple of hashable tokens.

    Cross-process stable (does not rely on builtins.hash, which is
    PYTHONHASHSEED-randomised by default).
    """
    raw = "|".join(repr(t) for t in tokens).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


@dataclass
class InferenceOnlyAdapter:
    """Mock LoRA adapter buffer keyed by tensor name.

    Attributes
    ----------
    out_dim
        LoRA B-matrix output dimension. 4096 in production (Qwen
        hidden size); smaller in tests.
    rank
        LoRA rank.
    seed
        Pins the initial delta state.
    """

    out_dim: int
    rank: int
    seed: int
    _deltas: dict[str, NDArray] = field(default_factory=dict, init=False)

    def current_delta(self, key: str) -> NDArray:
        """Return the current delta for ``key`` (zero-init on first read)."""
        if key not in self._deltas:
            self._deltas[key] = np.zeros(
                (self.out_dim, self.rank), dtype=np.float32,
            )
        return self._deltas[key]

    def set_delta(self, key: str, value: NDArray) -> None:
        if value.shape != (self.out_dim, self.rank):
            raise ValueError(
                f"delta shape {value.shape} != expected "
                f"({self.out_dim}, {self.rank})"
            )
        self._deltas[key] = value.astype(np.float32, copy=False)


def adapt_subdomain(
    *,
    adapter: InferenceOnlyAdapter,
    subdomain: str,
    train: Sequence[MMLURecord],
    seed: int,
    key: str = "layer_0_lora_B",
    step_magnitude: float = 0.05,
) -> None:
    """Apply a deterministic perturbation modelling per-subdomain LoRA training.

    The perturbation magnitude scales with the number of training
    records (linear) and is reproducible from ``(subdomain, seed,
    len(train))``. This is a stand-in for real LoRA training; the
    point is to give the dream-episode handlers a non-trivial state
    surface to operate on under Path B.

    Parameters
    ----------
    adapter
        InferenceOnlyAdapter to mutate.
    subdomain
        Subject name; folded into the seed so different subjects
        yield distinct perturbations.
    train
        Training records. Only ``len(train)`` is used (perturbation
        magnitude scales linearly).
    seed
        Cell-level seed.
    key
        LoRA tensor key to perturb.
    step_magnitude
        Per-record perturbation scale. Default 0.05 — small enough
        that several subdomains accumulate without saturating the
        bf16 numeric range, large enough that dream-handler downscale
        (factor 0.99) produces a measurable post-handler delta.
    """
    rng_seed = _stable_seed("g6-adapt", subdomain, seed, len(train))
    rng = np.random.default_rng(rng_seed)
    delta_step = (
        step_magnitude
        * len(train)
        * rng.standard_normal((adapter.out_dim, adapter.rank)).astype(np.float32)
    )
    cur = adapter.current_delta(key)
    adapter.set_delta(key, cur + delta_step)


__all__ = ["InferenceOnlyAdapter", "adapt_subdomain"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g6_micro_kiki_inference.py -v --no-cov`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/micro_kiki_inference.py \
        tests/unit/experiments/test_g6_micro_kiki_inference.py
git commit -m "feat(g6): Path B inference-only adapter shim"
```

---

## Task 6: Implement Path A LoRA training shim (with mocked seam)

**Files:**
- Create: `experiments/g6_mmlu_stream/micro_kiki_train.py`
- Test: `tests/unit/experiments/test_g6_micro_kiki_train.py`

> The shim's seam (`_run_lora_iters`) is mocked in tests so the suite
> stays network-free, mlx-lm-free, and < 100 ms. The seam itself
> calls `mlx_lm.tuner.lora.iterate_iter_steps` (or, on degraded Path A,
> a hand-rolled SGD loop) on the real run host. The contract test
> below asserts the shim's *interface*, not the underlying training.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/experiments/test_g6_micro_kiki_train.py` :
```python
"""Tests for the Path A LoRA training shim (seam mocked)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from experiments.g6_mmlu_stream.micro_kiki_train import (
    LoRATrainConfig,
    train_subdomain_lora,
)
from harness.real_benchmarks.mmlu import MMLURecord


def _record(subject: str, idx: int) -> MMLURecord:
    return MMLURecord(
        question=f"{subject}-Q{idx}?",
        choices=("A", "B", "C", "D"),
        answer=idx % 4,
        subject=subject,
    )


def test_train_subdomain_lora_calls_seam_with_expected_args() -> None:
    train = [_record("anatomy", i) for i in range(4)]
    cfg = LoRATrainConfig(
        lr=5e-5, inner_steps=10, batch_size=2, rank=4, alpha=4.0,
    )
    fake_wrapper = object()
    fake_tokenizer = object()

    with patch(
        "experiments.g6_mmlu_stream.micro_kiki_train._run_lora_iters",
    ) as seam:
        seam.return_value = {"final_loss": 1.234, "steps_run": 10}
        result = train_subdomain_lora(
            wrapper=fake_wrapper,
            tokenizer=fake_tokenizer,
            train=train,
            config=cfg,
            seed=0,
            subdomain="anatomy",
        )

    seam.assert_called_once()
    kwargs = seam.call_args.kwargs
    assert kwargs["wrapper"] is fake_wrapper
    assert kwargs["tokenizer"] is fake_tokenizer
    assert kwargs["records"] == train
    assert kwargs["config"] is cfg
    assert kwargs["seed"] == 0
    assert kwargs["subdomain"] == "anatomy"
    assert result["final_loss"] == pytest.approx(1.234)
    assert result["steps_run"] == 10


def test_train_subdomain_lora_rejects_empty_train() -> None:
    cfg = LoRATrainConfig(
        lr=5e-5, inner_steps=10, batch_size=2, rank=4, alpha=4.0,
    )
    with pytest.raises(ValueError, match="empty train split"):
        train_subdomain_lora(
            wrapper=object(),
            tokenizer=object(),
            train=[],
            config=cfg,
            seed=0,
            subdomain="anatomy",
        )


def test_train_subdomain_lora_rejects_invalid_config() -> None:
    train = [_record("anatomy", 0)]
    with pytest.raises(ValueError, match="lr must be > 0"):
        LoRATrainConfig(
            lr=0.0, inner_steps=10, batch_size=2, rank=4, alpha=4.0,
        )
    with pytest.raises(ValueError, match="inner_steps must be > 0"):
        LoRATrainConfig(
            lr=5e-5, inner_steps=0, batch_size=2, rank=4, alpha=4.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/experiments/test_g6_micro_kiki_train.py -v --no-cov`
Expected: 3 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `experiments/g6_mmlu_stream/micro_kiki_train.py` :
```python
"""Path A LoRA training shim for the G6 pilot.

Provides ``train_subdomain_lora(wrapper, tokenizer, train, config,
seed, subdomain)`` over a real Qwen wrapper + mlx_lm LoRA training.
The actual ``mlx_lm.lora.iterate_iter_steps`` (or the degraded
hand-rolled SGD path) lives behind the ``_run_lora_iters`` seam so
unit tests can patch it without spinning up MLX.

The shim is invoked from the G6 cell runner (Task 7) only on Path A.
Path B uses ``micro_kiki_inference.adapt_subdomain`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from harness.real_benchmarks.mmlu import MMLURecord


@dataclass(frozen=True)
class LoRATrainConfig:
    """Frozen LoRA training hyperparameters."""

    lr: float
    inner_steps: int
    batch_size: int
    rank: int
    alpha: float

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.inner_steps <= 0:
            raise ValueError(
                f"inner_steps must be > 0, got {self.inner_steps}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        if self.rank <= 0:
            raise ValueError(f"rank must be > 0, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


def _run_lora_iters(  # pragma: no cover - mocked in tests, real-host only
    *,
    wrapper: Any,
    tokenizer: Any,
    records: Sequence[MMLURecord],
    config: LoRATrainConfig,
    seed: int,
    subdomain: str,
) -> dict[str, Any]:
    """Production seam — calls mlx_lm.lora.iterate_iter_steps.

    Replaced by `unittest.mock.patch` in the unit suite. On the real
    run host this materialises an MMLU-style chat dataset, builds a
    LoRA config from ``config``, and runs ``inner_steps`` SGD steps
    over the records. Returns a dict with at least ``final_loss`` +
    ``steps_run`` so the shim's caller can log per-cell training
    diagnostics.

    On Path A degraded (no `mlx_lm.lora`), this falls back to a
    hand-rolled CE-loss SGD loop mirroring
    ``scripts/pilot_cycle3_real.py:_replay_qwen_handler``.
    """
    import mlx.core as mx

    try:
        from mlx_lm.tuner.lora import iterate_iter_steps  # type: ignore[import-not-found,unused-ignore]

        # Production path : delegate to mlx_lm.lora. Detailed
        # parameter wiring (dataset adapter, optimiser, callback) is
        # documented in `docs/specs/2026-04-17-dreamofkiki-master-design.md`
        # §G6.A. Calling convention pinned to mlx-lm 0.20+.
        mx.random.seed(seed)
        adapter_state = iterate_iter_steps(
            model=wrapper.model,
            tokenizer=tokenizer,
            dataset=_records_to_dataset(records, tokenizer),
            iters=config.inner_steps,
            batch_size=config.batch_size,
            learning_rate=config.lr,
            lora_rank=config.rank,
            lora_alpha=config.alpha,
        )
        final_loss = float(adapter_state.get("loss", 0.0))
        steps_run = int(adapter_state.get("iters_completed", config.inner_steps))
    except ImportError:
        # Degraded Path A — hand-rolled SGD over wrapper.parameters().
        final_loss, steps_run = _handrolled_sgd(
            wrapper=wrapper,
            tokenizer=tokenizer,
            records=records,
            config=config,
            seed=seed,
        )
    return {
        "final_loss": final_loss,
        "steps_run": steps_run,
        "subdomain": subdomain,
    }


def _records_to_dataset(  # pragma: no cover - real-host only
    records: Sequence[MMLURecord],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Materialise MMLU records as an mlx_lm.lora dataset.

    Each record becomes a chat-style sample
    ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
    where the assistant turn is the gold-letter completion.
    """
    samples: list[dict[str, Any]] = []
    for r in records:
        prompt = (
            "Question: " + r.question + "\n"
            f"A. {r.choices[0]}\nB. {r.choices[1]}\n"
            f"C. {r.choices[2]}\nD. {r.choices[3]}\n"
            "Answer:"
        )
        gold = "ABCD"[r.answer]
        samples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": gold},
            ],
        })
    return samples


def _handrolled_sgd(  # pragma: no cover - real-host only
    *,
    wrapper: Any,
    tokenizer: Any,
    records: Sequence[MMLURecord],
    config: LoRATrainConfig,
    seed: int,
) -> tuple[float, int]:
    """Degraded Path A — CE-loss SGD over wrapper.parameters().

    Mirrors `scripts/pilot_cycle3_real.py:_replay_qwen_handler` but
    iterates ``config.inner_steps`` outer steps over the per-subdomain
    records (loops the records list if it is shorter than
    ``inner_steps``).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    qwen = wrapper.model
    optimizer = optim.SGD(learning_rate=config.lr)

    def _ce_loss(qwen_inner: Any, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = qwen_inner(input_ids)
        logits_fp32 = logits.astype(mx.float32)
        return nn.losses.cross_entropy(
            logits_fp32.reshape(-1, logits_fp32.shape[-1]),
            target_ids.reshape(-1),
            reduction="mean",
        )

    grad_fn = nn.value_and_grad(qwen, _ce_loss)
    mx.random.seed(seed)
    n = len(records)
    last_loss = 0.0
    for step in range(config.inner_steps):
        r = records[step % n]
        prompt = (
            "Question: " + r.question + "\n"
            f"A. {r.choices[0]}\nB. {r.choices[1]}\n"
            f"C. {r.choices[2]}\nD. {r.choices[3]}\n"
            "Answer:"
        )
        gold = "ABCD"[r.answer]
        full_ids = list(tokenizer.encode(prompt + " " + gold))
        if len(full_ids) < 2:
            continue
        input_ids_arr = mx.array([full_ids[:-1]])
        target_ids_arr = mx.array([full_ids[1:]])
        loss, grads = grad_fn(qwen, input_ids_arr, target_ids_arr)
        optimizer.update(qwen, grads)
        mx.eval(qwen.parameters())
        last_loss = float(loss)
    return last_loss, config.inner_steps


def train_subdomain_lora(
    *,
    wrapper: Any,
    tokenizer: Any,
    train: Sequence[MMLURecord],
    config: LoRATrainConfig,
    seed: int,
    subdomain: str,
) -> dict[str, Any]:
    """Train (or further-train) the LoRA adapter on a single subdomain.

    Parameters
    ----------
    wrapper
        QwenMLXFP16Wrapper or compatible (must expose `.model` +
        `.tokenizer`).
    tokenizer
        Tokenizer with at least an ``encode`` method.
    train
        MMLU records for the subdomain training split.
    config
        LoRA hyperparameters.
    seed
        Cell-level seed; pinned for the per-subdomain SGD.
    subdomain
        Subject name (logged into the returned dict).

    Returns
    -------
    dict
        At minimum ``{"final_loss": float, "steps_run": int,
        "subdomain": str}``.

    Raises
    ------
    ValueError
        ``train`` is empty.
    """
    if not train:
        raise ValueError(
            f"empty train split for subdomain {subdomain!r}; G6 "
            "requires at least 1 training record per subdomain"
        )
    return _run_lora_iters(
        wrapper=wrapper,
        tokenizer=tokenizer,
        records=train,
        config=config,
        seed=seed,
        subdomain=subdomain,
    )


__all__ = ["LoRATrainConfig", "train_subdomain_lora"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/experiments/test_g6_micro_kiki_train.py -v --no-cov`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/micro_kiki_train.py \
        tests/unit/experiments/test_g6_micro_kiki_train.py
git commit -m "feat(g6): Path A LoRA training shim"
```

---

## Task 7: Implement the cell runner + retention metric + R1 wiring

**Files:**
- Modify: `experiments/g6_mmlu_stream/run_g6.py`
- Test: extend `tests/unit/experiments/test_g6_run_pilot.py`

- [ ] **Step 1: Write the failing test (extend the existing file)**

Append to `tests/unit/experiments/test_g6_run_pilot.py` :
```python
import json
from unittest.mock import patch

import numpy as np

from experiments.g6_mmlu_stream.run_g6 import (
    AccMatrix,
    CellResult,
    compute_retention,
    run_pilot,
)


def test_compute_retention_identity_when_no_forgetting() -> None:
    matrix: AccMatrix = {
        "anatomy": [0.5, 0.5, 0.5, 0.5, 0.5],
        "astronomy": [None, 0.5, 0.5, 0.5, 0.5],
        "business_ethics": [None, None, 0.5, 0.5, 0.5],
        "clinical_knowledge": [None, None, None, 0.5, 0.5],
        "college_biology": [None, None, None, None, 0.5],
    }
    retention = compute_retention(matrix, subdomains=(
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ))
    assert retention == 1.0


def test_compute_retention_drops_with_forgetting() -> None:
    matrix: AccMatrix = {
        "anatomy": [0.8, 0.7, 0.6, 0.5, 0.4],
        "astronomy": [None, 0.8, 0.7, 0.6, 0.5],
        "business_ethics": [None, None, 0.8, 0.7, 0.6],
        "clinical_knowledge": [None, None, None, 0.8, 0.7],
        "college_biology": [None, None, None, None, 0.8],
    }
    # Mean of (0.4/0.8, 0.5/0.8, 0.6/0.8, 0.7/0.8) = 0.75
    retention = compute_retention(matrix, subdomains=(
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ))
    assert retention == pytest.approx(0.75, abs=1e-9)


def test_compute_retention_excludes_zero_initial() -> None:
    """When acc[S_j after S_j] is 0, the ratio is undefined and the term is dropped."""
    matrix: AccMatrix = {
        "anatomy": [0.0, 0.0, 0.0, 0.0, 0.0],  # underperforming
        "astronomy": [None, 0.8, 0.4, 0.4, 0.4],
        "business_ethics": [None, None, 0.8, 0.4, 0.4],
        "clinical_knowledge": [None, None, None, 0.8, 0.4],
        "college_biology": [None, None, None, None, 0.8],
    }
    retention = compute_retention(matrix, subdomains=(
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ))
    # Only 3 ratios contribute: 0.4/0.8 = 0.5 each -> mean 0.5
    assert retention == pytest.approx(0.5, abs=1e-9)


def test_run_pilot_path_b_smoke(tmp_path) -> None:
    """End-to-end Path B smoke run with mocked Qwen wrapper."""
    fixture = tmp_path / "mmlu.jsonl"
    rows = []
    for subj in (
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ):
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4,
                "subject": subj,
            })
    fixture.write_text("\n".join(json.dumps(r) for r in rows))
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    db = tmp_path / "registry.sqlite"

    # Mock evaluate_mmlu to return a deterministic accuracy seeded by
    # (subdomain, arm, train_step) — keeps the smoke deterministic
    # while exercising the full pipeline.
    call_log = []

    def fake_eval(*args, **kwargs):
        records = kwargs.get("fixture_path") or "no-fixture"
        call_log.append(records)
        # Stable accuracy per call : argmax over a hash, mod-2 normalised.
        return {"accuracy": 0.5 + 0.01 * (len(call_log) % 5), "n": kwargs.get("n_samples", 0)}

    with patch(
        "experiments.g6_mmlu_stream.run_g6._evaluate_subdomain",
        side_effect=lambda *a, **k: 0.5 + 0.01 * (a[1] if len(a) > 1 else 0),
    ):
        payload = run_pilot(
            fixture_path=fixture,
            out_json=out_json,
            out_md=out_md,
            registry_db=db,
            seeds=(0,),
            n_train=4,
            n_eval=4,
            inner_steps=2,
            lr=5e-5,
            rank=4,
            alpha=4.0,
            path="B",
            scale_slot="qwen3p5-1p5b-fp16",
        )
    assert "cells" in payload
    assert len(payload["cells"]) == 4  # 4 arms × 1 seed
    for cell in payload["cells"]:
        assert "run_id" in cell
        assert "retention" in cell
        assert "acc_matrix" in cell
        assert cell["arm"] in ("baseline", "P_min", "P_equ", "P_max")
    assert out_json.exists()
    assert out_md.exists()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_g6_run_pilot.py -v --no-cov`
Expected: 4 new FAIL (ImportError on `compute_retention`, `AccMatrix`, etc.); 2 prior PASS.

- [ ] **Step 3: Replace `run_g6.py` with the full driver**

Replace `experiments/g6_mmlu_stream/run_g6.py` (overwrite the stub) with :
```python
"""G6 pilot driver — micro-kiki Qwen-35B × MMLU subdomain CL stream.

**Gate ID** : G6 — first empirical evidence on a real production LLM.
**Validates** : H1' / H3' / H_DR4' / H_NEW per `docs/osf-prereg-g6-pilot.md`.
**Path branches** :
- A : full LoRA pilot (Studio + KIKI-Mac_tunner + mlx_lm.lora).
- B : inference-only exploratory.
**Mode** : exploratory at first-pilot scale (3 seeds × 4 arms = 12 sequences).
**Expected output** :
    - docs/milestones/g6-pilot-<date>.json
    - docs/milestones/g6-pilot-<date>.md

Per-cell pipeline (per OSF pre-reg §1) :
    1. Fresh adapter / wrapper.
    2. For i in 1..5 :
       a. Adapt to subdomain S_i (Path A: train_subdomain_lora;
          Path B: adapt_subdomain inference-only shim).
       b. (Optional) Inject DreamEpisode (G6DreamRunner.run_episode).
       c. Eval on S_1..S_i.
    3. Compute retention per cell.
    4. Register cell in RunRegistry.

Usage ::

    # Path B Smoke (1 cell — P_min seed 0)
    uv run python experiments/g6_mmlu_stream/run_g6.py --smoke --path B

    # Path A full pilot (Studio overnight, ~15 h)
    uv run python experiments/g6_mmlu_stream/run_g6.py --path A
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
    JAVADI_2024_OVERALL,
)
from harness.real_benchmarks.mmlu import MMLURecord  # noqa: E402
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate  # noqa: E402

from experiments.g6_mmlu_stream.dream_wrap import G6DreamRunner  # noqa: E402
from experiments.g6_mmlu_stream.micro_kiki_inference import (  # noqa: E402
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from experiments.g6_mmlu_stream.micro_kiki_train import (  # noqa: E402
    LoRATrainConfig,
    train_subdomain_lora,
)
from experiments.g6_mmlu_stream.stream import (  # noqa: E402
    SubdomainSplit,
    build_subdomain_stream,
)


# AccMatrix[subject] = list of length-N_subdomains. accuracy after
# training step i ; None if i precedes the subject's introduction.
AccMatrix = dict[str, list[Optional[float]]]


class _CellPartial(TypedDict):
    arm: str
    seed: int
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    acc_matrix: AccMatrix


class CellResult(_CellPartial):
    run_id: str


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
)
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "milestones" / "g6-pilot-2026-05-03.json"
DEFAULT_OUT_MD = REPO_ROOT / "docs" / "milestones" / "g6-pilot-2026-05-03.md"
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
UNDERPERFORM_THRESHOLD = 0.30  # acc[S_1 after S_1] < 0.30 → exclude


def _resolve_commit_sha() -> str:
    """Mirror `experiments/g4_split_fmnist/run_g4.py:_resolve_commit_sha`."""
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def compute_retention(
    matrix: AccMatrix,
    *,
    subdomains: tuple[str, ...],
) -> float:
    """Compute retention = mean(acc_final / acc_initial) over subdomains.

    For each subdomain S_j (j < N), the ratio is
    ``acc_matrix[S_j][N-1] / max(acc_matrix[S_j][j-1], eps)`` where
    ``j-1`` is the column index immediately after S_j was trained.
    Subdomains with zero (≤ eps) initial accuracy contribute zero
    weight to the mean (the ratio would be undefined). The last
    subdomain (j == N-1) is excluded — there is no post-training step
    to forget.

    Returns 0.0 if no subdomain contributes a usable ratio.
    """
    n = len(subdomains)
    ratios: list[float] = []
    for j, subj in enumerate(subdomains[:-1]):  # exclude last
        col = matrix.get(subj, [])
        if len(col) <= n - 1:
            continue
        initial = col[j]
        final = col[n - 1]
        if initial is None or final is None:
            continue
        if initial < RETENTION_EPS:
            continue  # exclude zero-initial subdomain
        ratios.append(float(final) / float(initial))
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))


def _evaluate_subdomain(
    wrapper: Any,
    eval_records: list[MMLURecord],
    *,
    fixture_path: Optional[Path],
    seed: int,
) -> float:
    """Evaluate ``wrapper`` on ``eval_records`` (returns letter-argmax accuracy).

    Path B short-circuits to a deterministic mock when ``wrapper`` is
    None (no Qwen) — a function of (records, seed). Path A delegates
    to ``harness.real_benchmarks.mmlu.evaluate_mmlu``.
    """
    if wrapper is None:
        # Deterministic Path B mock : mean(record.answer == seed % 4).
        if not eval_records:
            return 0.0
        n_correct = sum(1 for r in eval_records if r.answer == (seed % 4))
        return n_correct / len(eval_records)
    from harness.real_benchmarks.mmlu import evaluate_mmlu

    # evaluate_mmlu loads its records from fixture_path; we pre-built
    # them via build_subdomain_stream so we materialise a temp fixture
    # for evaluate_mmlu to read.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for r in eval_records:
            fh.write(json.dumps({
                "question": r.question,
                "choices": list(r.choices),
                "answer": r.answer,
                "subject": r.subject,
            }) + "\n")
        tmp_fixture = Path(fh.name)
    try:
        result = evaluate_mmlu(
            wrapper, wrapper.tokenizer,
            n_samples=len(eval_records),
            seed=seed,
            fixture_path=tmp_fixture,
        )
    finally:
        tmp_fixture.unlink(missing_ok=True)
    return float(result["accuracy"])


def _run_cell(
    *,
    arm: str,
    seed: int,
    splits: list[SubdomainSplit],
    path: str,
    scale_slot: str,
    n_eval: int,
    train_config: LoRATrainConfig,
) -> _CellPartial:
    """Execute one (arm, seed) cell and return a `_CellPartial`."""
    start = time.time()
    subdomains = tuple(s.subject for s in splits)
    n_steps = len(splits)
    acc_matrix: AccMatrix = {
        subj: [None] * n_steps for subj in subdomains
    }

    # ----- substrate + adapter wiring -----
    substrate = MicroKikiSubstrate(num_layers=20, rank=train_config.rank, seed=seed)
    runner = G6DreamRunner(substrate=substrate, profile_name=arm)

    if path == "A":
        from harness.real_models.qwen_mlx_fp16 import load_qwen_fp16

        wrapper = load_qwen_fp16(scale_slot)
        adapter: InferenceOnlyAdapter | None = None
    else:
        wrapper = None
        adapter = InferenceOnlyAdapter(
            out_dim=8, rank=train_config.rank, seed=seed,
        )

    # ----- per-subdomain CL loop -----
    for i, split in enumerate(splits):
        # 1. Adapt to S_i
        if path == "A":
            train_subdomain_lora(
                wrapper=wrapper,
                tokenizer=wrapper.tokenizer,
                train=split.train,
                config=train_config,
                seed=seed,
                subdomain=split.subject,
            )
        else:
            assert adapter is not None
            adapt_subdomain(
                adapter=adapter,
                subdomain=split.subject,
                train=split.train,
                seed=seed,
            )

        # 2. Optional dream episode
        if arm != "baseline":
            runner.run_episode(seed=seed, subdomain=split.subject)

        # 3. Evaluate on S_1..S_i
        for j in range(i + 1):
            past = splits[j]
            acc = _evaluate_subdomain(
                wrapper, past.eval_,
                fixture_path=None, seed=seed,
            )
            acc_matrix[past.subject][i] = acc

    initial_first = acc_matrix[subdomains[0]][0]
    excluded = bool(
        initial_first is not None
        and initial_first < UNDERPERFORM_THRESHOLD
    )
    retention = compute_retention(acc_matrix, subdomains=subdomains)
    return {
        "arm": arm,
        "seed": seed,
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
        "acc_matrix": acc_matrix,
    }


def _retention_by_arm(cells: list[CellResult]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h1_prime_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    p_equ, base = retention["P_equ"], retention["baseline"]
    if len(p_equ) < 2 or len(base) < 2:
        return {"insufficient_samples": True,
                "n_p_equ": len(p_equ), "n_base": len(base)}
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 4)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 4,
        "n_p_equ": len(p_equ), "n_base": len(base),
    }


def _h3_prime_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    p_min, base = retention["P_min"], retention["baseline"]
    if len(p_min) < 2 or len(base) < 2:
        return {"insufficient_samples": True,
                "n_p_min": len(p_min), "n_base": len(base)}
    g = compute_hedges_g(p_min, base)
    welch = welch_one_sided(p_min, base, alpha=0.05 / 4)
    return {
        "hedges_g": g,
        "is_within_javadi_2024_ci": JAVADI_2024_OVERALL.is_within_ci(abs(g)),
        "below_javadi_2024_lower_ci_decrement": bool(
            g <= -JAVADI_2024_OVERALL.ci_low
        ),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 4,
        "n_p_min": len(p_min), "n_base": len(base),
    }


def _h_dr4_prime_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    groups = [retention["P_min"], retention["P_equ"], retention["P_max"]]
    if any(len(g) < 2 for g in groups):
        return {"insufficient_samples": True,
                "n_per_arm": [len(g) for g in groups]}
    res = jonckheere_trend(groups, alpha=0.05)
    means = [float(sum(g) / len(g)) for g in groups]
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": means[0], "mean_p_equ": means[1], "mean_p_max": means[2],
        "monotonic_observed": means[0] <= means[1] <= means[2],
    }


def _h_new_verdict(
    g_h1_prime: float | None,
    g_synthetic: float | None,
    margin: float = 0.10,
) -> dict[str, Any]:
    """H_NEW : non-inferiority of real vs synthetic effect size."""
    if g_h1_prime is None or g_synthetic is None:
        return {"insufficient_samples": True}
    diff = g_h1_prime - g_synthetic
    return {
        "g_h1_prime": g_h1_prime,
        "g_synthetic": g_synthetic,
        "diff": diff,
        "margin": -margin,
        "non_inferiority_observed": diff >= -margin,
    }


def _aggregate_verdict(
    cells: list[CellResult],
    *,
    g_synthetic: float | None,
) -> dict[str, Any]:
    retention = _retention_by_arm(cells)
    h1 = _h1_prime_verdict(retention)
    h3 = _h3_prime_verdict(retention)
    h4 = _h_dr4_prime_verdict(retention)
    h_new = _h_new_verdict(
        h1.get("hedges_g") if not h1.get("insufficient_samples") else None,
        g_synthetic,
    )
    return {
        "h1_prime_p_equ_vs_baseline": h1,
        "h3_prime_p_min_vs_baseline": h3,
        "h_dr4_prime_jonckheere": h4,
        "h_new_synth_to_real": h_new,
        "retention_by_arm": retention,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# G6 pilot — micro-kiki Qwen-{payload['scale_slot']} × MMLU CL stream",
        "",
        f"**Date** : {payload['date']}",
        f"**Path** : {payload['path']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        f"**Cells** : {len(payload['cells'])}",
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypotheses",
        "",
        "Pre-registration : `docs/osf-prereg-g6-pilot.md`",
        "",
    ]
    h1 = payload["verdict"]["h1_prime_p_equ_vs_baseline"]
    h3 = payload["verdict"]["h3_prime_p_min_vs_baseline"]
    h4 = payload["verdict"]["h_dr4_prime_jonckheere"]
    h_new = payload["verdict"]["h_new_synth_to_real"]
    lines.append("### H1' — P_equ retention vs Hu 2020 (g >= 0.21)")
    lines.append(f"```\n{json.dumps(h1, indent=2, sort_keys=True)}\n```")
    lines.append("### H3' — P_min retention vs Javadi 2024 (g <= -0.13)")
    lines.append(f"```\n{json.dumps(h3, indent=2, sort_keys=True)}\n```")
    lines.append("### H_DR4' — Jonckheere monotonicity")
    lines.append(f"```\n{json.dumps(h4, indent=2, sort_keys=True)}\n```")
    lines.append("### H_NEW — Synthetic-to-real non-inferiority")
    lines.append(f"```\n{json.dumps(h_new, indent=2, sort_keys=True)}\n```")
    lines.append("")
    lines.append("## Cells (R1 traceability)")
    lines.append("")
    lines.append("| arm | seed | retention | excluded | run_id |")
    lines.append("|-----|------|-----------|----------|--------|")
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | {c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | "
            f"`{c['run_id']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def run_pilot(
    *,
    fixture_path: Path,
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    seeds: tuple[int, ...],
    n_train: int,
    n_eval: int,
    inner_steps: int,
    lr: float,
    rank: int,
    alpha: float,
    path: str,
    scale_slot: str,
    g_synthetic: float | None = None,
    subdomains: tuple[str, ...] = DEFAULT_SUBDOMAINS,
) -> dict[str, Any]:
    """Execute the G6 pilot sweep and return the verdict payload."""
    if path not in ("A", "B"):
        raise ValueError(f"path must be 'A' or 'B', got {path!r}")

    splits = build_subdomain_stream(
        fixture_path=fixture_path,
        subdomains=subdomains,
        n_train=n_train,
        n_eval=n_eval,
        seed=0,  # subdomain split seed pinned at 0; cell seed varies separately
    )
    train_config = LoRATrainConfig(
        lr=lr, inner_steps=inner_steps, batch_size=4,
        rank=rank, alpha=alpha,
    )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[CellResult] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            cell = _run_cell(
                arm=arm, seed=seed, splits=splits,
                path=path, scale_slot=scale_slot,
                n_eval=n_eval, train_config=train_config,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g6/{path}/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cells.append(CellResult(**cell, run_id=run_id))
    wall = time.time() - sweep_start

    verdict = _aggregate_verdict(cells, g_synthetic=g_synthetic)
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "path": path,
        "scale_slot": scale_slot,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "subdomains": list(subdomains),
        "fixture_path": str(fixture_path),
        "wall_time_s": wall,
        "cells": cells,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G6 pilot driver")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1 cell (P_min, seed 0) on the sanity fixture.")
    parser.add_argument("--path", choices=("A", "B"), default="B",
                        help="Path A (full LoRA pilot) or B (inference-only).")
    parser.add_argument("--scale", default="qwen3p5-1p5b-fp16",
                        help="Base model slot (Path A only).")
    parser.add_argument("--fixture-path", type=Path,
                        default=REPO_ROOT / "tests" / "fixtures" / "mmlu_sanity.jsonl")
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--registry-db", type=Path, default=DEFAULT_REGISTRY_DB)
    parser.add_argument("--g-synthetic", type=float, default=None,
                        help="G4-bis synthetic g_h1 for H_NEW non-inferiority test.")
    args = parser.parse_args(argv)

    if args.smoke:
        seeds = (0,)
        n_train, n_eval, inner_steps = 4, 4, 2
    else:
        seeds = tuple(args.seeds)
        n_train, n_eval, inner_steps = args.n_train, args.n_eval, args.inner_steps

    payload = run_pilot(
        fixture_path=args.fixture_path,
        out_json=args.out_json, out_md=args.out_md,
        registry_db=args.registry_db,
        seeds=seeds,
        n_train=n_train, n_eval=n_eval,
        inner_steps=inner_steps, lr=args.lr,
        rank=args.rank, alpha=args.alpha,
        path=args.path, scale_slot=args.scale,
        g_synthetic=args.g_synthetic,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all G6 tests to verify they pass**

Run: `uv run pytest tests/unit/experiments/ -v --no-cov`
Expected: all G6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_mmlu_stream/run_g6.py \
        tests/unit/experiments/test_g6_run_pilot.py
git commit -m "feat(g6): cell runner + retention metric + R1"
```

---

## Task 8: Add property tests for determinism (hypothesis-driven)

**Files:**
- Test: `tests/unit/experiments/test_g6_determinism.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/experiments/test_g6_determinism.py` :
```python
"""Determinism property tests for the G6 pilot pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from experiments.g6_mmlu_stream.dream_wrap import G6DreamRunner
from experiments.g6_mmlu_stream.micro_kiki_inference import (
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from experiments.g6_mmlu_stream.stream import build_subdomain_stream
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate
from harness.real_benchmarks.mmlu import MMLURecord


@settings(max_examples=20, deadline=2_000)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_dream_runner_deterministic_for_fixed_seed(seed: int) -> None:
    """Same seed + same profile + same subdomain → same substrate state."""
    s1 = MicroKikiSubstrate(num_layers=4, rank=4, seed=seed)
    s2 = MicroKikiSubstrate(num_layers=4, rank=4, seed=seed)
    r1 = G6DreamRunner(substrate=s1, profile_name="P_equ", out_dim=4, rank=2)
    r2 = G6DreamRunner(substrate=s2, profile_name="P_equ", out_dim=4, rank=2)
    r1.run_episode(seed=seed, subdomain="anatomy")
    r2.run_episode(seed=seed, subdomain="anatomy")
    assert r1.last_episode_id == r2.last_episode_id
    assert s1.recombine_state.last_output_shape == s2.recombine_state.last_output_shape
    assert s1.restructure_state.last_episode_id == s2.restructure_state.last_episode_id


@settings(max_examples=10, deadline=2_000)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_inference_adapter_deterministic_for_fixed_seed(seed: int) -> None:
    """Two adapters with the same seed produce the same delta."""
    a1 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=seed)
    a2 = InferenceOnlyAdapter(out_dim=8, rank=4, seed=seed)
    train = [
        MMLURecord(question=f"Q{i}?", choices=("A", "B", "C", "D"),
                   answer=i % 4, subject="anatomy")
        for i in range(4)
    ]
    adapt_subdomain(adapter=a1, subdomain="anatomy", train=train, seed=seed)
    adapt_subdomain(adapter=a2, subdomain="anatomy", train=train, seed=seed)
    import numpy as np
    np.testing.assert_array_equal(
        a1.current_delta("layer_0_lora_B"),
        a2.current_delta("layer_0_lora_B"),
    )


def test_subdomain_stream_split_seed_isolated_from_cell_seed(tmp_path: Path) -> None:
    """Stream-split seed pins the splits ; cell seed must not affect the split."""
    rows = []
    for subj in ("anatomy", "astronomy"):
        for i in range(8):
            rows.append({"question": f"{subj}-Q{i}?", "choices": ["A", "B", "C", "D"],
                         "answer": i % 4, "subject": subj})
    fixture = tmp_path / "f.jsonl"
    fixture.write_text("\n".join(json.dumps(r) for r in rows))
    a = build_subdomain_stream(
        fixture_path=fixture, subdomains=("anatomy", "astronomy"),
        n_train=4, n_eval=2, seed=0,
    )
    b = build_subdomain_stream(
        fixture_path=fixture, subdomains=("anatomy", "astronomy"),
        n_train=4, n_eval=2, seed=0,
    )
    for sa, sb in zip(a, b):
        assert [r.question for r in sa.train] == [r.question for r in sb.train]
        assert [r.question for r in sa.eval_] == [r.question for r in sb.eval_]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/unit/experiments/test_g6_determinism.py -v --no-cov`
Expected: 3 PASS (the implementations from prior tasks already satisfy determinism).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/experiments/test_g6_determinism.py
git commit -m "test(g6): determinism property tests"
```

---

## Task 9: Smoke test on 1 subdomain × 1 seed (Path B)

**Files:**
- (no new files)

- [ ] **Step 1: Run the Path B smoke**

Run :
```bash
DREAMOFKIKI_COMMIT_SHA=$(git rev-parse HEAD) \
  uv run python experiments/g6_mmlu_stream/run_g6.py \
    --smoke --path B \
    --fixture-path tests/fixtures/mmlu_sanity.jsonl \
    --out-json /tmp/g6-smoke.json \
    --out-md /tmp/g6-smoke.md \
    --registry-db /tmp/g6-smoke-registry.sqlite
```
Expected: prints `Cells : 4`; both /tmp/g6-smoke.{json,md} exist.

> **Note** : the sanity fixture has < 5 records per subject. The
> stream loader will raise ``ValueError: not enough records`` on
> the production subjects (anatomy, astronomy, ...) which the
> sanity fixture does not cover. The smoke must therefore use the
> sanity fixture's actual subjects via a temporary override or
> a pre-built per-subject fixture. Implement that override here :

Add a `--smoke-subdomains` CLI flag to `run_g6.py:main` that defaults
to the sanity fixture's actual subjects when `--smoke` is set :
```python
SMOKE_SUBDOMAINS = (
    "world_facts",
    "astronomy",
    "elementary_mathematics",
    "world_literature",
    "chemistry",
)
```
Wire `subdomains=SMOKE_SUBDOMAINS` into `run_pilot` when `args.smoke`.

- [ ] **Step 2: Verify the smoke output**

Run :
```bash
uv run python -c "import json; d=json.load(open('/tmp/g6-smoke.json')); \
  print('cells', len(d['cells'])); \
  print('arms', sorted({c['arm'] for c in d['cells']})); \
  print('verdict_keys', sorted(d['verdict'].keys()))"
```
Expected :
```
cells 4
arms ['P_equ', 'P_max', 'P_min', 'baseline']
verdict_keys ['h1_prime_p_equ_vs_baseline', 'h3_prime_p_min_vs_baseline', 'h_dr4_prime_jonckheere', 'h_new_synth_to_real', 'retention_by_arm']
```

- [ ] **Step 3: Commit the smoke-subdomain wiring**

```bash
git add experiments/g6_mmlu_stream/run_g6.py
git commit -m "feat(g6): --smoke uses sanity fixture subjects"
```

---

## Task 10: Run the full pilot (overnight; mark as long-running)

**Files:**
- Output: `docs/milestones/g6-pilot-2026-05-03.{json,md}` (created by the run)

> **WARNING** : This task commits 10–25 h of Studio compute (Path A)
> or ~30 min (Path B). Verify the Task 0.5 path lock before
> launching. Path A pilot must run on Studio with `KIKI-Mac_tunner`
> accessible.

- [ ] **Step 1: Pre-flight check**

Run :
```bash
# Verify pre-reg is locked
git log --oneline docs/osf-prereg-g6-pilot.md | head -1

# Verify decisions doc is committed
git log --oneline docs/milestones/g6-pilot-decisions-2026-05-03.md | head -1

# Verify smoke run dump still parses
uv run python -c "import json; json.load(open('/tmp/g6-smoke.json'))"

# Verify G4-bis synthetic g_h1 is recorded (read from
# docs/milestones/g4-bis-pilot-*.json once that lands)
ls -la docs/milestones/ | grep -i g4-bis || echo "WARN g4-bis not yet landed"
```
Expected: all four lines succeed (the WARN is allowed iff Task 0.5
recorded `g6_unblocked = false` and the executing agent is rebuilding
G4-bis evidence in parallel).

- [ ] **Step 2: Launch the full pilot (Path B reduced-scope or Path A overnight)**

Path A (Studio) :
```bash
DREAMOFKIKI_COMMIT_SHA=$(git rev-parse HEAD) \
  nohup uv run python experiments/g6_mmlu_stream/run_g6.py \
    --path A \
    --scale qwen3p6-35b-bf16-local \
    --fixture-path /path/to/full-cais-mmlu.jsonl \
    --n-train 100 --n-eval 100 --inner-steps 50 \
    --lr 5e-5 --rank 16 --alpha 16 \
    --seeds 0 1 2 \
    --g-synthetic "$(uv run python -c \
        'import json,glob; \
         d=json.load(open(sorted(glob.glob(\"docs/milestones/g4-bis-pilot-*.json\"))[-1])); \
         print(d[\"verdict\"][\"h1_p_equ_vs_baseline\"][\"hedges_g\"])')" \
    > /tmp/g6-full.log 2>&1 &
```
Path B (any host) :
```bash
DREAMOFKIKI_COMMIT_SHA=$(git rev-parse HEAD) \
  uv run python experiments/g6_mmlu_stream/run_g6.py \
    --path B \
    --fixture-path /path/to/full-cais-mmlu.jsonl \
    --n-train 100 --n-eval 100 \
    --seeds 0 1 2 \
    --g-synthetic "<from g4-bis>"
```
Expected (Path A) : runs overnight; on completion writes
`docs/milestones/g6-pilot-2026-05-03.{json,md}`. Wall time line in JSON
should record between 36000 and 90000 seconds (10–25 h).

- [ ] **Step 3: Verify the dump**

Run :
```bash
uv run python -c "
import json
d = json.load(open('docs/milestones/g6-pilot-2026-05-03.json'))
assert len(d['cells']) == 12, f'expected 12 cells, got {len(d[\"cells\"])}'
assert d['path'] in ('A', 'B')
assert all('run_id' in c and len(c['run_id']) == 32 for c in d['cells']), \
    'missing run_id or wrong length'
print('OK', d['path'], 'wall_time_s =', d['wall_time_s'])
print('H1 g =', d['verdict']['h1_prime_p_equ_vs_baseline'].get('hedges_g'))
print('H3 g =', d['verdict']['h3_prime_p_min_vs_baseline'].get('hedges_g'))
print('H_DR4 mono =', d['verdict']['h_dr4_prime_jonckheere'].get('monotonic_observed'))
print('H_NEW non-inf =', d['verdict']['h_new_synth_to_real'].get('non_inferiority_observed'))
"
```
Expected: prints OK with 12 cells and four verdict lines.

- [ ] **Step 4: Commit the milestone**

```bash
git add docs/milestones/g6-pilot-2026-05-03.json \
        docs/milestones/g6-pilot-2026-05-03.md
git commit -m "feat(g6): pilot dump 2026-05-03"
```

---

## Task 11: Update Paper 2 §7.1.4 (NEW) with G6 results — EN + FR

**Files:**
- Modify: `docs/papers/paper2/results.md`
- Modify: `docs/papers/paper2-fr/results.md`

> **Critical** : Paper 1 / Paper 2 require synchronised EN→FR edits
> in the same commit (see `docs/CLAUDE.md` "Bilingual mirror" rule
> + `CONTRIBUTING.md`). Both files must be edited; the FR mirror
> follows the same structure.

- [ ] **Step 1: Append §7.1.4 to results.md (EN)**

Insert after §7.1.1 (line 105 in `docs/papers/paper2/results.md`) and
before §7.2, this new subsection :
```markdown
## 7.1.4 G6 pilot (real LLM continual learning — 2026-05-03)

First validation of framework C on a real production LLM
(Qwen3.6-35B-A3B + r=16 LoRA via the
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate`). The pilot
exposes the substrate to a 5-subdomain MMLU continual-learning
stream (anatomy → astronomy → business_ethics → clinical_knowledge
→ college_biology) and measures forgetting under four arms
(`baseline`, `P_min`, `P_equ`, `P_max`).

**Pre-registration** : `docs/osf-prereg-g6-pilot.md` (locked
2026-05-03, hypotheses H1', H3', H_DR4', H_NEW with Bonferroni
α/4 = 0.0125 across the inferential family).

**Path branches** :
- *Path A* — Studio + `~/KIKI-Mac_tunner/` + `mlx_lm.lora`
  fine-tune per subdomain. The publishable G6 result.
- *Path B* — any host, inference-only adapter-tensor mutation
  (no real LoRA training). Exploratory; never triggers a STABLE
  promotion.

**Cells** : 4 arms × 3 seeds = 12 sequences (each touches 5
subdomains, yielding 60 forgetting measurements).
**Compute** : Path A — ~15 h on Studio M3 Ultra; Path B — ~30 min
on any Apple Silicon.

**Verdict scalars** (see `docs/milestones/g6-pilot-2026-05-03.json`):

| Hypothesis | Direction | Anchor | Observed | Reject H0 |
|---|---|---|---|---|
| H1' | g >= 0.21 | Hu 2020 lower CI | g_h1' = TBA | TBA |
| H3' | g <= -0.13 | Javadi 2024 lower CI | g_h3' = TBA | TBA |
| H_DR4' | mono(P_min, P_equ, P_max) | DR-4 | TBA | TBA |
| H_NEW | g_real >= g_synth - 0.10 | G4-bis | TBA | TBA |

(Numbers materialise after Task 10 lands the milestone dump; this
section is committed alongside the dump in the same PR.)

**Interpretation** : G6 closes the synthetic-to-real gap that
G4-bis (MLX coupling) and G5 (cross-substrate E-SNN) leave open.
A confirmatory positive verdict on H1' / H3' / H_DR4' / H_NEW is
the EC-axis trigger for promoting the cumulative G4-G6 scope from
`PARTIAL` to `STABLE`. A null or falsifying verdict triggers the
§12.3 transition rule documented in
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`.

Reference : `docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md`
+ `docs/osf-prereg-g6-pilot.md`.
```

- [ ] **Step 2: Append §7.1.4 to results.md (FR mirror)**

Insert the equivalent section into `docs/papers/paper2-fr/results.md`
at the same structural position. Keep the same anchor labels and
verdict-table structure; translate prose. Use the canonical FR
glossary terms from `docs/glossary.md` (e.g. "ré-tention" not
"retention", "oubli catastrophique" for "catastrophic forgetting").

```markdown
## 7.1.4 Pilote G6 (apprentissage continu sur LLM réel — 2026-05-03)

Première validation du framework C sur un LLM de production
(Qwen3.6-35B-A3B + LoRA r=16 via le substrat
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate`). Le pilote
expose le substrat à un flux d'apprentissage continu de 5
sous-domaines MMLU (anatomy → astronomy → business_ethics →
clinical_knowledge → college_biology) et mesure l'oubli
catastrophique sous quatre conditions (`baseline`, `P_min`,
`P_equ`, `P_max`).

**Pré-enregistrement** : `docs/osf-prereg-g6-pilot.md` (verrouillé
le 2026-05-03, hypothèses H1', H3', H_DR4', H_NEW avec correction
de Bonferroni α/4 = 0.0125).

**Branches d'exécution** :
- *Voie A* — Studio + `~/KIKI-Mac_tunner/` + `mlx_lm.lora`
  fine-tuning par sous-domaine. Résultat publiable.
- *Voie B* — n'importe quel hôte, mutation des tenseurs LoRA
  uniquement (pas de fine-tuning réel). Exploratoire; ne déclenche
  jamais de promotion STABLE.

**Cellules** : 4 bras × 3 graines = 12 séquences (chacune touche
5 sous-domaines, soit 60 mesures d'oubli au total).

**Verdict** : voir `docs/milestones/g6-pilot-2026-05-03.json` (le
tableau complet est inséré ici dès l'atterrissage du dump).

**Interprétation** : G6 clôt l'écart synthétique-vers-réel laissé
par G4-bis (couplage MLX) et G5 (cross-substrat E-SNN). Une
confirmation positive de H1' / H3' / H_DR4' / H_NEW déclenche la
promotion EC du scope cumulé G4-G6 de `PARTIAL` à `STABLE`. Un
verdict nul ou falsifiant déclenche la règle de transition §12.3
documentée dans
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`.

Référence : `docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md`
+ `docs/osf-prereg-g6-pilot.md`.
```

- [ ] **Step 3: Verify both files were updated**

Run :
```bash
grep -n "## 7.1.4" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
```
Expected: each file emits exactly one match for `## 7.1.4`.

- [ ] **Step 4: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): G6 pilot section 7.1.4 EN+FR"
```

---

## Task 12: Update CHANGELOG + STATUS — DualVer branch

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Append a CHANGELOG entry under [Unreleased]**

Find the `[Unreleased]` block in `CHANGELOG.md` and append a new
`### Empirical` row. **Do not delete or rewrite the prior G4 entry**
(append-only convention).

```markdown
### Empirical (no FC bump — empirical pilot only)
- G6 pilot 2026-05-03 (micro-kiki Qwen-35B × MMLU subdomain CL stream)
  returned <STATUS> evidence on the four pre-registered hypotheses
  in `docs/osf-prereg-g6-pilot.md`. See
  `docs/milestones/g6-pilot-2026-05-03.{json,md}` for per-cell
  R1 traceability. Path = <A | B>.
- 12 cells under
  `(C-v0.12.0+PARTIAL, g6/<path>/{baseline,P_min,P_equ,P_max}, seed)`
  registered in `.run_registry.sqlite` with R1 bit-stable run_ids.
- Per framework-C §12.3 :
  - If H1' AND H3' AND H_DR4' AND H_NEW all confirm in the
    predicted direction (Path A only) → propose
    `PARTIAL → STABLE` for the cumulative G4 + G5 + G6 scope.
  - If any predicted direction is violated → propose
    `PARTIAL → UNSTABLE`.
  - Otherwise (partial confirmation, or Path B) → stay PARTIAL.
```

(Replace `<STATUS>`, `<A | B>`, and `<path>` after Task 10 lands.)

- [ ] **Step 2: Update the Versioning sub-section under [Unreleased]**

Append :
```markdown
### Versioning (G6 outcome branch)
- DualVer bump deferred until Task 10 dump exists. Rules :
  - Path A all-confirm : EC PARTIAL → STABLE, FC unchanged at v0.12.0.
    Tag `c-v0.13.0+STABLE`. Version line in run_g6.py + STATUS.md +
    CHANGELOG.md updated atomically.
  - Path A any-falsify : EC PARTIAL → UNSTABLE, FC unchanged.
    Tag `c-v0.12.1+UNSTABLE`.
  - Path A partial / Path B (any) : no version bump; the empirical
    row above documents the run, EC stays PARTIAL.
```

- [ ] **Step 3: Append a G6 row to the STATUS.md gates table**

In `STATUS.md`, find the gates table and append (after `Paper 1 v0.2 PLOS CB`) :
```markdown
| G6 — micro-kiki real LLM CL stream | 2026-05-03 → milestone | <STATUS_AFTER_TASK_10> |
```

If the Task 10 result triggers a STABLE promotion, also update the
`DualVer status` table : EC PARTIAL → STABLE, scope = G4 + G5 + G6.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(g6): CHANGELOG + STATUS row"
```

---

## Task 13: Self-review

**Files:**
- (no edits — review-only task; emit a concise report)

- [ ] **Step 1: Re-read the spec section by section against the plan**

Open `/Users/electron/hypneum-lab/dream-of-kiki/docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md`
in side-by-side with the user's spec (the message that triggered this
plan write). For each spec bullet, find the task that implements it.
Record any gaps below :

```
Spec bullet → Task mapping :
- Goal first validation real LLM → Tasks 0.5, 7, 10
- Substrate kiki_oniric.substrates.micro_kiki → Task 4 (dream_wrap), Task 5 (Path B), Task 6 (Path A)
- Benchmark MMLU 57 subdomains via harness/real_benchmarks/mmlu.py → Task 3 (stream loader)
- CL protocol train T1, eval T1, train T2, ... → Task 7 (_run_cell)
- 4 arms baseline / P_min / P_equ / P_max → Task 7 (ARMS) + Task 4 (PROFILE_OPS)
- Replay = sample β buffer + LoRA training step → Task 5 / Task 6
- Downscale = LoRA weight magnitude shrink → Task 4 (downscale handler invocation)
- Restructure = TIES-merge across LoRA deltas → Task 4 (restructure_handler call wired)
  *Note: spec says "TIES-merge in restructure"; framework C wires OPLoRA in restructure
  and TIES-merge in recombine — plan follows framework C, not the spec wording, and
  documents the discrepancy in the OSF pre-reg.*
- Recombine = generative replay → Task 4 (recombine_handler invocation; production-side
  generative replay is a Path A future-work item, not blocking)
- Compute MAJOR risk → Task 0.5 + Task 9 (smoke first) + Task 10 (overnight gate)
- 5 subdomains × 4 arms × 3 seeds = 60 cells → Task 7 + Task 0.5 decisions doc
- 10-30 h budget → Task 0.5 acceptance + Task 10 launch gate
- H1' Hu 2020 g >= 0.21 → Task 7 _h1_prime_verdict
- H3' Javadi 2024 g >= 0.13 → Task 7 _h3_prime_verdict
- H_DR4' Jonckheere → Task 7 _h_dr4_prime_verdict
- H_NEW synth-to-real → Task 7 _h_new_verdict + Task 1 pre-reg + Task 11 paper section
- Task 0 investigate → Task 0
- Task 0.5 decisions → Task 0.5
- Task 1 OSF pre-reg → Task 1
- Task 2 stub → Task 2
- Task 3 stream loader → Task 3
- Task 4 dream-episode coupling → Task 4
- Task 5 cell runner → Task 7
- Task 6 retention metric → Task 7 compute_retention
- Task 7 R1 register → Task 7 (run_pilot)
- Task 8 property tests → Task 8
- Task 9 smoke → Task 9
- Task 10 full run → Task 10
- Task 11 Paper 2 EN+FR → Task 11
- Task 12 CHANGELOG + STATUS → Task 12
- Task 13 self-review → Task 13
- KIKI-Mac_tunner blocker → Task 0.5 path matrix + Task 6 (production seam) + Task 5 (Path B fallback)
- LoRA training stability → Task 0.5 hyperparams (lr 5e-5, inner_steps 50, batch 4)
- G4-bis dependency → Task 0.5 g6_unblocked field + Task 10 pre-flight
- OSF pre-reg before launch → Task 1 + Task 10 step 1 pre-flight
- NO FC bump → Task 12 versioning rules
- Possible STABLE promotion → Task 12 step 2
- MMLU SHA-256 pinning → Task 3 (uses MMLULoader path, fixture_path arg)
```

- [ ] **Step 2: Placeholder scan**

Run :
```bash
grep -n "TBD\|TODO\|FIXME\|fill in details\|implement later" \
  experiments/g6_mmlu_stream/ tests/unit/experiments/ \
  docs/osf-prereg-g6-pilot.md docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md
```
Expected: only the deliberate `TBA` cells in §7.1.4 of paper2/results.md
(filled by Task 10's milestone dump). Anything else is a plan failure
and must be patched in-place.

- [ ] **Step 3: Type-consistency scan**

Run :
```bash
uv run mypy experiments/g6_mmlu_stream
```
Expected: `Success: no issues found`. Fix any drift between the types
declared in `stream.py` (`SubdomainSplit`), `dream_wrap.py` (`PROFILE_OPS`),
`micro_kiki_inference.py` (`InferenceOnlyAdapter`),
`micro_kiki_train.py` (`LoRATrainConfig`), and `run_g6.py`
(`AccMatrix`, `CellResult`).

- [ ] **Step 4: Coverage gate**

Run :
```bash
uv run pytest tests/unit/experiments/ tests/unit/test_osf_prereg_g6_present.py
```
Expected: all PASS, coverage gate (90%) holds for the new package.

- [ ] **Step 5: Emit a one-line review verdict**

Append to the executing agent's summary :
```
G6 plan self-review : <PASS | NEEDS FIX>
gaps : <list any spec bullets without an implementing task>
```

---

## Self-review (plan author, run during writing)

**Spec coverage** : every spec bullet (Goal, Architecture, Tasks 0-13,
Critical caveats 1-8) maps to at least one task. The "TIES-merge in
restructure" wording in the spec is a discrepancy with framework C
(framework C: OPLoRA = restructure, TIES-Merge = recombine). The plan
follows framework C and documents the discrepancy explicitly in Task 13
step 1 self-review map.

**Placeholder scan** : the only `TBA` markers are in the verdict table
of §7.1.4 (Task 11), which is intentional — the numbers materialise
after Task 10. The plan's pre-reg + tests + driver code contain zero
placeholders.

**Type consistency** :
- `SubdomainSplit(subject: str, train: list[MMLURecord], eval_: list[MMLURecord])`
  — used identically in stream.py + run_g6.py
- `LoRATrainConfig(lr: float, inner_steps: int, batch_size: int, rank: int, alpha: float)`
  — used identically in micro_kiki_train.py + run_g6.py
- `AccMatrix = dict[str, list[Optional[float]]]` — defined in run_g6.py,
  consumed by `compute_retention` (same module), no cross-module drift.
- `G6DreamRunner.run_episode(*, seed: int, subdomain: str)` — same
  signature in dream_wrap.py + run_g6.py call site.
- `MicroKikiSubstrate` constructor signature respected
  (`num_layers`, `rank`, `seed`).

**Open issues / risks (recorded for the executing agent)** :

1. **G4-bis dependency** : G6 should not launch full pilot until G4-bis
   confirms non-zero coupling effect. The Task 0.5 decisions doc has an
   explicit `g6_unblocked` field; Task 10 step 1 enforces it.
2. **KIKI-Mac_tunner workspace** : verified MISSING on this host
   (`/Users/electron/hypneum-lab/` does not contain it). The default
   path on this host is **Path B** (inference-only); Path A becomes
   accessible only when running on Studio with `~/KIKI-Mac_tunner/`.
3. **Sanity fixture coverage** : `tests/fixtures/mmlu_sanity.jsonl`
   does not contain the production target subjects (anatomy, astronomy,
   …) — Task 9 step 1 substitutes its actual subjects via a
   `--smoke-subdomains` flag wired in the same task.
4. **EC promotion gating** : Task 12 versioning rules are conditional
   on the Task 10 verdict. The executing agent must read the milestone
   JSON before editing CHANGELOG/STATUS. No automatic STABLE promotion.
5. **Path B is exploratory only** : pre-reg §6 forbids STABLE/UNSTABLE
   promotion under Path B. Task 12 versioning rules enforce this.
6. **MMLU full export not committed to repo** : Path A pilot requires
   a pre-materialised `cais/mmlu` JSONL file on disk. The harness
   `MMLULoader` is network-free by design; exporting the dataset is
   a Studio-side prerequisite, not a step in this plan. The pre-flight
   in Task 10 step 1 checks for it.
