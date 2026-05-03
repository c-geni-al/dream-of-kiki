# G4-sexto CIFAR-100 + Tiny-ImageNet escalation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the G4-quinto H5-C verdict (RECOMBINE empirically empty across FMNIST + CIFAR-10 × {MLP, CNN}) holds at mid-large scale by escalating to Split-CIFAR-100 (100 classes) and Split-Tiny-ImageNet (200 classes, 64x64 RGB) ; emit confirmatory verdicts H6-A / H6-B / H6-C and revise DR-4 evidence accordingly.

**Architecture:** Two sequential steps (no parallel third step — H6-C is a derived conjunction over H6-A and H6-B). Step 1 = port `G4SmallCNN` to multi-class output and run RECOMBINE strategy placebo (mog vs none) on Split-CIFAR-100 (10 tasks × 10 classes). Step 2 = build a taller `G4MediumCNN` (3 Conv2d + 3 MaxPool2d, 64x64 input) and run the same placebo design on Split-Tiny-ImageNet (10 tasks × 20 classes). Each step independently emits an H6-x verdict ; the aggregator computes H6-C as `H6-A_confirmed AND H6-B_confirmed`. All steps reuse the existing pre-reg / Welch / Bonferroni / R1 run-registry pipeline ; no axiom signature changes (no FC bump). EC stays PARTIAL across all outcomes.

**Tech Stack:** Python 3.12, MLX (`mlx.core`, `mlx.nn` — `nn.Conv2d`, `nn.MaxPool2d`), numpy, scipy.stats (Welch), Pillow (PNG/JPEG decode), pyarrow (parquet), `kiki_oniric.eval.statistics` (Welch + Hedges'g), `harness.storage.run_registry.RunRegistry` (R1 bit-stable run_ids), pure-numpy CIFAR-100 binary loader (~163 MB) + HF parquet Tiny-ImageNet fallback (~250 MB).

---

## Read-first context

Skim — do **not** rewrite — these files. They are the load-bearing patterns reused verbatim or with surgical substitutions :

- `docs/osf-prereg-g4-quinto-pilot.md` — pre-reg template (§1-§9 layout, deviation envelopes, honest-reading clause). Mirror structure.
- `docs/milestones/g4-quinto-aggregate-2026-05-03.md` — H5-C confirmed verdict (Welch p=0.9918, g=-0.0026, N=30). Cited in §1 of new pre-reg.
- `docs/proofs/dr4-profile-inclusion.md` §"Empirical-evidence amendment — G4-quinto" (lines 161-229) — current v0.4 amendment that this plan **extends to v0.5** (does not supersede ; append-only per `docs/proofs/CLAUDE.md`).
- `experiments/g4_quinto_test/cifar10_dataset.py` — pure-numpy + HF parquet fallback loader pattern. Tasks 2-3 mirror it.
- `experiments/g4_quinto_test/small_cnn.py` — `G4SmallCNN` substrate. Task 4 (CIFAR-100 head) reuses it modulo `n_classes` ; Task 5 (`G4MediumCNN`) extends the architecture.
- `experiments/g4_quinto_test/run_step3_cnn_recombine.py` — H5-C placebo driver (3 strategies × 4 arms × N seeds). Tasks 6+7 mirror it modulo strategy set (mog + none only) and substrate.
- `experiments/g4_quinto_test/aggregator.py` — verdict aggregation with deferred-block handling (`step3_path=None` → deferred). Pattern reused for the conjunction H6-C.
- `experiments/g4_quater_test/recombine_strategies.py` — `RecombineStrategy` Literal + `sample_synthetic_latents` dispatcher. **Reused as-is** (latent-dim agnostic).
- `experiments/g4_split_fmnist/dream_wrap.py:130` — `build_profile(name, seed)` — substrate-agnostic profile builder ; reused unchanged.
- `experiments/g4_ter_hp_sweep/hp_grid.py:17,41` — `HPCombo` + `representative_combo()` (C5 anchor). Reused unchanged.
- `kiki_oniric/eval/statistics.py` — `compute_hedges_g`. Welch via scipy direct.
- `harness/storage/run_registry.py` — `RunRegistry.register(c_version, profile, seed, commit_sha) -> run_id`.
- `docs/papers/paper2/results.md:460` — §7.1.7 G4-quinto subsection ; §7.1.8 lands directly after.

**Do NOT** modify `kiki_oniric/dream/episode.py`, `kiki_oniric/profiles/*`, or any axiom module. This pilot is empirical extension, not a framework change. FC stays at v0.12.0 across all outcomes.

---

## File structure

| Status | Path | Responsibility |
|--------|------|----------------|
| Create | `experiments/g4_sexto_test/__init__.py` | package marker |
| Create | `experiments/g4_sexto_test/cifar100_dataset.py` | pure-numpy CIFAR-100 binary loader + HF parquet fallback ; 10-task class-incremental split |
| Create | `experiments/g4_sexto_test/tiny_imagenet_dataset.py` | HF parquet/zip Tiny-ImageNet loader ; 10-task class-incremental split |
| Create | `experiments/g4_sexto_test/data/.gitkeep` | data dir for downloaded payloads (gitignored) |
| Create | `experiments/g4_sexto_test/medium_cnn.py` | `G4MediumCNN` — 3 Conv2d + 3 MaxPool2d + 2 Linear, 64×64 input |
| Create | `experiments/g4_sexto_test/run_step1_cifar100.py` | Step 1 driver — H6-A on `G4SmallCNN` (multi-class head, mog + none) |
| Create | `experiments/g4_sexto_test/run_step2_tiny_imagenet.py` | Step 2 driver — H6-B on `G4MediumCNN` (mog + none) |
| Create | `experiments/g4_sexto_test/aggregator.py` | load 2 step JSONs, emit H6-A / H6-B / H6-C (conjunction) verdict |
| Create | `tests/unit/test_g4_sexto_cifar100_loader.py` | unit tests for CIFAR-100 loader |
| Create | `tests/unit/test_g4_sexto_tiny_imagenet_loader.py` | unit tests for Tiny-ImageNet loader |
| Create | `tests/unit/test_g4_sexto_medium_cnn.py` | unit tests for `G4MediumCNN` |
| Create | `tests/unit/test_g4_sexto_aggregator.py` | unit tests for conjunction aggregator |
| Create | `docs/osf-prereg-g4-sexto-pilot.md` | OSF pre-reg G4-sexto, locked **before** Step 1 run |
| Create | `docs/milestones/g4-sexto-step1-2026-05-03.{json,md}` | Step 1 outputs (driver-emitted) |
| Create | `docs/milestones/g4-sexto-step2-2026-05-03.{json,md}` | Step 2 outputs (Option A only) |
| Create | `docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}` | aggregator outputs |
| Modify | `experiments/g4_quinto_test/small_cnn.py` | already supports `n_classes` ; **reused unchanged** for Step 1 (only docstring update OK if needed) |
| Modify | `docs/papers/paper2/results.md` | add §7.1.8 G4-sexto subsection (after §7.1.7) |
| Modify | `docs/papers/paper2-fr/results.md` | mirror §7.1.8 (FR translation, parallel maintenance) |
| Modify | `docs/proofs/dr4-profile-inclusion.md` | add §"Empirical-evidence amendment — G4-sexto" appended after the v0.4 G4-quinto block |
| Modify | `CHANGELOG.md` | new entry under [Unreleased] (no FC/EC bump) |
| Modify | `STATUS.md` | append G4-sexto line under "Critical risks watched" → none ; under header "As of" + Gates G4 row |
| Modify | `.gitignore` | add `experiments/g4_sexto_test/data/cifar-100-*` and `experiments/g4_sexto_test/data/tiny-imagenet-*` |

**Decomposition rationale**: each step is a self-contained driver with its own milestone artefact (append-only per `docs/CLAUDE.md`). The aggregator combines the two verdicts into the derived H6-C conjunction. CIFAR-100 (Task 6) can land before Tiny-IN code (Tasks 3, 5, 7) exists if Option B is chosen. Resilience pattern lifted from G4-quinto (commit-per-Step lesson learned).

---

## Architecture decisions

### Compute budget — pick **before** Task 1

| Option | Steps | Cells | Wall time M1 Max | Wall time Studio | Use when |
|--------|-------|-------|------------------|------------------|----------|
| **A — full** | Step 1 + Step 2 | 240 + 240 = **480** | ≈ 32–52 h (2 nights) | ≈ 6–10 h (1 night) | Studio available, full scientific verdict |
| **B — CIFAR-100 only** | Step 1 only | **240** | ≈ 12–20 h (1 night) | ≈ 2–4 h | M1 Max only, defer Step 2 to G4-septimo |
| **C — smoke** | 2 tasks × 2 seeds × 4 arms × 2 strategies (mog,none), Step 1 only | **32** | ≈ 30 min | ≈ 5 min | Pipeline validation only — never produces milestone artefacts under `docs/milestones/` |

**Recommend Option A if Mac Studio is available** ; otherwise **Default to Option B**. Option C is only for CI / smoke gates and must use `--smoke` flag (driver writes to `/tmp/...` and JSON header carries `"smoke": true`).

The **decision is recorded as a Task 0.5 commit** before Task 1 starts ; the pre-reg §3 N value reflects the chosen option (Option A: 30 seeds/arm × 2 strategies ; Option B: 30 seeds/arm × 2 strategies, Step 2 deferred).

### Datasets

**CIFAR-100** — binary version, canonical mirror `https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz` (~163 MB). Records: 1 coarse-label byte + 1 fine-label byte + 3072 image bytes (CHW: 1024 R + 1024 G + 1024 B). Two binary files: `train.bin` (50000 records) + `test.bin` (10000 records). HF parquet fallback: `uoft-cs/cifar100` (commit pinned at first download). SHA-256 on canonical tar pinned in loader header; HF SHAs pinned same way as G4-quinto pattern.

Class-incremental 10-task split (use **fine labels** 0..99) :

```
task 0 : fine classes {0..9}
task 1 : fine classes {10..19}
...
task 9 : fine classes {90..99}
```

Per task, labels remapped to `{0, 1, ..., 9}` (10-class head shared across tasks). Layout `(N, 32, 32, 3)` NHWC float32 in `[0, 1]`.

**Tiny-ImageNet** — HF dataset `zh-plus/tiny-imagenet` (commit pinned at first download), parquet shards. 200 classes × 500 train + 50 val images, 64×64 RGB JPEG. Split into 10 sequential 20-class binary-equivalent tasks ; per-task label remap to `{0, 1, ..., 19}`. Layout `(N, 64, 64, 3)` NHWC float32 in `[0, 1]`. Deviation envelope (§9) covers HF outage → abort + amendment.

### Substrates

**`G4SmallCNN` (Step 1)** — already in `experiments/g4_quinto_test/small_cnn.py`. Constructor accepts `n_classes` ; we set `n_classes=10` for CIFAR-100 per-task head. Op site mapping unchanged.

**`G4MediumCNN` (Step 2, new)** — taller MLX CNN for 64×64 input :

```
(N, 64, 64, 3)
  -> Conv2d(3, 16, 3, pad=1) ReLU MaxPool2d(2, 2)   -> (N, 32, 32, 16)
  -> Conv2d(16, 32, 3, pad=1) ReLU MaxPool2d(2, 2)  -> (N, 16, 16, 32)
  -> Conv2d(32, 64, 3, pad=1) ReLU MaxPool2d(2, 2)  -> (N, 8, 8, 64)
  -> Flatten -> Linear(4096, 128) ReLU -> Linear(128, n_classes)
```

Op site mapping (mirrors `G4SmallCNN`):
- **REPLAY** — full-model SGD on a batch from beta buffer (records may carry flat 12288 floats or NHWC `(64,64,3)`).
- **DOWNSCALE** — multiply every weight + bias of `{conv1, conv2, conv3, fc1, fc2}` by `factor`. Bound `(0, 1]`.
- **RESTRUCTURE** — perturb `conv2.weight` only (middle feature extractor — analogue choice, preserves input projection conv1 + classifier fc2).
- **RECOMBINE** — synthetic latents (dim 128 = `latent_dim`) per active strategy ∈ {mog, none}, single CE-loss SGD step on `fc2`.

### Strategy set: mog + none (no AE)

H6-C focuses on the placebo : RECOMBINE=mog vs RECOMBINE=none. AE strategy is dropped (it is a secondary observation in G4-quinto Step 3, not part of the H5-C / H6-C verdict). 2 strategies × 4 arms × N=30 seeds = **240 cells per step**.

### Statistics & multiplicity

- **H6-A** : Welch two-sided `retention(P_max with mog)` vs `retention(P_max with none)` on CIFAR-100, α = 0.05/3 = 0.0167 (Bonferroni for 3 hypotheses). **Failing to reject** confirms H6-A.
- **H6-B** : same on Tiny-ImageNet, α = 0.0167.
- **H6-C** : derived conjunction `H6-A_confirmed AND H6-B_confirmed`. No additional Welch test ; H6-C is a logical aggregation (no Bonferroni adjustment needed beyond the per-step α).

**Honest reading clause** (must be embedded verbatim in every milestone MD) : *"Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H6-A/H6-B specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the substrate at this scale."*

### Pre-reg discipline

OSF pre-reg G4-sexto (Task 1) is locked at the commit *that introduces it* — **before Task 6 (Step 1 driver run)**. The pre-reg cites G4-quinto verdict (H5-A False, H5-B False, H5-C **CONFIRMED**) as exploratory baseline ; H6-A, H6-B, H6-C are the confirmatory hypotheses. Deviation envelopes mirror G4-quinto §9 (network outages, low acc_initial, wall-time overrun, Option B deferral).

---

## Common commands

```bash
# Install (mlx + scipy + pillow + pyarrow already in pyproject)
uv sync --all-extras

# Smoke (any step)
uv run python experiments/g4_sexto_test/run_step1_cifar100.py --smoke
uv run python experiments/g4_sexto_test/run_step2_tiny_imagenet.py --smoke

# Full Option A (~6-10 h Studio, 32-52 h M1 Max)
uv run python experiments/g4_sexto_test/run_step1_cifar100.py --n-seeds 30
uv run python experiments/g4_sexto_test/run_step2_tiny_imagenet.py --n-seeds 30

# Aggregate
uv run python experiments/g4_sexto_test/aggregator.py

# Aggregate Option B (Step 2 deferred — pass /dev/null per G4-quinto pattern)
uv run python experiments/g4_sexto_test/aggregator.py --step2 /dev/null

# Lint + types + tests (must pass before each commit)
uv run ruff check experiments/g4_sexto_test tests/unit/test_g4_sexto_*.py
uv run mypy experiments/g4_sexto_test
uv run pytest tests/unit/test_g4_sexto_*.py -v
```

---

## Tasks

### Task 0: Investigate state + MLX Conv2d for taller CNN

**Files:**
- Read only: `experiments/g4_quinto_test/{small_cnn,run_step3_cnn_recombine,aggregator,cifar10_dataset}.py`
- Read only: `experiments/g4_quater_test/recombine_strategies.py`
- Read only: `kiki_oniric/dream/episode.py` (Operation enum + DreamEpisode signature)
- Read only: `harness/storage/run_registry.py`

- [ ] **Step 1: Confirm MLX Conv2d / MaxPool2d with stacked layers behaves on 64x64 input**

```bash
uv run python -c "
import mlx.core as mx, mlx.nn as nn
x = mx.random.normal((2, 64, 64, 3))
c1 = nn.Conv2d(3, 16, 3, padding=1); p1 = nn.MaxPool2d(2, 2)
c2 = nn.Conv2d(16, 32, 3, padding=1); p2 = nn.MaxPool2d(2, 2)
c3 = nn.Conv2d(32, 64, 3, padding=1); p3 = nn.MaxPool2d(2, 2)
y = p3(c3(p2(c2(p1(c1(x))))))
print('shape:', y.shape)  # expect (2, 8, 8, 64)
"
```
Expected: `shape: (2, 8, 8, 64)`. Layout is **NHWC**.

- [ ] **Step 2: Confirm reused signatures**

Skim and note (no code change) :
- `G4SmallCNN(latent_dim, n_classes, seed)` — already accepts arbitrary `n_classes` (verified in `__post_init__`, line 82 : `nn.Linear(self.latent_dim, self.n_classes)`).
- `RecombineStrategy = Literal["mog", "ae", "none"]` (in `experiments/g4_quater_test/recombine_strategies.py:29`) — latent-dim agnostic.
- `build_profile(name, seed)` — substrate-agnostic ; reused unchanged.
- `representative_combo() -> HPCombo` — C5 anchor.

No code changes in Task 0. No commit.

---

### Task 0.5: Record compute option (A / B / C)

**Files:**
- (No file change — decision logged in executor session log + reflected in pre-reg §3 N value)

This is a **decision step**, not code.

- [ ] **Step 1: Decide A / B / C**

Match against:
- Available compute (Studio overnight → Option A ; M1 Max only → Option B ; smoke gate → Option C).
- Whether the user accepts a 2-night run.
- Pre-reg §6 row matrix below.

- [ ] **Step 2: Record decision**

(No commit at this step — committed implicitly via the `--n-seeds` value embedded in the OSF pre-reg in Task 1, and the choice of which steps run.)

---

### Task 1: OSF pre-registration — G4-sexto pilot

**Files:**
- Create: `docs/osf-prereg-g4-sexto-pilot.md`

- [ ] **Step 1: Write the pre-registration**

Mirror the §1-§9 structure of `docs/osf-prereg-g4-quinto-pilot.md` verbatim. Replace pilot-specific content as follows :

- **Header**: pilot = G4-sexto ; sister pilot = G4-quinto (H5-A/B falsified, H5-C confirmed — Welch p=0.9918, g=-0.0026, mean P_max(mog)=0.9842, mean P_max(none)=0.9845, N=30) ; substrates = `G4SmallCNN` (Step 1, multi-class head n_classes=10) + `G4MediumCNN` (Step 2, n_classes=20) ; benchmarks = Split-CIFAR-100 (10 tasks × 10 classes) + Split-Tiny-ImageNet (10 tasks × 20 classes).
- **§1 Background**: cite G4-quinto aggregate (`docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`) verbatim. Summarise H5-C confirmed (RECOMBINE empty universalises across FMNIST + CIFAR-CNN). Cite G4-quinto pre-reg §6 row 6 "ImageNet escalation prerequisite for any STABLE promotion" as the trigger for this escalation. Note the Hu 2020 directional anchor caveat (not a magnitude calibrator).
- **§2 Hypotheses (confirmatory)** :
  - **H6-A (mid-large class count)** — on Split-CIFAR-100 with `G4SmallCNN` (n_classes=10 per-task head), `retention(P_max with mog)` is not statistically distinguishable from `retention(P_max with none)`. Welch two-sided fails to reject H0 at α = 0.05/3 = 0.0167. **Failing** to reject = positive empirical claim that the H5-C finding generalises to CIFAR-100 at 100-class scale.
  - **H6-B (mid-large resolution + class count)** — on Split-Tiny-ImageNet with `G4MediumCNN` (n_classes=20 per-task head, 64×64 input, 3-Conv layers), same Welch contract at α = 0.0167.
  - **H6-C (universality of RECOMBINE-empty across 4 benchmarks × 2 substrate families)** — derived conjunction `H6-A_confirmed AND H6-B_confirmed`. If both H6-A and H6-B confirm, the universality claim spans {Split-FMNIST, Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer MLP, small CNN, medium CNN}. If only one confirms, universality is **partial** (scope-bound).
- **§3 Power**: N = 30 seeds per arm at α = 0.0167 detects |g| ≥ 0.74 at 80% power (Welch two-sided, identical to G4-quinto). The lower N (30 vs G4-quater's 95) is dictated by the compute envelope — Option A targets 6-10 h Studio / 32-52 h M1 Max overnight × 2 nights.
- **§4 Exclusion**: identical to G4-quinto (`acc_initial < 0.5` for binary OR `acc_initial < 1/n_classes_head` for multi-class → exclude cell ; non-finite `acc_final` → exclude ; run_id collision → abort + amendment).
- **§5 Paths**: drivers = `experiments/g4_sexto_test/run_step{1,2}_*.py` ; substrates = `experiments.g4_quinto_test.small_cnn.G4SmallCNN` (reused) + `experiments.g4_sexto_test.medium_cnn.G4MediumCNN` (new) ; loaders = `experiments.g4_sexto_test.cifar100_dataset.load_split_cifar100_10tasks_auto` + `tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto` ; sources = `https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz` + HF `zh-plus/tiny-imagenet`.
- **§6 DualVer outcome matrix** (6 rows) :
  1. H6-A and H6-B both confirmed → EC PARTIAL ; H6-C confirmed (RECOMBINE-empty universalises across 4 benchmarks × 2 substrates) ; DR-4 partial refutation universalises further ; DR-4 evidence file revised to v0.5.
  2. H6-A confirmed only → EC PARTIAL ; H6-C partial (scope-bound to CIFAR-100, Tiny-IN open Q) ; DR-4 evidence v0.5 records partial extension.
  3. H6-B confirmed only → EC PARTIAL ; H6-C partial (scope-bound to Tiny-IN, CIFAR-100 anomaly) ; DR-4 evidence v0.5 records partial extension.
  4. Both falsified (Welch rejects H0 on at least one) → EC PARTIAL ; **H5-C universality breaks at mid-large scale** ; DR-4 partial refutation tagged as low-scale-bound ; DR-4 evidence v0.5 records the boundary.
  5. Option B chosen (Step 2 deferred) → EC PARTIAL ; aggregator reports `h6b_deferred=True`, no H6-B verdict in this pilot ; H6-C deferred ; document deferral in CHANGELOG.
  6. Option C smoke → no science verdict, milestone artefacts NOT committed (use `--smoke`).

  EC stays PARTIAL across **all** rows. FC stays at v0.12.0 across all rows.
- **§7 Reporting**: Welch fail-to-reject = "absence of evidence" — except H6-A and H6-B, where it **is** the predicted positive claim. Verbatim honest-reading clause from G4-quinto §7, embedded in every G4-sexto milestone MD.
- **§8 Audit trail**: profile keys `g4-sexto/{step1,step2}/<arm>/<combo>/<strategy>` ; milestones at `docs/milestones/g4-sexto-step{1,2}-2026-05-03.{json,md}` + aggregate.
- **§9 Deviations**: (a) CIFAR-100 / Tiny-IN download fails → abort + §9.1 amendment ; (b) acc_initial below threshold for majority of seeds → epochs 3 → 5 ; (c) Step 2 wall time > 16 h (overnight ceiling) → reduce N from 30 to 20 for Step 2 only, tag exploratory ; (d) Option B chosen → Step 2 deferred to G4-septimo follow-up ; (e) **Path A Studio recommended** (verbatim note in §6 banner) — M1-Max-only execution implicitly forces Option B unless multi-night session accepted.

- [ ] **Step 2: Lint + commit the pre-reg**

```bash
uv run ruff check docs/osf-prereg-g4-sexto-pilot.md 2>/dev/null  # markdown — ruff skips, OK
git add docs/osf-prereg-g4-sexto-pilot.md
git commit -m "docs(g4-sexto): lock OSF pre-reg pilot"
```

---

### Task 2: Pure-numpy CIFAR-100 loader + 10-task split

**Files:**
- Create: `experiments/g4_sexto_test/__init__.py` (empty)
- Create: `experiments/g4_sexto_test/cifar100_dataset.py`
- Create: `experiments/g4_sexto_test/data/.gitkeep` (empty)
- Modify: `.gitignore` — add `experiments/g4_sexto_test/data/cifar-100-*` lines
- Test: `tests/unit/test_g4_sexto_cifar100_loader.py`

CIFAR-100 binary format (per https://www.cs.toronto.edu/~kriz/cifar.html) : each record = 1 coarse-label byte + 1 fine-label byte + 3072 image bytes. Two binary files: `train.bin` (50000 records) + `test.bin` (10000 records). The loader uses **fine labels** (0..99) for the 10-task split.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_g4_sexto_cifar100_loader.py
"""Unit tests for G4-sexto CIFAR-100 loader (synthetic tmp_path fixture)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from experiments.g4_sexto_test.cifar100_dataset import (
    CIFAR100_RECORD_SIZE, decode_cifar100_bin, load_split_cifar100_10tasks,
)


def _write_batch(path: Path, fine_labels: list[int], rng: np.random.Generator) -> None:
    rows = []
    for fl in fine_labels:
        coarse = fl // 5  # arbitrary deterministic mapping for fixture
        img = rng.integers(0, 256, size=3072, dtype=np.uint8).tobytes()
        rows.append(bytes([coarse, fl]) + img)
    path.write_bytes(b"".join(rows))


def test_decode_cifar100_bin_shape(tmp_path: Path) -> None:
    f = tmp_path / "train.bin"
    _write_batch(f, [0, 9, 10, 99], np.random.default_rng(0))
    images, fine = decode_cifar100_bin(f)
    assert images.shape == (4, 32, 32, 3) and images.dtype == np.uint8
    assert fine.tolist() == [0, 9, 10, 99]


def test_decode_cifar100_bin_truncated_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * (CIFAR100_RECORD_SIZE - 1))
    with pytest.raises(ValueError, match="truncated"):
        decode_cifar100_bin(f)


def test_load_split_cifar100_10tasks_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bin_dir = tmp_path / "cifar-100-binary"
    bin_dir.mkdir()
    # 2 examples per fine class so each task has data
    _write_batch(bin_dir / "train.bin",
                 [c for c in range(100) for _ in range(2)], rng)
    _write_batch(bin_dir / "test.bin", list(range(100)), rng)
    tasks = load_split_cifar100_10tasks(bin_dir)
    assert len(tasks) == 10
    for k, task in enumerate(tasks):
        assert task["x_train_nhwc"].shape[1:] == (32, 32, 3)
        assert task["x_train"].shape[1] == 3072
        assert set(task["y_train"].tolist()) <= set(range(10))
        # remap : fine labels {10k..10k+9} -> {0..9}
        assert task["x_train_nhwc"].shape[0] == 20
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/unit/test_g4_sexto_cifar100_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g4_sexto_test.cifar100_dataset'`.

- [ ] **Step 3: Implement the loader**

**Strategy** : copy `experiments/g4_quinto_test/cifar10_dataset.py` whole-file and apply the surgical changes below. The file structure (canonical-then-HF-fallback, `_http_get`, `_verify_sha256`, `download_if_missing`, `download_if_missing_hf`, `_decode_parquet_shard`, `_build_tasks_from_arrays`, `load_split_cifar10_5tasks_auto`) is **kept verbatim** ; only the constants, the record decode, and the task split loop change.

Constants (replace at top of file) :

```python
CIFAR100_LABEL_BYTES: Final[int] = 2  # coarse + fine
CIFAR100_IMAGE_BYTES: Final[int] = 32 * 32 * 3
CIFAR100_RECORD_SIZE: Final[int] = CIFAR100_LABEL_BYTES + CIFAR100_IMAGE_BYTES  # 3074
CIFAR100_URL: Final[str] = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
CIFAR100_TAR_SHA256: Final[str] = "...replace_in_task9..."
CIFAR100_HF_TRAIN_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/"
    "cifar100/train-00000-of-00001.parquet"
)
CIFAR100_HF_TEST_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/"
    "cifar100/test-00000-of-00001.parquet"
)
CIFAR100_HF_TRAIN_SHA256: Final[str] = "...replace_at_first_download..."
CIFAR100_HF_TEST_SHA256:  Final[str] = "...replace_at_first_download..."
```

`decode_cifar100_bin` reads byte 0 as coarse, byte 1 as fine label, bytes 2..3074 as the image ; returns `(nhwc_uint8, fine_labels_uint8)` (coarse dropped).

`_decode_parquet_shard` reads the HF schema `{"img": {"bytes": ...}, "fine_label": int, "coarse_label": int}` — read `fine_label` only ; column name `"img"` matches the cifar10 path verbatim.

`_build_tasks_from_arrays` becomes a 10-iteration loop with range filter (no class-pair logic). Per task k :

```python
for k in range(10):
    lo, hi = k * 10, (k + 1) * 10
    tr = (y_train_raw >= lo) & (y_train_raw < hi)
    te = (y_test_raw  >= lo) & (y_test_raw  < hi)
    y_tr = (y_train_raw[tr].astype(np.int64) - lo)  # remap to {0..9}
    y_te = (y_test_raw[te].astype(np.int64)  - lo)
    tasks.append(SplitCIFAR100Task(
        x_train=x_tr_flat[tr], x_train_nhwc=x_tr_nhwc[tr], y_train=y_tr,
        x_test =x_te_flat[te], x_test_nhwc =x_te_nhwc[te], y_test =y_te,
    ))
```

The `SplitCIFAR100Task` TypedDict has the same 6 fields as `SplitCIFAR10Task`. The public entry point is `load_split_cifar100_10tasks_auto(data_dir)` (rename of `load_split_cifar10_5tasks_auto`). The canonical extracted directory is `cifar-100-binary/` (single dir with `train.bin` + `test.bin`, vs cifar10's 6-batch split).

- [ ] **Step 4: Verify SHA-256 once at first download (manual)**

After Task 6 (or earlier) actually downloads the file :

```bash
sha256sum experiments/g4_sexto_test/data/cifar-100-binary.tar.gz
# replace the placeholder, commit:
# git commit -m "chore(g4-sexto): pin CIFAR-100 SHA-256"
```

Same procedure for HF parquet shards if the canonical mirror is unavailable.

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/unit/test_g4_sexto_cifar100_loader.py -v
uv run mypy experiments/g4_sexto_test/cifar100_dataset.py
git add experiments/g4_sexto_test/__init__.py experiments/g4_sexto_test/cifar100_dataset.py experiments/g4_sexto_test/data/.gitkeep tests/unit/test_g4_sexto_cifar100_loader.py .gitignore
git commit -m "feat(g4-sexto): CIFAR-100 loader 10-task"
```

---

### Task 3: Tiny-ImageNet loader + 10-task split

**Files:**
- Create: `experiments/g4_sexto_test/tiny_imagenet_dataset.py`
- Modify: `.gitignore` — add `experiments/g4_sexto_test/data/tiny-imagenet-*` lines (already covered by `data/` dir wildcard if Task 2 added one ; otherwise add per-pattern).
- Test: `tests/unit/test_g4_sexto_tiny_imagenet_loader.py`

**Format note** : Tiny-ImageNet is **JPEG inside parquet shards** on the HF mirror (`zh-plus/tiny-imagenet`). The canonical mirror (https://image-net.org/data/tiny-imagenet-200.zip) is a ZIP of JPEGs ; this loader **prefers the HF parquet path** (consistent with the G4-quinto §9.1 amendment pattern) and treats the ZIP path as a fallback.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_g4_sexto_tiny_imagenet_loader.py
"""Unit tests for G4-sexto Tiny-ImageNet loader (synthetic parquet fixture)."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.g4_sexto_test.tiny_imagenet_dataset import (
    decode_tiny_imagenet_parquet, load_split_tiny_imagenet_10tasks_from_parquet,
)


def _make_jpeg_bytes(rng: np.random.Generator) -> bytes:
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _write_parquet(path: Path, labels: list[int], rng: np.random.Generator) -> None:
    rows_img = [{"bytes": _make_jpeg_bytes(rng), "path": f"img_{i}.jpeg"}
                for i in range(len(labels))]
    table = pa.table({"image": rows_img, "label": labels})
    pq.write_table(table, path)


def test_decode_parquet_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = tmp_path / "shard.parquet"
    _write_parquet(p, [0, 1, 2, 199], rng)
    images, labels = decode_tiny_imagenet_parquet(p)
    assert images.shape == (4, 64, 64, 3) and images.dtype == np.uint8
    assert labels.tolist() == [0, 1, 2, 199]


def test_split_10tasks_remap(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = tmp_path / "train.parquet"
    val = tmp_path / "valid.parquet"
    # 2 examples per class so every task has data
    _write_parquet(train, [c for c in range(200) for _ in range(2)], rng)
    _write_parquet(val, list(range(200)), rng)
    tasks = load_split_tiny_imagenet_10tasks_from_parquet(train, val)
    assert len(tasks) == 10
    for k, task in enumerate(tasks):
        assert task["x_train_nhwc"].shape[1:] == (64, 64, 3)
        assert set(task["y_train"].tolist()) <= set(range(20))
        assert task["x_train_nhwc"].shape[0] == 40  # 20 classes × 2 examples
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/unit/test_g4_sexto_tiny_imagenet_loader.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the loader**

**Strategy** : copy the HF parquet path of `experiments/g4_quinto_test/cifar10_dataset.py` (lines 134-274 — `_http_get`, `_verify_sha256`, `download_if_missing_hf`, `_decode_parquet_shard`, `load_split_cifar10_5tasks_hf`, `load_split_cifar10_5tasks_auto`) and apply the surgical changes below. There is **no canonical-binary path** for Tiny-IN ; HF parquet is the default acquisition (canonical ImageNet ZIP is fallback only — out of scope for this loader).

Constants (top of file) :

```python
TINY_IMAGENET_HF_TRAIN_URL: Final[str] = (
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/"
    "resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet"
)
TINY_IMAGENET_HF_VALID_URL: Final[str] = (
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/"
    "resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet"
)
TINY_IMAGENET_HF_TRAIN_SHA256: Final[str] = "...replace_at_first_download..."
TINY_IMAGENET_HF_VALID_SHA256: Final[str] = "...replace_at_first_download..."
HTTP_USER_AGENT: Final[str] = "g4-sexto-pilot/1 (mlx-on-m1max)"
```

`decode_tiny_imagenet_parquet(path)` mirrors cifar10's `_decode_parquet_shard` modulo two changes : (a) `(64, 64, 3)` output shape, (b) image column name `"image"` (Tiny-IN HF schema) with fallback to `"img"` for compatibility, label column `"label"` (fallback `"fine_label"`). Shape is asserted to be 64×64 ; if a row deviates, defensively `pil_img.convert("RGB").resize((64, 64))`.

`_build_tasks_from_arrays(x_tr, y_tr, x_te, y_te)` is identical to the CIFAR-100 version (Task 2) modulo the constants `lo, hi = k*20, (k+1)*20` and the `SplitTinyImageNetTask` TypedDict (same 6 fields).

`download_if_missing_hf(data_dir) -> (train_path, valid_path)` copies the body of cifar10's `download_if_missing_hf` (lines 184-213) — only constants change (URLs + SHA-256s, output filenames `tiny-imagenet-{train,valid}.parquet`, timeout 600 s for the 250 MB train shard).

The public entry point is `load_split_tiny_imagenet_10tasks_auto(data_dir)` :

```python
def load_split_tiny_imagenet_10tasks_auto(data_dir: Path) -> list[SplitTinyImageNetTask]:
    """Locate or download HF parquet shards then build the 10-task split."""
    train_path, valid_path = download_if_missing_hf(data_dir)
    return load_split_tiny_imagenet_10tasks_from_parquet(train_path, valid_path)
```

with `load_split_tiny_imagenet_10tasks_from_parquet(train_parquet, valid_parquet)` decoding both shards via `decode_tiny_imagenet_parquet` then calling `_build_tasks_from_arrays`. Both paths raise `FileNotFoundError` if the shard is absent (same contract as cifar10).

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/unit/test_g4_sexto_tiny_imagenet_loader.py -v
uv run mypy experiments/g4_sexto_test/tiny_imagenet_dataset.py
git add experiments/g4_sexto_test/tiny_imagenet_dataset.py tests/unit/test_g4_sexto_tiny_imagenet_loader.py .gitignore
git commit -m "feat(g4-sexto): Tiny-ImageNet loader 10-task"
```

---

### Task 4: G4SmallCNN reuse for CIFAR-100 (n_classes=10)

**Files:**
- Read only: `experiments/g4_quinto_test/small_cnn.py`

**No new substrate file.** `G4SmallCNN(latent_dim=64, n_classes=10, seed=...)` already supports an arbitrary `n_classes` via line 82 (`nn.Linear(self.latent_dim, self.n_classes)`). All op-site methods (`restructure_step`, `downscale_step`, `replay_optimizer_step`, `recombine_step`) are class-count-agnostic.

- [ ] **Step 1: Confirm n_classes=10 path is sound**

```bash
uv run python -c "
import numpy as np
from experiments.g4_quinto_test.small_cnn import G4SmallCNN
m = G4SmallCNN(latent_dim=64, n_classes=10, seed=0)
x = np.random.default_rng(0).standard_normal((4, 32, 32, 3)).astype(np.float32)
print('logits shape:', m.predict_logits(x).shape)  # (4, 10)
print('latent shape:', m.latent(x).shape)          # (4, 64)
"
```
Expected output : `logits shape: (4, 10)` then `latent shape: (4, 64)`. No commit (read-only validation).

---

### Task 5: G4MediumCNN — taller CNN for Tiny-ImageNet 64×64

**Files:**
- Create: `experiments/g4_sexto_test/medium_cnn.py`
- Test: `tests/unit/test_g4_sexto_medium_cnn.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_g4_sexto_medium_cnn.py
"""Unit tests for G4MediumCNN (3 Conv2d + 3 MaxPool2d + 2 Linear, 64×64)."""
from __future__ import annotations
import numpy as np
import pytest
from experiments.g4_sexto_test.medium_cnn import G4MediumCNN


def _clf(n_classes: int = 20) -> G4MediumCNN:
    return G4MediumCNN(latent_dim=128, n_classes=n_classes, seed=0)


def test_predict_logits_shape() -> None:
    x = np.random.default_rng(0).standard_normal((4, 64, 64, 3)).astype(np.float32)
    assert _clf().predict_logits(x).shape == (4, 20)


def test_latent_shape() -> None:
    x = np.random.default_rng(0).standard_normal((4, 64, 64, 3)).astype(np.float32)
    assert _clf().latent(x).shape == (4, 128)


def test_restructure_step_zero_factor_is_noop() -> None:
    clf = _clf()
    w = np.asarray(clf._conv2.weight).copy()
    clf.restructure_step(factor=0.0, seed=0)
    np.testing.assert_array_equal(np.asarray(clf._conv2.weight), w)


def test_restructure_step_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _clf().restructure_step(factor=-0.1, seed=0)


def test_downscale_bounds() -> None:
    clf = _clf()
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        clf.downscale_step(factor=1.1)


def test_replay_optimizer_step_flat_and_nhwc() -> None:
    """Buffer records may carry flat 12288-d or nested NHWC ; both go."""
    clf = _clf()
    rng = np.random.default_rng(0)
    flat = [{"x": rng.standard_normal(12288).astype(np.float32).tolist(),
             "y": int(rng.integers(0, 20))} for _ in range(4)]
    clf.replay_optimizer_step(flat, lr=0.001, n_steps=1)


def test_recombine_step_empty_is_noop() -> None:
    _clf().recombine_step(latents=[], n_synthetic=8, lr=0.01, seed=0)


def test_recombine_step_single_class_is_noop() -> None:
    clf = _clf()
    clf.recombine_step(
        latents=[([0.0] * 128, 5)] * 4, n_synthetic=8, lr=0.01, seed=0,
    )  # all label 5 -> S1-trivial -> no-op
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/unit/test_g4_sexto_medium_cnn.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.g4_sexto_test.medium_cnn'`.

- [ ] **Step 3: Implement the substrate**

`medium_cnn.py` is structurally identical to `experiments/g4_quinto_test/small_cnn.py` — copy it and apply the following surgical changes :

1. **Header docstring** updated to reference 64×64 RGB input + 3-Conv layout.
2. **Class name** : `G4SmallCNN` → `G4MediumCNN`.
3. **Architecture** (`__post_init__`) — add a third Conv layer + Pool, change `_fc1` input :

```python
def __post_init__(self) -> None:
    mx.random.seed(self.seed)
    np.random.seed(self.seed)
    self._conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self._conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self._conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self._pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self._fc1 = nn.Linear(8 * 8 * 64, self.latent_dim)  # = 4096
    self._fc2 = nn.Linear(self.latent_dim, self.n_classes)
    self._model = nn.Sequential(
        self._conv1, nn.ReLU(), self._pool1,
        self._conv2, nn.ReLU(), self._pool2,
        self._conv3, nn.ReLU(), self._pool3,
        _Flatten(), self._fc1, nn.ReLU(), self._fc2,
    )
    mx.eval(self._model.parameters())
```

4. **`latent()`** updated to traverse all 3 Conv blocks then `_fc1` ReLU :

```python
def latent(self, x: np.ndarray) -> np.ndarray:
    """Post-relu fc1 activations shape (N, latent_dim) — RECOMBINE site."""
    h = self._pool1(nn.relu(self._conv1(mx.array(x))))
    h = self._pool2(nn.relu(self._conv2(h)))
    h = self._pool3(nn.relu(self._conv3(h)))
    h = mx.reshape(h, (h.shape[0], -1))
    h = nn.relu(self._fc1(h))
    mx.eval(h)
    return np.asarray(h)
```

5. **`downscale_step()`** — extend the layer tuple to include `_conv3` :

```python
for layer in (self._conv1, self._conv2, self._conv3, self._fc1, self._fc2):
    layer.weight = layer.weight * factor
    if getattr(layer, "bias", None) is not None:
        layer.bias = layer.bias * factor
```

6. **`replay_optimizer_step()`** — flat-input reshape becomes `(-1, 64, 64, 3)` :

```python
if x_np.ndim == 2:
    # Flat 12288-d records reshape into NHWC for medium-CNN consumption.
    x_np = x_np.reshape(-1, 64, 64, 3)
```

7. **`restructure_step()`** unchanged (still perturbs `_conv2.weight` only — middle feature extractor analogue).

8. **`recombine_step()`** unchanged (operates on `_fc2` only via synthetic latents of dim `latent_dim`).

9. **`_Flatten`** — copy verbatim from `small_cnn.py:43-51`.

The dataclass `field(init=False)` declarations gain `_conv3` and `_pool3` — pattern is otherwise identical.

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/unit/test_g4_sexto_medium_cnn.py -v
uv run mypy experiments/g4_sexto_test/medium_cnn.py
git add experiments/g4_sexto_test/medium_cnn.py tests/unit/test_g4_sexto_medium_cnn.py
git commit -m "feat(g4-sexto): MediumCNN substrate"
```

---

### Task 6: Step 1 driver — H6-A on CIFAR-100 (G4SmallCNN multi-class head)

**Files:**
- Create: `experiments/g4_sexto_test/run_step1_cifar100.py`

This driver mirrors `experiments/g4_quinto_test/run_step3_cnn_recombine.py` (the H5-C placebo design) **modulo** : (a) CIFAR-100 loader, (b) `n_classes=10` per-task head, (c) strategies = `("mog", "none")` only (no AE), (d) outputs paths under `g4-sexto-step1-2026-05-03.{json,md}`.

- [ ] **Step 1: Build the driver**

The driver is a near-verbatim port of `run_step3_cnn_recombine.py` (lines 1-726). Differences below ; everything else (beta buffer FIFO with NHWC + latent records, dream-episode wrapper, registry registration, MD/JSON emission) is **copied unchanged** :

1. **Imports** — replace cifar10 imports with cifar100, keep small_cnn :

```python
from experiments.g4_sexto_test.cifar100_dataset import (
    SplitCIFAR100Task, load_split_cifar100_10tasks_auto,
)
from experiments.g4_quinto_test.small_cnn import G4SmallCNN
```

2. **Constants** :

```python
C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
STRATEGIES: tuple[RecombineStrategy, ...] = ("mog", "none")  # H6-A : no AE
DEFAULT_N_SEEDS = 30
N_CLASSES_PER_TASK = 10
LATENT_DIM = 64
DEFAULT_DATA_DIR = (
    REPO_ROOT / "experiments" / "g4_sexto_test" / "data"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step1-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step1-2026-05-03.md"
)
SMOKE_OUT_JSON = Path("/tmp") / "g4-sexto-step1-smoke.json"
SMOKE_OUT_MD = Path("/tmp") / "g4-sexto-step1-smoke.md"
```

3. **Substrate construction** (`_run_cell` body) — pass `n_classes=N_CLASSES_PER_TASK` :

```python
cnn = G4SmallCNN(latent_dim=LATENT_DIM, n_classes=N_CLASSES_PER_TASK, seed=seed)
```

4. **Exclusion criterion** — `acc_initial` is multi-class accuracy ; chance = 1/10 = 0.10. Update :

```python
RANDOM_CHANCE = 1.0 / N_CLASSES_PER_TASK  # 0.10 for CIFAR-100 head
# in _run_cell :
excluded = bool(acc_initial < 2.0 * RANDOM_CHANCE)  # i.e. < 0.20 = 2× chance
```

The `2× chance` threshold is a defensive widening over the binary `< 0.5` ; it preserves the §4 exclusion semantics ("seed failed to learn task 1") while adapting to the multi-class regime. Document this in the milestone MD header (deviation acknowledgement).

5. **Verdict function** — copy `_h5c_verdict` body verbatim from `run_step3_cnn_recombine.py:397-430`, rename function to `_h6a_verdict`, rename the output key `h5c_recombine_empty_confirmed` → `h6a_recombine_empty_confirmed`. The Welch-two-sided + Hedges-g logic is unchanged ; only the docstring references "CIFAR-100 at 100-class scale" instead of "CNN substrate at CIFAR-10 scale".

6. **MD report** — update title (G4-sexto Step 1 — H6-A …), hypothesis text, and embed the verbatim honest-reading clause adapted to H6-A + CIFAR-100. AE-observation block is dropped.

7. **`run_pilot`** payload — verdict block key is `h6a_recombine_strategy` (NOT `h5c_…`). No AE secondary block.

8. **`main`** — strategies default `("mog", "none")` ; smoke uses `("mog",)`.

The driver shape mirrors `run_step3_cnn_recombine.py:589-725` modulo substitutions above. The dream-episode wrapper `_dream_episode_strategy` is **copied verbatim** from `run_step3_cnn_recombine.py:200-299` (substrate-agnostic w.r.t. `n_classes`). The `_BetaBufferCNN`, `_strategy_aware_recombine`, `_resolve_commit_sha`, `_run_cell` (modulo the substrate construction line above), `_retention_by_arm_strategy` are also **copied verbatim**.

- [ ] **Step 2: Smoke test**

```bash
uv run python experiments/g4_sexto_test/run_step1_cifar100.py --smoke
ls /tmp/g4-sexto-step1-smoke.{json,md}
```
Expected: 2 files written ; smoke flag visible in JSON header.

- [ ] **Step 3: Lint + types + commit**

```bash
uv run ruff check experiments/g4_sexto_test/run_step1_cifar100.py
uv run mypy experiments/g4_sexto_test/run_step1_cifar100.py
git add experiments/g4_sexto_test/run_step1_cifar100.py
git commit -m "feat(g4-sexto): step1 CIFAR-100 driver"
```

---

### Task 7: Step 2 driver — H6-B on Tiny-ImageNet (G4MediumCNN)

**Files:**
- Create: `experiments/g4_sexto_test/run_step2_tiny_imagenet.py`

The driver mirrors Task 6 with three substitutions : (a) substrate = `G4MediumCNN`, (b) loader = `tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto`, (c) `n_classes=20` per task, (d) flat input dim 12288 (vs 3072 for CIFAR-100).

- [ ] **Step 1: Build the driver**

Copy `experiments/g4_sexto_test/run_step1_cifar100.py` (just landed in Task 6) and apply :

```python
from experiments.g4_sexto_test.tiny_imagenet_dataset import (
    SplitTinyImageNetTask, load_split_tiny_imagenet_10tasks_auto,
)
from experiments.g4_sexto_test.medium_cnn import G4MediumCNN

C_VERSION = "C-v0.12.0+PARTIAL"
N_CLASSES_PER_TASK = 20
LATENT_DIM = 128
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step2-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step2-2026-05-03.md"
)
SMOKE_OUT_JSON = Path("/tmp") / "g4-sexto-step2-smoke.json"
SMOKE_OUT_MD = Path("/tmp") / "g4-sexto-step2-smoke.md"
RANDOM_CHANCE = 1.0 / N_CLASSES_PER_TASK  # 0.05 for Tiny-IN 20-class head

# in _run_cell :
cnn = G4MediumCNN(latent_dim=LATENT_DIM, n_classes=N_CLASSES_PER_TASK, seed=seed)
```

The `excluded_underperforming_baseline` test stays at `2 * RANDOM_CHANCE` (= 0.10 for Tiny-IN). The verdict function is renamed `_h6b_verdict` and emits the key `h6b_recombine_empty_confirmed`. The MD report substitutes H6-A→H6-B, CIFAR-100→Tiny-ImageNet throughout.

The dream-episode wrapper `_dream_episode_strategy` is again **copied verbatim** ; the only place that touches the substrate dimensionality is the beta-buffer record `x` field which already auto-dispatches between flat and NHWC in `replay_optimizer_step` (the `medium_cnn.py` reshape rule we added in Task 5).

- [ ] **Step 2: Smoke test**

```bash
uv run python experiments/g4_sexto_test/run_step2_tiny_imagenet.py --smoke
```
Expected: 2 files in `/tmp/g4-sexto-step2-smoke.*` ; first download triggers HF parquet fetch (~250 MB) — log time.

- [ ] **Step 3: Pin SHAs, commit**

After the first download succeeds, replace `TINY_IMAGENET_HF_TRAIN_SHA256` and `TINY_IMAGENET_HF_VALID_SHA256` with the real SHA-256s :

```bash
sha256sum experiments/g4_sexto_test/data/tiny-imagenet-train.parquet \
          experiments/g4_sexto_test/data/tiny-imagenet-valid.parquet
# edit the constants in tiny_imagenet_dataset.py
git add experiments/g4_sexto_test/{run_step2_tiny_imagenet.py,tiny_imagenet_dataset.py}
git commit -m "feat(g4-sexto): step2 Tiny-IN driver + SHA pin"
```

---

### Task 8: Aggregator — H6-A / H6-B / H6-C conjunction verdict

**Files:**
- Create: `experiments/g4_sexto_test/aggregator.py`
- Test: `tests/unit/test_g4_sexto_aggregator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_g4_sexto_aggregator.py
"""Unit tests for G4-sexto aggregator (H6-A / H6-B / H6-C conjunction)."""
from __future__ import annotations
import json
from pathlib import Path
from experiments.g4_sexto_test.aggregator import aggregate_g4_sexto_verdict


def _step1_payload(*, fail_to_reject: bool) -> dict:
    return {
        "verdict": {"h6a_recombine_strategy": {
            "n_p_max_mog": 30, "n_p_max_none": 30,
            "mean_p_max_mog": 0.50, "mean_p_max_none": 0.50,
            "welch_t": 0.0, "welch_p_two_sided": 0.99 if fail_to_reject else 0.001,
            "alpha_per_test": 0.0167,
            "fail_to_reject_h0": fail_to_reject,
            "h6a_recombine_empty_confirmed": fail_to_reject,
            "hedges_g_mog_vs_none": 0.0,
        }},
    }


def _step2_payload(*, fail_to_reject: bool) -> dict:
    return {
        "verdict": {"h6b_recombine_strategy": {
            "n_p_max_mog": 30, "n_p_max_none": 30,
            "mean_p_max_mog": 0.30, "mean_p_max_none": 0.30,
            "welch_t": 0.0, "welch_p_two_sided": 0.99 if fail_to_reject else 0.001,
            "alpha_per_test": 0.0167,
            "fail_to_reject_h0": fail_to_reject,
            "h6b_recombine_empty_confirmed": fail_to_reject,
            "hedges_g_mog_vs_none": 0.0,
        }},
    }


def _write(p: Path, payload: dict) -> None:
    p.write_text(json.dumps(payload))


def test_both_confirmed_yields_h6c(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"; _write(s1, _step1_payload(fail_to_reject=True))
    s2 = tmp_path / "step2.json"; _write(s2, _step2_payload(fail_to_reject=True))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is True
    assert v["summary"]["h6b_confirmed"] is True
    assert v["summary"]["h6c_confirmed"] is True


def test_only_h6a_yields_partial(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"; _write(s1, _step1_payload(fail_to_reject=True))
    s2 = tmp_path / "step2.json"; _write(s2, _step2_payload(fail_to_reject=False))
    v = aggregate_g4_sexto_verdict(s1, s2)
    assert v["summary"]["h6a_confirmed"] is True
    assert v["summary"]["h6b_confirmed"] is False
    assert v["summary"]["h6c_confirmed"] is False
    assert v["summary"]["h6c_partial"] is True


def test_step2_deferred_yields_deferred(tmp_path: Path) -> None:
    s1 = tmp_path / "step1.json"; _write(s1, _step1_payload(fail_to_reject=True))
    v = aggregate_g4_sexto_verdict(s1, None)
    assert v["summary"]["h6b_deferred"] is True
    assert v["summary"]["h6c_confirmed"] is False
```

- [ ] **Step 2: Run to confirm fail**

```bash
uv run pytest tests/unit/test_g4_sexto_aggregator.py -v
```
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement the aggregator**

**Strategy** : copy `experiments/g4_quinto_test/aggregator.py` whole-file ; the deferred-block handling is the load-bearing pattern. Surgical changes :

- Two step files (step1, step2) instead of three.
- Drop `h5a_*` and `h5b_*` Jonckheere blocks (no Jonckheere here — H6-A and H6-B are both Welch placebo tests).
- Conjunction logic : `h6c_confirmed = h6a_confirmed AND h6b_confirmed`, `h6c_partial = (h6a_confirmed XOR h6b_confirmed) AND NOT h6b_deferred`.
- Default paths target `g4-sexto-step{1,2}-2026-05-03.json` + aggregate `g4-sexto-aggregate-2026-05-03.{json,md}`.

The signature is `aggregate_g4_sexto_verdict(step1_path: Path, step2_path: Path | None) -> dict[str, Any]`. `step2_path is None` triggers the deferred branch (Option B). The `confirmed` flag for each step reads `h6{a,b}_recombine_empty_confirmed` (matching the driver's verdict-block key from Tasks 6+7).

Output dict shape :

```python
{
    "h6a_cifar100": {**h6a, "confirmed": h6a_confirmed},
    "h6b_tiny_imagenet": {**h6b, "deferred": False, "confirmed": h6b_confirmed}
                          # OR {"deferred": True, "confirmed": False} if Option B,
    "h6c_universality": {
        "confirmed": h6c_confirmed,
        "partial": h6c_partial,
        "deferred": h6b_deferred,
    },
    "summary": {
        "h6a_confirmed": ..., "h6b_confirmed": ..., "h6c_confirmed": ...,
        "h6c_partial": ..., "h6b_deferred": ...,
        "h5c_to_h6c_universality_extension": h6c_confirmed,
    },
}
```

The MD renderer follows `experiments/g4_quinto_test/aggregator.py:119-266` adapted to two steps. Title `# G4-sexto aggregate verdict` ; the Summary block has 5 bullets (h6a_confirmed, h6b_confirmed, h6c_confirmed, h6c_partial, h5c_to_h6c_extension). Per-step blocks copy the H5-C numerical-table template (mean P_max(mog), mean P_max(none), Hedges' g, Welch t, Welch p, alpha, fail_to_reject_h0). Each block embeds the verbatim honest-reading clause adapted to the step. The DEFERRED branch for H6-B is copied verbatim from `experiments/g4_quinto_test/aggregator.py:197-201`. The Verdict-DR-4-evidence trailing paragraph is rewritten to reference 4 benchmarks × 4 substrates (vs 2×2 in G4-quinto).

`main()` accepts `--step1`, `--step2`, `--out-json`, `--out-md`. Treat `/dev/null` or any non-existing `--step2` path as Option B (deferred) — copy the convention verbatim from `experiments/g4_quinto_test/aggregator.py:284-287`.

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/unit/test_g4_sexto_aggregator.py -v
uv run mypy experiments/g4_sexto_test/aggregator.py
git add experiments/g4_sexto_test/aggregator.py tests/unit/test_g4_sexto_aggregator.py
git commit -m "feat(g4-sexto): aggregator H6-A/B/C"
```

---

### Task 9: Run pilot (overnight) + emit milestones

**Files:**
- Create: `docs/milestones/g4-sexto-step1-2026-05-03.{json,md}`
- Create: `docs/milestones/g4-sexto-step2-2026-05-03.{json,md}` (Option A only)
- Create: `docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}`

This task is the actual scientific run. Each sub-step **commits** at the end so a partial run is recoverable (lesson learned : G4-quinto Step 2 wall-time scare).

- [ ] **Step 1: Pre-flight smoke**

```bash
uv run python experiments/g4_sexto_test/run_step1_cifar100.py --smoke
uv run python experiments/g4_sexto_test/run_step2_tiny_imagenet.py --smoke  # Option A only
ls /tmp/g4-sexto-step{1,2}-smoke.{json,md} 2>/dev/null
```
Expected: 4 files (Option A) or 2 files (Option B). Verifies HF downloads + smoke pipeline.

- [ ] **Step 2: Run Step 1 (CIFAR-100, ~12-20 h M1 Max / ~2-4 h Studio)**

```bash
uv run python experiments/g4_sexto_test/run_step1_cifar100.py --n-seeds 30
ls docs/milestones/g4-sexto-step1-2026-05-03.{json,md}
git add docs/milestones/g4-sexto-step1-2026-05-03.{json,md}
git commit -m "data(g4-sexto): step1 CIFAR-100 milestone"
```
Expected: 240 cells in JSON ; H6-A verdict block populated.

- [ ] **Step 3: Run Step 2 (Tiny-ImageNet, Option A only)**

```bash
uv run python experiments/g4_sexto_test/run_step2_tiny_imagenet.py --n-seeds 30
ls docs/milestones/g4-sexto-step2-2026-05-03.{json,md}
git add docs/milestones/g4-sexto-step2-2026-05-03.{json,md}
git commit -m "data(g4-sexto): step2 Tiny-IN milestone"
```
**Skip if Option B was chosen at Task 0.5** ; jump to Step 4 with `--step2 /dev/null`.

- [ ] **Step 4: Run aggregator**

```bash
# Option A (both step files exist) :
uv run python experiments/g4_sexto_test/aggregator.py
# Option B (step 2 deferred) :
uv run python experiments/g4_sexto_test/aggregator.py --step2 /dev/null

ls docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}
git add docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}
git commit -m "data(g4-sexto): aggregate H6 verdict"
```

---

### Task 10: Update Paper 2 §7.1.8 (EN + FR)

**Files:**
- Modify: `docs/papers/paper2/results.md` — add §7.1.8 after §7.1.7
- Modify: `docs/papers/paper2-fr/results.md` — mirror §7.1.8 (FR translation)

The §7.1.8 subsection mirrors the §7.1.7 G4-quinto block (`docs/papers/paper2/results.md:460-`) modulo the substantive content (CIFAR-100 + Tiny-IN, H6-A/B/C verdicts, scope of universality extension).

- [ ] **Step 1: Insert §7.1.8 in EN paper**

Insert after §7.1.7 (line 460+). Use the §7.1.7 G4-quinto block (`docs/papers/paper2/results.md:460-`) as the structural template. Section title : `## 7.1.8 G4-sexto pilot — RECOMBINE-empty universality extension to mid-large scales (2026-05-03)`. Required content :

1. **Opening** — *"Following the G4-quinto §7.1.7 finding that RECOMBINE was empirically empty across {Split-FMNIST, Split-CIFAR-10} × {3-layer MLP, small CNN}, the G4-sexto pilot tests universality at mid-large benchmark + class-count scale ..."*
2. **Step description** — Step 1 = Split-CIFAR-100 × G4SmallCNN (n_classes=10 per-task head, 240 cells = 4 arms × 30 seeds × 2 strategies × 1 HP) ; Step 2 = Split-Tiny-ImageNet × G4MediumCNN (n_classes=20, 64×64 RGB, 240 cells).
3. **Hypotheses** : H6-A (CIFAR-100 Welch fail-to-reject at α=0.0167), H6-B (same on Tiny-IN), H6-C (conjunction → universality across 4 benchmarks × 2 substrate families).
4. **Verdict block** — populated literally from `docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}`. Use the same numerical-table template as §7.1.7 (rows : Welch t, p, alpha, Hedges' g, mean P_max(mog), mean P_max(none), N, fail_to_reject, confirmed). Embed the verbatim honest-reading clause once.
5. **DR-4 implication** — point at `docs/proofs/dr4-profile-inclusion.md` v0.5 G4-sexto amendment (Task 11) ; the empirical scope now spans the literal grid that lands in the v0.5 amendment (do not list a "*-conditional" — list the actual confirmed benchmarks).
6. **Provenance** — pre-reg, aggregate, step1, step2 (Option A only).

**Do not** leave any `[Verdict block populated...]` / `{value}` placeholder in the committed file. Numbers come from Task 9's aggregate JSON.

- [ ] **Step 2: Mirror §7.1.8 in FR paper**

Translate §7.1.8 to French keeping all numerical values, file paths, axiom IDs, and section references identical (these are load-bearing per `docs/papers/CLAUDE.md`). The opening verb is *"À la suite du résultat G4-quinto §7.1.7 …"* ; "fail-to-reject" → *"non-rejet"* ; "RECOMBINE-empty" → *"RECOMBINE empiriquement vide"*. The hypothesis-table headers H6-A/H6-B/H6-C are kept unchanged (canonical IDs).

- [ ] **Step 3: Lint + commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): G4-sexto §7.1.8 EN+FR"
```

The EN→FR pair MUST land in the same commit per `CONTRIBUTING.md`'s parallel-maintenance rule.

---

### Task 11: DR-4 v0.5 amendment in evidence file

**Files:**
- Modify: `docs/proofs/dr4-profile-inclusion.md` — append a new section after the v0.4 G4-quinto amendment

Per `docs/proofs/CLAUDE.md`, proof revisions are **append-only** — never edit the v0.4 block. Add a new section following the template of the v0.4 G4-quinto amendment (lines 161-229) with G4-sexto evidence rows.

- [ ] **Step 1: Append §"Empirical-evidence amendment — G4-sexto" block**

Append a new section after the v0.4 G4-quinto block (line 229 of the current file). Mirror the v0.4 amendment structure (lines 161-229) with these substantive replacements :

- Header : `## Empirical-evidence amendment — G4-sexto (2026-05-03)` ; sub-header `v0.5 (2026-05-03 G4-sexto addendum)` extending v0.4 from "two-benchmark × two-substrate" to "{up-to-four}-benchmark × {up-to-four}-substrate" scope.
- Repeat the load-bearing invariants (structural inclusions remain proven ; Lemma DR-4.L remains formally true ; within-arm differences within ±0.001).
- Bullet list of all four benchmark × substrate cells with Welch p / Hedges' g / mean P_max(mog) / mean P_max(none) / N. The first two rows (FMNIST/G4-quater, CIFAR-10/G4-quinto) carry the **already-known** values verbatim from the v0.4 block ; the third (CIFAR-100/G4-sexto H6-A) and fourth (Tiny-IN/G4-sexto H6-B) carry literal values from `docs/milestones/g4-sexto-aggregate-2026-05-03.json`. **Do not** leave any `{p_h6a}` / placeholder pattern in the committed file — substitute literal numbers from the aggregate JSON.
- Three conditional outcome paragraphs :
  1. **H6-C confirmed** → "framework-C claim ... partially refuted across four benchmarks × four substrates" ; explicit list of the {FMNIST, CIFAR-10, CIFAR-100, Tiny-IN} × {3-layer MLP, 5-layer MLP, small CNN, medium CNN} grid.
  2. **H6-C partial** → "universality extends to the confirming benchmark only ; the falsifying benchmark restores the predicted DR-4 ordering at mid-large scale".
  3. **H6-C falsified** (both Welch reject) → "RECOMBINE is empty at {FMNIST, CIFAR-10} but contributes statistically at {CIFAR-100, Tiny-IN} ... the framework-C claim is rehabilitated for mid-large scales".
  Keep only the matching paragraph in the committed file (delete the other two — the verdict is known after Task 9).
- Closing paragraph : same as v0.4 ("does not weaken the inclusion proof ... future work pre-registered in `docs/osf-prereg-g4-sexto-pilot.md` §6 row 6 testing ImageNet-1k / transformer / hierarchical E-SNN remains the path forward").
- Provenance bullets : pre-reg, §9.1 amendment if applicable, aggregate, step1, step2 (Option A only), Paper 2 §7.1.8.

- [ ] **Step 2: Commit (no FC bump)**

```bash
git add docs/proofs/dr4-profile-inclusion.md
git commit -m "docs(dr4): v0.5 G4-sexto amendment"
```

Note : per `docs/proofs/CLAUDE.md`, the proof header version line stays at v0.5 ; this is an empirical-evidence amendment, not a structural revision.

---

### Task 12: CHANGELOG + STATUS sync

**Files:**
- Modify: `CHANGELOG.md` — new entry under `[Unreleased]`
- Modify: `STATUS.md` — update the "As of" header + Gates G4 row

- [ ] **Step 1: CHANGELOG entry**

Append under `## [Unreleased]` :

```markdown
- **2026-05-03 — G4-sexto pilot (mid-large escalation).** 2-step sequential
  (480 cells Option A or 240 Option B), Studio overnight ~6-10 h or M1 Max
  ~32-52 h. Substrates : `G4SmallCNN` (n_classes=10) on CIFAR-100 + new
  `G4MediumCNN` (n_classes=20, 64×64) on Tiny-ImageNet. Verdicts :
  H6-A `{confirmed/falsified}` (Welch p=`{p_h6a}`, g=`{g_h6a}`, N=30) ;
  H6-B `{confirmed/falsified/deferred}` (Welch p=`{p_h6b}`, g=`{g_h6b}`, N=30) ;
  H6-C universality conjunction `{confirmed/partial/falsified/deferred}`.
  EC stays PARTIAL ; FC stays C-v0.12.0 (no axiom signature change).
  DR-4 evidence file revised to v0.5 (`docs/proofs/dr4-profile-inclusion.md`
  G4-sexto amendment).
```

Replace `{...}` placeholders with concrete values from the aggregate JSON before commit.

- [ ] **Step 2: STATUS.md update**

In the `**As of**` header (line 3 of `STATUS.md`), prepend a G4-sexto sentence :

```
**As of** : 2026-05-03 G4-sexto pilot (2-step sequential, ~480 cells Option A,
~6-10 h Studio overnight) — H6-A {confirmed/falsified}, H6-B
{confirmed/falsified/deferred}, H6-C universality
{confirmed/partial/falsified/deferred} ; DR-4 partial refutation scope
{universalised across 4 benchmarks × 4 substrates / scope-bound to
{FMNIST + CIFAR-10}} ; EC stays PARTIAL per pre-reg §6 ; FC stays
C-v0.12.0. Prior : 2026-05-03 G4-quinto pilot (3-step sequential, 600 cells, ...
```

In the Gates table row for G4, append a new bullet after the existing G4-quinto entry (do not delete the G4-quinto bullet — append-only audit trail) :

```
G4-sexto 2026-05-03 PARTIAL — H6-A {result}, H6-B {result/deferred},
H6-C {result} ; RECOMBINE-empty {extends to / breaks at} mid-large scale ;
EC stays PARTIAL ; FC stays C-v0.12.0 ; STABLE promotion blocked pending
ImageNet-1k / transformer / hierarchical E-SNN follow-up.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(g4-sexto): CHANGELOG + STATUS"
```

---

## Execution order

1. Task 0 (read-first) — no commit
2. Task 0.5 (decision) — no commit
3. Task 1 → commit `docs(g4-sexto): lock OSF pre-reg pilot`
4. Task 2 → commit `feat(g4-sexto): CIFAR-100 loader 10-task`
5. Task 3 → commit `feat(g4-sexto): Tiny-ImageNet loader 10-task`  *(skip if Option B)*
6. Task 4 (read-only validation) — no commit
7. Task 5 → commit `feat(g4-sexto): MediumCNN substrate`  *(skip if Option B)*
8. Task 6 → commit `feat(g4-sexto): step1 CIFAR-100 driver`
9. Task 7 → commit `feat(g4-sexto): step2 Tiny-IN driver + SHA pin`  *(skip if Option B)*
10. Task 8 → commit `feat(g4-sexto): aggregator H6-A/B/C`
11. Task 9 — runs + 3 commits (`data(g4-sexto): step1 CIFAR-100 milestone`, `data(g4-sexto): step2 Tiny-IN milestone`, `data(g4-sexto): aggregate H6 verdict`)
12. Task 10 → commit `docs(paper2): G4-sexto §7.1.8 EN+FR`
13. Task 11 → commit `docs(dr4): v0.5 G4-sexto amendment`
14. Task 12 → commit `docs(g4-sexto): CHANGELOG + STATUS`

Coverage gate: `pytest --cov` must remain ≥ 90% after each commit (per `pyproject.toml`). Loader and aggregator tests carry the new module coverage ; the drivers are exercised via `--smoke` integration in CI (no unit-test mock for full drivers).

---

## Self-review

**1. Spec coverage.** All 12 tasks called out in the prompt are mapped : Task 0 (investigate), 0.5 (option), 1 (pre-reg), 2 (CIFAR-100 loader), 3 (Tiny-IN loader), 4 (G4SmallCNN reuse for multi-class), 5 (G4MediumCNN), 6 (Step 1 driver), 7 (Step 2 driver), 8 (aggregator), 9 (run + milestones), 10 (Paper 2 §7.1.8 EN+FR), 11 (DR-4 v0.5), 12 (CHANGELOG+STATUS). Compute Options A/B/C are explicit at Task 0.5 + §"Architecture decisions". Pre-reg discipline (lock before pilot run) is enforced by the Task 9 ordering. EN→FR mirror is enforced at Task 10 step 2.

**2. Placeholder scan.** `{p_h6a}`, `{g_h6a}`, etc. are explicitly flagged as "fill from aggregate JSON before commit" in Tasks 10, 11, 12 — they are run-time-substituted scalars, not plan placeholders. SHA-256 hashes labeled `...replace_at_first_download...` are explicitly pinned in Task 2 step 4 + Task 7 step 3 (matching the G4-quinto §9.1 first-download lock pattern). No "TODO" / "TBD" / "implement later" patterns.

**3. Type consistency.** Verdict-block keys are stable across tasks : `h6a_recombine_strategy` / `h6a_recombine_empty_confirmed` (Task 6) ↔ `h6a_recombine_strategy` (Task 8 aggregator). Same for h6b. Aggregator output keys `h6a_cifar100`, `h6b_tiny_imagenet`, `h6c_universality`, `summary.h6c_partial`, `summary.h6b_deferred` are referenced consistently in Task 8 tests, Task 10 paper text, Task 11 amendment, Task 12 STATUS update. `RANDOM_CHANCE` / `2 * RANDOM_CHANCE` exclusion threshold is documented at Tasks 6 and 7 with the same definition. Strategy literal `("mog", "none")` is consistent (no AE, contrary to G4-quinto Step 3 which uses `("mog", "ae", "none")`).

**4. Resilience.** Per-step commit (Task 9 sub-steps) preserves milestone artefacts even on partial run (G4-quinto lesson learned — Step 2 wall-time scare). Option B path is wired through every layer (aggregator deferred-block, paper §7.1.8 conditional text, DR-4 v0.5 conditional rows, CHANGELOG conditional values). HF parquet fallback for Tiny-IN is the default (canonical ImageNet ZIP path is fallback) — pattern lifted from G4-quinto §9.1.

**5. EC/FC.** No FC bump. EC stays PARTIAL across all H6-A/B/C outcomes — explicitly stated in pre-reg §6 (6 rows) and DR-4 v0.5 amendment.

**6. Append-only discipline.** DR-4 amendment is appended (not edited) per `docs/proofs/CLAUDE.md`. STATUS Gates G4 row gains a new bullet (G4-sexto) without deleting G4-quinto. Milestones are dated 2026-05-03 immutables under `docs/milestones/`.
