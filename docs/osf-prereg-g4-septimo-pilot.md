# G4-septimo pilot pre-registration

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pilot:** G4-sexto Step 1 (H6-A **CONFIRMED**, H6-B
**deferred** under locked Option B). H6-A re-confirmed by N=95
Studio confirmatory : Welch p = 0.3457, g = 0.153, mean_mog =
0.3701 vs mean_none = 0.3592 (76 cells per arm × strategy after
acc_initial < 0.20 exclusion). M1 Max main run N = 30 : Welch
p = 0.8450, g = 0.057.
**Substrate:** MLX medium CNN (`G4MediumCNN`, 3 Conv2d + 3
MaxPool2d + 2 Linear, `latent_dim = 128`, `n_classes = 20` per
task, 64×64 RGB input).
**Benchmark:** Split-Tiny-ImageNet (10 sequential 20-class tasks,
fine labels remapped to `{0..19}` per task ; 64×64 RGB JPEG
decoded from HF parquet shards `zh-plus/tiny-imagenet`).
**Compute:** N = 30 seeds/arm × 2 strategies × 4 arms = 240 cells.
Per-cell wall ≈ 30-45 s on Mac Studio M3 Ultra (RGB input + 3
Conv heavier than G4-sexto small CNN). Total ≈ 2-3 h overnight
on Studio (Path A); M1 Max would be 6-9 h, single-overnight
feasible. **Option A locked** — Studio path is now available
(SSH 100.116.92.12 user `clems`, repo at
`/Users/clems/Projets/dream-of-kiki`, env synced via `uv sync`,
SpikingKiki-V4 unrelated to this pilot).
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run).

## §1 Background

G4-sexto Step 1 (commit `0728a18`) confirmed H6-A on
Split-CIFAR-100 100-class scale at both N = 30 (M1 Max) and
N = 95 (Studio confirmatory). The H6-C universality conjunction
`H6-A ∧ H6-B` was reported `deferred` by the G4-sexto aggregator
because Step 2 (Tiny-ImageNet H6-B) was locked under Option B.
G4-septimo executes that deferred Step 2 as a standalone pilot
to close the H6-C conjunction across {Split-FMNIST,
Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} ×
{3-layer MLP, 5-layer MLP, small CNN, medium CNN}.

Under H6-B confirmed, the H6-C conjunction promotes from
`deferred` to `confirmed`, and the empirical-emptiness universality
of RECOMBINE extends to its full pre-registered four-benchmark
scope. Under H6-B falsified (Welch rejects H0 with the predicted
positive g for mog vs none), the universality breaks at the
200-class / 64×64 RGB scale and the framework's RECOMBINE channel
is shown to contribute measurably at that scale.

The Hu 2020 anchor (g = 0.29) remains a directional reference
only ; cross-class biological-vs-numerical magnitude calibration
is a category error.

## §2 Hypotheses (confirmatory)

- **H6-B (mid-large resolution + class count)** — on
  Split-Tiny-ImageNet with the medium CNN substrate
  (`G4MediumCNN`, `latent_dim = 128`, `n_classes = 20` per-task
  head, 64×64 RGB input, Conv2d×3 + MaxPool2d×3 + Linear×2),
  `retention(P_max with RECOMBINE = mog)` is not statistically
  distinguishable from `retention(P_max with RECOMBINE = none)`.
  Test : Welch two-sided fails to reject H0 at α = 0.05 / 1 =
  **0.05** (only one new test ; the H6-A test is already locked
  in G4-sexto, no Bonferroni inheritance). **Failing** to reject
  **is** the predicted positive empirical claim that the H5-C /
  H6-A RECOMBINE-empty finding generalises to Tiny-ImageNet at
  200-class / 64×64 RGB scale.

- **H6-C (universality of RECOMBINE-empty across 4 benchmarks
  × 4 substrates)** — derived conjunction
  `H6-A_confirmed AND H6-B_confirmed`. H6-A is already confirmed
  by G4-sexto Step 1 (both N = 30 and N = 95). G4-septimo
  resolves H6-C by adding the H6-B clause :
    - if H6-B confirms → H6-C **confirmed** (full universality)
    - if H6-B falsifies → H6-C **partial** (H6-A scope-bound to
      {FMNIST, CIFAR-10, CIFAR-100} ; Tiny-IN anomaly)

No additional Welch test for H6-C.

## §3 Power analysis

N = 30 seeds per arm at α = 0.05 detects |g| ≥ 0.74 at 80 % power
(Welch two-sided). Identical N to G4-sexto Step 1 main run for
direct comparability. Sub-threshold effect sizes remain
exploratory.

## §4 Exclusion criteria

- multi-class exclusion floor : `acc_initial < 2 × random_chance =
  0.10` for n_classes = 20 — exclude cell.
- `acc_final` non-finite — exclude cell.
- run_id collision with prior pilot's registry — abort + amend.

The 0.10 floor is half the G4-sexto 0.20 floor because random
chance halves from 0.10 to 0.05 with the doubled class budget per
task. The "2 × random_chance" rule remains symmetric across
G4-sexto and G4-septimo.

## §5 Substrate / driver paths

- Driver : `experiments/g4_septimo_test/run_step1_tiny_imagenet.py`
- Substrate : `experiments.g4_septimo_test.medium_cnn.G4MediumCNN`
  (new, 3 Conv2d + 3 MaxPool2d + 2 Linear, latent_dim = 128,
  64×64 RGB input, n_classes = 20)
- Loader : `experiments.g4_septimo_test.tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto`
  (new, HF parquet decode + SHA-256 pin per §9.1-style amendment
  pattern)
- Aggregator : `experiments/g4_septimo_test/aggregator.py`
  (emits H6-B verdict + H6-C conjunction with H6-A from G4-sexto
  aggregate)
- Sources :
  - HF `zh-plus/tiny-imagenet` parquet shards (commit pinned at
    first download, SHA-256 of decoded train+val tarball stored
    in `tiny_imagenet_dataset.TINY_IN_*_SHA256` constants).
  - Canonical https://image-net.org/data/tiny-imagenet-200.zip is
    fallback only (Stanford site occasionally HTTP 503 ; HF
    parquet is preferred path per the same operational logic as
    G4-quinto §9.1 CIFAR-10 fallback).

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H6-B confirmed | EC stays PARTIAL ; H6-C confirmed (RECOMBINE-empty universalises across 4 benchmarks × 4 substrates) ; DR-4 evidence file revised to v0.6 with G4-septimo addendum. | FC stays C-v0.12.0 |
| Row 2 — H6-B falsified (Welch rejects with predicted positive g) | EC stays PARTIAL ; H6-C partial (scope-bound to CIFAR-100 + below) ; DR-4 evidence v0.6 records the boundary at 200-class / 64×64 RGB scale. | FC stays C-v0.12.0 |
| Row 3 — H6-B falsified (Welch rejects with negative g, i.e. RECOMBINE *hurts*) | EC stays PARTIAL ; framework's "richer ops yield richer consolidation" claim **further weakened** (RECOMBINE actively reduces retention at this scale) ; DR-4 evidence v0.6 records the negative-direction boundary. | FC stays C-v0.12.0 |
| Row 4 — H6-B exclusion-rate > 50 % (insufficient cells) | abort and amend pre-reg with raised epochs (per §9 envelope b pattern) ; do not commit milestone. | n/a |

EC stays PARTIAL across all rows. FC stays at v0.12.0 across all
rows (no formal-axis bump scheduled).

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome.
H6-B confirmation specifically requires Welch failing to reject
difference between RECOMBINE = mog and RECOMBINE = none —
*"Welch fail-to-reject = absence of evidence at this N for a
difference between mog and none — under H6-B specifically, this
**is** the predicted positive empirical claim that RECOMBINE adds
nothing measurable beyond REPLAY+DOWNSCALE on the medium CNN
substrate at Tiny-ImageNet 200-class / 64×64 RGB scale."*
(Verbatim honest-reading clause adapted from G4-sexto §7.)

If H6-B is confirmed, the H6-C conjunction promotes from `deferred`
to `confirmed` and DR-4 evidence v0.6 amends the v0.5 G4-sexto
addendum with the four-benchmark universality flag.

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with
profile keys `g4-septimo/step1/<arm>/<combo>/<strategy>` and R1
bit-stable run_ids. Milestone artefacts under
`docs/milestones/g4-septimo-step1-2026-05-04.{json,md}` plus
aggregate `docs/milestones/g4-septimo-aggregate-2026-05-04.{json,md}`.

## §9 Deviations

Pre-known envelopes :

a. Tiny-ImageNet HF parquet download fails — abort and file §9.1
   amendment (Stanford zip mirror as second fallback).
b. `acc_initial < 0.10` for majority of seeds — raise epochs from
   3 to 8 (mirroring G4-sexto §9.1 pattern). Document in step
   milestone header.
c. Per-cell wall > 60 s sustained — extrapolated total > 4 h ;
   escalate to user before committing milestone, propose N = 20
   reduced run if Studio access constrained.
d. SHA-256 mismatch on first download — abort, file amendment
   with new pinned SHA-256.

Any deviation outside the envelopes requires an amendment commit
*before* the affected cell runs, OR a post-hoc honest disclosure
in Paper 2 §7.1.11 acknowledging the deviation and its impact on
confirmatory status.

### §9.1 — TBD on first run if data download issues surface

Reserved per the G4-quinto §9.1 / G4-sexto §9.1 pattern.
