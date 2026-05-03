# OSF Pre-Registration — G4-ter pilot (HP sweep + richer substrate)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G4-ter pilot — confirmatory follow-up to G4-bis null
  finding (`g_h1 = -2.31`, `H_DR4` degenerate equal-means)
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G4-ter run is registered in
  `harness/storage/run_registry.RunRegistry`

## 1. Background — G4-bis baseline

The G4-bis pilot (milestone `docs/milestones/g4-pilot-2026-05-03-bis.md`)
re-ran G4 after wiring `dream_episode()` to mutate classifier weights
via the REPLAY + DOWNSCALE channels. It produced:

- `g_h1 = -2.3067` (sign-reversed relative to Hu 2020 anchor `g >= 0.21`)
- `H_DR4` degenerate equal means: `mean retention[P_min] =
  mean retention[P_equ] = mean retention[P_max] = 0.5609`

Diagnostic in §H_DR4 of the G4-bis milestone identifies the binary
MLP head as exposing only REPLAY + DOWNSCALE; RESTRUCTURE and
RECOMBINE remain spectator-only (no hierarchy nor VAE latents). The
identical retention vectors across P_min/P_equ/P_max are therefore
mechanically identical, not substantively monotonic.

G4-ter is the confirmatory N≥30 follow-up scheduled by the
exploratory-positive-evidence rule of `osf-prereg-g4-pilot.md` §4.

## 2. Hypotheses

### H1 — HP artefact

**Statement** : observed `g_h1 = -2.31` is an artefact of the
G4-bis HP combo (`replay_lr=0.01`, `replay_n_steps=1`,
`replay_n_records=32`, `downscale_factor=0.95`). A curated 10-combo
HP grid sweep on the binary MLP head yields at least one combo
with `g_h1 >= 0` (sign reversed back to the Hu 2020 anchor's side)
on N=10 seeds.

**Operationalization** :
- `g_hp_best = max(compute_hedges_g(retention[P_equ, hp_i],
  retention[baseline]) for hp_i in HP_GRID)`
- Reject H0_HP iff `g_hp_best >= 0.0`
- Statistical test : Welch's one-sided t-test
  `(retention[baseline], retention[P_equ, hp_best])` at
  α = 0.05 / 4 = 0.0125 (Bonferroni for 4 hypothesis families:
  H1, H2, H3, H_DR4)

### H2 — Substrate-level limitation

**Statement** : observed `g_h1 = -2.31` is a substrate-level
limitation of the binary MLP head. A hierarchical MLP head that
exposes RESTRUCTURE (perturbation of hidden_2 weights) and
RECOMBINE (Gaussian-MoG synthetic-latent injection) yields
`g_h1 >= 0.0` on N=30 seeds with a single representative HP combo.

**Operationalization** :
- `g_h2 = compute_hedges_g(retention[P_equ, richer], retention[baseline, richer])`
- Reject H0_substrate iff `g_h2 >= 0.0`
- Statistical test : Welch's one-sided t-test
  `(retention[baseline, richer], retention[P_equ, richer])`
  at α = 0.0125

### H3 — Combined effect

**Statement** : neither HP sweep nor richer substrate alone
recovers the Hu 2020 anchor floor `g >= 0.21`, but the combination
(richer substrate + best HP combo) does. Validated by a tertiary
non-pre-registered exploratory cell.

**Operationalization** : exploratory only. Reject H0_combined iff
`g_h2 >= 0.21` and `g_hp_best >= 0.21` are *both* false but a
post-hoc richer-substrate × `hp_best` cell yields `g >= 0.21`. The
exploratory status is recorded in the milestone; it does **not**
trigger an EC bump on its own.

### H_DR4-ter — Monotonicity recovered on richer substrate

**Statement** : on the richer substrate at N=30 seeds, mean
retention is monotonically ordered `P_max >= P_equ >= P_min` and
the Jonckheere-Terpstra trend test rejects H0 at α = 0.0125. This
re-tests the H_DR4 monotonicity hypothesis after the degenerate
G4-bis tie is structurally broken by the richer head.

**Operationalization** :
- `mean_retention[P_max] >= mean_retention[P_equ] >= mean_retention[P_min]`
- Statistical test : `kiki_oniric.eval.statistics.jonckheere_trend`
  on the three retention groups at α = 0.0125

## 3. HP grid (curated 10 combos)

Grid : 4 × 3 × 3 × 4 = 144 candidates → curated to 10 along the
qualitative gradient hypothesised most likely to flip the sign of
g_h1. Listed in `experiments/g4_ter_hp_sweep/hp_grid.py`.

| combo_id | downscale_factor | replay_batch | replay_n_steps | replay_lr |
|----------|------------------|--------------|----------------|-----------|
| C0       | 0.85             | 16           | 1              | 0.001     |
| C1       | 0.85             | 32           | 5              | 0.001     |
| C2       | 0.90             | 32           | 1              | 0.001     |
| C3       | 0.90             | 32           | 5              | 0.01      |
| C4       | 0.95             | 32           | 1              | 0.001     |
| C5       | 0.95             | 32           | 5              | 0.01      |
| C6       | 0.95             | 64           | 10             | 0.001     |
| C7       | 0.99             | 16           | 1              | 0.001     |
| C8       | 0.99             | 32           | 5              | 0.01      |
| C9       | 0.99             | 64           | 10             | 0.05      |

Rationale: each axis probes a separate G4-bis suspect — SHY
over-shrinkage (`downscale_factor`), replay/downscale balance
(`replay_lr`), insufficient consolidation (`replay_n_steps`),
under-sampled replay (`replay_batch`). C5 is the G4-bis-aligned
anchor and the representative combo for the richer-substrate sweep
(only `replay_n_steps` differs : 1 → 5).

## 4. Sample size / power

- HP sub-grid : N=10 seeds × 3 dream arms (P_min, P_equ, P_max) ×
  10 combos = 300 cells. Baseline arm is **not** swept across
  combos (HP changes do not affect baseline) — its 10 cells are
  taken from the richer-substrate sweep below.
- Richer-substrate sweep : N=30 seeds × 4 arms (baseline, P_min,
  P_equ, P_max) × 1 combo (C5) = 120 cells.
- Total : 420 cells.
- Power floor : N=30 vs N=30, minimum detectable Hedges' g at
  80 % power, α = 0.0125 (Bonferroni × 4 hypothesis families)
  one-sided is ~0.6. Effects below g ≈ 0.3 remain exploratory
  and trigger a deferred N≥95 confirmatory follow-up.
- HP sub-grid power : N=10 vs N=10, min detectable g ≈ 1.0 — HP
  sweep is **screening**, not confirmatory. A `g_hp_best >= 0.0`
  outcome triggers a confirmatory N=30 sweep on `hp_best` only,
  scheduled as G4-quater.

## 5. Pre-specified analyses

- H1, H2, H_DR4-ter : `kiki_oniric.eval.statistics.welch_one_sided`
  + Hedges' g via `compute_hedges_g`.
- H_DR4-ter trend : `kiki_oniric.eval.statistics.jonckheere_trend`
  on the three retention groups in `[P_min, P_equ, P_max]` order
  at α = 0.0125.
- Multiple-comparison correction : Bonferroni at family size 4
  (H1, H2, H3, H_DR4-ter), α_per_test = 0.0125. H3 is exploratory
  and excluded from the inferential family — it inherits the
  same α/4 anyway by construction.

## 6. Data exclusion rules

- Cells where the substrate raises any BLOCKING invariant (S1
  retained-non-regression, S2 finite weights) are excluded from
  H1/H2/H_DR4-ter and logged with `excluded=true`.
- Cells with `acc_task1_initial < 0.5` are excluded as
  underperforming-baseline (same rule as G4-bis).
- Cells where `dream_episode_hier()` exits with NotImplementedError
  surface as a *plan* failure, not a data issue.

## 7. DualVer outcome rules (binding)

| Outcome | EC bump | Rationale |
|---------|---------|-----------|
| H_DR4-ter rejected H0 in predicted direction (Jonckheere monotonic, p < 0.0125) **and** H1 or H2 rejected H0 | PARTIAL → STABLE | Empirical confirmation crosses §12.3 STABLE bar for G4 scope |
| H_DR4-ter inconclusive **or** all of H1/H2 inconclusive | stays PARTIAL | Partial confirmation, schedule N≥95 G4-quater |
| H_DR4-ter falsified (Jonckheere reverses: P_min > P_max statistically) | PARTIAL → UNSTABLE | §12.3 transition rule on falsification |

No FC bump in any outcome (no axiom or primitive change).

## 8. Deviations from pre-registration

Any post-hoc deviation will be documented in
`docs/osf-deviations-g4-ter-<date>.md` (separate file, dated
immutable). Deviations include : seed-count change, statistical
test substitution, exclusion-rule relaxation, HP grid pruning at
smoke-time (allowed only via the 4-combo subset specified in the
plan §"Decision log").

## 9. Data and code availability

- Pre-reg : this file, locked at `git rev-parse HEAD` before
  the first run-registry insert.
- Pilot driver : `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- Effect-size helpers : `kiki_oniric.eval.statistics.{compute_hedges_g, welch_one_sided, jonckheere_trend}`
- Verdict anchors : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g4-ter-pilot-2026-05-03.{json,md}`

## 10. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G4-ter cell is registered in the
run registry.**
