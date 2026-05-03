# OSF Pre-Registration — G5-bis pilot (richer-head cross-substrate transfer)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G5-bis pilot — cross-substrate transfer test of the
  G4-ter MLX richer-head positive finding (`g_h2 = +2.77`,
  REPLAY+DOWNSCALE on a 3-layer hierarchy) onto the E-SNN
  thalamocortical substrate.
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G5-bis run is registered in
  `harness/storage/run_registry.RunRegistry`

## 0. Background

Two prior cycle-3 results frame this pilot :

1. **G4-ter MLX richer-head positive finding** (milestone
   `docs/milestones/g4-ter-pilot-2026-05-03.md`). On a 3-layer
   hierarchical MLP with REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE
   coupling at HP combo C5, P_equ retention exceeded the no-dream
   baseline with `g_h2 = +2.77`, Welch one-sided p = 4.9e-14,
   N = 30 seeds per arm. This is the only positive cross-arm
   effect surviving the cycle-3 ablation cascade.

2. **G5 binary-head cross-substrate replication** (milestone
   `docs/milestones/g5-cross-substrate-2026-05-03.md`). The
   single-hidden-layer G4-bis spectator pattern transferred to the
   E-SNN substrate (within-substrate spectator on both sides), but
   the absolute retention level diverged across substrates : the
   per-arm cross-substrate Welch two-sided test rejected the null
   on baseline (g = +9.98) and dream (g = +3.49) at α/4 = 0.0125.
   I.e. the qualitative pattern transfers, the absolute level does
   not.

Both citations are mandatory : the G4-ter positive finding is
*only* a substrate-agnostic claim if it transfers to the E-SNN
substrate at the positive-effect channel ; G5 establishes that
cross-substrate level divergence is already the empirical norm at
the spectator-pattern channel, so we cannot assume level parity in
G5-bis a priori.

## 1. Purpose — three pre-registered hypotheses

The decision rule is exhaustive : every observed
(`g_h7a`, Welch outcome, P_equ-row two-sided cross-substrate
consistency) tuple maps to exactly one classification.

### H7-A — positive transfer with level-divergence

**Statement** : the G4-ter MLX positive effect transfers
*qualitatively* to the E-SNN substrate (`g_h7a > 0`, P_equ
retention exceeds baseline retention own-substrate), but the
absolute level diverges across substrates (cross-substrate Welch
two-sided on P_equ rejects the null).

**Operationalization** :
- `g_h7a = compute_hedges_g(retention[P_equ, esnn_richer], retention[baseline, esnn_richer])`
- own-substrate : `g_h7a >= H7B_G_THRESHOLD = 0.5` AND Welch
  one-sided rejects at α/4 = 0.0125
- cross-substrate : two-sided Welch on P_equ between
  `retention_richer_by_arm[P_equ]` (G4-ter MLX milestone) and
  `retention_by_arm[P_equ]` (G5-bis E-SNN milestone) rejects at
  α/4 = 0.0125

### H7-B — MLX-only artefact

**Statement** : the G4-ter MLX positive effect is bound to the
MLX substrate ; no positive effect is observable on the E-SNN
substrate at this protocol scale.

**Operationalization** :
- `|g_h7a| < H7B_G_THRESHOLD = 0.5`
- own-substrate Welch one-sided fails to reject at α/4 = 0.0125
  (insufficient evidence to claim a positive effect)

### H7-C — universal cross-substrate

**Statement** : the G4-ter MLX positive effect transfers to the
E-SNN substrate both qualitatively (`g_h7a > 0`, own-substrate
Welch rejects) and quantitatively (cross-substrate Welch two-sided
on P_equ fails to reject — means match within a tolerance window
defined by the test).

**Operationalization** :
- own-substrate : `g_h7a >= H7B_G_THRESHOLD = 0.5` AND Welch
  one-sided rejects at α/4 = 0.0125
- cross-substrate : two-sided Welch on P_equ fails to reject at
  α/4 = 0.0125 (empirical level parity within the N=10 vs N=30
  power envelope ; observed divergence within a factor-2 of
  `g_h2_mlx = 2.77`)

### Decision rule (locked)

```
classify(g_h7a, own_welch_reject, p_equ_consistency):
    if abs(g_h7a) < 0.5 and not own_welch_reject:
        return "H7-B"
    if g_h7a >= 0.5 and own_welch_reject and p_equ_consistency is True:
        return "H7-C"
    if g_h7a >= 0.5 and own_welch_reject:
        return "H7-A"
    return "ambiguous"  # logged for post-hoc inspection
```

Note : `H7B_G_THRESHOLD = 0.5` is the decision knob (Option B's
detection floor at N=10 is `g ≈ 1.27` at 80 % power, two-sided
α = 0.0125 ; 0.5 is a conservative lower bound on the
"positive-effect channel survives" claim and is *not* the Hu 2020
0.21 anchor).

## 2. Sweep design (Option B, locked)

| Parameter | Value |
|-----------|-------|
| Arms | `["baseline", "P_min", "P_equ", "P_max"]` |
| Seeds | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (Option B, N=10) |
| Cells | 4 arms × 10 seeds × 1 HP combo = **40** |
| HP combo | `representative_combo()` C5 (downscale_factor=0.95, replay_batch=32, replay_n_steps=5, replay_lr=0.01) |
| Classifier | `EsnnG5BisHierarchicalClassifier` (in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, n_steps=20, tau=10.0, threshold=1.0) |
| Beta buffer | `EsnnHierBetaBuffer` capacity 256, fill 32 per task |
| RESTRUCTURE_FACTOR | 0.05 |
| RECOMBINE_N_SYNTHETIC | 16 |
| RECOMBINE_LR | 0.01 |
| Wrapper | `dream_episode_hier_esnn` couples REPLAY → SGD-with-STE on (W_in, W_h, W_out) ; DOWNSCALE → multiplies (W_in, W_h, W_out) by `factor` ; RESTRUCTURE → adds `factor·sigma·N(0,1)` to W_h only ; RECOMBINE → MoG synthetic latents through W_out only |

## 3. Effect-size anchors

- **Hu 2020 overall** : `g = 0.29`, 95 % CI [0.21, 0.38] —
  directional anchor only (positive sign).
- **G4-ter MLX richer-head** : `g_h2 = +2.77`, Welch p = 4.9e-14,
  N=30 — the source-side empirical reference.
- **G5 binary-head cross-substrate divergence** : per-arm
  cross-substrate Welch two-sided rejects on baseline (g = +9.98)
  and dream (g = +3.49) — the null reference for "level diverges
  across substrates".

## 4. Power

- Option B : N=10 per arm (own-substrate). Min detectable Hedges'
  g at 80 % power, two-sided α = 0.0125 is **~ 1.27**.
- Detection floor is well below the G4-ter MLX `g_h2 = +2.77`, so
  Option B is sufficient *if* the effect transfers at comparable
  magnitude. If `g_h7a` lands between 0.5 and 1.27, Welch will
  not reject and the verdict will be H7-B by construction even
  if the effect is real but small ; this is a known limitation of
  the exploratory budget.
- A confirmatory **Option A** (N=30) follow-up is scheduled if
  the G5-bis verdict is H7-A or H7-C ; the N=30 budget gives a
  detection floor of `g ≈ 0.74` at 80 % power.

## 5. Data exclusion rules

- Cells with `acc_task1_initial < 0.5` are flagged
  (`excluded_underperforming_baseline=True`) and dropped from the
  per-arm retention vectors before H7-A/B/C testing — same rule
  as G4-ter / G5.
- Cells where the substrate raises any BLOCKING invariant (S1
  retained-non-regression, S2 finite weights) are excluded from
  H7-A/B/C and logged with `excluded=true`.
- Cells where `dream_episode_hier_esnn()` exits with
  NotImplementedError surface as a *plan* failure, not a data
  issue.

## 6. Outputs

- `docs/milestones/g5-bis-richer-esnn-2026-05-03.json` — per-cell
  records + own-substrate H7-A verdict.
- `docs/milestones/g5-bis-richer-esnn-2026-05-03.md` — markdown
  rendering of the same.
- `docs/milestones/g5-bis-aggregate-2026-05-03.json` —
  cross-substrate Welch table + H7-A/B/C classification.
- `docs/milestones/g5-bis-aggregate-2026-05-03.md` — markdown
  rendering of the aggregate verdict.
- `docs/proofs/dr3-substrate-evidence.md` — append-only DR-3
  evidence revision per H7 outcome.
- Paper 2 §7.1.9 (EN + FR) — section narrative anchored to the
  observed verdict.

## 7. DualVer outcome rules (binding)

| Outcome | EC | FC | Rationale |
|---------|----|----|-----------|
| Any of H7-A / H7-B / H7-C | stays PARTIAL | stays C-v0.12.0 | Option B is exploratory ; no axiom or primitive change |

Confirmatory Option A (N≥30) is scheduled if H7-A or H7-C ; that
follow-up may bump EC to STABLE on H7-C, or formalise the
level-divergence claim under H7-A. No FC bump is triggered by
this pilot regardless of H7 outcome.

## 8. Amendments

This pre-registration is append-only. Any post-hoc deviation
(seed-count change, statistical-test substitution, exclusion-rule
relaxation, decision-knob shift) is documented in
`docs/osf-deviations-g5-bis-<date>.md` (separate file, dated
immutable). Deviations include : `H7B_G_THRESHOLD` change,
addition of an extra arm, change of HP combo away from C5,
reduction of N below 10.

## 9. Data and code availability

- Pre-reg : this file, locked at `git rev-parse HEAD` before the
  first run-registry insert.
- Pilot driver : `experiments/g5_bis_richer_esnn/run_g5_bis.py`
- Aggregator : `experiments/g5_bis_richer_esnn/aggregator.py`
- Classifier : `experiments/g5_bis_richer_esnn/esnn_hier_classifier.py`
- Wrapper : `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py`
- Effect-size helpers : `kiki_oniric.eval.statistics.{compute_hedges_g, welch_one_sided}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dumps : `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}`
  and `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}`

## 10. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G5-bis cell is registered in the
run registry.**
