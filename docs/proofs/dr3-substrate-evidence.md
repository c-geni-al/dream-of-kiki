# DR-3 Conformance Criterion — substrate evidence (C2.10)

**Status** : two-substrate evidence, synthetic substitute.

DR-3 claims substrate-agnosticism : any substrate that
satisfies the three conditions below inherits the framework's
guarantees. This document records the **evidence per
substrate** for each condition, with pointers to the test
files and source modules that back the verdicts.

## Conformance conditions (spec §6.2)

1. **Signature typing** — the 8 primitives are declared as
   typed `Protocol`s (awake→dream α/β/γ/δ + 4 channels
   dream→awake). A substrate conforms to C1 by exposing
   handlers / factories compatible with these signatures.
2. **Axiom property tests** — DR-0..DR-4 property tests pass
   on the substrate's state representation.
3. **BLOCKING invariants enforceable** — the S2 finite and
   S3 topology guards can be applied to the substrate's
   state and refuse ill-formed values.

## Evidence per substrate

### `mlx_kiki_oniric`

Evidence summary :

- **C1 — signature typing (typed Protocols)** : `PASS` — all 8 primitives declared as Protocols + registry complete
  - evidence : `tests/conformance/axioms/test_dr3_substrate.py`
- **C2 — axiom property tests pass** : `PASS` — DR-0, DR-1, DR-3, DR-4 axiom suites pass on MLX
  - evidence : `tests/conformance/axioms/`
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on MLX
  - evidence : `tests/conformance/invariants/`

### `esnn_thalamocortical`

Evidence summary *(synthetic substitute — numpy LIF skeleton)* :

- **C1 — signature typing (typed Protocols)** : `PASS` — 4 op factories callable + core registry shared with MLX (spike-rate numpy LIF skeleton, synthetic substitute)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C2 — axiom property tests pass** : `PASS` — DR-3 E-SNN conformance suite passes on numpy LIF skeleton (synthetic substitute — no Loihi-2 HW)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- **C3 — BLOCKING invariants enforceable** : `PASS` — S2 finite + S3 topology guards enforceable on LIFState (synthetic substitute — spike-rate numpy LIF)
  - evidence : `tests/conformance/axioms/test_dr3_esnn_substrate.py`

### `hypothetical_cycle3` (placeholder — not yet implemented)

This row is reserved for a future (cycle-3) substrate. All cells are `N/A — placeholder for cycle 3`. The DR-3 evidence set is *two* substrates in cycle 2.

## Synthetic-data caveat

The E-SNN substrate rows are backed by a numpy LIF spike-
rate skeleton, not by real Loihi-2 hardware. No fMRI or
behavioural cohort is involved in either substrate's C2
axiom tests. DR-3 two-substrate replication therefore
strengthens the *architectural* claim (the framework's
Conformance Criterion is operational across two
independent implementations of the 8 primitives) — it does
not yet carry a cross-substrate empirical claim on real
biological data.

## Cross-references

- Spec §6.2 : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- Matrix dump : `docs/milestones/conformance-matrix.md`
- JSON dump : `docs/milestones/conformance-matrix.json`
- MLX substrate : `kiki_oniric/substrates/mlx_kiki_oniric.py`
- E-SNN substrate : `kiki_oniric/substrates/esnn_thalamocortical.py`
- C2 axiom suites : `tests/conformance/axioms/`
- C3 invariant suites : `tests/conformance/invariants/`


## Empirical-evidence amendment — G5-bis (2026-05-03)

**Context.** The G5 binary-head pilot (§7.1.3 paper) found
that absolute retention diverges between MLX and E-SNN at the
spectator-baseline level. The G4-ter MLX richer head returned
g_h2 = +2.77 (Welch p = 4.9e-14, N=30). The G5-bis pilot
ports the richer head to E-SNN to test whether the positive
effect transfers cross-substrate.

**G5-bis findings (2026-05-03)** :
- Own-substrate g_h7a = 0.1043 (E-SNN richer P_equ vs baseline)
- Welch one-sided p = 0.4052 at α/4 = 0.0125 → fail-to-reject H0
- H7B_G_THRESHOLD = 0.5 not reached
- Cross-substrate aggregate : all 4 arms reject H0 with
  g_mlx_minus_esnn ∈ [3.23, 4.20]
- **Classification : H7-B (MLX-only artefact at this N)**

**Scope amendment.** The DR-3 substrate-agnosticism guarantee
provided by this evidence file covers **only the Conformance
Criterion structural checks (axiom property tests + BLOCKING
invariants)**. The guarantee does **not** extend to empirical
effect-size transferability. At the scales tested
(Split-FMNIST 5 binary tasks, N ≤ 30, MLX MLP/CNN + E-SNN
LIF), the dream positive effect is empirically MLX-bound :
spike-rate quantization, LIF non-linearity, and STE-backward
approximation of the E-SNN substrate wash out the dream effect
that emerged on MLX richer head.

**Lemma DR-3 axiom-level guarantee** : preserved formally and
empirically. Both substrates pass the axiom property tests
DR-0/1/2'/4.

**Empirical effect-size transferability** : refuted for the
G4-ter / G5-bis comparison at this N. Future work : confirmatory
N=30 G5-bis Option A, spiking-CNN G5-ter (testing whether the
spike non-linearity is the load-bearing washout mechanism),
and ImageNet-scale + transformer-substrate escalation are
listed in pre-reg G5-bis §6 row 6 for any future
substrate-agnosticism STABLE promotion.

**Citations** :
- Pre-reg : `docs/osf-prereg-g5-bis-richer-esnn.md`
- Milestone : `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}`
- Aggregate : `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}`
- Paper : `docs/papers/paper2/results.md` §7.1.9


## Empirical-evidence amendment — G5-ter (2026-05-03)

**Context.** Following the G5-bis amendment above, Plan G5-ter
ports the G4-quinto Step 2 small-CNN architecture onto the
E-SNN substrate as a 4-layer spiking CNN
(`EsnnG5TerSpikingCNN` : Conv2d-LIF + Conv2d-LIF + avg-pool + FC-LIF
+ Linear, STE backward, pure-numpy Conv2d) and tests whether
convolutional inductive bias recovers the cross-arm positive
effect that the LIF MLP failed to express. The pre-registered
decision rule (LOCKED thresholds 0.5 / 1.0 / 2.0) maps the
observed (`g_h8`, own-Welch outcome, `g_p_equ_cross`) tuple to
H8-A (LIF non-linearity is the load-bearing washout), H8-B
(architecture mismatch was the issue), or H8-C (partial — both
contribute).

**G5-ter findings (2026-05-03)** :

- 4 arms × N=10 seeds × HP combo C5, 40 cells, 36 min wall on
  M1 Max ; train shard subsampled to 1500 examples per task per
  cell per `docs/osf-deviations-g5-ter-2026-05-03.md` (test
  shard intact)
- Own-substrate `g_h8 = -0.1093` (E-SNN spiking-CNN P_equ vs
  baseline) ; Welch one-sided p = 0.5992 at α/4 = 0.0125 →
  fail-to-reject H₀ ; H7B_G_THRESHOLD = 0.5 not reached
- Cross-substrate aggregate vs G4-quinto Step 2 MLX small-CNN :
  all 4 arms reject H₀ at α/4 = 0.0125 with `g_mlx_minus_esnn ∈
  [+1.21, +1.32]` ; `g_p_equ_cross = +1.31` falls between the
  H8-A floor (2.0) and the H8-B ceiling (1.0)
- **Classification : H8-C (partial — both architecture and LIF
  non-linearity contribute)**

**Scope amendment.** DR-3 substrate-agnosticism at the
positive-effect channel is now refuted at *two* architectural
depths : (i) G5-bis H7-B for the 3-layer LIF MLP (`g_h7a =
+0.1043`, MLP washout) ; (ii) G5-ter H8-C for the 4-layer
spiking CNN (`g_h8 = -0.1093`, CNN washout but with a
~2/3-reduced cross-substrate level gap). Architectural inductive
bias contributes partially — moving from a dense 3-layer head to
a 4-layer convolutional stack closes about two thirds of the
G5-bis cross-substrate retention level gap (`+4.02 → +1.31` at
P_equ) but does **not** close the own-substrate gap : the
cycle-3 positive effect remains absent on E-SNN regardless of
whether the architecture is convolutional or fully connected.

**Lemma DR-3 axiom-level guarantee** : preserved formally and
empirically. Both substrates pass the axiom property tests
DR-0/1/2'/4 at every architectural depth tested.

**Empirical effect-size transferability** : refuted at this N
across two architectural conditions. The H8-C verdict supports a
"both mechanisms contribute" reading rather than the strong-form
H8-A "LIF non-linearity is the load-bearing washout" reading.
Per Critic precedent, fail-to-reject is read as
absence-of-evidence at this N (Option B detection floor `g ≈
1.27`), not evidence-of-absence. A confirmatory N=30 Option A
follow-up is scheduled to tighten the H8-C reading ; the Option B
detection floor would already have surfaced any g ≥ 1.27
own-substrate effect at the chosen α. ImageNet-scale +
transformer-substrate escalations remain on the future-work
backlog before any DR-3 STABLE promotion at the
positive-effect-channel level.

**Citations** :
- Pre-reg : `docs/osf-prereg-g5-ter-spiking-cnn.md`
- Deviation log : `docs/osf-deviations-g5-ter-2026-05-03.md`
- Milestone : `docs/milestones/g5-ter-spiking-cnn-2026-05-03.{json,md}`
- Aggregate : `docs/milestones/g5-ter-aggregate-2026-05-03.{json,md}`
- Paper : `docs/papers/paper2/results.md` §7.1.10
