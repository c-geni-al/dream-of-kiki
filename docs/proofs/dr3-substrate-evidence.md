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
