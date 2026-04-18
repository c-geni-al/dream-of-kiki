# Paper 3 — Outline (cycle-3 amorçage, PROVISIONAL)

**Status banner** : **PROVISIONAL — drafted 2026-04-19 as C2.18
amorçage.** Formal planning + brainstorming happens at cycle-3
kickoff, not here. The first actual items on the cycle-3 agenda
are the deferred Phase 3 + Phase 4 of cycle 2 (finish the
ablation, write Paper 2) — Paper 3 emerges only if cycle-3
research threads (HW substrates, live fMRI) deliver standalone
new material.

**Working title placeholder** : "Cycle-3 research agenda
(provisional) — from simulated to embodied substrate validation".

**Target venue** : TBD at cycle-3 kickoff. Candidates : NeurIPS,
Nature Machine Intelligence, Neuromorphic Computing and
Engineering, Frontiers in Neuroscience (Neuromorphic Engineering
section).

**Format** : TBD. Likely a **bridge paper** (~6-8 pages) between
Paper 2 (engineering ablation, simulated substrates) and a
future Paper 4 (embodied cognition / live data) if cycle 3
generates enough data for two deliverables.

---

## Working hypothesis

Cycle 2 demonstrated that the substrate-agnosticism claim of
Framework C (Paper 1) holds across **two simulated substrates**
(MLX kiki-oniric + E-SNN thalamocortical, both numpy / MLX
compute). Cycle 3 would test the same claim in a **more
stringent regime** : (a) actual neuromorphic hardware (Loihi-2
or SpiNNaker) where spike dynamics are not simulated but
physically realized, and / or (b) live fMRI cohort data where
the benchmark cues come from human subjects rather than
synthetic generators.

The hypothesis is that the Conformance Criterion (DR-3) remains
a **necessary and sufficient** characterization : substrates
that pass it replicate the profile-chain effects (H1-H4)
within the same statistical envelope as MLX ; substrates that
fail it show systematic deviation.

---

## Relationship to Paper 2

Paper 2 closes cycle 2 : engineering ablation paper, simulated
substrates only, cross-substrate matrix with MLX + E-SNN as the
two data points. Paper 2 is an **intensive** contribution (deep
on methodology + Conformance Criterion practice).

Paper 3 (future) opens cycle 3 : **extensive** contribution — new
substrate modalities (HW or live data) stress-test the Framework
C claim. Paper 3 **does not replicate** Paper 2 ; it extends the
Conformance Criterion test set from 2 data points to 3 or 4 (MLX
+ E-SNN simulation + Loihi-2 HW + fMRI cohort).

Paper 3 is **not committed** at this stage. It becomes plausible
only if cycle-3 item (3) or (4) in the G9 amorçage agenda
materializes with real data.

---

## Provisional sections (TOC)

### §1 Introduction (~1 page)

- Recap of Framework C (Paper 1) and the engineering ablation
  (Paper 2).
- Motivation : why move from simulated substrates to HW /
  embodied data.
- Contribution : extend the Conformance Criterion test set from
  2 simulated substrates (Paper 2) to N≥3 with at least one
  non-simulated member.

### §2 Residual cycle-2 ablation (carry-over)

If Paper 2 (cycle 2 Phase 4) did not exhaustively cover the
ablation matrix — e.g. some P_max × E-SNN cells were skipped due
to compute budget — Paper 3 §2 runs them to completion. This
also includes :

- Re-running cycle-2 Phase 3 cells that Paper 2 authored against
  partial data.
- Tightening the seeds_per_cell from 3 to 5 if statistical power
  calculations at Paper 2 review flagged any H1-H4 cells as
  underpowered.

### §3 E-SNN on Loihi-2 (or equivalent HW)

- Intel NRC partnership status (action externe G6).
- Mapping from E-SNN thalamocortical substrate (cycle 2 numpy
  LIF) to Loihi-2 neuron cores.
- Conformance Criterion re-verification on HW : conditions
  (1) signature typing, (2) axiom property tests, (3) BLOCKING
  invariants S2 + S3 enforceable on HW state representations.
- H1-H4 ablation on HW-executed dream cycles. Comparison with
  MLX + E-SNN simulation baselines from Paper 2.
- Fallback if Loihi-2 access does not materialize : SpiNNaker or
  Norse-accelerated sim with full hardware-in-loop
  instrumentation.

### §4 fMRI cohort integration

- T-Col partnership status (action externe G3).
- Replacement of synthetic mega-v2 placeholder (cycle 1-2) with
  real Studyforrest RSA pipeline output.
- Impact on H1-H4 statistical tests : power + effect size
  re-estimation with real subject-level variance.
- Ethical / IRB notes and data-sharing plan (OSF + HAL).

### §5 Discussion

- Substrate-agnosticism claim after N≥3 Conformance passes :
  framework saturates empirical validation budget.
- Open questions : substrates that **fail** the Conformance
  Criterion — do they systematically fail the H1-H4 tests, as
  the theory predicts ? This requires an adversarial-substrate
  construction exercise (out of scope for Paper 3, flagged as
  Paper 4 bait).
- Limitations : compute cost of HW-in-loop runs, IRB gating on
  fMRI cohort, commercial licensing on Loihi-2.

### §6 Conclusion + Future Work

- Summary : Framework C validated on N simulated + M
  non-simulated substrates, Conformance Criterion acts as the
  predictive filter.
- Future : adversarial-substrate Paper 4 ; clinical-population
  fMRI cohort ; production-grade engine (post-research) if
  industry partnerships materialize.

---

## Open questions for cycle-3 kickoff brainstorming

1. Is Paper 3 a **single paper** or should it split into Paper 3a
   (HW substrate) + Paper 3b (fMRI cohort) ?
2. Is the target venue Nature MI (high bar, substrate
   integration story) or NCE (more focused on neuromorphic
   engineering angle) ?
3. What is the minimum Loihi-2 / SpiNNaker access level that
   makes §3 publishable ? Single-chip run or cluster run ?
4. Does fMRI cohort require a fresh IRB or is Studyforrest's
   public licensing sufficient for our re-use + publication ?
5. Should cycle 3 adopt a **4-paper cycle** (Paper 2 late + Paper
   3 + Paper 4 adversarial + Paper 5 clinical) or remain
   2-paper-per-cycle cadence ?

---

## Cross-references

- G9 cycle-2 closeout : `docs/milestones/g9-cycle2-publication.md`
- Paper 1 (cycle 1, framework) : `docs/papers/paper1/`
- Paper 2 (cycle 2, engineering ablation ; narrative **deferred**
  to cycle 3 Phase 2) : `docs/papers/paper2/outline.md`
- Cycle-3 gate amorçage (first items on the agenda) : Phase 3
  (cross-substrate ablation) + Phase 4 (Paper 2 narrative) of
  the original cycle-2 plan, inherited as cycle-3 Phase 1 + 2
- Framework C spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- DR-3 Conformance Criterion : framework-C spec §6.2
