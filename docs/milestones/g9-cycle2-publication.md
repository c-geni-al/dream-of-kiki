# G9 — Cycle-2 Publication-Ready Gate Report

**Gate** : G9 (cycle-2 publication-ready : Paper 2 ready for arXiv +
NeurIPS submission + cycle-3 amorçage)
**Target week** : cycle 2 closeout (after C2.17)
**Status** : **CONDITIONAL-GO / PARTIAL** — engineering + substrate
deliverables COMPLETE (Phase 1 + 2 + 5) ; empirical ablation
(Phase 3) and Paper 2 narrative (Phase 4) DEFERRED per user scope
decision ("objectif : phase 5").

---

## Executive summary

Cycle 2 was executed in a **scoped-down mode** : Phases 1 (E-SNN
substrate), 2 (P_max full wiring) and 5 (async worker + this
closeout gate) were delivered end-to-end ; Phases 3 (cross-substrate
ablation) and 4 (Paper 2 narrative authoring) were explicitly
deferred by the user to a later cycle or a dedicated writing sprint.

As a consequence, the original G9 criterion — "Paper 2 ready for
arXiv + NeurIPS submission" — is **not** satisfied. What IS
satisfied is the **engineering foundation** that Paper 2 will
ultimately rely on : E-SNN substrate DR-3 Conformance, P_max fully
wired (4 ops × 4 channels), and a real concurrent dream worker. The
gate is closed as **CONDITIONAL-GO / PARTIAL** to reflect the
scope-honest status (research-discipline rule §3 : never report
deferred work as complete).

---

## DualVer bump proposal

Current tag : `C-v0.5.0+STABLE` (end of cycle-1, per `STATUS.md`).

Proposed cycle-2 tag : **`C-v0.6.0+PARTIAL`** (formal minor bump,
empirical axis marked PARTIAL).

Rationale :

- **Formal axis MINOR bump** (`0.5` → `0.6`) : the Conformance
  Criterion (DR-3) has been extended to a **second substrate**
  (E-SNN thalamocortical) and its 3 conditions verified (G7). This
  is a formal extension of the framework surface, not a breaking
  change — hence MINOR.
- **Empirical axis : `+PARTIAL`** (not `+STABLE`) : the cross-
  substrate ablation matrix (Phase 3) has not been executed ; the
  cycle-2 empirical validation target is not met. `+PARTIAL` is the
  honest qualifier until Phase 3 lands.

Cross-references (**proposed edits, not yet applied**) :

- `CHANGELOG.md` : add `[C-v0.6.0+PARTIAL] — 2026-04-19` entry with
  cycle-2 Phase 1 + 2 + 5 deliverables and explicit DEFERRED
  subsection for Phase 3 + 4.
- `STATUS.md` : bump **Version** field to `C-v0.6.0+PARTIAL` and
  **Phase** field to `end of cycle-2 Phase 1+2+5 / Phase 3+4
  deferred`.

Do **not** apply these edits as part of the G9 commit ; they belong
to a separate `docs(changelog): bump to C-v0.6.0+PARTIAL` commit
that the user can review and sequence at their discretion.

---

## Deliverables inventory

### Phase 1 — E-SNN substrate (C2.1 – C2.4)

| Task | Commit | Deliverable |
|------|--------|-------------|
| C2.1 | `fba1ac5` | `refactor(substrate)` : `kiki_oniric/substrates/` module split (MLX extracted) |
| C2.2 | `2ceaef6` | `feat(esnn)` : thalamocortical skeleton (LIF numpy, 4 op factory stubs) |
| C2.3 | `39ecd43` | `feat(esnn)` : 4 operations wired (replay, downscale, restructure, recombine) |
| C2.4 | `e884217` | `test(dr3)` : E-SNN passes Conformance Criterion (3/3 conditions) |

Supporting fix : `47c89e6` `fix(esnn)` : robust replay record
validation (CodeRabbit follow-up).

G7 LOCKED per `docs/milestones/g7-esnn-conformance.md` : E-SNN
passes DR-3 conditions (1) signature typing, (2) axiom property
tests DR-0/DR-2/R1, (3) BLOCKING invariants S2 + S3.

### Phase 2 — P_max full wiring (C2.5 – C2.8)

| Task | Commit | Deliverable |
|------|--------|-------------|
| C2.5 | `8ee452b` | `feat(alpha)` : α-stream raw traces ring buffer (1024 FIFO, canal-α input) |
| C2.6 | `9906520` | `feat(recombine)` : full MLX VAE variant (KL divergence upgrade from VAE-light) |
| C2.7 | `63af87d` | `feat(canal4)` : ATTENTION_PRIOR output + S4 invariant guard |
| C2.8 | `450ad3e` | `feat(profile)` : P_max wired (4 ops + 4 channels) |

G8 LOCKED per `docs/milestones/g8-p-max-functional.md` : P_max is
now the empirically-wired ceiling profile (previously skeleton-only
in cycle 1 S16.1).

### Phase 5 — Closeout (C2.17 – C2.18)

| Task | Commit | Deliverable |
|------|--------|-------------|
| C2.17 | `018fd05` | `feat(dream)` : real async dream worker (concurrent execution, no longer a skeleton) |
| C2.18 | *this commit* | `docs(milestone)` : G9 cycle-2 publication-ready gate report (this file) + cycle-3 amorçage |

### Phase 3 + 4 — DEFERRED

No commits, no code, no narrative — explicitly out of scope for
this cycle-2 closeout (see "What's deferred" below).

---

## Test + coverage summary

Run on `main` @ `018fd05` (just before C2.18), Python 3.12, uv :

```
$ uv run python -m pytest -q 2>&1 | tail -3
TOTAL                                             1067     94    91%
Required test coverage of 90% reached. Total coverage: 91.19%
169 passed in 1.86s
```

- **169 tests PASS** (up from ~16 at end of S4 setup per
  `STATUS.md`, and from the cycle-1 closeout baseline).
- **Coverage 91.19 %** (gate threshold 90 %, enforced by
  `pyproject.toml`).
- **Zero failures, zero skips, zero xfails**.

---

## Conformance status (DR-0..DR-4 + invariants)

### Axiom test suite — 5/5 active

| Axiom | Status | Evidence |
|-------|--------|----------|
| DR-0 accountability | ✅ PASS | replay produces observable output (MLX + E-SNN) |
| DR-1 episodic conservation | ✅ PASS | substrate-agnostic, framework-level |
| DR-2 op properties | 🟡 4/5 internally ; external reviewer **pending** | downscale commutative non-idempotent empirically confirmed (MLX + E-SNN) ; G3 DR-2 reviewer feedback still action externe utilisateur |
| DR-3 Conformance Criterion | ✅ PASS | 3 conditions green on **2 substrates** (MLX kiki-oniric + E-SNN thalamocortical) |
| DR-4 profile chain inclusion | ✅ PASS | ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max) ; channels likewise |

### Invariant guards — active on all 3 profiles

- **S1** retained non-regression — canal / swap protocol level
- **S2** finite (no NaN / Inf) — enforced on MLX tensors + LIFState.v
- **S3** topology ortho species — ortho chain validated pre-swap
- **S4** attention_budget ≤ 1.5 — new in cycle 2, canal-4 guard

### Substrate count — 2

- MLX kiki-oniric (primary, cycle 1)
- E-SNN thalamocortical (added cycle 2 Phase 1, G7 LOCKED)

### Profile count — 3 wired (was 2 wired + 1 skeleton at end of
cycle 1)

- **P_min** : replay only, canal-1 only — WIRED cycle 1
- **P_equ** : replay + downscale + restructure, canal-1 + canal-2 +
  canal-3 — WIRED cycle 1
- **P_max** : 4 ops (+ recombine), 4 channels (+ α input +
  ATTENTION_PRIOR) — WIRED cycle 2 Phase 2

### DR-3 Conformance Criterion — 3 conditions × 2 substrates

| Condition | MLX kiki-oniric | E-SNN thalamocortical |
|-----------|-----------------|-----------------------|
| (1) Typed Protocols (signature + registry) | ✅ | ✅ |
| (2) Axiom property tests green | ✅ | ✅ |
| (3) BLOCKING invariants enforceable | ✅ | ✅ |

Substrate-agnosticism claim (Paper 1 Discussion §8.3) is therefore
**empirically validated across two independent substrates** — the
strongest piece of engineering evidence in the cycle-2 basket.

---

## What's deferred (scope-honest section)

### Phase 3 — Cross-substrate ablation (C2.9 – C2.12)

**Scope** : 3 profiles (P_min, P_equ, P_max) × 2 substrates (MLX +
E-SNN) × H1-H4 hypothesis tests (Welch / TOST / Jonckheere /
one-sample t), executed through the `harness/` matrix with 3 seeds
per cell.

**Rationale for deferral** : user prioritized the direct path to
closeout ("objectif : phase 5"). Data generation + statistical
analysis can follow on a later cycle or a separate G-gate. The
engineering foundation (E-SNN conformant, P_max wired, async
worker real) is in place — the ablation is a data-collection +
statistical-analysis sprint, not a blocking engineering item.

**Proposed landing** : cycle 3 Phase 1 (first item on the agenda),
or a dedicated mid-cycle "Phase 3 catch-up" sprint with its own
G-gate (tentative name G10-ablation).

### Phase 4 — Paper 2 narrative (C2.13 – C2.16)

**Scope** : Paper 2 (ablation, engineering-facing) full draft —
Abstract, Introduction, Methods, Results, Discussion, Limitations,
bibtex. Target venues : arXiv preprint + NeurIPS / TMLR submission.

**Rationale for deferral** : Paper 2's Results section depends on
Phase 3 data. Without that data, the narrative would be speculative
or synthetic-placeholder, which directly violates research-
discipline rule §3 ("never report synthetic results as empirical
claims"). Deferring Phase 4 until Phase 3 data exists is the
honest path.

**Proposed landing** : cycle 3 Phase 2 (after Phase 3 data), or a
dedicated 2-3 week writing sprint once the ablation matrix is
populated.

---

## External user actions still required

Inherited from cycle-1 closeout gates (G3 / G5 / G6) and augmented
by cycle-2 scope :

| Action | Gate | Status |
|--------|------|--------|
| arXiv submission of Paper 1 | G5 | ⏳ pending (action externe) |
| arXiv submission of Paper 2 | G9 | ⏸ BLOCKED on Phase 3 + 4 |
| Nature HB editorial decision (Paper 1) | G5 | ⏳ pending (action externe) |
| NeurIPS / TMLR submission (Paper 2) | G9 | ⏸ BLOCKED on Phase 3 + 4 |
| OSF DOI lock | G5 | ⏳ pending (action externe) |
| DR-2 external reviewer feedback (T-Col) | G3 | ⏳ pending (action externe) |
| fMRI lab partnership formalization | G6 | ⏳ pending (action externe) |
| HAL FR deposit (post-arXiv) | G5 | ⏳ pending (action externe) |
| Intel NRC Loihi-2 access / E-SNN HW mapping | G6 | ⏳ pending (action externe) |

---

## G9 decision criteria table

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Paper 2 full-draft | ready for submit | **deferred (Phase 4)** | ⚠ PENDING |
| Paper 2 arXiv submission | preprint live | **deferred** | ⚠ PENDING (action externe) |
| Paper 2 NeurIPS / TMLR submission | under review | **deferred** | ⚠ PENDING (action externe) |
| Cross-substrate ablation | 3 profiles × 2 substrates × H1-H4 | **deferred (Phase 3)** | ⚠ PENDING |
| E-SNN substrate conformance | DR-3 conditions (1)+(2)+(3) | achieved (G7) | ✅ |
| P_max profile functional | 4 ops + 4 channels wired | achieved (G8) | ✅ |
| Async dream worker | real concurrent execution | achieved (C2.17) | ✅ |
| Coverage ≥ 90 % | ≥ 90 % | 91.19 % | ✅ |
| Test suite green | 0 failures | 169 pass / 0 fail | ✅ |
| DualVer bump proposal documented | yes | yes (C-v0.6.0+PARTIAL proposed) | ✅ |

**Decision** : **CONDITIONAL-GO / PARTIAL**. Engineering
deliverables complete and coherent ; publication deliverables
deferred by user scope decision. Close cycle-2 on this partial
basis ; do not claim Paper 2 ready.

---

## Cycle-3 amorçage

A provisional Paper 3 outline has been filed at
`docs/papers/paper3/outline.md` as C2.18 amorçage material. It is
explicitly marked PROVISIONAL — formal cycle-3 planning and
brainstorming happen at cycle-3 kickoff, not here.

### Likely cycle-3 agenda (ranked by priority)

1. **Finish cycle-2 Phase 3** — cross-substrate ablation matrix
   (MLX × E-SNN × 3 profiles × H1-H4). First item on the cycle-3
   critical path.
2. **Finish cycle-2 Phase 4** — Paper 2 narrative (depends on #1
   data). Authoring sprint, arXiv + NeurIPS target.
3. **E-SNN HW substrate** — Loihi-2 mapping or equivalent neuromorphic
   HW (if Intel NRC partnership materializes) ; otherwise extend the
   simulation-only path with SpiNNaker / Norse deeper integration.
4. **Live fMRI cohort data** — if T-Col / Studyforrest partnership
   formalizes, replace synthetic placeholders with real RSA pipeline
   output (closes the last synthetic caveat of Paper 1 v1).
5. **Paper 3 emergence** — a new cycle-3 paper becomes plausible
   only if (3) or (4) lands with strong data. Until then, Paper 3
   is a provisional placeholder, not a commitment.

---

## Closing statement

Cycle 2 closes at **C-v0.6.0+PARTIAL** (proposed) — formal axis
minor-bumped on the E-SNN Conformance extension, empirical axis
flagged PARTIAL because Phase 3 + 4 were explicitly out of scope.
The engineering foundation is coherent : 2 substrates DR-3-
conformant, 3 profiles wired, 4 invariants enforced, 4 channels
operational, real async worker, 169 tests green at 91.19 %
coverage.

The gap between this status and the original G9 criterion (Paper 2
ready for submission) is **honest scope deferral**, not hidden
failure. Paper 2 submission is a cycle-3 agenda item, and the
cycle-3 amorçage outline (`docs/papers/paper3/outline.md`) records
that commitment alongside any additional cycle-3 research threads.

STATUS.md lineage : propose a surgical 1-line update for the user
to apply, **not** performed as part of this commit :

```
**As of** : 2026-04-19 end of cycle-2 (Phase 1+2+5)
**Version** : C-v0.6.0+PARTIAL (proposed — pending changelog edit)
**Phase** : cycle-2 closeout ; Phase 3+4 deferred to cycle 3
```

---

## Cross-references

- Cycle-2 atomic plan : `docs/superpowers/plans/2026-04-19-dreamofkiki-cycle2-atomic.md`
- Cycle-2 Phase 1 commits : `fba1ac5`, `2ceaef6`, `39ecd43`, `e884217`, `47c89e6`
- Cycle-2 Phase 2 commits : `8ee452b`, `9906520`, `63af87d`, `450ad3e`
- Cycle-2 Phase 5 commits : `018fd05`, *this commit*
- G7 report (E-SNN Conformance) : `docs/milestones/g7-esnn-conformance.md`
- G8 report (P_max functional) : `docs/milestones/g8-p-max-functional.md`
- Cycle-3 amorçage outline : `docs/papers/paper3/outline.md`
- Paper 1 (cycle 1) : `docs/papers/paper1/` + `docs/papers/paper1-fr/`
- Paper 2 (cycle 2, pending) : `docs/papers/paper2/outline.md` + `docs/papers/paper2-fr/`
- Framework C spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- DualVer rules : framework-C spec §12
