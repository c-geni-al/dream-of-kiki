# docs/proofs/ — formal proofs (DR-2..DR-4)

Formal proofs supporting framework spec §6.2 axioms. Index lives in
`__init__.md` (status table). Each proof has an executable
counterpart under `tests/conformance/axioms/`.

## Files

| File | Proves | Status |
|------|--------|--------|
| `dr2-compositionality.md` | DR-2 (weakened) closure + additivity + effect-chaining under precondition `¬(∃ i<j : π_i=RESTRUCTURE ∧ π_j=REPLAY)` | v0.2-draft (2026-04-21), G3-draft |
| `dr3-substrate-evidence.md` | DR-3 Conformance Criterion two-substrate evidence (MLX + E-SNN synthetic substitute) | C2.10 |
| `dr4-profile-inclusion.md` | DR-4 ops/channels chain inclusion + Lemma DR-4.L | draft (S7-S8) |
| `op-pair-analysis.md` | Pairwise non-commutativity table (24 permutations) | analysis support |
| `g3-decision-log.md`, `g3-draft-circulation.md` | G3 gate decision audit trail | append-only |
| `pivot-b-decision.md` | Pivot B fallback rationale | dated |

## Conventions

- **Format** : Markdown + LaTeX-in-`$$` blocks. No Lean / Coq yet
  (deferred to cycle 3 if reviewers request mechanisation).
- **Header block** every proof file MUST carry: version, supersedes,
  amendment pointer, target venue, executable counterpart (path to
  test file + line if applicable). Cf. `dr2-compositionality.md`
  header for the canonical template.
- **Pinning** : papers (`docs/papers/<paperN>/`) cite proofs via
  filename + version (`dr2-compositionality.md v0.2`), never via
  `HEAD`. A proof revision = new `vN` header block + amendment in
  `docs/specs/amendments/`.
- **Executable link bidirectional** : every proof references its
  test file ; every test docstring references its proof. Search
  for orphan proofs with `grep -L test_dr` style audits.

## Coupling

- **Proof status ↔ submission gate**. DR-2 proof status (Draft /
  Reviewed / Final) is read by `STATUS.md` and gates Paper 1
  submission via `ops/formal-reviewer-recruitment.md`. Submission
  blocks on `Status: external-review-complete` in the ops tracker.
- **Proof revision ↔ axiom test** : weakening or strengthening an
  axiom changes both the proof and `tests/conformance/axioms/test_dr<N>_*.py`
  in the same PR + FC bump (cf. 2026-04-21 DR-2 case: spec
  amendment + proof v0.2 + 12 xfail empirical witnesses).
- **Proof revision ↔ paper §6.2** : papers (paper1, paper1-fr)
  must cite the new version tag in same PR; FR mirror in lockstep.

## Anti-patterns

- Editing a proof body without bumping its header version + adding
  the amendment under `docs/specs/amendments/` — silent proof drift
  invalidates every cited tag.
- Publishing a draft proof outside the repo (arXiv, blog) before
  internal formal-reviewer review (`ops/formal-reviewer-recruitment.md`
  Status ≠ external-review-complete).
- Removing the `Executable counterpart:` header line — the proof ↔
  test bidirectional pin is the only safeguard against axiom drift
  between formal statement and conformance suite.
- Citing `dr2-compositionality.md` (no version) from a paper —
  always pin `v0.2-draft` (or current). HEAD-citations in papers
  break re-submission reproducibility.
- Treating `g3-decision-log.md` like a regular doc (rewriting
  history). It is append-only audit trail; supersede via new dated
  entry.
