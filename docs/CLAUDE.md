# docs/ — specifications, proofs, papers, milestones

Documentary hub for the formal framework C. Specs are authoritative
(numbers, IDs, axioms cited from here are load-bearing). All code,
tests and milestones cite back into this tree.

## Layout

| Subdir | Role | CLAUDE.md |
|--------|------|-----------|
| `specs/` | Master + framework-C designs (canonical EN) | — |
| `specs-fr/` | FR mirror of `specs/` (parallel maintenance) | — |
| `specs/amendments/` | Dated FC-bump amendments (e.g. DR-2 falsif.) | — |
| `proofs/` | Formal proofs DR-2..DR-4, decision logs | yes |
| `papers/` | Paper 1 / 1-fr / 2 / 2-fr / 3 source trees | yes |
| `milestones/` | G-gate dumps (md + json), append-only artefacts | — |
| `interfaces/` | `primitives.md`, `eval-matrix.yaml`, `fmri-schema.yaml` | — |
| `axioms/` | `AXIOMS.md` summary (canonical IDs DR-0..DR-4) | — |
| `invariants/` | `registry.md` (I/S/K families) | — |
| `feasibility/`, `drafts/`, `superpowers/` | Notes, scratch, plans | — |
| `osf-*.md` | Pre-registration draft, amendments, upload checklists | — |

## Routing rules

- **Spec change** (axiom statement, primitive signature, channel set,
  invariant ID) → edit `specs/`, propagate `specs-fr/` in same PR,
  bump FC axis, append amendment under `specs/amendments/` if it
  weakens an axiom (cf. DR-2 2026-04-21 falsification).
- **Proof draft / revision** → `proofs/` (see `proofs/CLAUDE.md`).
- **Paper section / figure** → `papers/<paperN>/` (see
  `papers/CLAUDE.md` for sync EN↔FR + citation pinning).
- **Milestone result** (G-gate dump) → `milestones/`, append-only,
  one md + one json sibling, dated immutable.
- **Interface contract** (`eval-matrix.yaml`, `primitives.md`,
  `fmri-schema.yaml`) → `interfaces/`. Schema change = FC bump
  + harness migration in `harness/config/eval_matrix.py`.

## Conventions

- **Pinning**. Cross-repo citations (e.g. Paper 1 §7.4 → `nerve-wml
  v0.9`) use a tag, never `HEAD`. Internal cross-doc references
  cite by file path and section number, not line.
- **Dated immutables**. `milestones/*.md`, `osf-*.md`,
  `specs/amendments/*.md`, `superpowers/plans/2026-*.md` are
  **append-only**. Status corrections add a new dated entry that
  supersedes; never rewrite the past.
- **Glossary discipline**. Every new term either lives in
  `glossary.md` or refers to an existing entry. No local synonyms
  in prose (papers, specs, amendments).
- **Bilingual mirror**. `specs/` ↔ `specs-fr/` parallel; EN leads,
  FR follows in same PR. Do not let `specs-fr/` drift.
- **Synthetic vs empirical**. Numbers in `milestones/` carry the
  R1 `run_id`; synthetic-pipeline runs must be flagged in their
  md header (cf. G2 pilot template).

## Anti-patterns

- Citing a paper draft or proof by `HEAD` (or by uncommitted state)
  — pin to a tag (e.g. `paper-1-v0.2`, `dr2-proof-v0.2`).
- Editing or deleting a dated milestone (e.g.
  `g2-pilot-results.md`, `cycle3-plan-adaptation-2026-04-20.md`)
  to "clean up" — append a superseding entry instead.
- Modifying `interfaces/eval-matrix.yaml` or
  `interfaces/primitives.md` without bumping the FC axis and
  updating the harness loader / conformance test in the same PR.
- Adding a new term in a paper or amendment without registering
  it in `glossary.md` — local synonyms break the cite-by-ID rule.
- Editing `specs/` without mirroring in `specs-fr/` (or vice
  versa) — the EN→FR propagation rule is enforced by
  `CONTRIBUTING.md` and reviewers will block the PR.
- Renaming or removing an OSF amendment after upload — the OSF
  DOI pins to that filename; supersede via a new dated amendment.
