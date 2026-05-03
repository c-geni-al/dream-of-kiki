# papers — venue tracker, submission discipline, byline policy

Submission package for Paper 1 (framework) and Paper 2 (ablation). **Sources live elsewhere**: `docs/papers/{paper1,paper1-fr,paper2,paper2-fr,paper3}/` carries the actual section files (`abstract.md`, `methodology.md`, `full-draft.md`, …) — see `docs/papers/CLAUDE.md` for sync EN↔FR + citation pinning + section conventions.

This dir owns: venue tracker, byline policy, cross-paper writing discipline.

## Venue targets

| Paper | Target | Status | Source tree |
|---|---|---|---|
| Paper 1 (framework C) | PLOS Comp Bio (since 2026-04-20; Nature HB retired) | submission-ready | `docs/papers/paper1/` + `paper1-fr/` (HAL mirror) |
| Paper 2 (kiki-oniric ablation) | NeurIPS / ICML / TMLR | drafting | `docs/papers/paper2/` + `paper2-fr/` |
| Paper 3 (cycle-3 amorçage) | TBD | outline only — **PROVISIONAL** banner | `docs/papers/paper3/` |

Byline (all venues) : *dreamOfkiki project contributors*. No AI attribution.

## Writing discipline (applies to all venues)

- **Empirical numbers ↔ `run_id`**. Every quantitative claim resolves to a registered `run_id` from `harness/storage/run_registry.py` + a dump under `docs/milestones/`. No "we observed ~0.6" — cite the run.
- **Canonical IDs**. Every axiom / invariant uses the canonical ID (`DR-0..DR-4`, `DR-2'`, `I/S/K` families) and matches `docs/glossary.md`. No local synonyms.
- **Proof references**. Every proof sketched in a paper has a full version under `docs/proofs/` (or is explicitly marked deferred). Pin by versioned filename (`dr2-compositionality.md v0.2`).
- **Synthetic vs empirical**. Synthetic-pipeline results allowed only in methodology / pipeline-validation sections, never in headline claims. Mark `(synthetic placeholder, G2 pilot)` in caption.
- **DualVer**. When citing framework C, always give the full `C-vX.Y.Z+{STABLE,UNSTABLE}` tag (per `CHANGELOG.md`). No bare `C-v0.5`.

## Submission state lifecycle

- **Draft → submission-ready**: section files closed in `docs/papers/paperN/`, `docs/proofs/` consistent with sketches, milestones cite required `run_id`s. Update venue table above.
- **Submitted**: tag source tree (`paper-1-v1.0-submitted`), record date, freeze `docs/papers/paperN/`. Forward edits go to branch `submitted-revN` per `docs/papers/CLAUDE.md`.
- **Revision requested**: branch `submitted-revN/`, EN→FR propagation in same PR, re-baseline `run_id`s if numbers change.

## Coupling

| Concern | Where |
|---|---|
| Reviewer recruitment, outreach, mail drafts | `ops/CLAUDE.md` |
| DR-2 proof status (gates Paper 1 submission) | `docs/proofs/CLAUDE.md` |
| Section-level conventions (EN↔FR sync, naming, building) | `docs/papers/CLAUDE.md` |

## Anti-patterns (cross-paper)

- Pasting numbers from a scratch notebook — they must resolve to a registry dump. If the dump doesn't exist, the claim doesn't exist.
- Editing `docs/papers/paperN/` after a venue submission snapshot — branch to `submitted-revN/`; the submitted state must stay reproducible.
- Mixing Paper 1 (formal, substrate-agnostic) and Paper 2 (engineering ablation) scope: Paper 1 must not reference `kiki_oniric` internals by name; Paper 2 cites Paper 1 axioms by version tag, never copies content.
- AI attribution in byline / acknowledgments / commit trailers. Project policy is *dreamOfkiki project contributors*.
- Citing framework C without the full DualVer tag (`C-v0.5.0+STABLE`, not bare `C-v0.5`).
- Regenerating figures without recording seed + harness version. Figure scripts live under `scripts/`, emit to `docs/milestones/`.
- Fleshing out `docs/papers/paper3/outline.md` before cycle-3 kickoff — the **PROVISIONAL** banner is load-bearing.
