# Paper 1 — PLOS Computational Biology submission readiness audit

**Date** : 2026-05-04
**Auditor scope** : process-state only ; no content review, no edits to the paper.
**Sources cross-checked** : `STATUS.md`, `docs/papers/paper1/`, `docs/papers/paper1-fr/`,
`docs/proofs/`, `ops/`, `docs/milestones/`, `CHANGELOG.md`.

## **Submission readiness : BLOCKED**

Two hard gates are open : (1) DR-2 external review has not started (per
`docs/proofs/CLAUDE.md` the submission gate blocks on
`Status: external-review-complete`, current state is *recruitment-pending*) ;
(2) the assembled `full-draft.md` is stale relative to its section sources
(re-render required). Two soft gates also open : arXiv ID not yet assigned ;
4 suggested-reviewer placeholders unfilled in cover letter.

---

## 1. Paper 1 current state

**Section inventory** (`docs/papers/paper1/`) :

| File | Mod date | Notes |
|---|---|---|
| `__init__.md` | 2026-04-30 | **stale — still says Nature HB primary** |
| `abstract.md` | 2026-04-30 | 346 words |
| `introduction.md` | 2026-05-02 | |
| `background.md` | 2026-05-02 | |
| `methodology.md` | 2026-05-03 | |
| `results-section.md` | 2026-04-30 | |
| `discussion.md` | 2026-05-02 | |
| `future-work.md`, `outline.md` | 2026-04-30 | |
| `references.bib` | 2026-05-02 | 62 `@` entries |
| `full-draft.md` | **2026-04-30** | 7 626 w / 1 102 ln — **stale vs. May-2/3 sections** |
| `full-draft-fr.md` | 2026-04-30 | |
| `cover-letter-plos-cb.md` | 2026-04-30 | |
| `reviewer-feedback.md` | 2026-04-30 | skeleton (S25.1) |
| `build/full-draft.tex` | 2026-04-30 | 408 lines |
| `build/full-draft.pdf` | **MISSING** | STATUS.md cites 22pp 296KB but only `.tex` in `build/` |

**Section structure of `full-draft.md`** : §1 Abstract → §10 References, full IMRaD shell
plus §4 Framework C and §5 Conformance Criterion validation. Self-reported budget ~10 000
words ; PLOS CB has no hard limit.

**PLOS CB structural gaps** :
- **Author Summary** (PLOS CB mandatory ~150–200 w lay block) — no `author-summary.md`,
  no `## Author Summary` header in `full-draft.md`. **MISSING.**
- No in-paper Competing Interests / Funding / Data Availability statements (only in cover
  letter). PLOS CB requires them in the manuscript.
- No separate Acknowledgments header.

---

## 2. Cover letter status

**Present** : `docs/papers/paper1/cover-letter-plos-cb.md`, dated 2026-04-20, 142 lines.
Hits novelty (DR-0..DR-4 + Conformance Criterion + cross-substrate evidence), fit (PLOS CB
remit, sleep-consolidation lineage), open-science contract (R1, R3, OSF DOI
`10.17605/OSF.IO/Q6JYN` + amendment `10.17605/OSF.IO/TPM5S`). Byline already pinned to
`Saillant, Clément` with ORCID `0000-0002-8414-185X`.

**Gaps in current cover letter** :
- Lines 106–121 : **4 suggested reviewers all `[PLACEHOLDER]`** (Cheng/Feld, Gershman/O'Reilly,
  Alexandre/Loihi-2, Spivak). PLOS CB submission portal requires ≥4 named reviewers with email
  + affiliation. Names must be cross-checked against the 5-year COI lookback before paste.
- No mention of DR-2 external-review status (currently *recruitment-pending*) — recommend
  one sentence explicitly stating reviewer recruitment state per `ops/formal-reviewer-recruitment.md`.
- No FR mirror : `paper1-fr/` does not contain a cover letter (PLOS CB submits in EN, so
  this is acceptable but flag for HAL deposit consistency).

---

## 3. DR-2 external review status

**Proof state** : `docs/proofs/dr2-compositionality.md` v0.2-draft (2026-04-21), G3-draft
status. Header lists "external-review-complete" as the gating state — **not yet reached**.

**Recruitment state** (per `ops/formal-reviewer-recruitment.md`) :
- 6 EN/FR + 2 JEPA-adjacent candidates identified ; `.eml` drafts staged in `Business OS/`.
- Tracking log shows **Milewski, Dumas, Bardes** marked `TODO W17 Mon` — **never marked sent**.
- `docs/proofs/g3-decision-log.md` — all 5 reviewer-status rows still `TODO`. Outcome section
  empty (Branch chosen : TBD).

**Submission gating impact** : per `docs/proofs/CLAUDE.md` and `ops/CLAUDE.md`, Paper 1
submission gate (G3-draft / G6-submit) blocks on `Status: external-review-complete`.
Current state = **recruitment-pending**, no reviewer has been contacted as of audit date.
Pivot B partial fallback (sub-agent `critic` + `validator` only) is documented in the
recruitment plan as an escape hatch if no reviewer confirmed by S6 ; this fallback has
**not** been formally activated in `g3-decision-log.md`.

---

## 4. arXiv prep status

**Present** :
- `docs/papers/paper1/build/full-draft.tex` (rendered 2026-04-30 from commit `22784f8`).
- `docs/papers/paper1/build/README-arxiv.md` — submission walkthrough, categories
  (`cs.LG` primary ; `q-bio.NC`, `cs.AI` cross-list).
- `docs/milestones/arxiv-submit-log.md` — `Status: PENDING manual user action`. Pre-submit
  checklist all green except final web-UI walkthrough.

**Gaps** :
- No PDF in `build/` despite STATUS.md citing 22 pages 296 KB — re-render needed.
- No figure assets : pre-submit checklist says "Figures embedded — `[ ]`" still unchecked.
  `full-draft.md` notes 5 planned figures (4-pillars, swap-protocol, profile-inclusion,
  cross-substrate matrix, §7.5 forgetting curves) — none in repo.
- `[INCLUDE: ...]` directives in `full-draft.md` not auto-resolved by current pandoc
  invocation (per README known limitation) ; the rendered `.tex` may inline literals.
- `full-draft.tex` is from the 2026-04-30 render and predates the May-2/3 section edits
  to introduction / background / methodology / discussion / references.bib — **stale**.

PLOS CB allows preprint coexistence ; arXiv is recommended but not blocking for journal
submission.

---

## 5. R1 bit-exact status for Paper 1 figures

**Citations in `full-draft.md`** :
- §5.6 / §5.7 → `docs/milestones/g7-esnn-conformance.md` — **EXISTS**.
- §6.5 / §7.1 / §7.3 / §8.2 / §9.4 → `docs/milestones/ablation-results.json` (run_id
  `syn_s15_3_g4_synthetic_pipeline_v1`) — **EXISTS** (`.md` + `.json`).
- §7.4 → cross-substrate portability from sibling `nerve-wml` v0.9 (external repo, pinned
  by tag per spec rule). Not re-verifiable here.

**Cited but no figure file** : 5 planned figures (see §4 above) — all referenced in
"Notes for revision" as TODO, none rendered, none registered to a `run_id`-keyed
`scripts/render_figures.py` invocation. **Figures are an open R1 item** — every figure must
resolve to a registered run + rendered artefact before submission per `papers/CLAUDE.md`
and `docs/papers/CLAUDE.md` rules.

**Orphans** : none detected. All milestone references in `full-draft.md` resolve to
existing files.

---

## 6. Bilingual mirror health

EN-only files (no FR mirror) : `cover-letter-plos-cb.md`, `reviewer-feedback.md`, `build/`,
`full-draft-fr.md` (FR full draft is misplaced under EN dir, not in FR tree).
FR-only files : `blog.md`. All other 10 section files present in both trees but contents
differ (expected ; needs cross-check that May-2/3 EN edits propagated to FR per same-PR
rule). `paper1/__init__.md` line 3 still says "Nature Human Behaviour (primary)" — stale.

---

## 7. Critic-flagged issues from prior session

Surveyed `docs/proofs/g3-decision-log.md`, `CHANGELOG.md`, `docs/superpowers/plans/` :

- **Q_CR.1 b** (critic) — DR-2 proof needs external peer review beyond sub-agent `critic`.
  **OPEN**, gates G3 → §3 above.
- **Critic finding #3** (CHANGELOG line 1105) — DR-3 Conformance Criterion strengthened
  post-critic. Status not explicitly closed in proof tree ; verify
  `dr3-substrate-evidence.md` v-tag matches paper §5 cite.
- **DR-2 empirical falsification 2026-04-21** — handled via FC-PATCH `C-v0.7.0 → C-v0.7.1`,
  proof v0.2 with explicit precondition. Documented in
  `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md` (per proof header).
  Paper §4.5 / §7 must cite the v0.2 (precondition-weakened) form ; verify before submit.
- **Reviewer placeholders in cover letter** — see §2 above.
- **Stale `__init__.md`** target journal — see §6 above.
- **`full-draft.md` ↔ section file drift** — section sources edited 2026-05-02/03,
  `full-draft.md` last touched 2026-04-30. Re-render before snapshot. See §1.

---

## 8. Submission checklist (priority order)

| # | Action | Effort | Owner / file |
|---|---|---|---|
| 1 | **Send DR-2 reviewer outreach emails** (Milewski, Dumas, Bardes — drafts already staged in `Business OS/`) ; update `ops/formal-reviewer-recruitment.md` tracking log with sent dates | 1 h | user (Apple Mail) ; `ops/formal-reviewer-recruitment.md` |
| 2 | **Decide DR-2 review path** : either wait for ≥1 reviewer confirmation (target S6–S8), or formally activate Pivot B partial in `docs/proofs/g3-decision-log.md` (sub-agent fallback, paper framed as "formal-leaning" per existing v0.2 wording) | 0.5 d (decision) ; up to 4 wk (review) | `docs/proofs/g3-decision-log.md` |
| 3 | **Re-render `full-draft.md` from latest section files** (intro, background, methodology, discussion, references all updated 2026-05-02/03 ; current `full-draft.md` predates them) | 1 h | `docs/papers/paper1/full-draft.md` ; `ops/build-arxiv.sh` |
| 4 | **Re-build `build/full-draft.tex` + PDF** (pandoc + bibtex + figure inclusion) and verify the 22-page 296 KB PDF claim from STATUS.md | 1 h | `docs/papers/paper1/build/` |
| 5 | **Produce 5 planned figures** with registered `run_id`s : 4-pillars conceptual, swap-protocol state diagram, profile-chain inclusion, cross-substrate conformance matrix, §7.5 forgetting curves | 1–2 d | `scripts/render_figures.py --gate G7` (and equivalents) ; `docs/milestones/` |
| 6 | **Add Author Summary block** (PLOS CB mandatory ~150–200 word lay-readership block) to `full-draft.md` and FR mirror | 0.5 d | `docs/papers/paper1/` |
| 7 | **Add Competing Interests / Funding / Data Availability statements** to `full-draft.md` per PLOS CB submission requirements (currently only in cover letter) | 1 h | `docs/papers/paper1/full-draft.md` |
| 8 | **Fill 4 suggested-reviewer placeholders** in `cover-letter-plos-cb.md` after COI screening (5-year PLOS lookback) | 2 h | `docs/papers/paper1/cover-letter-plos-cb.md` |
| 9 | **Update stale `paper1/__init__.md`** target journal banner from Nature HB → PLOS CB | 5 min | `docs/papers/paper1/__init__.md` |
| 10 | **Cross-check FR mirror parity** : verify EN edits dated 2026-05-02/03 propagated to `paper1-fr/` in same PR per CONTRIBUTING.md rule ; resolve any drift | 0.5 d | `docs/papers/paper1-fr/` |
| 11 | **arXiv submission walkthrough** (web UI, after DR-2 review path decided) ; record `2604.XXXXX` ID in `docs/milestones/arxiv-submit-log.md` ; back-fill into §6.1 + `CITATION.cff` | 1 h | user ; `docs/milestones/arxiv-submit-log.md` |
| 12 | **Mint Zenodo DOI** for repo snapshot (cover letter mentions ; arXiv README pre-submit checklist line "Zenodo DOI inserted (post-mint)" still pending visibility) | 1 h | external (Zenodo) ; repo `CITATION.cff` |
| 13 | **PLOS CB submission portal walkthrough** : upload manuscript PDF, cover letter, suggested reviewers, COI declarations, OSF DOI links | 0.5 d | user (PLOS Editorial Manager) |
| 14 | **Tag `paper-1-v1.0-submitted`** in repo on submit ; freeze `docs/papers/paper1/`, branch forward edits to `submitted-rev1/` per `papers/CLAUDE.md` | 15 min | git ; `papers/` |

**Critical-path** : items #1, #2, #3, #5, #6 are the hard blockers. Items #4, #7–#10 can run
in parallel. Items #11–#14 are post-blocker ordering.

---

*End of audit. No paper content was authored or modified ; this document reports
process state only.*
