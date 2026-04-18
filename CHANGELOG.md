# Changelog

All notable changes to dream-of-kiki are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
+ [Conventional Commits](https://www.conventionalcommits.org/).

Versioning scheme : **DualVer** (framework C formal+empirical axes,
see `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12).

---

## [C-v0.5.0+STABLE] — 2026-04-17

End of setup phase (S1-S4). Program enters implementation phase (S5+).

### Added

- Formal framework C specs (master + framework C, 977 lines)
- Implementation plan phase-1 detailed (1416 lines, 80 checkboxes)
- Implementation plan S3-S4 atomic (1464 lines, 6 tasks)
- DR-3 Conformance Criterion strengthened (post-critic finding #3)
- Python project skeleton (uv, Python 3.12+, mlx, numpy, hypothesis,
  pytest, duckdb, pyarrow, plotly, yaml, click)
- RunRegistry with deterministic run_id (SHA-256-based, R1 contract)
- Invariants & Axioms registry (I/S/K families + DR-0..DR-4)
- Canonical glossary (primitives, profiles, DualVer, gates, metrics)
- T-Col outreach plan (3 fMRI labs + formal reviewer candidates)
- GitHub Actions CI workflow (lint + types + pytest + invariants)
- Fork decision document (kiki-oniric r3 jalonné S1/S8/S18)
- kiki_oniric skeleton
- Studyforrest RSA feasibility note (G1 Branch A locked)
- Story 0 expose typed Protocols (8 primitives, Conformance condition 1)
- Interface contracts : primitives.md + eval-matrix.yaml
- EvalMatrix loader with 6 contract tests
- OSF pre-registration draft (H1-H4 operationalized)
- Formal reviewer recruitment tracker + email template
- Retained benchmark (SHA-256 integrity, 50 synthetic items)
- fMRI schema lock (Studyforrest Branch A)
- Framework version bumped C-v0.3.1 → C-v0.5.0+STABLE

### Changed

- sqlite3 context manager fixed (leak + contextlib.closing)
- .gitignore now excludes .coverage artifacts

### Stats

- 21 commits across brainstorm → spec → plan → execution flow
- 16 tests passing, coverage 93.62% (gate ≥90%)
- 5 source files with C-v0.5.0+STABLE version consistency
- 0 BLOCKING invariant violations
- 0 AI attribution in any commit

### Milestones achieved

- **G1** — T-Col fallback locked Branch A (Studyforrest feasibility)
- **DR-3 Conformance Criterion condition (1)** — typed Protocols
  exposed, 3 tests passing

### Pending (S5+)

- DR-2 compositionality proof draft (G3-draft S6)
- P_min runtime functional (G2 S8)
- OSF pre-registration lock (upload via checklist)
- fMRI lab outreach replies (S3-S5)
- Formal reviewer recruitment (Q_CR.1 b, S3-S5)
