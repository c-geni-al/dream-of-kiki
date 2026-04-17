# kiki-oniric fork decision

## Context

Master spec §3.1 and Q2.3 (c) decision: dedicated fork
`kiki-oniric` from `kiki-flow-core`, total isolation from the 3
active stories in kiki-flow-research that launched 2026-04-16.

## Rebase policy (r3, jalonné)

- **S1** — initial fork from kiki-flow-core `main` HEAD (this task
  S1.6/S1.7 documents intent; actual fork population happens in S2.2
  Story 0 expose-primitives)
- **S8** — mid-program rebase (capture upstream major improvements)
- **S18** — pre-paper final rebase (freeze for paper 1 submission)

## Source repo location

- Parent : `~/Documents/Projets/kiki-flow-research/` (branche main)
- Fork target : `~/Documents/Projets/dreamOfkiki/kiki_oniric/`
  (nested in dreamOfkiki repo, not a git submodule — simplicity,
  avoid cross-repo sync overhead)

## Naming convention

- **Logical name** : `kiki-oniric` (kebab-case for filesystem clarity)
- **Python package name** : `kiki_oniric` (underscore, PEP 8 compliant)
- Filesystem directory uses underscore to match Python package
  identifier directly.

## Action sequence

1. **S1.7** — create placeholder `kiki_oniric/README.md` + `.gitkeep`
   (this sprint)
2. **S2.2** — Story 0 expose-primitives populates
   `kiki_oniric/core/primitives.py` with typed Protocols (DR-3
   Conformance Criterion condition 1)
3. **S5+** — progressive implementation per plan Phase 2 milestones
4. **S8** — first rebase checkpoint
5. **S18** — final rebase checkpoint
