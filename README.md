# dreamOfkiki

A substrate-agnostic formal framework for dream-based
knowledge consolidation in artificial cognitive systems.

**Status** : Research program cycle 1, S1 (2026-04-17)
**Author** : Clement Saillant (L'Electron Rare)
**License** : MIT (code) + CC-BY-4.0 (docs)

## Overview

dreamOfkiki investigates how AI systems can learn, memorize,
and organize knowledge through dream-inspired offline
consolidation. The program produces two complementary papers:

- **Paper 1** (Nature HB / PLoS Comp Bio): formal framework C
  with axioms DR-0..DR-4
- **Paper 2** (NeurIPS / ICML / TMLR): empirical ablation on
  kiki-oniric substrate (P_min, P_equ, P_max profiles)

## Structure

- `docs/specs/` — design specs (master + framework C)
- `docs/invariants/` — invariant registry (I, S, K, DR)
- `harness/` — shared eval harness (stratified matrix, bit-exact repro)
- `kiki-oniric/` — fork of kiki-flow-core (Track A implementation)
- `papers/` — paper drafts
- `tests/` — testing suite

## Specs

See `docs/specs/`:
- `2026-04-17-dreamofkiki-master-design.md` — vision, 5 tracks
- `2026-04-17-dreamofkiki-framework-C-design.md` — formal framework

## Public resources (planned)

- Dashboard: `dream.saillant.cc` (public read-only)
- Models: `huggingface.co/clemsail/kiki-oniric-{P_min,P_equ,P_max}`
- OSF pre-registration: H1-H4 locked at S3
- Zenodo DOIs: harness + models + datasets
