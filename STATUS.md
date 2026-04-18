# dream-of-kiki — Status

**As of** : 2026-04-17 end of S4
**Version** : C-v0.5.0+STABLE
**Phase** : end of setup (S1-S4) → start of implementation (S5+)

Public repo : https://github.com/electron-rare/dream-of-kiki

---

## Program progress

Cycle 1 calendar : 28 weeks total (S1-S28)
Completed : S1-S4 (setup + formalization phase)
In progress : S5 (dream runtime + P_min + property tests DR-0/DR-1)
Upcoming gates : G2 (P_min viable, S8), G3 (DR-2 proof, S8),
  G4 (P_equ fonctionnel, S12), G5 (PUBLICATION-READY, S18)

## Test suite

```
16 tests passing
coverage 93.62% (gate 90%)

breakdown:
- tests/unit/test_run_registry.py         : 2 tests
- tests/unit/test_eval_matrix.py          : 6 tests
- tests/unit/test_retained_benchmark.py   : 5 tests
- tests/conformance/axioms/test_dr3_substrate.py : 3 tests
```

## DualVer status

| Axis | Value | Meaning |
|------|-------|---------|
| FC   | v0.5.0 | MINOR bump (added Conformance Criterion, Protocols, infra) |
| EC   | STABLE | No empirical invalidation (pre-S5) |

Next target : C-v0.7.0+STABLE at S6 (DR-2 draft, G3-draft milestone)

## Gates

| Gate | Target week | Status |
|------|-------------|--------|
| G1 — T-Col fallback lock | S2 | ✅ LOCKED Branch A Studyforrest |
| G2 — P_min viable | S8 | ⏳ Pending S5-S8 |
| G3 — DR-2 proof peer-reviewed | S8 | ⏳ Draft S6 + review S6-S8 |
| G3-draft — DR-2 proof circulated | S6 | ⏳ Pending |
| G4 — P_equ fonctionnel | S12 | ⏳ Pending |
| G5 — PUBLICATION-READY | S18 | ⏳ Pending |
| G6 — Cycle 2 decision | S28 | ⏳ Pending |

## Critical risks watched

| ID | Risk | Status |
|----|------|--------|
| R-EXT-01 | fMRI lab outreach fail | **MITIGATED** via Branch A Studyforrest |
| R-CHA-01 | Cognitive overload > 15h/sem | Monitoring Dream-sync Monday |
| R-FRM-01 | DR-2 proof fails | Fallback DR-2' canonical order ready |
| R-IMP-01 | Swap guards too strict | Configurable thresholds, permissive start |
| R-CAL-01 | Paper 1 reject | Fallback PLoS CB / Cognitive Science |

## Outstanding human actions

1. **OSF upload** — follow `docs/osf-upload-checklist.md`,
   lock H1-H4 pre-registration before S5 experiments.
   Blocking : pre-reg confirmatory status for S5+ results.
2. **Emails T-Col fMRI labs** — Huth, Norman, Gallant outreach
   using `ops/formal-reviewer-email-template.md` adapted.
3. **Formal reviewer recruitment** — Q_CR.1 b, 3 candidates from
   academic network for DR-2 proof review (S3-S5 target).

## Open science artifacts (planned)

- [x] Repo public GitHub `electron-rare/dream-of-kiki`
- [ ] OSF project + pre-registration DOI (S3 human action)
- [ ] HuggingFace models `clemsail/kiki-oniric-{P_min,P_equ,P_max}` (S22)
- [ ] Zenodo DOI for harness (S22)
- [ ] Zenodo DOI for models (S22)
- [ ] Public dashboard `dream.saillant.cc` (S13+)

## License

- Code : MIT
- Docs : CC-BY-4.0
- Authorship byline (Paper 1) : dreamOfkiki project contributors
