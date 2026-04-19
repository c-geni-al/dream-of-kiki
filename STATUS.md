# dream-of-kiki — Status

**As of** : 2026-04-19 cycle-3 Phase 1 launch (C3.6 runner +
C3.10 bump)
**Version** : C-v0.7.0+PARTIAL
**Phase** : cycle 3 Phase 1 in flight — real-data multi-scale
ablation matrix enumerated (1080 configs), sanity pilot 1.5B
scripted (C3.7). G10 cycle-3 gate D = CONDITIONAL-GO/PARTIAL,
empirical cells not yet validated.

Public repo : https://github.com/electron-rare/dream-of-kiki

---

## Program progress

Cycle 1 calendar : 28 weeks total (S1-S28) — closed at C-v0.5.0+STABLE
Cycle 2 calendar : all 5 phases delivered (Phase 1 E-SNN substrate,
Phase 2 P_max wiring, Phase 3 cross-substrate ablation, Phase 4
Paper 2 narrative, Phase 5 async worker + closeout) ;
G9 = FULL-GO/STABLE
Cycle 3 calendar : 6 weeks (S47-S52). Phase 1 in flight : pre-
cycle-3 locks landed (SHA-pinned models + datasets + Studyforrest
init + H5 trivariant + Bonferroni 8-test) ; C3.6 1080-config runner
landed + C3.10 DualVer bump landed + C3.7 sanity pilot scripted.
Active gate : **G10 cycle-3 Gate D (CONDITIONAL-GO / PARTIAL)**

## Test suite

```
240 tests passing
coverage 91.13% (gate 90%)
```

## DualVer status

| Axis | Value | Meaning |
|------|-------|---------|
| FC   | v0.7.0 | MINOR bump (H6 profile-ordering derived constraint surface added per framework-C §12.2 ; scale-axis glossary entry ; cycle-3 cross-scale DR-3 formal feature add) |
| EC   | PARTIAL | Phase 1 + 2 + 5 cycle-2 + pre-cycle-3 locks delivered, cycle-3 Phase 1 scripted but 1080-matrix not yet executed ; Phase 2 tracks (neuromorph + fMRI, C3.11-C3.22) scoped-deferred until sem 4-6. STABLE → PARTIAL per framework-C §12.3 transition rule |

Next target : C-v0.7.0+STABLE post-C3.22 (Gate D = FULL-GO, Phase 2 cells re-closed)

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
| G7 — E-SNN substrate conformance | cycle-2 Phase 1 | ✅ LOCKED |
| G8 — P_max profile wired | cycle-2 Phase 2 | ✅ LOCKED |
| G9 — cycle-2 publication-ready | cycle-2 closeout | ✅ FULL-GO/STABLE |
| G10 — cycle-3 Gate D (H1-H6) | cycle-3 sem 3 | ⏳ CONDITIONAL-GO/PARTIAL (C3.6 runner + C3.7 sanity scripted, matrix execution pending) |

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
