# Formal Reviewer Recruitment (DR-2 Compositionality Proof)

**Context** : critic finding Q_CR.1 b — DR-2 proof needs external
peer review beyond sub-agent `critic`. Target recruitment S3-S5,
draft circulation S6 (G3-draft milestone), final review S6-S8.

**Status 2026-04-20** : 6+2 candidates identified across EN + FR
tracks ; draft `.eml` outreach files held in
`Business OS/mail-*.eml` ready to send from Apple Mail.

## Target profile

- TCS / category theory / monoid structures familiarity (ideal)
- OR cognitive modeling with formal background (acceptable)
- OR predictive-processing / FEP / world-model ML (acceptable — fits DR-3 substrate-agnostic lens)
- Available for 2-3 hours of proof review work
- Willing to accept acknowledgment (not byline) on Paper 1, or
  named contributor entry in `CONTRIBUTORS.md`, per their
  preference

## Candidates — EN track (DR-2 formal proof focus)

| Priority | Name | Institution | Relevance to DR-2 | Contact | Status | `.eml` |
|----|------|-------------|-------------------|---------|--------|--------|
| 1 | Bartosz Milewski | independent / consultant | category theory, monoids, applied TCS — *the* person for DR-2 closure + associativity review | bartosz@bartosz.com | draft ready | `mail-milewski.eml` |
| 2 | Francesco Locatello | ISTA | disentanglement + compositionality in representation learning | francesco.locatello@ist.ac.at | draft ready | `mail-locatello.eml` |
| 3 | Sam Gershman | Harvard | computational cognitive architecture, CL + memory | gershman@fas.harvard.edu | draft ready | `mail-gershman.eml` |

## Candidates — FR track (FEP / cognitive-AI focus)

| Priority | Name | Institution | Relevance | Contact | Status | `.eml` |
|----|------|-------------|-----------|---------|--------|--------|
| 1 | **Guillaume Dumas** | UMontréal + Sorbonne / Pasteur | active inference, open-science, pre-registration advocacy — bilingual FR-EN bridge, high response probability | guillaume.dumas@umontreal.ca | draft ready 2026-04-20 | `mail-dumas-reviewer.eml` |
| 2 | Frédéric Alexandre | Inria Bordeaux | predictive processing, neuro-AI | frederic.alexandre@inria.fr | draft ready | `mail-alexandre.eml` |
| 3 | Pierre-Yves Oudeyer | Inria Bordeaux | developmental AI, cognitive | pierre-yves.oudeyer@inria.fr | draft ready | `mail-oudeyer.eml` |
| 4 | Stanislas Dehaene | NeuroSpin / Collège de France | cognitive neuroscience — star, low response rate | stanislas.dehaene@cea.fr | draft ready | `mail-dehaene.eml` |

## Candidates — JEPA-adjacent (world-models + substrate-agnostic lens)

| Priority | Name | Institution | Relevance | Contact | Status | `.eml` |
|----|------|-------------|-----------|---------|--------|--------|
| A | Adrien Bardes | Meta Paris (V-JEPA lead) | direct overlap §4.3 `restructure` primitive ; FR-based, accessible | adrien.bardes@meta.com | draft ready 2026-04-20 | `mail-bardes-vjepa.eml` |
| B | Alex LeBrun | AMI Labs CEO (ex-Nabla) | scientific alignment ; AMI is Paris-based JEPA-family pivot | alex.lebrun@ami-labs.ai | draft ready 2026-04-20 | `mail-ami-labs-lebrun.eml` |

Fallback : sub-agent `critic` + `validator` (always available).

## Outreach sequence

- **S17 (2026-W16, 2026-04-19–20)** : draft all 8 `.eml` files
- **S17 Mon 2026-04-20** : send Milewski (EN priority 1) + Dumas (FR priority 1) + Bardes (JEPA-adjacent) — 3 high-probability responders in parallel
- **S17 Thu 2026-04-23** : if no reply from Milewski, send Locatello (priority 2)
- **S17 Fri 2026-04-24** : send Alexandre if no Dumas reply yet
- **S18 Mon 2026-04-27** : send Gershman + Oudeyer as priority 3
- **S19** : send LeBrun (after paper 1 v0.2 arXiv ID received — can cite preprint)
- **S20** : Dehaene as last-resort low-probability star contact
- **S6-S8** : review iteration with whoever confirmed
- **S8** : G3 gate decision based on reviewer returns

## Tracking log

| Date | Action | Status | Notes |
|------|--------|--------|-------|
| 2026-04-17 | Tracker created (S3.3.1) | DONE | template |
| 2026-04-18 | DR-2 proof draft v0.1 created | DONE | `docs/proofs/dr2-compositionality.md` |
| 2026-04-19 | 3 EN + 3 FR DR-2 `.eml` drafted | DONE | `Business OS/mail-*.eml` |
| 2026-04-19 | DR-2 proof elevated from "TO BE PROVEN" to "proved theorem" in Paper 1 v0.2 abstract + §4.5 | DONE | closure + budget additivity + functional composition + associativity |
| 2026-04-20 | Bardes V-JEPA + LeBrun AMI Labs drafts added (JEPA-adjacent track) | DONE | `mail-bardes-vjepa.eml`, `mail-ami-labs-lebrun.eml` |
| 2026-04-20 | Dumas FR priority-1 draft added | DONE | `mail-dumas-reviewer.eml` — highest response probability FR track |
| TBD | Milewski sent | TODO W17 Mon | from Apple Mail |
| TBD | Dumas sent | TODO W17 Mon | from Apple Mail |
| TBD | Bardes sent | TODO W17 Mon | from Apple Mail |

## Paper-1 framing based on reviewer returns

- **≥1 reviewer confirms DR-2 proof as stated** (closure + budget additivity + functional composition + associativity) → Paper 1 claims "proved theorem" with cited reviewer acknowledgment in §Acknowledgments
- **≥1 reviewer flags a gap in the DR-2 proof** → iterate proof until gap closed OR adopt DR-2' fallback (canonical ordering subset, weaker but provable by inspection)
- **Zero reviewer confirmation by S8** → Pivot B partial : paper submitted with DR-2 framed as "generated-semigroup theorem with freeness left open" (the current v0.2 framing), target kept at PLOS CB

## Fallback protocol (if no human reviewer by S6)

If no human reviewer confirmed by S6, **Pivot B partial** activates :
- DR-2 proof reviewed by sub-agent `critic` + `validator` only
- Paper 1 framed as "formal-leaning" rather than "formally proven" (already the v0.2 wording — "we do not claim the universal property of freeness")
- Target journal : PLOS Computational Biology (already the v0.2 target)
- Document decision in `docs/proofs/g3-decision-log.md` at S8
