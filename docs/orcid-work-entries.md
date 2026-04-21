# ORCID Profile — Work Entries to Paste

**Target profile** : https://orcid.org/0000-0002-8414-185X
(Clément Saillant, corresponding author of the PLOS CB submission)

Post-audit finding (2026-04-21) : the ORCID record is public but
empty (zero works, zero affiliations), and the name fields are
swapped. For a cover letter in a journal that publishes
affiliation metadata alongside author names, an empty ORCID is a
weak credibility surface. The goal of this file is to provide
ready-to-paste content for a 10–15 minute fix session on
orcid.org.

---

## 1. Immediate fixes on the profile page

### Swap name fields

- Current : `given_names = "saillant"`, `family_name = "clément"`
- Correct : `given_names = "Clément"`, `family_name = "Saillant"`

Edit under *Names* → *Published name*, *Other names* if you want
to keep "Saillant, Clément" as alternate display.

### Add affiliation

- Organisation : L'Electron Rare
- Department : (leave blank or "Independent research")
- City / region / country : Grandris / Auvergne-Rhône-Alpes / France
- Role : Founder / Independent Researcher
- Start date : 2023-01 (or your actual incorporation date)
- Employment / Invited position : Employment

### Keywords

Add in the *Keywords* section :

```
formal framework; cognitive AI; substrate agnosticism;
continual learning; sleep consolidation;
category theory in cognitive science; spiking neural networks;
MLX; Apple Silicon ML; dream-based consolidation
```

---

## 2. Works to add (one ORCID work entry each)

### 2.1 dreamOfkiki pre-registration

- *Work type* : Research resources → Preregistration
- *Title* : dreamOfkiki Cycle 1 — Substrate-Agnostic Formal
  Framework for Dream-Based Knowledge Consolidation
- *Publication date* : 2026-04-19
- *URL* : `https://osf.io/q6jyn`
- *External identifier* : DOI `10.17605/OSF.IO/Q6JYN`
- *Citation* :
  ```
  Saillant, C. (2026). dreamOfkiki Cycle 1: A Substrate-Agnostic
  Formal Framework for Dream-Based Knowledge Consolidation
  [Pre-registration]. Open Science Framework.
  https://doi.org/10.17605/OSF.IO/Q6JYN
  ```

### 2.2 dream-of-kiki software repository

- *Work type* : Software → Research software
- *Title* : dreamOfkiki — substrate-agnostic formal framework
  (reference implementation)
- *Version* : 0.8.0 (SemVer alias of DualVer C-v0.8.0+PARTIAL)
- *Publication date* : 2026-04-19
- *URL* : `https://github.com/hypneum-lab/dream-of-kiki`
- *External identifier* : (pending Zenodo DOI at tag `v0.8.0`)
- *Citation* :
  ```
  Saillant, C. (2026). dreamOfkiki: A Substrate-Agnostic Formal
  Framework for Dream-Based Knowledge Consolidation
  [Software, version 0.8.0].
  https://github.com/hypneum-lab/dream-of-kiki
  ```

### 2.3 Related Hypneum Lab outputs (Zenodo DOIs minted 2026-04-20)

Look up each exact DOI on Zenodo (search "Hypneum Lab" and
select "All versions" toggle) and add each as a separate ORCID
work entry under *Research software* :

- nerve-wml v1.3.0 — GammaThetaMultiplexer + oscillatory binding
  infrastructure.
- kiki-flow-research paper-v0.8-draft — teacher–student
  distillation research infrastructure.
- micro-kiki v0.3.0 — micro-kiki concept DOI + version DOI pair.

For each : use the Zenodo concept DOI (not the version DOI) as
the ORCID external identifier, and link the version DOI in the
*Notes* field.

---

## 3. Suggested cross-links from the new PLOS CB cover letter

After filling the ORCID profile, link back to it :

- Cover letter sign-off block already uses
  `ORCID : [0000-0002-8414-185X](https://orcid.org/0000-0002-8414-185X)`.
- `CITATION.cff` lists the author without ORCID — add an `orcid`
  field under the author entry for the preferred-citation metadata.
- Push the ORCID to the OSF profile (OSF Settings → ORCID) so
  the registration Q6JYN is auto-linked to the ORCID identifier.

---

## 4. Time budget

- Swap name fields : 2 min
- Add affiliation : 3 min
- Add keywords : 1 min
- Add the three priority works (2.1, 2.2, 2.3 × 3) : 8–10 min
- Push ORCID to OSF profile : 2 min
- **Total** : ~15 min

## 5. Post-fix verification

After pushing the above, re-fetch the public record :

```bash
curl -s -H "Accept: application/xml" \
  https://pub.orcid.org/v3.0/0000-0002-8414-185X/record | head -80
```

Expect : non-empty `<activities-summary>` with at least 3 works,
1 employment entry, 10 keywords, and the correct given / family
name. Re-run before submitting to PLOS CB.
