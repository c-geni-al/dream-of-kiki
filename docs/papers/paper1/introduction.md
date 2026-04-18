# Introduction (Paper 1, draft)

**Target length** : ~1.5 pages markdown (≈ 1200 words)

---

## 1. Catastrophic forgetting and the consolidation gap

Modern artificial cognitive systems excel at single-task learning
but degrade rapidly when trained sequentially across tasks — a
phenomenon known as **catastrophic forgetting** [McCloskey & Cohen
1989, French 1999]. Despite two decades of mitigation strategies
(elastic weight consolidation [Kirkpatrick et al. 2017], generative
replay [Shin et al. 2017], rehearsal-based memory [Rebuffi et al.
2017]), the field still lacks a *unified theoretical account* of
why these mechanisms work and when they should compose.

Biological cognition solves this problem during **sleep**.
Hippocampal replay during NREM, synaptic downscaling, predictive
restructuring of cortical representations, and creative
recombination during REM together form a multi-stage
consolidation pipeline [Diekelmann & Born 2010, Tononi & Cirelli
2014]. Yet existing AI work has integrated only fragments of this
biology, typically focusing on a single mechanism (e.g., replay
alone) without a principled account of how mechanisms interact.

## 2. Four pillars of dream-based consolidation

We identify four theoretical pillars that any complete
dream-inspired AI consolidation framework must address :

- **A — Walker/Stickgold consolidation** : episodic-to-semantic
  transfer via replay [Walker & Stickgold 2004, Stickgold 2005].
- **B — Tononi SHY** : synaptic homeostasis renormalizing weights
  during sleep [Tononi & Cirelli 2014].
- **C — Hobson/Solms creative dreaming** : recombination and
  abstraction during REM [Hobson 2009, Solms 2021].
- **D — Friston FEP** : minimization of free energy as a unifying
  account of inference and consolidation [Friston 2010].

Prior AI work has implemented A (van de Ven et al. 2020), B
(Kirkpatrick et al. 2017 as a SHY-adjacent regularization), and
elements of D (Rao & Ballard 1999, Whittington & Bogacz 2017),
but **no work has formalized how the four pillars compose** in a
substrate-agnostic manner amenable to ablation and proof.

## 3. The compositional gap

Why does composition matter ? Empirically, the order in which
consolidation operations apply changes the resulting cognitive
state — replay before downscaling preserves episodic specificity,
while downscaling before restructuring may erase the very
representations that restructuring is meant to refine. Our
analysis (`docs/proofs/op-pair-analysis.md`) enumerates the 16
op-pairs and finds 12 cross-pairs are non-commutative, reinforcing
that *order is part of the framework*, not an implementation
detail.

A proper formal framework must therefore (i) specify the
operations as composable primitives with well-defined types, (ii)
make explicit which compositions are valid, (iii) provide an
**executable** account that any compliant substrate can implement,
and (iv) support empirical ablation comparing different operation
profiles. None of the prior art does all four.

## 4. Contribution roadmap

In this paper we present **dreamOfkiki**, the first formal
framework for dream-based consolidation in artificial cognitive
systems with the following contributions :

1. **Framework C** : 8 typed primitives, 4 canonical operations
   forming a free semigroup, 4 OutputChannels, 5-tuple Dream
   Episode ontology, axioms DR-0..DR-4 with executable
   Conformance Criterion (§4).
2. **Implementation** : `kiki-oniric` MLX substrate on Apple
   Silicon, three ablation profiles (P_min, P_equ, P_max) with
   profile chain inclusion verified by axiom test (DR-4) (§5).
3. **Pre-registered ablation** : OSF-registered hypotheses H1-H4
   evaluated via Welch / TOST / Jonckheere / one-sample t-test
   with Bonferroni correction (§6, §7).
4. **Open-science artifacts** : code (MIT), models, pre-
   registration, raw data with deterministic run_id contract
   (§5.5, Methods).
5. **Roadmap to substrate generalization** (E-SNN
   thalamocortical, cycle 2) and real fMRI representational
   alignment (Studyforrest baseline locked G1, real lab
   partnership pursued via T-Col outreach).

The remainder of the paper is organized as follows : §3 reviews
the four pillars in depth ; §4 develops Framework C with axioms
and proofs ; §5 describes the kiki-oniric implementation ; §6
details the methodology ; §7 reports ablation results ; §8
discusses implications and limitations ; §9 outlines cycle-2
future work.

---

## Notes for revision

- Insert proper bibtex citations once reference manager is set up
- Cross-reference §3-§9 line numbers once full paper is laid out
  in target journal template
- Tighten to ≤1500 words for Nature HB main text discipline
