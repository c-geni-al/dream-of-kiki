# Abstract (Paper 1, draft)

**Word count target** : 250 words

---

## Draft v0.1 (S17.2, 2026-04-18)

Catastrophic forgetting remains a central obstacle for artificial
cognitive systems learning sequentially across tasks. Sleep-inspired
consolidation has been proposed as a remedy, with prior work
exploring replay (Walker, van de Ven), synaptic homeostasis
(Tononi), creative recombination (Hobson), and predictive coding
(Friston) — but no unified formal framework integrates these four
pillars into composable, substrate-agnostic operations.

We introduce **dreamOfkiki**, a formal framework with executable
axioms (DR-0 accountability, DR-1 episodic conservation, DR-2
compositionality on a free semigroup of dream operations, DR-3
substrate-agnosticism via a Conformance Criterion, DR-4 profile
chain inclusion). The framework defines 8 typed primitives (α, β,
γ, δ inputs ; 4 output channels), 4 canonical operations (replay,
downscale, restructure, recombine), and a 5-tuple Dream Episode
ontology. We instantiate the framework as `kiki-oniric`, an MLX
substrate on Apple Silicon, with three ablation profiles (P_min,
P_equ, P_max) wired against a SHA-256 frozen retained benchmark.

Pre-registered hypotheses (OSF DOI : pending) are evaluated via
Welch's t-test, TOST equivalence, Jonckheere-Terpstra trend, and
one-sample t-test against compute budget. On a synthetic
mega-v2-style placeholder, P_equ significantly reduces forgetting
versus baseline (Welch one-sided p < 0.001) and the dream compute
overhead remains within budget (ratio < 2.0, p = 0.01). Real
mega-v2 ablation and fMRI representational similarity analysis
follow in cycle 2. All code, models, and pre-registration are open
under MIT/CC-BY-4.0.

---

## Notes for revision

- Replace synthetic results with real ablation numbers post-S20+
- Insert OSF DOI once locked (currently pending action)
- Insert Zenodo DOI for code+model artifacts at submission tag
- Tighten to ≤250 words (current ~265)
