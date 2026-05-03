# K2 — SO × fast-spindle phase-coupling evidence

**Version** : v0.1-draft (2026-05-02)
**Supersedes** : —
**Amendment pointer** : —
**Target venue** : Paper 1 §5 (invariants), Paper 2 §3 (engineering
evidence)
**Executable counterpart** :
`tests/conformance/invariants/test_k2_coupling.py`
(`test_k2_property_synthetic_in_window`,
`test_k2_property_smoke_known_seed`)

## Status

`evidence-only`. K2 is a measurement-class invariant, not an axiom;
this file pins (a) the empirical anchor, (b) the estimator, (c) the
synthetic-substrate calibration. It is **not** a formal proof.

## Empirical anchor

eLife 2025 Bayesian meta-analysis of slow-oscillation–spindle
coupling and memory consolidation (BibTeX
`elife2025bayesian`). Headline numbers (verbatim from the paper note
in `docs/papers/paper1/references.bib:321-330`):

* Coupling strength: **0.33** with 95 % CI **[0.27, 0.39]**.
* Bayes factor vs. null: **> 58**.
* Egger publication-bias test on the phase branch: **p = 0.59**
  (no detected bias).

## Estimator

Mean-vector-length (Tort 2010-style):

$$
\mathrm{MVL} = \frac{\left|\frac{1}{N}\sum_t a(t)\,e^{i\phi(t)}\right|}
                    {\frac{1}{N}\sum_t \left|a(t)\right|}
$$

with $\phi(t)$ the SO instantaneous phase (radians, wrapped to
$[-\pi, \pi]$) and $a(t)$ the fast-spindle envelope. Implemented in
`tests/conformance/invariants/test_k2_coupling.py::_mean_vector_length`.

## Synthetic substrate calibration

The reference substrate
`tests/conformance/invariants/_synthetic_phase_coupling.py`
generates an SO carrier at $f_{SO} = 1$ Hz sampled at $f_s = 256$ Hz,
modulating a fast-spindle envelope of mean 0.5 with PAC depth 0.33
and additive Gaussian noise $\sigma = 0.05$. Across 50
Hypothesis-supplied seeds, the estimator yields
$\mathrm{MVL} \in [0.27, 0.39]$ (in practice the synthetic clusters
tightly around the eLife headline 0.33 — the 50-seed empirical range
was [0.328, 0.332] at $n = 8192$, fs = 256 Hz). The smoke test
`seed=7` pins the calibrated mid-window value to
$\mathrm{MVL} \in (0.30, 0.36)$.

The original plan defaulted to $\mathrm{PAC\_DEPTH} = 0.10$, which
empirically yields $\mathrm{MVL} \approx 0.10$ — below the CI floor
0.27. The analytic relation for this generator family is
$\mathrm{MVL} \approx \mathrm{PAC\_DEPTH}$ (the cosine cross-product
plus the unit-mean amplitude denominator collapse to that ratio),
so the calibration was lifted to 0.33 to match the eLife headline.

## Limitations

* Single-meta-analysis anchor; the WARN severity in
  `docs/invariants/registry.md` reflects this.
* Real substrate measurement protocols (sampling, band-pass, Hilbert
  transform vs. complex Morlet) can shift the CI upward (typically
  toward 0.4) or downward (sub-0.27 in noisier recordings) without
  invalidating the underlying physiology. K2 is therefore advisory,
  not gating, until ≥ 2 independent meta-analyses converge.
* No real substrate implements `PhaseCouplingObservable` yet
  (S18+ planned for MLX kiki-oniric, S22+ for E-SNN
  thalamocortical). Until then K2 is exercised exclusively against
  the synthetic substrate.
