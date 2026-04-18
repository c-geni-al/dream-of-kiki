# G7 — E-SNN Conformance Gate Report

**Gate** : G7 (E-SNN substrate passes DR-3 Conformance Criterion)
**Target week** : cycle 2 (after C2.4)
**Status** : **LOCKED — all 3 conditions verified**

## Context

DR-3 Conformance Criterion (framework spec §6.2) defines what it
means for a substrate to instantiate the framework. Cycle 1
validated the criterion on MLX kiki-oniric (the primary substrate).
Cycle 2 C2.4 validates the criterion on E-SNN thalamocortical — a
structurally very different substrate (spike-rate dynamics via
numpy LIF instead of dense-matrix gradient updates via MLX).

If E-SNN passes all 3 conditions, the substrate-agnosticism claim
of Framework C (Paper 1) is **empirically validated across 2
independent substrates** — a strong piece of evidence for
reviewers at Nature Human Behaviour.

## Criterion status

### Condition 1 — Signature typing : ✅ PASS

- `ESNN_SUBSTRATE_NAME`, `ESNN_SUBSTRATE_VERSION` identity
  constants exported
- `EsnnBackend` enum with NORSE + NXNET options
- `EsnnSubstrate` dataclass instantiable with backend parameter
- 4 op factory methods return callable handlers
  (`replay_handler_factory`, `downscale_handler_factory`,
  `restructure_handler_factory`, `recombine_handler_factory`)
- `esnn_substrate_components()` returns dotted-path registry that
  mirrors `mlx_substrate_components()` keys

Evidence : 4 tests in `tests/conformance/axioms/test_dr3_esnn_substrate.py`
covering the condition-1 signature typing.

### Condition 2 — Axiom property tests : ✅ PASS

- DR-0 accountability : every replay execution produces an
  observable output (spike-rate numpy array, non-None)
- DR-2 op properties : downscale commutative non-idempotent
  confirmed empirically (shrink_f ∘ shrink_f = f²) on E-SNN
- R1 reproducibility : recombine deterministic with same seed

Evidence : 3 tests covering DR-0/DR-2 property verification on
E-SNN op handlers.

Note : DR-1 (episodic conservation), DR-4 (profile chain inclusion)
are substrate-agnostic properties already proven at the framework
level ; E-SNN inherits them by construction.

### Condition 3 — BLOCKING invariants enforceable : ✅ PASS

- S2 finite guard : `check_finite()` accepts LIFState.v arrays and
  rejects NaN/Inf injection
- S3 topology guard : `validate_topology()` accepts canonical
  ortho species chain (rho_phono → rho_lex → rho_syntax →
  rho_sem) and rejects self-loops. Substrate-agnostic since the
  graph is pure connectivity info.

Evidence : 2 tests covering S2 + S3 enforcement on E-SNN state
representations.

Note : S1 retained non-regression and I1 episodic conservation
are substrate-agnostic (operate on benchmark + β buffer, not on
substrate-specific state). Verified at the framework level in
cycle 1.

## Gate decision

**G7 LOCKED** — E-SNN substrate passes all 3 conditions of the
DR-3 Conformance Criterion. Substrate-agnosticism claim of Framework
C is now empirically validated across 2 independent substrates
(MLX kiki-oniric + E-SNN thalamocortical).

## Implications for Paper 1 and Paper 2

### Paper 1 (cycle 1) update

The Discussion section (`docs/papers/paper1/discussion.md` §8.3
limitations) currently notes "Single-substrate validation" as a
cycle-1 caveat. Cycle 2 C2.4 closes this caveat. A v2 arXiv
preprint update (cycle 2 C2.12) will incorporate the cross-
substrate evidence.

### Paper 2 (cycle 2) foundational claim

Paper 2's engineering contribution centers on the Conformance
Criterion in practice (§4 per outline). The G7 LOCKED result is
the **central experimental evidence** for Paper 2. The conformance
matrix (C2.10) extends this to a multi-condition × multi-substrate
grid.

## Cross-references

- Framework spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  §6.2 DR-3 Conformance Criterion
- E-SNN substrate : `kiki_oniric/substrates/esnn_thalamocortical.py`
- Conformance test file : `tests/conformance/axioms/test_dr3_esnn_substrate.py`
- MLX substrate (cycle 1) : `kiki_oniric/substrates/mlx_kiki_oniric.py`
- Paper 1 Discussion §8.3 : `docs/papers/paper1/discussion.md`
- Paper 2 outline §4 : `docs/papers/paper2/outline.md`
