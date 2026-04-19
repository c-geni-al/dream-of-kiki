"""MLX kiki-oniric substrate (cycle-1 canonical implementation).

This module is a **namespace marker** for cycle 2. Existing cycle-1
code is not moved — it stays at its current locations under
`kiki_oniric/{core, dream, profiles, eval}/`. This module
documents which components belong to the MLX substrate and
provides convenient re-exports.

Cycle 2 adds `kiki_oniric.substrates.esnn_thalamocortical` as a
sibling, also implementing the framework C primitives and
profiles per the DR-3 Conformance Criterion.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion : signature typing + axiom
property tests passing + BLOCKING invariants enforceable).
"""
from __future__ import annotations

# Substrate identity
MLX_SUBSTRATE_NAME = "mlx_kiki_oniric"
# DualVer empirical axis `+PARTIAL` : the substrate is green against
# the current framework-C (DR-3 conformance all three conditions pass
# on MLX, all BLOCKING invariants S1/S2/S3/I1 enforced, 240 tests
# passing ≥90% coverage). Bumped C-v0.6.0+STABLE → C-v0.7.0+PARTIAL
# on 2026-04-19 at cycle-3 Phase 1 launch : FC MINOR (+0.1.0) adds
# H6 profile-ordering derived constraint + scale-axis per
# framework-C §12.2 ; EC demoted STABLE → PARTIAL per §12.3 because
# cycle-3 Phase 2 cells (Norse LIF + fMRI alignment + Paper 1 v2)
# are scoped-deferred. See CHANGELOG.md [C-v0.7.0+PARTIAL] for the
# bump log and STATUS.md for gate G10 CONDITIONAL-GO/PARTIAL status.
MLX_SUBSTRATE_VERSION = "C-v0.7.0+PARTIAL"


def mlx_substrate_components() -> dict[str, str]:
    """Return the canonical map of MLX substrate components.

    Each value is the dotted path to the cycle-1 implementation
    of the named primitive / operation / profile / guard. Cycle-2
    E-SNN substrate provides the same API at sibling paths.
    """
    return {
        # 8 typed Protocols (substrate-agnostic, defined in core)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations (skeleton + MLX backend per file)
        "replay": "kiki_oniric.dream.operations.replay",
        "downscale": "kiki_oniric.dream.operations.downscale",
        "restructure": "kiki_oniric.dream.operations.restructure",
        "recombine": "kiki_oniric.dream.operations.recombine",
        # 3 invariant guards
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
        # Evaluation harness
        "eval_retained": "kiki_oniric.dream.eval_retained",
        "ablation": "kiki_oniric.eval.ablation",
        "statistics": "kiki_oniric.eval.statistics",
    }
