"""Conformance test for invariant S3 — topology valid post-swap.

Spec : invariant S3 (BLOCKING) defined in
`docs/invariants/registry.md` and
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §5.2.

The current `HierarchyChangeChannel` is a typed `Protocol` with
no concrete ortho-graph implementation in cycle 1 ; the
post-restructure invariant is therefore exercised through
`validate_topology` directly, which is what the production swap
path calls before commit.
"""
from __future__ import annotations

import pytest

from kiki_oniric.dream.guards.topology import (
    TopologyGuardError,
    validate_topology,
)


def test_s3_invariant_blocks_invalid_post_topology() -> None:
    """S3 must abort swap when post-restructure topology is invalid.

    Spec : S3 §5.2 (BLOCKING). Surfaces via `TopologyGuardError`.
    """
    bad_graph = {
        "rho_phono": ["rho_phono"],  # self-loop
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    with pytest.raises(TopologyGuardError):
        validate_topology(bad_graph)


def test_s3_invariant_passes_canonical_topology() -> None:
    """S3 should pass canonical kiki ortho chain silently.

    Spec : S3 §5.2 (canonical species chain phono→lex→syntax→sem).
    """
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)
