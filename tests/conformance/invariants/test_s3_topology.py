"""Conformance test for invariant S3 — topology valid post-swap."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.guards.topology import (
    TopologyGuardError,
    validate_topology,
)


def test_s3_invariant_blocks_invalid_post_topology() -> None:
    """S3 must abort swap when post-restructure topology is invalid."""
    bad_graph = {
        "rho_phono": ["rho_phono"],  # self-loop
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    with pytest.raises(TopologyGuardError):
        validate_topology(bad_graph)


def test_s3_invariant_passes_canonical_topology() -> None:
    """S3 should pass canonical kiki ortho chain silently."""
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)
