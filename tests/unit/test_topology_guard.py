"""Unit tests for S3 topology guard (validate_topology)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.guards.topology import (
    DEFAULT_MAX_LAYERS,
    REQUIRED_SPECIES,
    TopologyGuardError,
    validate_topology,
)


def _ortho_chain() -> dict[str, list[str]]:
    """Canonical kiki ortho species chain: phono → lex → syntax → sem."""
    return {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }


def test_validate_topology_accepts_canonical_chain() -> None:
    validate_topology(_ortho_chain())  # No exception


def test_validate_topology_rejects_self_loop() -> None:
    graph = _ortho_chain()
    graph["rho_lex"] = ["rho_lex", "rho_syntax"]
    with pytest.raises(TopologyGuardError, match="self-loop"):
        validate_topology(graph)


def test_validate_topology_rejects_missing_species() -> None:
    graph = _ortho_chain()
    del graph["rho_syntax"]
    # rho_lex still points to rho_syntax (now dangling)
    with pytest.raises(TopologyGuardError, match="rho_syntax"):
        validate_topology(graph)


def test_validate_topology_rejects_disconnected_sem() -> None:
    graph = {
        "rho_phono": ["rho_lex"],
        "rho_lex": [],  # broken: no edge to syntax
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    with pytest.raises(TopologyGuardError, match="unreachable"):
        validate_topology(graph)


def test_validate_topology_rejects_too_many_layers() -> None:
    graph = _ortho_chain()
    for i in range(DEFAULT_MAX_LAYERS + 5):
        graph[f"extra_layer_{i}"] = []
    with pytest.raises(TopologyGuardError, match="layer count"):
        validate_topology(graph)


def test_required_species_constant_matches_kiki_ortho() -> None:
    assert REQUIRED_SPECIES == frozenset(
        {"rho_phono", "rho_lex", "rho_syntax", "rho_sem"}
    )
