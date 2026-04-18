"""S3 topology guard — validates ortho species topology integrity.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.2
Invariant S3 — BLOCKING. Enforced post-restructure (canal 3) before
swap commit.

Checks:
- All required ortho species present (rho_phono, rho_lex,
  rho_syntax, rho_sem)
- No self-loops (a node cannot point to itself)
- All edges target existing nodes (no dangling references)
- rho_sem reachable from rho_phono (full chain connectivity)
- Total layer count within DEFAULT_MAX_LAYERS bound
"""
from __future__ import annotations


REQUIRED_SPECIES: frozenset[str] = frozenset(
    {"rho_phono", "rho_lex", "rho_syntax", "rho_sem"}
)
DEFAULT_MAX_LAYERS: int = 64


class TopologyGuardError(Exception):
    """Raised when S3 invariant is violated."""


def _check_required_species(graph: dict[str, list[str]]) -> None:
    missing = REQUIRED_SPECIES - graph.keys()
    if missing:
        raise TopologyGuardError(
            f"missing required species: {sorted(missing)}"
        )


def _check_self_loops(graph: dict[str, list[str]]) -> None:
    for node, edges in graph.items():
        if node in edges:
            raise TopologyGuardError(
                f"self-loop detected on node {node!r}"
            )


def _check_dangling_edges(graph: dict[str, list[str]]) -> None:
    nodes = set(graph.keys())
    for node, edges in graph.items():
        for target in edges:
            if target not in nodes:
                raise TopologyGuardError(
                    f"edge {node!r} -> {target!r}: target node "
                    f"{target!r} does not exist"
                )


def _check_sem_reachable(graph: dict[str, list[str]]) -> None:
    """BFS from rho_phono — rho_sem must be reachable."""
    visited: set[str] = set()
    queue: list[str] = ["rho_phono"]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        queue.extend(graph.get(node, []))
    if "rho_sem" not in visited:
        raise TopologyGuardError(
            "rho_sem unreachable from rho_phono — chain broken"
        )


def _check_layer_count(
    graph: dict[str, list[str]], max_layers: int
) -> None:
    if len(graph) > max_layers:
        raise TopologyGuardError(
            f"layer count {len(graph)} exceeds bound {max_layers}"
        )


def validate_topology(
    graph: dict[str, list[str]],
    max_layers: int = DEFAULT_MAX_LAYERS,
) -> None:
    """Verify graph is a valid kiki ortho species topology.

    Raises TopologyGuardError on first violation with descriptive
    message. Order of checks: required species → self-loops →
    dangling edges → reachability → layer count.

    Designed to be called post-restructure (canal 3) before swap
    commit, mirroring the S2 finite check pattern from
    `guards/finite.py`.
    """
    _check_required_species(graph)
    _check_self_loops(graph)
    _check_dangling_edges(graph)
    _check_sem_reachable(graph)
    _check_layer_count(graph, max_layers)
