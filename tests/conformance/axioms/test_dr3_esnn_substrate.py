"""DR-3 Conformance Criterion — E-SNN substrate (C2.4).

Validates that E-SNN thalamocortical substrate satisfies the 3
conditions of DR-3 Conformance Criterion :
1. Signature typing : substrate exports the expected Protocol-
   compatible factories (callable + correct signatures)
2. Axiom property tests : DR-0/DR-1 reusable from MLX tests ;
   DR-3 itself trivially holds (by instantiation)
3. BLOCKING invariants enforceable : S2 finite guard + S3
   topology guard operate on E-SNN state representations

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import check_finite
from kiki_oniric.dream.guards.topology import validate_topology
from kiki_oniric.substrates.esnn_thalamocortical import (
    ESNN_SUBSTRATE_NAME,
    EsnnBackend,
    EsnnSubstrate,
    LIFState,
    esnn_substrate_components,
)


# ===== Condition 1: signature typing =====


def test_c1_esnn_exports_all_required_identity_constants() -> None:
    """E-SNN substrate module exports required identity."""
    assert ESNN_SUBSTRATE_NAME == "esnn_thalamocortical"
    assert isinstance(EsnnBackend.NORSE, EsnnBackend)
    assert isinstance(EsnnBackend.NXNET, EsnnBackend)


def test_c1_esnn_substrate_instantiable_with_backend() -> None:
    substrate_norse = EsnnSubstrate(backend=EsnnBackend.NORSE)
    substrate_nxnet = EsnnSubstrate(backend=EsnnBackend.NXNET)
    assert substrate_norse.backend == EsnnBackend.NORSE
    assert substrate_nxnet.backend == EsnnBackend.NXNET


def test_c1_esnn_provides_4_op_factory_methods() -> None:
    """All 4 ops return callable handlers."""
    substrate = EsnnSubstrate()
    assert callable(substrate.replay_handler_factory())
    assert callable(substrate.downscale_handler_factory())
    assert callable(substrate.restructure_handler_factory())
    assert callable(substrate.recombine_handler_factory())


def test_c1_esnn_substrate_components_registry_mirrors_mlx() -> None:
    """E-SNN component registry shares the MLX keys."""
    from kiki_oniric.substrates.mlx_kiki_oniric import (
        mlx_substrate_components,
    )
    esnn = esnn_substrate_components()
    mlx = mlx_substrate_components()
    # E-SNN might have a subset (eval modules are MLX-specific)
    # but the core substrate keys must be present in both
    core_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
    }
    assert core_keys <= set(esnn.keys())
    assert core_keys <= set(mlx.keys())


# ===== Condition 2: axiom property tests =====


def test_c2_esnn_replay_op_respects_dr0_accountability() -> None:
    """DR-0 : every execution produces observable output (spike rates).

    For E-SNN, replay with non-empty records returns a non-None
    numpy array (the spike-rate tensor). Empty records return zeros.
    This is the E-SNN analogue of DR-0 "every DE produces a log
    entry" — the log-entry analogue here is the spike-rate output.
    """
    substrate = EsnnSubstrate()
    replay = substrate.replay_handler_factory()

    empty_out = replay([], n_steps=5)
    assert isinstance(empty_out, np.ndarray)

    records = [{"input": [0.5, 0.7, 0.3]}]
    non_empty_out = replay(records, n_steps=10)
    assert isinstance(non_empty_out, np.ndarray)
    assert non_empty_out.shape == (3,)


def test_c2_esnn_downscale_non_idempotent() -> None:
    """DR-2 op-pair analysis : downscale is commutative but NOT
    idempotent (shrink_f ∘ shrink_f = f²). Must hold on E-SNN."""
    substrate = EsnnSubstrate()
    downscale = substrate.downscale_handler_factory()

    weights = np.array([0.8, 1.2, 0.5, 0.9])
    factor = 0.5
    once = downscale(weights, factor=factor)
    twice = downscale(once, factor=factor)

    # twice == weights * factor^2, NOT weights * factor
    np.testing.assert_allclose(twice, weights * (factor ** 2))
    np.testing.assert_allclose(once, weights * factor)
    # Not idempotent : twice != once
    assert not np.allclose(twice, once)


def test_c2_esnn_recombine_deterministic_with_seed() -> None:
    """DR-2 : same inputs + same seed → same output (R1 contract)."""
    substrate = EsnnSubstrate()
    recombine = substrate.recombine_handler_factory()

    latents = np.array([[0.8, 0.2, 0.5], [0.1, 0.9, 0.3]])
    out_a = recombine(latents, seed=42, n_steps=10)
    out_b = recombine(latents, seed=42, n_steps=10)
    np.testing.assert_array_equal(out_a, out_b)


# ===== Condition 3: BLOCKING invariants enforceable =====


def test_c3_s2_finite_guard_works_on_esnn_state() -> None:
    """S2 invariant check_finite accepts valid E-SNN spike rates
    and rejects NaN/Inf in the weight representation."""
    lif_state = LIFState(n_neurons=8)
    # LIF v and spikes initialize to zero — finite OK
    check_finite(lif_state.v)
    check_finite(lif_state.spikes.astype(float))

    # Inject NaN and verify rejection
    bad = lif_state.v.copy()
    bad[3] = float("nan")
    from kiki_oniric.dream.guards.finite import FiniteGuardError
    with pytest.raises(FiniteGuardError):
        check_finite(bad)


def test_c3_s3_topology_guard_works_on_esnn_topology() -> None:
    """S3 topology validator accepts the canonical ortho chain
    (substrate-agnostic) and rejects a self-loop."""
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)  # pass

    bad = {
        "rho_phono": ["rho_phono"],  # self-loop
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    from kiki_oniric.dream.guards.topology import TopologyGuardError
    with pytest.raises(TopologyGuardError):
        validate_topology(bad)
