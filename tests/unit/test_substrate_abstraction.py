"""Unit tests for substrate abstraction (C2.1 cycle 2)."""
from __future__ import annotations

import importlib

from kiki_oniric.substrates import (
    MLX_SUBSTRATE_NAME,
    MLX_SUBSTRATE_VERSION,
    mlx_substrate_components,
)


def test_substrate_identity_constants() -> None:
    assert MLX_SUBSTRATE_NAME == "mlx_kiki_oniric"
    assert MLX_SUBSTRATE_VERSION.startswith("C-v")


def test_mlx_substrate_components_listed() -> None:
    components = mlx_substrate_components()
    expected_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
        "eval_retained", "ablation", "statistics",
    }
    assert set(components.keys()) == expected_keys


def test_all_mlx_substrate_modules_importable() -> None:
    """Every component path in the registry must be importable."""
    components = mlx_substrate_components()
    for name, dotted_path in components.items():
        try:
            importlib.import_module(dotted_path)
        except ImportError as exc:
            raise AssertionError(
                f"MLX substrate component {name!r} at "
                f"{dotted_path!r} not importable: {exc}"
            )
