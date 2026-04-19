"""Real-model wrappers and base-model registry for cycle-3.

Exposes :

- :mod:`harness.real_models.base_model_registry` — SHA-pinned base
  model entries (Qwen2.5 MLX Q4 fallback for the Qwen3.5 slot
  per cycle-3 spec §5 pre-cycle-3 lock #2).

Future cycle-3 C3.2 deliverable : ``qwen_mlx.py`` wrappers that
consume these pins for deterministic weight loading on Studio.
"""
from __future__ import annotations

from harness.real_models.base_model_registry import (
    REGISTRY,
    BaseModelPin,
    get_pin,
    verify_all,
)

__all__ = [
    "REGISTRY",
    "BaseModelPin",
    "get_pin",
    "verify_all",
]
