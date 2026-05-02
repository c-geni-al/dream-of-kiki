"""Wake-Sleep Consolidated Learning baseline adapter (Paper 2 §7 row).

This module wires Alfarano et al. 2024 [IEEE TNNLS, arXiv
2401.08623] into the dreamOfkiki ablation matrix as a
**baseline-only** adapter — it is registered alongside the
DR-3-conformant substrates (MLX, E-SNN, micro-kiki) but it does
NOT implement the 4 dream operations (replay / downscale /
restructure / recombine). It only exposes the comparator
contract `evaluate_continual(seed, task_split) -> dict`, which
is the minimum surface the Paper 2 §7 results table needs.

Variant choice (set by `docs/superpowers/plans/
2026-05-02-wake-sleep-cl-ablation-baseline.md` Task 0.5) :
- `c` (default — this file) — published reference metrics from
  Alfarano et al. 2024 IEEE TNNLS Tables 2-3, frozen as a
  module-level constant. The seed argument is passed through but
  does not influence the numbers (they are reference values, not
  a re-run). Caveat documented in Paper 2 §6.4 style.
- `a` / `b` — re-run with seeded RNG ; signature unchanged. Swap
  the `_REFERENCE_METRICS_BY_TASKSPLIT` constant for a real
  Avalanche-driven training loop in those variants.

References :
- docs/papers/paper1/references.bib L454 (`alfarano2024wakesleep`)
- docs/papers/paper1/introduction.md L94, L108 (Paper 1 framing)
- docs/papers/paper2/architecture.md §5.8 (this file's role)
- docs/interfaces/eval-matrix.yaml `baselines.wake_sleep_cl`
- docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
  (DR-3 — the baseline is exempt from conformance)
"""
from __future__ import annotations

from dataclasses import dataclass

WAKE_SLEEP_BASELINE_NAME = "wake_sleep_cl_baseline"
WAKE_SLEEP_BASELINE_VERSION = "C-v0.12.0+PARTIAL"

# Variant-c reference metrics. Source : Alfarano et al. 2024 IEEE
# TNNLS Tables 2-3 (Split-FMNIST 5-task class-incremental).
#
# PLACEHOLDER — these numerical values MUST be cross-checked
# against the Alfarano paper PDF before publication. They are
# documented as published-reference placeholders in the Paper 2
# milestone dump and the §7.X table caption ; the discipline is
# "expose the placeholder so reviewers can re-derive it" rather
# than "ship a verified number we cannot defend".
_REFERENCE_METRICS_BY_TASKSPLIT: dict[str, dict[str, float]] = {
    "split_fmnist_5tasks": {
        "forgetting_rate": 0.082,
        "avg_accuracy": 0.847,
    },
}

_SUPPORTED_TASKSPLITS = frozenset(_REFERENCE_METRICS_BY_TASKSPLIT.keys())


@dataclass(frozen=True)
class WakeSleepCLBaseline:
    """Stub adapter for the Wake-Sleep CL baseline.

    Variant `c` (default) : returns frozen reference metrics from
    Alfarano 2024 Tables 2-3. The `seed` argument round-trips
    into the output for R1-style provenance but does NOT
    influence the numerical values — by design, see Paper 2 §6.4
    caveat style.
    """

    n_tasks: int = 5

    def evaluate_continual(
        self, *, seed: int, task_split: str
    ) -> dict[str, float | int | str]:
        """Return forgetting + avg-accuracy for the requested split.

        Raises ``ValueError`` if ``task_split`` is not one of the
        supported reference splits (currently
        ``split_fmnist_5tasks`` only ; variants a/b may extend this).
        """
        if task_split not in _SUPPORTED_TASKSPLITS:
            raise ValueError(
                f"task_split={task_split!r} unsupported. "
                f"Choose from {sorted(_SUPPORTED_TASKSPLITS)}"
            )
        ref = _REFERENCE_METRICS_BY_TASKSPLIT[task_split]
        return {
            "forgetting_rate": ref["forgetting_rate"],
            "avg_accuracy": ref["avg_accuracy"],
            "n_tasks": self.n_tasks,
            "seed": seed,
            "source": "published_reference_alfarano2024",
        }


def wake_sleep_substrate_components() -> dict[str, str]:
    """Return the canonical map of WS-CL baseline components.

    Note : NO op factories. The 4 dream operations are
    intentionally absent — this is a baseline, not a DR-3
    substrate. The conformance test in
    ``tests/conformance/test_baseline_registration.py`` pins
    this distinction.
    """
    return {
        "evaluate_continual": (
            "kiki_oniric.substrates.wake_sleep_cl_baseline."
            "WakeSleepCLBaseline.evaluate_continual"
        ),
        "predictor": (
            "kiki_oniric.substrates.wake_sleep_cl_baseline."
            "WakeSleepCLBaseline"
        ),
    }
