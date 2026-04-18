"""Ablation runner harness — (profile × seed × benchmark) matrix.

Executes the cartesian product of profile specifications and
seeds against a frozen benchmark, collecting metrics into a
pandas.DataFrame ready for S15.1 statistical tests.

Profile specs are intentionally minimal — they bind a name to a
predictor callable. Real ablation (S15.3) wraps PMin/PEqu profile
inference into predictors. Tests use mock predictors directly.

Reference: docs/specs/2026-04-17-dreamofkiki-master-design.md §5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from harness.benchmarks.retained.retained import RetainedBenchmark
from kiki_oniric.dream.eval_retained import evaluate_retained


ItemPredictor = Callable[[dict], str]


@dataclass(frozen=True)
class ProfileSpec:
    """Binding of profile name to predictor callable."""

    name: str
    predictor: ItemPredictor


@dataclass
class AblationRunner:
    """Run (profile × seed) matrix on a frozen benchmark.

    Each cell calls `evaluate_retained(spec.predictor, benchmark)`
    and records a row in the output DataFrame. Seeds are recorded
    alongside the result for downstream statistical handling
    (e.g., paired tests across seeds).
    """

    profile_specs: list[ProfileSpec]
    seeds: list[int]
    benchmark: RetainedBenchmark

    def run(self) -> pd.DataFrame:
        """Execute the full grid and return results DataFrame."""
        rows: list[dict] = []
        for spec in self.profile_specs:
            for seed in self.seeds:
                acc = evaluate_retained(spec.predictor, self.benchmark)
                rows.append({
                    "profile": spec.name,
                    "seed": seed,
                    "accuracy": acc,
                    "benchmark_hash": self.benchmark.source_hash,
                })
        return pd.DataFrame(rows)
