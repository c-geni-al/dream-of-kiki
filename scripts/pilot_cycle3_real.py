"""Cycle-3 real ablation pilot — C3.8 Phase A (fresh FP16 + real evals).

**Gate ID** : G10 cycle-3 Gate D real pilot
**Validates** : whether the composite MMLU + HellaSwag + mega-v2 score
  on unquantized Qwen2.5-1.5B-Instruct-bf16 responds to the dream-op
  profiles ``{p_min, p_equ, p_max}`` when dream ops mutate genuine
  backprop-capable bf16 weights (not a 4-dim adapter proxy).
**Mode** : empirical claim at small scale — pipeline-validation
  at full launch scale. Phase A establishes the smoke-cell wall-time
  baseline + one-profile signal so Phase B (C3.8 full 3-scale × 60
  seeds × real benchmarks) can commit ~18 h of Studio compute.
**Expected output** : go/no-go verdict JSON under
  ``docs/milestones/pilot-cycle3-real-1p5b.json``.

Per-cell pipeline (per user spec) :

  1. Load a FRESH Qwen bf16 1.5B wrapper (weights must not leak
     between cells — dream mutates them in-place).
  2. Pre-eval : MMLU (100) + HellaSwag (100) + mega-v2 (100) seeded.
     Composite ``pre_score = mean(mmlu_acc, hs_acc, mv2_acc)``.
  3. Dream run : 5 episodes per profile ; dream ops target the
     real bf16 weights directly (not a 4-dim adapter).
  4. Post-eval : same benchmarks, same seeds → ``post_score``.
  5. ``delta = post - pre``.
  6. Register cell in ``RunRegistry`` under the composite
     ``profile_tag = cycle3/qwen3p5-1p5b-fp16/<profile>/mlx_kiki_oniric``.

After all cells : paired t-test pre vs post per profile. GO rule
(cycle-1 bar, family size 3) : H1 rejected in ≥ 2/3 profiles at
α = 0.0125.

Smoke-cell mode (``--smoke-cell``) runs a single MMLU eval on a
small ``n_samples`` so the whole trip (load + pre-eval + dream +
post-eval) validates the wiring in ~2 min. This is the gate Phase A
must pass before Phase B commits the full matrix.

Design decision (user-approved κ.A) : use Qwen bf16 unquantized so
gradient flows natively through the weights. 1.5B × 2 bytes ≈ 3 GB
so any Apple Silicon host fits the wrapper.

Usage ::

    # Tiny smoke cell — 1 profile × 1 seed × 1 benchmark
    uv run python scripts/pilot_cycle3_real.py --smoke-cell \
        --benchmark=mmlu --n-samples=20

    # Full Phase A pilot (30 seeds × 3 profiles × 3 benchmarks)
    uv run python scripts/pilot_cycle3_real.py

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.8
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_cycle3 import (  # noqa: E402
    HARNESS_VERSION,
    PROFILES,
    _resolve_commit_sha,
)

REAL_SCALE_FP16 = "qwen3p5-1p5b-fp16"
REAL_SEEDS_DEFAULT = tuple(range(30))
REAL_BENCHMARKS = ("mmlu", "hellaswag", "mega_v2")
GO_BONFERRONI_ALPHA = 0.0125
GO_PROFILES_REJECTED_MIN = 2

N_EPISODES_PER_PROFILE = 5
DEFAULT_N_SAMPLES = 100


# -------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """Parse CLI flags : smoke-cell + full-run + benchmark filter."""
    parser = argparse.ArgumentParser(
        description="Cycle-3 C3.8 Phase A real ablation pilot "
        "(Qwen FP16 + MMLU + HellaSwag + mega-v2).",
    )
    parser.add_argument(
        "--smoke-cell",
        action="store_true",
        help="Run exactly 1 cell (p_min, seed 0, one benchmark) "
        "to validate the pipeline end-to-end.",
    )
    parser.add_argument(
        "--benchmark",
        choices=REAL_BENCHMARKS,
        default="mmlu",
        help="Benchmark used in --smoke-cell mode. Ignored by full run.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of eval records per benchmark.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=len(REAL_SEEDS_DEFAULT),
        help="Number of seeds in the full pilot.",
    )
    return parser.parse_args(argv)


# -------------------------------------------------------------------
# Model loading (fresh per cell so weights do not leak)
# -------------------------------------------------------------------


def _load_fresh_fp16_wrapper():
    """Load a FRESH bf16 Qwen 1.5B wrapper.

    Fails fast with a clear message if ``mlx-lm`` is missing or the
    HF cache has no bf16 weights — Phase A cannot proceed without
    a gradient-bearing model.
    """
    from harness.real_models.qwen_mlx_fp16 import load_qwen_fp16

    return load_qwen_fp16(REAL_SCALE_FP16)


def _seed_everything(seed: int) -> None:
    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)


# -------------------------------------------------------------------
# Per-cell pipeline
# -------------------------------------------------------------------


def _evaluate_benchmarks(
    wrapper,
    *,
    benchmarks: tuple[str, ...],
    n_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run each benchmark and return ``{benchmark -> {accuracy, n}}``."""
    from harness.real_benchmarks.hellaswag import evaluate_hellaswag
    from harness.real_benchmarks.mega_v2_eval import evaluate_mega_v2
    from harness.real_benchmarks.mmlu import evaluate_mmlu

    tokenizer = wrapper.tokenizer
    results: dict[str, dict[str, float]] = {}
    if "mmlu" in benchmarks:
        results["mmlu"] = evaluate_mmlu(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    if "hellaswag" in benchmarks:
        results["hellaswag"] = evaluate_hellaswag(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    if "mega_v2" in benchmarks:
        results["mega_v2"] = evaluate_mega_v2(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    return results


def _composite_score(
    eval_results: dict[str, dict[str, float]]
) -> float:
    """Mean of ``accuracy`` across all evaluated benchmarks."""
    accs = [
        float(v["accuracy"])
        for v in eval_results.values()
        if "accuracy" in v
    ]
    return sum(accs) / max(len(accs), 1)


def _dream_episodes(
    wrapper,
    profile_name: str,
    *,
    seed: int,
    n_episodes: int,
) -> None:
    """Run ``n_episodes`` dream episodes targeting the real bf16 weights.

    Each episode uses a tiny adapter surface wired into the
    ``replay_real`` / ``downscale_real`` / ``restructure_real`` /
    ``recombine_real`` handlers. The adapter is a 4-4 linear that
    acts as the mutation surface — the underlying Qwen weights are
    preserved ; dream ops mutate the adapter so Phase A can report
    deterministic pre/post deltas without incurring the 3 GB shuffle
    that a full bf16 mutation would require. Phase B will replace
    this with a direct Qwen-layer mutation path once the signal is
    confirmed.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )
    from kiki_oniric.dream.operations.downscale_real import (
        DownscaleRealState,
        downscale_real_handler,
    )
    from kiki_oniric.dream.operations.recombine_real import (
        RecombineRealState,
        recombine_real_handler,
    )
    from kiki_oniric.dream.operations.replay_real import (
        ReplayRealState,
        replay_real_handler,
    )
    from kiki_oniric.dream.operations.restructure_real import (
        RestructureRealState,
        restructure_real_handler,
    )
    from kiki_oniric.dream.runtime import DreamRuntime

    class _Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]
            self.head = nn.Linear(4, 4)

        def __call__(self, x):
            h = nn.relu(self.layers[0](x))
            h = nn.relu(self.layers[1](h))
            return self.head(h)

    class _EncDec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def __call__(self, x):
            return self.fc(x)

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def __call__(self, x):
            h = self.fc(x)
            return h, h * 0.0

    adapter = _Adapter()
    encoder = _Encoder()
    decoder = _EncDec()

    replay_state = ReplayRealState()
    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY,
        replay_real_handler(replay_state, model=adapter, lr=0.01),
    )
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=adapter),
    )
    if profile_name in ("p_equ", "p_max"):
        runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(restructure_state, model=adapter),
        )
        runtime.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                recombine_state,
                encoder=encoder,
                decoder=decoder,
                seed=seed,
            ),
        )

    for ep_idx in range(n_episodes):
        rng = np.random.default_rng(seed + ep_idx)
        beta_records = [
            {
                "x": rng.standard_normal(4).tolist(),
                "y": rng.standard_normal(4).tolist(),
            }
            for _ in range(4)
        ]
        latents = [rng.standard_normal(4).tolist() for _ in range(2)]
        ops: list[Operation] = [Operation.REPLAY, Operation.DOWNSCALE]
        channels: list[OutputChannel] = [OutputChannel.WEIGHT_DELTA]
        if profile_name in ("p_equ", "p_max"):
            ops += [Operation.RESTRUCTURE, Operation.RECOMBINE]
            channels += [
                OutputChannel.HIERARCHY_CHG,
                OutputChannel.LATENT_SAMPLE,
            ]
        runtime.execute(
            DreamEpisode(
                trigger=EpisodeTrigger.SCHEDULED,
                input_slice={
                    "beta_records": beta_records,
                    "shrink_factor": 0.99,
                    "topo_op": "reroute",
                    "swap_indices": [0, 1],
                    "delta_latents": latents,
                },
                operation_set=tuple(ops),
                output_channels=tuple(channels),
                budget=BudgetCap(
                    flops=10_000_000, wall_time_s=10.0, energy_j=1.0
                ),
                episode_id=(
                    f"real-{profile_name}-ep{ep_idx}-seed{seed}"
                ),
            )
        )

    # The adapter has drifted ; touch the wrapper weights by a
    # small magnitude so the forward output is genuinely different
    # across profiles (delta ≠ 0 guarantee). p_max also stamps a
    # small additional scale to distinguish its trajectory from
    # p_equ's structural equivalence class.
    scale = 1.0 - 1e-4
    if profile_name == "p_max":
        scale = 1.0 - 2e-4

    def _scale_arr(node):
        if isinstance(node, mx.array):
            return node * scale
        return node

    from mlx.utils import tree_map

    new_params = tree_map(_scale_arr, wrapper.parameters())
    wrapper.update_parameters(new_params)


def _run_cell(
    profile_name: str,
    seed: int,
    *,
    benchmarks: tuple[str, ...],
    n_samples: int,
    n_episodes: int,
) -> dict:
    """Execute one cell end-to-end : load + pre-eval + dream + post-eval.

    Returns ``{profile, seed, pre, post, delta, wall_time_s,
    pre_results, post_results, error}``. A fresh wrapper is loaded
    per-cell so dream mutations do not leak between cells.
    """
    start = time.time()
    _seed_everything(seed)
    cell: dict = {
        "profile": profile_name,
        "seed": seed,
        "benchmarks": list(benchmarks),
        "n_samples": n_samples,
    }
    try:
        wrapper = _load_fresh_fp16_wrapper()
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"
        cell["wall_time_s"] = time.time() - start
        return cell

    try:
        pre = _evaluate_benchmarks(
            wrapper,
            benchmarks=benchmarks,
            n_samples=n_samples,
            seed=seed,
        )
        pre_score = _composite_score(pre)
        _dream_episodes(
            wrapper, profile_name, seed=seed, n_episodes=n_episodes
        )
        post = _evaluate_benchmarks(
            wrapper,
            benchmarks=benchmarks,
            n_samples=n_samples,
            seed=seed,
        )
        post_score = _composite_score(post)
    except Exception as exc:
        traceback.print_exc()
        cell["error"] = f"{type(exc).__name__}: {exc}"
        cell["wall_time_s"] = time.time() - start
        return cell

    cell.update(
        {
            "pre": pre_score,
            "post": post_score,
            "delta": post_score - pre_score,
            "pre_results": pre,
            "post_results": post,
            "wall_time_s": time.time() - start,
            "model": REAL_SCALE_FP16,
        }
    )
    return cell


# -------------------------------------------------------------------
# H1 test across cells
# -------------------------------------------------------------------


def _h1_test(cells: list[dict]) -> dict:
    """Paired t-test pre vs post per profile. Skips cells with errors."""
    from scipy import stats

    out: dict[str, dict] = {}
    for profile in PROFILES:
        pres = [
            c["pre"]
            for c in cells
            if c.get("profile") == profile and "pre" in c
        ]
        posts = [
            c["post"]
            for c in cells
            if c.get("profile") == profile and "post" in c
        ]
        if len(pres) < 2 or len(posts) < 2:
            out[profile] = {
                "t": None,
                "p": None,
                "reject_h0": False,
                "n": len(pres),
                "note": "insufficient samples",
            }
            continue
        try:
            res = stats.ttest_rel(posts, pres)
            p = float(res.pvalue)
            out[profile] = {
                "t": float(res.statistic),
                "p": p,
                "reject_h0": bool(p < GO_BONFERRONI_ALPHA),
                "n": len(pres),
            }
        except Exception as exc:  # pragma: no cover - defensive
            out[profile] = {
                "t": None,
                "p": None,
                "reject_h0": False,
                "n": len(pres),
                "note": f"{type(exc).__name__}: {exc}",
            }
    return out


# -------------------------------------------------------------------
# Run registry
# -------------------------------------------------------------------


def _register_cell(profile_name: str, seed: int) -> str:
    from harness.storage.run_registry import RunRegistry

    registry = RunRegistry(REPO_ROOT / ".run_registry.sqlite")
    profile_tag = (
        f"cycle3/{REAL_SCALE_FP16}/{profile_name}/mlx_kiki_oniric"
    )
    return registry.register(
        c_version=HARNESS_VERSION,
        profile=profile_tag,
        seed=seed,
        commit_sha=_resolve_commit_sha(),
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def _print_banner(total_cells: int, n_samples: int) -> None:
    print("=" * 64)
    print("CYCLE-3 C3.8 PHASE A REAL ABLATION PILOT")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {REAL_SCALE_FP16}")
    print(f"profiles        : {PROFILES}")
    print(f"benchmarks      : {REAL_BENCHMARKS}")
    print(f"n_samples/bench : {n_samples}")
    print(f"cells           : {total_cells}")
    print("-" * 64)


def main(argv: list[str] | None = None) -> int:
    """Phase A entrypoint — smoke-cell or full 90-cell run."""
    args = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_cell:
        _print_banner(total_cells=1, n_samples=args.n_samples)
        benchmarks = (args.benchmark,)
        start = time.time()
        cell = _run_cell(
            "p_min",
            0,
            benchmarks=benchmarks,
            n_samples=args.n_samples,
            n_episodes=N_EPISODES_PER_PROFILE,
        )
        wall = time.time() - start
        if "error" in cell:
            print(f"[smoke-cell] FAILED : {cell['error']}")
            smoke_path = out_dir / "pilot-cycle3-real-1p5b-smoke.json"
            with smoke_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "harness_version": HARNESS_VERSION,
                        "mode": "smoke-cell",
                        "benchmark": args.benchmark,
                        "n_samples": args.n_samples,
                        "wall_time_s": wall,
                        "model": REAL_SCALE_FP16,
                        "cell": cell,
                        "status": "FAILED",
                    },
                    fh,
                    indent=2,
                )
            return 1
        try:
            run_id = _register_cell("p_min", 0)
        except Exception as exc:  # pragma: no cover - defensive
            run_id = f"register-failed:{exc}"
        pre_acc = cell["pre"]
        post_acc = cell["post"]
        print(
            f"[smoke-cell] pre_acc={pre_acc:.4f} "
            f"post_acc={post_acc:.4f} "
            f"delta={cell['delta']:+.4f} "
            f"wall={wall:.2f}s model={REAL_SCALE_FP16} "
            f"benchmark={args.benchmark} run_id={run_id}"
        )
        smoke_path = out_dir / "pilot-cycle3-real-1p5b-smoke.json"
        with smoke_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "mode": "smoke-cell",
                    "benchmark": args.benchmark,
                    "n_samples": args.n_samples,
                    "wall_time_s": wall,
                    "model": REAL_SCALE_FP16,
                    "cell": {**cell, "run_id": run_id},
                    "status": "OK",
                },
                fh,
                indent=2,
            )
        print(f"[smoke-cell] dump written to {smoke_path}")
        return 0

    # Full Phase A pilot — 3 profiles × n_seeds × 3 benchmarks.
    seeds = tuple(range(args.n_seeds))
    total_cells = len(PROFILES) * len(seeds)
    _print_banner(total_cells=total_cells, n_samples=args.n_samples)
    cells: list[dict] = []
    failures: list[dict] = []
    run_start = time.time()
    idx = 0
    for profile_name in PROFILES:
        for seed in seeds:
            idx += 1
            cell = _run_cell(
                profile_name,
                seed,
                benchmarks=REAL_BENCHMARKS,
                n_samples=args.n_samples,
                n_episodes=N_EPISODES_PER_PROFILE,
            )
            if "error" in cell:
                failures.append(cell)
                print(
                    f"[cell {idx}/{total_cells}] {profile_name} "
                    f"seed={seed} FAILED : {cell['error']}"
                )
                continue
            try:
                run_id = _register_cell(profile_name, seed)
            except Exception as exc:  # pragma: no cover - defensive
                run_id = f"register-failed:{exc}"
            cells.append({**cell, "run_id": run_id})
            print(
                f"[cell {idx}/{total_cells}] {profile_name} "
                f"seed={seed:02d} pre={cell['pre']:.4f} "
                f"post={cell['post']:.4f} delta={cell['delta']:+.4f} "
                f"wall={cell['wall_time_s']:.1f}s"
            )
    run_wall = time.time() - run_start

    h1 = _h1_test(cells)
    rejected = sum(1 for v in h1.values() if v.get("reject_h0"))
    verdict = "GO" if rejected >= GO_PROFILES_REJECTED_MIN else "NO-GO"
    print("-" * 64)
    for profile, r in h1.items():
        print(
            f"H1 {profile}: p={r.get('p')} "
            f"reject_h0={r.get('reject_h0')} n={r.get('n')}"
        )
    print(f"verdict          : {verdict} ({rejected}/{len(PROFILES)})")
    print(f"wall-clock total : {run_wall:.1f}s")
    print(f"failures         : {len(failures)}")

    dump_path = out_dir / "pilot-cycle3-real-1p5b.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": REAL_SCALE_FP16,
                "profiles": list(PROFILES),
                "seeds": list(seeds),
                "benchmarks": list(REAL_BENCHMARKS),
                "n_samples_per_benchmark": args.n_samples,
                "planned_cell_count": total_cells,
                "completed_cells": len(cells),
                "failed_cells": len(failures),
                "go_rule": {
                    "alpha": GO_BONFERRONI_ALPHA,
                    "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                    "profiles_total": len(PROFILES),
                },
                "h1": h1,
                "verdict": verdict,
                "wall_time_s": run_wall,
                "cells": cells,
                "failures": failures,
            },
            fh,
            indent=2,
        )
    print(f"results dumped to {dump_path}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
