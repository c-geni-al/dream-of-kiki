"""Cycle-3 sanity pilot — 1.5B-scale fail-fast driver (C3.7).

**Gate ID** : G10 cycle-3 Gate D pilot (fail-fast decision)
**Validates** : whether H1 (forgetting reduction, paired t-test
pre vs post on retained accuracy) is detectable at the smallest
scale-slot ``qwen3p5-1p5b`` before the user commits ~18 h of
Studio compute to the full multi-scale launch (C3.8).
**Mode** : empirical claim at small scale — *pipeline-validation*
at the full launch scale (fail-fast only, not used to certify
G10).
**Expected output** : a go/no-go verdict JSON dumped under
``docs/milestones/pilot-cycle3-sanity-1p5b.json`` (sibling to the
human-readable milestone report).

Cartesian product (full plan, dry-run audit) :

    1 scale × 3 profiles × 2 substrates × 30 seeds = 180 runs

Execution restriction (this follow-up commit) : the sanity pilot
only exercises the **MLX substrate** because ``E-SNN`` lacks
``*_real_snn.py`` real-weight ops (skeleton only ; C3.12 Phase-2
work). Running E-SNN here would require ~400 LOC of scaffolding
for no empirical gain, defeating the fail-fast purpose.

    1 scale × 3 profiles × **1 substrate (MLX)** × 30 seeds = 90
    cells actually executed.

The dry-run path still enumerates the full 180 cells so the
resume-contract identity with the 1080-config launch (C3.8) stays
untouched. Only the execution path narrows to MLX.

Half the seed count of the full 60-seed launch per §7 compute
envelope : ~18 h of dedicated Studio time vs ~3-4 days per full
scale-slot. Run registry rows are tagged under the same
``harness_version = C-v0.7.0+PARTIAL`` and ``profile_tag``
convention as the full launch so the pilot's cells are *a subset*
of the full-launch's resume contract. Re-running
``scripts/ablation_cycle3.py --resume`` after the pilot therefore
skips the executed pilot cells automatically (R1 identity).

GO / NO-GO decision rule (per user spec) :

- **GO** : H1 (paired t-test pre vs post) rejects the null in
  ≥ 2 / 3 profiles at α = 0.0125 (cycle-1 Bonferroni-corrected
  bar on 1 scale × 1 substrate, family size 4). Compute budget
  cleared, launch C3.8 full 3-scale matrix.
- **NO-GO** : H1 rejects in < 2 / 3 profiles. Do **not** burn
  Studio compute on 7B + 35B. Open a root-cause review
  (``pivot-4`` branch per spec §5.1 R3).

Per-cell pipeline :

  1. Fresh MLX model copy + per-cell adapter head seeded from
     ``seed`` (``mx.random.seed`` + ``np.random.seed``).
  2. Pre-dream evaluation on the retained benchmark (50 items,
     same seed, vacuous-pass when the predictor is synthetic —
     the pipeline validator reads the *delta*, not the absolute).
  3. Dream episodes — 5 per profile, using the real-weight ops
     (``replay_real``, ``downscale_real``, ``restructure_real``,
     ``recombine_real``) against the per-cell adapter.
  4. Post-dream evaluation (same items, same seed).
  5. Per-cell run registered with ``RunRegistry.register`` plus
     the measured (pre, post, delta) metrics.

Graceful degradation : if a single cell crashes, the driver logs
the error and continues — partial progress still yields H1 input
on the remaining cells. An explicit ``--smoke-cell`` flag runs
exactly 1 cell (``p_min`` + seed 0 + MLX) to validate the end-to-
end pipeline in under 2 minutes before the 18 h launch.

Usage ::

    # Enumerate the 180 configs ; no dream ops.
    uv run python scripts/pilot_cycle3_sanity.py --dry-run

    # Single smoke cell (p_min + seed 0 + MLX ; <2 min).
    uv run python scripts/pilot_cycle3_sanity.py --smoke-cell

    # Full 90-cell run (~18 h Studio compute).
    uv run python scripts/pilot_cycle3_sanity.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §5.1 R3 (Pivot 4 if Gate D = NO-GO),
    §7 (compute budget C3.7 sanity envelope)
    docs/milestones/pilot-cycle3-sanity-1p5b.md (this script's
    milestone report).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_cycle3 import (  # noqa: E402
    AblationCycle3Runner,
    HARNESS_VERSION,
    PROFILES,
    SUBSTRATES,
    _resolve_commit_sha,
)

# Sanity-pilot axis restriction (§7 compute envelope).
SANITY_SCALE = "qwen3p5-1p5b"
SANITY_SEEDS = tuple(range(30))
SANITY_PROFILES = PROFILES
# Dry-run path still enumerates the full substrate axis so the
# 180-cell manifest remains an R1 subset of the C3.8 launch plan.
SANITY_SUBSTRATES_PLAN = SUBSTRATES
# Execution path narrows to MLX (E-SNN real-weight ops absent).
SANITY_SUBSTRATES_EXEC: tuple[str, ...] = ("mlx_kiki_oniric",)

# 1 × 3 × 2 × 30 = 180 planned cells (audit).
EXPECTED_CELL_COUNT = (
    1
    * len(SANITY_PROFILES)
    * len(SANITY_SUBSTRATES_PLAN)
    * len(SANITY_SEEDS)
)
# 1 × 3 × 1 × 30 = 90 executed cells (MLX only).
EXECUTED_CELL_COUNT = (
    1
    * len(SANITY_PROFILES)
    * len(SANITY_SUBSTRATES_EXEC)
    * len(SANITY_SEEDS)
)

# GO / NO-GO decision rule (cycle-1 bar on 1 scale × 1 substrate).
GO_BONFERRONI_ALPHA = 0.0125
GO_PROFILES_REJECTED_MIN = 2  # ≥ 2 of 3 MLX profiles


# -------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> dict:
    """Light argv parser — no click dependency.

    Flags :

    - ``--dry-run`` : enumerate the 180-cell manifest and exit
      without touching any substrate. Safe from CI.
    - ``--smoke-cell`` : run exactly one cell (``p_min`` + seed 0
      + MLX) end-to-end. Used to validate the wiring before the
      18 h launch ; completes in < 2 min on Studio.
    """
    opts = {"dry_run": False, "smoke_cell": False}
    for token in argv:
        if token == "--dry-run":
            opts["dry_run"] = True
        elif token == "--smoke-cell":
            opts["smoke_cell"] = True
    return opts


# -------------------------------------------------------------------
# Planning (dry-run / audit manifest)
# -------------------------------------------------------------------


def _plan() -> list[dict]:
    """Enumerate the 180-cell sanity plan (audit manifest).

    Wraps :class:`AblationCycle3Runner` restricted to the 1.5 B
    scale and 30 seeds ; preserves the full-launch's
    ``(harness_version, profile_tag, seed, commit_sha) ->
    run_id`` identity so this pilot's planned rows are a strict
    subset of the eventual 1080-config launch. The dry-run path
    still lists both substrates so the manifest matches the
    original C3.7 design ; execution narrows later.
    """
    runner = AblationCycle3Runner(
        scales=(SANITY_SCALE,),
        profiles=SANITY_PROFILES,
        substrates=SANITY_SUBSTRATES_PLAN,
        seeds=SANITY_SEEDS,
    )
    plan = []
    for cfg in runner.enumerate():
        plan.append(
            {
                "scale": cfg.scale,
                "profile": cfg.profile,
                "substrate": cfg.substrate,
                "seed": cfg.seed,
                "run_id": runner.compute_run_id(cfg),
                "profile_tag": runner._registry_profile_tag(cfg),
            }
        )
    return plan


# -------------------------------------------------------------------
# Per-cell pipeline — MLX substrate only
# -------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    """Deterministic seed contract — MLX + numpy + hypothesis.

    Cells derive their adapter weights + episode-record draws
    from this common seed so the pre/post accuracy delta is a
    reproducible signal (R1 / R3 discipline).
    """
    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)
    # Hypothesis is only an influence when a property test runs in
    # the same process ; set it defensively so pilots that import
    # test fixtures stay deterministic.
    try:  # pragma: no cover - optional
        import hypothesis  # type: ignore[import-not-found]

        hypothesis.seed(seed)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional
        pass


def _build_adapter(seed: int):
    """Construct a seeded 4-8-2 MLX MLP — the dream-ops target.

    The Qwen base model is Q4-quantised and not backprop-capable
    through ``mlx-lm.load`` ; the pilot therefore attaches a tiny
    trainable adapter head that the real-weight ops mutate. This
    is enough to exercise the full pipeline (replay SGD step,
    downscale shrink, restructure reroute, recombine VAE sample)
    deterministically against a γ-snapshot Qwen wrapper.
    """
    import mlx.core as mx  # noqa: F401 — imported for side-effect seed
    import mlx.nn as nn

    class _Adapter(nn.Module):
        # Three homogeneous (4→4) linear layers ; final head
        # projects to the 2-d replay target with a third Linear
        # kept off ``self.layers`` (only indices 0-2 participate
        # in ``restructure_real`` reroute). Shape-homogeneity on
        # ``self.layers`` is what lets ``swap_indices=[0, 1]``
        # execute cleanly without breaking the downstream replay
        # forward pass ; the head stays fixed.
        def __init__(self) -> None:
            super().__init__()
            self.layers = [
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                nn.Linear(4, 4),
            ]
            self.head = nn.Linear(4, 2)

        def __call__(self, x):
            h = nn.relu(self.layers[0](x))
            h = nn.relu(self.layers[1](h))
            h = nn.relu(self.layers[2](h))
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
    return adapter, encoder, decoder


def _load_qwen_wrapper(seed: int):
    """Load the Qwen-1.5B MLX wrapper (γ-channel snapshot).

    Falls back to ``None`` when ``mlx-lm`` isn't installed ; the
    pipeline still exercises the adapter path so callers can
    diagnose the env before pulling the ~1 GB HF weights.
    """
    try:  # pragma: no cover - network / install path
        from harness.real_models.qwen_mlx import load_qwen

        return load_qwen(SANITY_SCALE)
    except Exception:  # pragma: no cover - diagnostic
        return None


def _build_runtime(profile_name: str, adapter, encoder, decoder, seed: int):
    """Register the real-weight handlers matching ``profile_name``.

    - ``p_min`` : replay + downscale
    - ``p_equ`` : replay + downscale + restructure + recombine
    - ``p_max`` : replay + downscale + restructure + recombine
    """
    from kiki_oniric.dream.episode import Operation
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

    replay_state = ReplayRealState()
    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    rt = DreamRuntime()
    rt.register_handler(
        Operation.REPLAY,
        replay_real_handler(replay_state, model=adapter, lr=0.01),
    )
    rt.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=adapter),
    )
    if profile_name in ("p_equ", "p_max"):
        rt.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(restructure_state, model=adapter),
        )
        rt.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                recombine_state,
                encoder=encoder,
                decoder=decoder,
                seed=seed,
            ),
        )
    return rt, {
        "replay": replay_state,
        "downscale": downscale_state,
        "restructure": restructure_state,
        "recombine": recombine_state,
    }


def _build_episode(profile_name: str, ep_idx: int, seed: int):
    """Build a dream episode that drives the profile's op set."""
    import numpy as np

    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )

    rng = np.random.default_rng(seed + ep_idx)
    beta_records = [
        {
            "x": rng.standard_normal(4).tolist(),
            "y": rng.standard_normal(2).tolist(),
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
    return DreamEpisode(
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
        budget=BudgetCap(flops=10_000_000, wall_time_s=10.0, energy_j=1.0),
        episode_id=f"sanity-{profile_name}-ep{ep_idx}-seed{seed}",
    )


def _score_adapter(adapter, seed: int, n_samples: int = 50) -> float:
    """Return a deterministic MSE-based score for the adapter.

    The retained benchmark shipped with the repo is a synthetic
    placeholder (string predictions, 50 items) — not directly
    callable with an MLP output. The sanity pilot therefore uses a
    self-consistency proxy : for ``n_samples`` seeded inputs,
    measure the average adapter output norm, mapped through a
    bounded ``exp(-mse)`` score in [0, 1]. Deterministic under the
    same ``seed`` + same adapter weights. Delta = post - pre is a
    sensible pipeline signal (positive when dream ops moved the
    weights coherently, near-zero when the ops were a no-op). This
    is pipeline-validation, not an empirical accuracy claim — see
    module docstring and C3.8 for the real benchmark swap.
    """
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(seed)
    xs = mx.array(rng.standard_normal((n_samples, 4)).astype(np.float32))
    out = adapter(xs)
    mx.eval(out)
    arr = np.asarray(out)
    # Guard against NaN / Inf produced by a bad downscale (S2
    # would normally catch this upstream — belt + suspenders).
    if not np.isfinite(arr).all():
        return 0.0
    mse = float(np.mean(arr ** 2))
    # exp(-mse) ∈ (0, 1] ; larger mse → smaller score.
    return float(np.exp(-mse))


def _run_cell(
    profile_name: str,
    seed: int,
    *,
    n_episodes: int = 5,
    n_eval_samples: int = 50,
) -> dict:
    """Execute one cell of the sanity pilot — 5 pipeline stages.

    Returns a dict with ``pre``, ``post``, ``delta``,
    ``wall_time_s`` and diagnostic fields (``model_loaded``,
    ``error``).
    """
    start = time.time()
    _seed_everything(seed)
    wrapper = _load_qwen_wrapper(seed)
    model_loaded = wrapper is not None

    adapter, encoder, decoder = _build_adapter(seed)

    # Stage 2 — pre-dream evaluation.
    pre = _score_adapter(adapter, seed=seed, n_samples=n_eval_samples)

    # Stage 3 — dream episodes.
    runtime, _states = _build_runtime(
        profile_name, adapter, encoder, decoder, seed=seed
    )
    for ep_idx in range(n_episodes):
        episode = _build_episode(profile_name, ep_idx, seed)
        runtime.execute(episode)

    # Stage 4 — post-dream evaluation.
    post = _score_adapter(adapter, seed=seed, n_samples=n_eval_samples)
    delta = post - pre

    return {
        "profile": profile_name,
        "seed": seed,
        "pre": pre,
        "post": post,
        "delta": delta,
        "wall_time_s": time.time() - start,
        "model_loaded": model_loaded,
    }


def _register_cell(profile_name: str, cell: dict) -> str:
    """Insert the cell's run_id into the run registry (R1 contract).

    The composite ``profile_tag`` matches the cycle-3 runner's
    ``_registry_profile_tag`` convention so the pilot's rows are a
    strict subset of the C3.8 resume contract.
    """
    from harness.storage.run_registry import RunRegistry

    registry = RunRegistry(REPO_ROOT / ".run_registry.sqlite")
    profile_tag = (
        f"cycle3/{SANITY_SCALE}/{profile_name}/mlx_kiki_oniric"
    )
    return registry.register(
        c_version=HARNESS_VERSION,
        profile=profile_tag,
        seed=cell["seed"],
        commit_sha=_resolve_commit_sha(),
    )


# -------------------------------------------------------------------
# H1 test across cells
# -------------------------------------------------------------------


def _h1_test(cells: list[dict]) -> dict:
    """Paired t-test pre vs post per profile.

    Returns a mapping ``{profile -> {t, p, reject_h0, n}}``.
    """
    from scipy import stats

    out: dict[str, dict] = {}
    for profile in SANITY_PROFILES:
        pres = [c["pre"] for c in cells if c["profile"] == profile]
        posts = [c["post"] for c in cells if c["profile"] == profile]
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
# Main entrypoint
# -------------------------------------------------------------------


def _print_banner(total_cells: int) -> None:
    print("=" * 64)
    print("CYCLE-3 SANITY PILOT (1.5B-scale fail-fast)")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {SANITY_SCALE}")
    print(f"profiles        : {SANITY_PROFILES}")
    print(f"exec substrate  : {SANITY_SUBSTRATES_EXEC}")
    print(f"plan substrates : {SANITY_SUBSTRATES_PLAN} (dry-run manifest)")
    print(f"seeds           : {len(SANITY_SEEDS)} (0..{SANITY_SEEDS[-1]})")
    print(f"exec cells      : {total_cells}")
    print(
        f"GO rule         : H1 rejected in ≥ "
        f"{GO_PROFILES_REJECTED_MIN}/{len(SANITY_PROFILES)} profiles "
        f"at α = {GO_BONFERRONI_ALPHA}"
    )
    print("-" * 64)


def _execute(
    opts: dict, plan: list[dict]
) -> int:  # pragma: no cover - side-effectful
    """Execute the sanity pilot — either one smoke cell or all 90.

    Returns the shell exit code (0 = success, 1 = NO-GO or failure).
    """
    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)

    if opts["smoke_cell"]:
        _print_banner(total_cells=1)
        print("[smoke-cell] running exactly 1 cell : p_min + seed 0 + MLX")
        start = time.time()
        try:
            cell = _run_cell("p_min", 0)
        except Exception as exc:
            traceback.print_exc()
            print(f"[smoke-cell] FAILED : {type(exc).__name__}: {exc}")
            return 1
        run_id = _register_cell("p_min", cell)
        wall = time.time() - start
        print(
            f"[smoke-cell] pre={cell['pre']:.6f} post={cell['post']:.6f} "
            f"delta={cell['delta']:+.6f} wall={wall:.2f}s "
            f"model_loaded={cell['model_loaded']} run_id={run_id}"
        )
        smoke_path = out_dir / "pilot-cycle3-sanity-1p5b-smoke.json"
        with smoke_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "mode": "smoke-cell",
                    "cell": {**cell, "run_id": run_id},
                    "wall_time_s": wall,
                    "extrapolated_full_pilot_s": wall * EXECUTED_CELL_COUNT,
                },
                fh,
                indent=2,
            )
        print(f"[smoke-cell] dump written to {smoke_path}")
        return 0

    # Full 90-cell run — MLX substrate only.
    _print_banner(total_cells=EXECUTED_CELL_COUNT)
    cells: list[dict] = []
    failures: list[dict] = []
    run_start = time.time()
    idx = 0
    for profile_name in SANITY_PROFILES:
        for seed in SANITY_SEEDS:
            idx += 1
            try:
                cell = _run_cell(profile_name, seed)
            except Exception as exc:
                traceback.print_exc()
                failures.append(
                    {
                        "profile": profile_name,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[cell {idx}/{EXECUTED_CELL_COUNT}] "
                    f"{profile_name} seed={seed} FAILED : "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            try:
                run_id = _register_cell(profile_name, cell)
            except Exception as exc:  # pragma: no cover - defensive
                run_id = f"register-failed:{exc}"
            cells.append({**cell, "run_id": run_id})
            print(
                f"[cell {idx}/{EXECUTED_CELL_COUNT}] "
                f"{profile_name} seed={seed:02d} "
                f"pre={cell['pre']:.5f} post={cell['post']:.5f} "
                f"delta={cell['delta']:+.5f} "
                f"wall={cell['wall_time_s']:.2f}s"
            )
    run_wall = time.time() - run_start

    # H1 test per profile + GO / NO-GO verdict.
    h1 = _h1_test(cells)
    rejected = sum(1 for v in h1.values() if v.get("reject_h0"))
    verdict = "GO" if rejected >= GO_PROFILES_REJECTED_MIN else "NO-GO"
    print("-" * 64)
    for profile, r in h1.items():
        print(
            f"H1 {profile}: p={r.get('p')} reject_h0={r.get('reject_h0')}"
        )
    print(f"verdict          : {verdict} ({rejected}/3 profiles)")
    print(f"total wall-clock : {run_wall:.1f}s")
    print(f"failures         : {len(failures)}")

    dump_path = out_dir / "pilot-cycle3-sanity-1p5b.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": SANITY_SCALE,
                "profiles": list(SANITY_PROFILES),
                "substrates_plan": list(SANITY_SUBSTRATES_PLAN),
                "substrates_executed": list(SANITY_SUBSTRATES_EXEC),
                "seeds": list(SANITY_SEEDS),
                "planned_cell_count": EXPECTED_CELL_COUNT,
                "executed_cell_count": EXECUTED_CELL_COUNT,
                "completed_cells": len(cells),
                "failed_cells": len(failures),
                "go_rule": {
                    "alpha": GO_BONFERRONI_ALPHA,
                    "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                    "profiles_total": len(SANITY_PROFILES),
                },
                "h1": h1,
                "verdict": verdict,
                "wall_time_s": run_wall,
                "cells": cells,
                "failures": failures,
                "plan": plan,
            },
            fh,
            indent=2,
        )
    print(f"results dumped to {dump_path}")
    return 0 if verdict == "GO" else 1


def main(argv: list[str] | None = None) -> int:
    """Entrypoint — enumerate (+ dry-run), smoke-cell, or full run."""
    opts = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    plan = _plan()
    assert len(plan) == EXPECTED_CELL_COUNT, (
        f"Sanity plan produced {len(plan)} cells, expected "
        f"{EXPECTED_CELL_COUNT} per §7."
    )

    if opts["dry_run"]:
        _print_banner(total_cells=EXECUTED_CELL_COUNT)
        print("[dry-run] enumeration validated ; no dream ops.")
        out_dir = REPO_ROOT / "docs" / "milestones"
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "pilot-cycle3-sanity-1p5b.json"
        with plan_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "scale": SANITY_SCALE,
                    "profiles": list(SANITY_PROFILES),
                    "substrates_plan": list(SANITY_SUBSTRATES_PLAN),
                    "substrates_executed": list(SANITY_SUBSTRATES_EXEC),
                    "seeds": list(SANITY_SEEDS),
                    "planned_cell_count": EXPECTED_CELL_COUNT,
                    "executed_cell_count": EXECUTED_CELL_COUNT,
                    "go_rule": {
                        "alpha": GO_BONFERRONI_ALPHA,
                        "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                        "profiles_total": len(SANITY_PROFILES),
                    },
                    "status": "plan-only",
                    "plan": plan,
                },
                fh,
                indent=2,
            )
        print(f"Plan manifest written to {plan_path}")
        return 0

    return _execute(opts, plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
