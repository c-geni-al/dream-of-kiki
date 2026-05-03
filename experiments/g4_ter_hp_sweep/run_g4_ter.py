"""G4-ter pilot driver - HP sweep + richer-substrate confirmatory.

Two sweeps run side-by-side (Option C) :

1. Richer-substrate sweep : 4 arms x N_richer seeds x 1 HP (C5) on
   ``G4HierarchicalClassifier``.
2. HP sub-grid sweep : 3 dream arms x N_hp seeds x 10 HP combos on
   the binary ``G4Classifier`` (no baseline arm - HP changes do not
   affect the no-dream branch).

Per-cell pipeline mirrors ``experiments/g4_split_fmnist/run_g4.py``.
Outputs :
    docs/milestones/g4-ter-pilot-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-ter-pilot.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    BetaBufferFIFO,
    G4Classifier,
    build_profile,
)
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (  # noqa: E402
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HP_COMBOS,
    HPCombo,
    representative_combo,
)
from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS_RICHER: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ARMS_HP: tuple[str, ...] = ("P_min", "P_equ", "P_max")
DEFAULT_SEEDS_RICHER: tuple[int, ...] = tuple(range(30))
DEFAULT_SEEDS_HP: tuple[int, ...] = tuple(range(10))
DEFAULT_HP_COMBO_IDS: tuple[str, ...] = tuple(c.combo_id for c in HP_COMBOS)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-ter-pilot-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-ter-pilot-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
HIDDEN_1 = 32
HIDDEN_2 = 16
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05


class CellRicher(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    run_id: str


class CellHP(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    run_id: str


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _run_cell_richer(
    arm: str,
    seed: int,
    combo: HPCombo,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4HierarchicalClassifier(
        in_dim=feat_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferHierFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            x = task["x_train"][i]
            latent = clf.latent(x[None, :])[0]
            buffer.push(x=x, y=int(task["y_train"][i]), latent=latent)

    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            clf.dream_episode_hier(
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                replay_n_records=combo.replay_batch,
                replay_n_steps=combo.replay_n_steps,
                replay_lr=combo.replay_lr,
                downscale_factor=combo.downscale_factor,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _run_cell_hp(
    arm: str,
    seed: int,
    combo: HPCombo,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int = 128,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4Classifier(
        in_dim=feat_dim,
        hidden_dim=hidden_dim,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            buffer.push(task["x_train"][i], int(task["y_train"][i]))

    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = build_profile(arm, seed=seed)
    for k in range(1, len(tasks)):
        clf.dream_episode(
            profile,
            seed=seed + k,
            beta_buffer=buffer,
            replay_n_records=combo.replay_batch,
            replay_n_steps=combo.replay_n_steps,
            replay_lr=combo.replay_lr,
            downscale_factor=combo.downscale_factor,
        )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h2_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H2 - richer substrate P_equ vs baseline (Hu 2020 anchor)."""
    p_equ = retention.get("P_equ", [])
    base = retention.get("baseline", [])
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 4)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "above_zero": bool(g >= 0.0),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 4,
        "n_p_equ": len(p_equ),
        "n_base": len(base),
    }


def _h_dr4_ter_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    groups = [
        retention.get("P_min", []),
        retention.get("P_equ", []),
        retention.get("P_max", []),
    ]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05 / 4)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": mean_p_min,
        "mean_p_equ": mean_p_equ,
        "mean_p_max": mean_p_max,
        "monotonic_observed": (
            mean_p_min <= mean_p_equ <= mean_p_max
        ),
        "alpha_per_test": 0.05 / 4,
    }


def _h1_hp_verdict(
    cells_hp: list[dict[str, Any]],
    base_retention: list[float],
) -> dict[str, Any]:
    """H1 - best HP combo on binary head vs richer-substrate baseline."""
    by_combo_arm: dict[tuple[str, str], list[float]] = {}
    for c in cells_hp:
        if c["excluded_underperforming_baseline"]:
            continue
        by_combo_arm.setdefault(
            (c["hp_combo_id"], c["arm"]), []
        ).append(c["retention"])
    best: dict[str, Any] | None = None
    if len(base_retention) < 2:
        return {
            "insufficient_samples": True,
            "n_base": len(base_retention),
        }
    for (combo_id, arm), rets in by_combo_arm.items():
        if arm != "P_equ" or len(rets) < 2:
            continue
        g = compute_hedges_g(rets, base_retention)
        if best is None or g > best["hedges_g"]:
            best = {
                "hp_combo_id": combo_id,
                "hedges_g": g,
                "n_p_equ": len(rets),
            }
    if best is None:
        return {"insufficient_samples": True}
    best["above_zero"] = bool(best["hedges_g"] >= 0.0)
    best["above_hu_2020_lower_ci"] = bool(
        best["hedges_g"] >= HU_2020_OVERALL.ci_low
    )
    best["alpha_per_test"] = 0.05 / 4
    best["n_base"] = len(base_retention)
    return best


def _aggregate_verdict(
    cells_richer: list[dict[str, Any]],
    cells_hp: list[dict[str, Any]],
) -> dict[str, Any]:
    retention_richer = _retention_by_arm(cells_richer)
    return {
        "h2_substrate_richer": _h2_verdict(retention_richer),
        "h_dr4_ter_richer": _h_dr4_ter_verdict(retention_richer),
        "h1_hp_artefact": _h1_hp_verdict(
            cells_hp, retention_richer.get("baseline", [])
        ),
        "retention_richer_by_arm": retention_richer,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h1 = payload["verdict"]["h1_hp_artefact"]
    h2 = payload["verdict"]["h2_substrate_richer"]
    h4 = payload["verdict"]["h_dr4_ter_richer"]
    lines: list[str] = [
        "# G4-ter pilot - HP sweep + richer substrate",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells (richer)** : {len(payload['cells_richer'])} "
            f"({len(ARMS_RICHER)} arms x {payload['n_seeds_richer']} "
            "seeds x 1 HP)"
        ),
        (
            f"**Cells (HP)** : {len(payload['cells_hp'])} "
            f"({len(ARMS_HP)} arms x {payload['n_seeds_hp']} seeds x "
            f"{payload['n_hp_combos']} combos)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypotheses",
        "",
        "Pre-registration : `docs/osf-prereg-g4-ter-pilot.md`",
        "",
        "### H1 - HP artefact (best HP combo on binary head)",
    ]
    if h1.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_base={h1.get('n_base')})")
    else:
        lines += [
            f"- best combo : `{h1['hp_combo_id']}`",
            f"- best Hedges' g : **{h1['hedges_g']:.4f}**",
            f"- above zero : {h1['above_zero']}",
            (
                "- above Hu 2020 lower CI 0.21 : "
                f"{h1['above_hu_2020_lower_ci']}"
            ),
        ]
    lines += [
        "",
        "### H2 - Substrate-level (richer head, P_equ vs baseline)",
    ]
    if h2.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_p_equ={h2.get('n_p_equ')}, "
            f"n_base={h2.get('n_base')})"
        )
    else:
        lines += [
            f"- observed Hedges' g : **{h2['hedges_g']:.4f}**",
            f"- above zero : {h2['above_zero']}",
            (
                "- above Hu 2020 lower CI 0.21 : "
                f"{h2['above_hu_2020_lower_ci']}"
            ),
            (
                f"- Welch one-sided p (alpha/4 = "
                f"{h2['alpha_per_test']:.4f}) : {h2['welch_p']:.4f} -> "
                f"reject_h0 = {h2['welch_reject_h0']}"
            ),
        ]
    lines += [
        "",
        "### H_DR4-ter - Jonckheere monotonicity on richer substrate",
    ]
    if h4.get("insufficient_samples"):
        lines.append(f"INSUFFICIENT SAMPLES (n_per_arm={h4['n_per_arm']})")
    else:
        lines += [
            f"- mean retention P_min : {h4['mean_p_min']:.4f}",
            f"- mean retention P_equ : {h4['mean_p_equ']:.4f}",
            f"- mean retention P_max : {h4['mean_p_max']:.4f}",
            (
                "- monotonic observed P_max >= P_equ >= P_min : "
                f"{h4['monotonic_observed']}"
            ),
            f"- Jonckheere J statistic : {h4['j_statistic']:.4f}",
            (
                f"- one-sided p (alpha/4 = {h4['alpha_per_test']:.4f}) : "
                f"{h4['p_value']:.4f} -> reject_h0 = {h4['reject_h0']}"
            ),
        ]
    lines += [
        "",
        "## Provenance",
        "",
        "- Pre-registration : "
        "[docs/osf-prereg-g4-ter-pilot.md](../osf-prereg-g4-ter-pilot.md)",
        "- Driver : `experiments/g4_ter_hp_sweep/run_g4_ter.py`",
        "- Substrate : `experiments.g4_ter_hp_sweep.dream_wrap_hier."
        "G4HierarchicalClassifier`",
        "- HP grid : `experiments.g4_ter_hp_sweep.hp_grid.HP_COMBOS`",
        "- Run registry : `harness/storage/run_registry.RunRegistry` "
        "(db `.run_registry.sqlite`)",
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds_richer: tuple[int, ...],
    seeds_hp: tuple[int, ...],
    hp_combo_ids: tuple[str, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    smoke: bool = False,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    # ---- Richer substrate sweep (C5 only) ----
    c5 = representative_combo()
    cells_richer: list[dict[str, Any]] = []
    sweep_start = time.time()
    for arm in ARMS_RICHER:
        for seed in seeds_richer:
            cell = _run_cell_richer(
                arm,
                seed,
                c5,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g4-ter/richer/{arm}/{c5.combo_id}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell["run_id"] = run_id
            cells_richer.append(cell)

    # ---- HP sub-grid sweep on binary head ----
    cells_hp: list[dict[str, Any]] = []
    selected_combos = [c for c in HP_COMBOS if c.combo_id in hp_combo_ids]
    for combo in selected_combos:
        for arm in ARMS_HP:
            for seed in seeds_hp:
                cell = _run_cell_hp(
                    arm,
                    seed,
                    combo,
                    tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                run_id = registry.register(
                    c_version=C_VERSION,
                    profile=f"g4-ter/hp/{arm}/{combo.combo_id}",
                    seed=seed,
                    commit_sha=commit_sha,
                )
                cell["run_id"] = run_id
                cells_hp.append(cell)

    wall = time.time() - sweep_start
    verdict = _aggregate_verdict(cells_richer, cells_hp)

    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds_richer": len(seeds_richer),
        "n_seeds_hp": len(seeds_hp),
        "n_hp_combos": len(selected_combos),
        "arms_richer": list(ARMS_RICHER),
        "arms_hp": list(ARMS_HP),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "smoke": smoke,
        "cells_richer": cells_richer,
        "cells_hp": cells_hp,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-ter pilot driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON
    )
    parser.add_argument(
        "--out-md", type=Path, default=DEFAULT_OUT_MD
    )
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument(
        "--seeds-richer", type=int, nargs="+",
        default=list(DEFAULT_SEEDS_RICHER),
    )
    parser.add_argument(
        "--seeds-hp", type=int, nargs="+",
        default=list(DEFAULT_SEEDS_HP),
    )
    parser.add_argument(
        "--hp-combo-ids", type=str, nargs="+",
        default=list(DEFAULT_HP_COMBO_IDS),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds_richer = (0, 1)
        seeds_hp = (0,)
        hp_combo_ids = ("C5",)
    else:
        seeds_richer = tuple(args.seeds_richer)
        seeds_hp = tuple(args.seeds_hp)
        hp_combo_ids = tuple(args.hp_combo_ids)

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds_richer=seeds_richer,
        seeds_hp=seeds_hp,
        hp_combo_ids=hp_combo_ids,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(
        f"Cells richer : {len(payload['cells_richer'])}  "
        f"HP : {len(payload['cells_hp'])}"
    )
    print(
        f"H2.hedges_g : "
        f"{payload['verdict']['h2_substrate_richer'].get('hedges_g')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
