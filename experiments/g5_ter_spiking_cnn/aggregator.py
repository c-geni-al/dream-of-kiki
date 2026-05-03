"""Cross-substrate H8-A/B/C aggregator for the G5-ter pilot.

Loads a G4-quinto Step 2 MLX small-CNN milestone (key
``verdict.retention_by_arm``) and a G5-ter E-SNN spiking-CNN
milestone (key ``verdict.retention_by_arm``), runs four Welch
two-sided per-arm consistency tests at Bonferroni alpha/4 = 0.0125,
classifies the cross-substrate outcome into one of three
pre-registered hypotheses :

- **H8-A** (LIF non-linearity is the load-bearing washout) :
  own-substrate ``g_h8 < 0.5`` AND own-Welch fails to reject AND
  cross-substrate ``g_p_equ_cross >= 2.0`` (large MLX-minus-ESNN
  gap). The convolutional inductive bias does not save the cycle-3
  positive-effect channel from LIF washout.
- **H8-B** (architecture-dependent : CNN closes the gap) :
  own-substrate ``g_h8 >= 0.5`` AND own-Welch rejects AND
  ``g_p_equ_cross < 1.0`` (small MLX-minus-ESNN gap). The G5-bis
  MLP washout was an architecture-mismatch artefact.
- **H8-C** (partial : both contribute) : anything else. Logged
  with the observed (`g_h8`, `g_p_equ_cross`) pair for post-hoc
  inspection.

Thresholds 0.5 / 1.0 / 2.0 are LOCKED at pre-reg time and may not
be moved post hoc.

Reference :
    docs/osf-prereg-g5-ter-spiking-cnn.md sec 1
    docs/proofs/dr3-substrate-evidence.md
    experiments/g5_bis_richer_esnn/aggregator.py (sister)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kiki_oniric.eval.statistics import compute_hedges_g, welch_one_sided


REQUIRED_ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
ALPHA_PER_ARM = 0.05 / 4  # Bonferroni across 4 arms
H7B_G_THRESHOLD = 0.5  # own-substrate cutoff (reused from G5-bis)
H8A_G_MLX_MINUS_ESNN_MIN = 2.0  # cross-substrate H8-A floor
H8B_G_MLX_MINUS_ESNN_MAX = 1.0  # cross-substrate H8-B ceiling


def _load_retention(
    milestone_path: Path, key: str
) -> dict[str, list[float]]:
    """Load ``verdict[key]`` as a {arm: [retention floats]} dict."""
    payload = json.loads(milestone_path.read_text())
    retention = payload.get("verdict", {}).get(key)
    if not isinstance(retention, dict):
        raise ValueError(
            f"milestone {milestone_path} missing verdict.{key}"
        )
    for arm in REQUIRED_ARMS:
        if arm not in retention:
            raise ValueError(
                f"milestone {milestone_path} missing arm {arm!r} "
                f"in verdict.{key}"
            )
    return {
        arm: [float(v) for v in retention[arm]] for arm in REQUIRED_ARMS
    }


def _welch_two_sided(
    a: list[float], b: list[float], alpha: float
) -> tuple[float, bool]:
    """Run Welch one-sided in both directions, return (p_two_sided, reject)."""
    welch_a = welch_one_sided(a, b, alpha=alpha)
    welch_b = welch_one_sided(b, a, alpha=alpha)
    p_two_sided = min(2.0 * min(welch_a.p_value, welch_b.p_value), 1.0)
    reject = bool(p_two_sided < alpha)
    return p_two_sided, reject


def aggregate_g5ter_verdict(
    mlx_milestone: Path, esnn_milestone: Path
) -> dict[str, Any]:
    """Compute per-arm cross-substrate Welch + H8-A/B/C classification."""
    mlx = _load_retention(mlx_milestone, "retention_by_arm")
    esnn = _load_retention(esnn_milestone, "retention_by_arm")

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in REQUIRED_ARMS:
        mlx_vals = mlx[arm]
        esnn_vals = esnn[arm]
        if len(mlx_vals) < 2 or len(esnn_vals) < 2:
            per_arm[arm] = {
                "insufficient_samples": True,
                "n_mlx": len(mlx_vals),
                "n_esnn": len(esnn_vals),
            }
            continue
        g = compute_hedges_g(mlx_vals, esnn_vals)
        p_two_sided, reject = _welch_two_sided(
            mlx_vals, esnn_vals, ALPHA_PER_ARM
        )
        per_arm[arm] = {
            "hedges_g_mlx_minus_esnn": g,
            "welch_p_two_sided": p_two_sided,
            "reject_h0": reject,
            "consistency": not reject,
            "n_mlx": len(mlx_vals),
            "n_esnn": len(esnn_vals),
        }

    # E-SNN own-substrate effect (P_equ vs baseline)
    if len(esnn["P_equ"]) < 2 or len(esnn["baseline"]) < 2:
        g_h8 = 0.0
        welch_p = 1.0
        welch_reject = False
        own_insufficient = True
    else:
        g_h8 = compute_hedges_g(esnn["P_equ"], esnn["baseline"])
        welch_h8 = welch_one_sided(
            esnn["baseline"], esnn["P_equ"], alpha=ALPHA_PER_ARM
        )
        welch_p = welch_h8.p_value
        welch_reject = bool(welch_h8.reject_h0)
        own_insufficient = False

    # Decision rule (locked, pre-reg sec 1)
    p_equ_row = per_arm.get("P_equ", {})
    g_p_equ_cross = float(
        p_equ_row.get("hedges_g_mlx_minus_esnn", 0.0)
    )

    classification: str
    if own_insufficient:
        classification = "ambiguous"
    elif (
        g_h8 < H7B_G_THRESHOLD
        and not welch_reject
        and g_p_equ_cross >= H8A_G_MLX_MINUS_ESNN_MIN
    ):
        classification = "H8-A"
    elif (
        g_h8 >= H7B_G_THRESHOLD
        and welch_reject
        and g_p_equ_cross < H8B_G_MLX_MINUS_ESNN_MAX
    ):
        classification = "H8-B"
    else:
        classification = "H8-C"

    return {
        "per_arm": per_arm,
        "h8_classification": classification,
        "g_h8_esnn": g_h8,
        "g_h8_welch_p": welch_p,
        "g_h8_welch_reject_h0": welch_reject,
        "alpha_per_arm": ALPHA_PER_ARM,
        "h7b_g_threshold": H7B_G_THRESHOLD,
        "h8a_g_mlx_minus_esnn_min": H8A_G_MLX_MINUS_ESNN_MIN,
        "h8b_g_mlx_minus_esnn_max": H8B_G_MLX_MINUS_ESNN_MAX,
        "g_p_equ_cross": g_p_equ_cross,
        "mlx_milestone": str(mlx_milestone),
        "esnn_milestone": str(esnn_milestone),
    }


def _render_md(verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "# G5-ter cross-substrate aggregate - H8-A/B/C verdict"
    )
    lines.append("")
    lines.append("**Date** : 2026-05-03")
    lines.append(
        f"**MLX milestone** : `{verdict['mlx_milestone']}`"
    )
    lines.append(
        f"**E-SNN milestone** : `{verdict['esnn_milestone']}`"
    )
    lines.append(
        f"**Bonferroni alpha / 4** : {verdict['alpha_per_arm']:.4f}"
    )
    lines.append(
        f"**Own-substrate g_h8 threshold** : "
        f"{verdict['h7b_g_threshold']:.2f}"
    )
    lines.append(
        f"**H8-A g_mlx_minus_esnn floor** : "
        f"{verdict['h8a_g_mlx_minus_esnn_min']:.2f}"
    )
    lines.append(
        f"**H8-B g_mlx_minus_esnn ceiling** : "
        f"{verdict['h8b_g_mlx_minus_esnn_max']:.2f}"
    )
    lines.append("")
    classification = verdict["h8_classification"]
    lines.append(f"## Verdict : {classification}")
    lines.append("")
    lines.append(
        f"- Observed E-SNN g_h8 (P_equ vs baseline) : "
        f"**{verdict['g_h8_esnn']:+.4f}**"
    )
    lines.append(
        f"- Welch one-sided p (alpha/4 = "
        f"{verdict['alpha_per_arm']:.4f}) : "
        f"{verdict['g_h8_welch_p']:.4f}"
    )
    lines.append(
        f"- reject_h0 (own-substrate) : "
        f"{verdict['g_h8_welch_reject_h0']}"
    )
    lines.append(
        f"- Observed cross-substrate g (MLX - E-SNN, P_equ) : "
        f"**{verdict['g_p_equ_cross']:+.4f}**"
    )
    lines.append("")
    lines.append("## Per-arm cross-substrate Welch consistency")
    lines.append("")
    lines.append(
        "| arm | g (MLX - E-SNN) | Welch p (two-sided) | "
        "reject H0 | consistent |"
    )
    lines.append(
        "|-----|------------------|----------------------|"
        "-----------|------------|"
    )
    for arm in REQUIRED_ARMS:
        row = verdict["per_arm"][arm]
        if row.get("insufficient_samples"):
            lines.append(
                f"| {arm} | INSUFFICIENT | INSUFFICIENT | n/a | False |"
            )
            continue
        lines.append(
            f"| {arm} | {row['hedges_g_mlx_minus_esnn']:+.4f} | "
            f"{row['welch_p_two_sided']:.4f} | "
            f"{row['reject_h0']} | {row['consistency']} |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- DR-3 spec : "
        "`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2"
    )
    lines.append(
        "- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`"
    )
    lines.append(
        "- Aggregator : `experiments/g5_ter_spiking_cnn/aggregator.py`"
    )
    lines.append(
        "- Pre-registration : `docs/osf-prereg-g5-ter-spiking-cnn.md`"
    )
    lines.append(
        "- Sister G5-bis aggregate : "
        "`docs/milestones/g5-bis-aggregate-2026-05-03.md`"
    )
    lines.append("")
    return "\n".join(lines)


def write_aggregate_dump(
    *,
    mlx_milestone: Path,
    esnn_milestone: Path,
    out_json: Path,
    out_md: Path,
) -> dict[str, Any]:
    """Compute the verdict and persist `.json` + `.md` siblings."""
    verdict = aggregate_g5ter_verdict(mlx_milestone, esnn_milestone)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    out_md.write_text(_render_md(verdict))
    return verdict
