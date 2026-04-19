"""Cycle-2 multi-substrate ablation runner (C2.9 + C2.11).

Pipeline-validation script (synthetic substrate × profile × seed).
Runs the full cartesian product :

    {mlx_kiki_oniric, esnn_thalamocortical}
        × {baseline, P_min, P_equ}
        × {42, 123, 7}

against the synthetic mega-v2 stratified benchmark, then re-runs
the cycle-1 H1-H4 statistical chain *per substrate* so the
cross-substrate replication signal is verifiable. C2.11 adds the
per-substrate H1-H4 dump and the cross-substrate consistency
report.

Both substrate columns share the same Python predictor objects in
this synthetic mode — the substrate axis is an *identity label*
for the row. A real wiring (post-cycle-2) will swap the predictor
to substrate-specific inference. Until then the cross-substrate
test is a *consistency* check : same H1-H4 verdict across both
substrate labels under the same seed grid (synthetic data — no
real cohort/HW).

Usage :
    uv run python scripts/ablation_cycle2.py

Output :
    docs/milestones/ablation-cycle2-results.json (machine data)
    docs/milestones/ablation-cycle2-results.md   (human report)

Reference : docs/specs/2026-04-17-dreamofkiki-master-design.md §5
            docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
            §6.2 (DR-3 Conformance Criterion)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.mega_v2.adapter import (
    load_megav2_stratified,
)
from harness.storage.run_registry import RunRegistry
from kiki_oniric.eval.ablation import (
    AblationRunner,
    ProfileSpec,
    SubstrateSpec,
)
from kiki_oniric.eval.statistics import (
    jonckheere_trend,
    one_sample_threshold,
    tost_equivalence,
    welch_one_sided,
)


HARNESS_VERSION = "C-v0.6.0+STABLE"
SUBSTRATES = ["mlx_kiki_oniric", "esnn_thalamocortical"]
SEEDS = [42, 123, 7]


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


def baseline_predictor_factory(seed: int):
    """Mock baseline: ~50% accuracy (no consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        return item["expected"] if (rid + seed) % 2 == 0 else "WRONG"
    return predict


def p_min_predictor_factory(seed: int):
    """Mock P_min: ~70% accuracy (basic consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        return item["expected"] if (rid + seed) % 10 < 7 else "WRONG"
    return predict


def p_equ_predictor_factory(seed: int):
    """Mock P_equ: ~85% accuracy (balanced consolidation)."""
    def predict(item: dict) -> str:
        rid = int(item["id"].split("-")[-1])
        return item["expected"] if (rid + seed) % 20 < 17 else "WRONG"
    return predict


def _profile_specs_for_seed(seed: int) -> list[ProfileSpec]:
    """Build per-seed profile specs (predictor closes over seed)."""
    return [
        ProfileSpec("baseline", baseline_predictor_factory(seed)),
        ProfileSpec("P_min", p_min_predictor_factory(seed)),
        ProfileSpec("P_equ", p_equ_predictor_factory(seed)),
    ]


def _per_seed_grid(benchmark, registry_path: Path) -> dict:
    """Run the substrate × profile × seed grid one seed at a time
    so the predictor closes correctly over each seed.

    Returns a dict of dicts :
        {substrate: {profile: [acc_seed0, acc_seed1, ...]}}
    plus the run_id of the last batch (all batches in this script
    share the same SubstrateSpec / ProfileSpec shape so the
    deterministic batch-seed only varies on `seeds`).
    """
    table: dict[str, dict[str, list[float]]] = {
        sub: {p: [] for p in ("baseline", "P_min", "P_equ")}
        for sub in SUBSTRATES
    }
    last_run_id = ""
    for seed in SEEDS:
        runner = AblationRunner(
            profile_specs=_profile_specs_for_seed(seed),
            seeds=[seed],
            benchmark=benchmark,
            substrate_specs=[SubstrateSpec(s) for s in SUBSTRATES],
            registry_path=registry_path,
        )
        df = runner.run()
        last_run_id = df["run_id"].iloc[0]
        for _, row in df.iterrows():
            table[row["substrate"]][row["profile"]].append(
                float(row["accuracy"])
            )
    return {"acc": table, "run_id": last_run_id}


def _run_h1_h4(
    baseline_acc: list[float],
    p_min_acc: list[float],
    p_equ_acc: list[float],
) -> dict:
    """Run the H1-H4 statistical chain on (baseline, P_min, P_equ).

    Returns a JSON-serialisable dict per hypothesis. Bonferroni
    α = 0.05/4 = 0.0125 per the OSF pre-registration.
    """
    forgetting_baseline = [1 - a for a in baseline_acc]
    forgetting_p_equ = [1 - a for a in p_equ_acc]
    h1 = welch_one_sided(
        treatment=forgetting_p_equ,
        control=forgetting_baseline,
        alpha=0.0125,
    )
    p_max_smoke = [
        a + 0.001 * (i - 1) for i, a in enumerate(p_equ_acc)
    ]
    h2 = tost_equivalence(
        treatment=p_max_smoke,
        control=p_equ_acc,
        epsilon=0.05,
        alpha=0.0125,
    )
    h3 = jonckheere_trend(
        groups=[p_min_acc, p_equ_acc], alpha=0.0125
    )
    energy_ratios = [1.5 + 0.1 * i for i in range(len(SEEDS))]
    h4 = one_sample_threshold(
        sample=energy_ratios, threshold=2.0, alpha=0.0125
    )
    return {
        "H1_forgetting": {
            "test_name": h1.test_name,
            "p_value": h1.p_value,
            "reject_h0": h1.reject_h0,
        },
        "H2_equivalence_self": {
            "test_name": h2.test_name,
            "p_value": h2.p_value,
            "reject_h0": h2.reject_h0,
        },
        "H3_monotonic": {
            "test_name": h3.test_name,
            "p_value": h3.p_value,
            "reject_h0": h3.reject_h0,
        },
        "H4_energy_budget": {
            "test_name": h4.test_name,
            "p_value": h4.p_value,
            "reject_h0": h4.reject_h0,
        },
    }


def _cross_substrate_consistency(per_substrate_h: dict) -> dict:
    """Cross-substrate consistency : same H1-H4 verdict across
    substrates? Returns a per-hypothesis agreement flag."""
    consistency = {}
    keys = ("H1_forgetting", "H2_equivalence_self",
            "H3_monotonic", "H4_energy_budget")
    for key in keys:
        verdicts = {
            sub: per_substrate_h[sub][key]["reject_h0"]
            for sub in SUBSTRATES
        }
        consistency[key] = {
            "verdicts": verdicts,
            "agree": len(set(verdicts.values())) == 1,
        }
    return consistency


def main() -> None:
    benchmark = load_megav2_stratified(
        real_path=None,
        items_per_domain=20,
        synthetic_seed=42,
    )
    registry_path = Path(os.environ.get(
        "DREAMOFKIKI_RUN_REGISTRY",
        REPO_ROOT / ".run_registry.sqlite",
    ))

    # Tag the cycle-2 batch under its own profile name so the
    # registry distinguishes it from the cycle-1 G4 batch.
    registry = RunRegistry(registry_path)
    cycle2_batch_id = registry.register(
        c_version=HARNESS_VERSION,
        profile="cycle2_multi_substrate_ablation",
        seed=min(SEEDS),
        commit_sha=_resolve_commit_sha(),
    )

    # Run the substrate × profile × seed grid
    grid = _per_seed_grid(benchmark, registry_path)
    acc_table = grid["acc"]

    # Per-substrate H1-H4
    per_substrate_h = {
        sub: _run_h1_h4(
            acc_table[sub]["baseline"],
            acc_table[sub]["P_min"],
            acc_table[sub]["P_equ"],
        )
        for sub in SUBSTRATES
    }

    consistency = _cross_substrate_consistency(per_substrate_h)
    sig_counts = {
        sub: sum(
            1 for h in per_substrate_h[sub].values() if h["reject_h0"]
        )
        for sub in SUBSTRATES
    }
    fully_consistent = all(
        v["agree"] for v in consistency.values()
    )

    results = {
        "cycle2_batch_id": cycle2_batch_id,
        "ablation_runner_run_id": grid["run_id"],
        "harness_version": HARNESS_VERSION,
        "is_synthetic": True,
        "synthetic_substitute": True,
        "data_provenance": (
            "synthetic data — no real cohort/HW, mock predictors "
            "shared across substrate labels (cycle-2 C2.9/C2.11)"
        ),
        "benchmark_size": len(benchmark.items),
        "benchmark_hash": benchmark.source_hash,
        "seeds": SEEDS,
        "substrates": SUBSTRATES,
        "accuracy": acc_table,
        "hypotheses_per_substrate": per_substrate_h,
        "cross_substrate_consistency": consistency,
        "significant_count_per_substrate": sig_counts,
        "fully_consistent": fully_consistent,
    }

    print("=" * 64)
    print("CYCLE-2 MULTI-SUBSTRATE ABLATION (synthetic substitute)")
    print("=" * 64)
    print(f"benchmark : {len(benchmark.items)} items, "
          f"hash {benchmark.source_hash[:24]}...")
    print(f"seeds     : {SEEDS}")
    print(f"substrates: {SUBSTRATES}")
    print("-" * 64)
    for sub in SUBSTRATES:
        print(f"\n[{sub}]")
        for prof in ("baseline", "P_min", "P_equ"):
            accs = acc_table[sub][prof]
            print(f"  {prof:10s} acc {[f'{a:.3f}' for a in accs]}")
        for hname, h in per_substrate_h[sub].items():
            flag = "PASS" if h["reject_h0"] else "fail"
            print(f"  {hname:22s} p={h['p_value']:.4f} {flag}")
        print(f"  significant : {sig_counts[sub]}/4")
    print("-" * 64)
    print("Cross-substrate consistency :")
    for hname, c in consistency.items():
        flag = "AGREE" if c["agree"] else "DISAGREE"
        print(f"  {hname:22s} {flag} {c['verdicts']}")
    print(
        f"\nfully consistent across substrates : "
        f"{'YES' if fully_consistent else 'NO'}"
    )
    print("=" * 64)

    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation-cycle2-results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {json_path}")

    # Lightweight markdown sibling for human consumption.
    md_path = out_dir / "ablation-cycle2-results.md"
    md_lines = [
        "# Cycle-2 multi-substrate ablation (synthetic substitute)",
        "",
        "**(synthetic data — no real cohort/HW)**  Pipeline-",
        "validation only ; substrate axis is an identity label, the",
        "Python predictor is shared across substrate rows. See",
        "`scripts/ablation_cycle2.py` docstring for rationale.",
        "",
        f"- harness_version : `{HARNESS_VERSION}`",
        f"- cycle2_batch_id : `{cycle2_batch_id}`",
        f"- ablation_runner_run_id : `{grid['run_id']}`",
        f"- benchmark_hash : `{benchmark.source_hash[:24]}...`",
        f"- seeds : `{SEEDS}`",
        f"- substrates : `{SUBSTRATES}`",
        "",
        "## Per-substrate accuracy (synthetic substitute)",
        "",
        "| substrate | profile | acc (per seed) |",
        "|-----------|---------|----------------|",
    ]
    for sub in SUBSTRATES:
        for prof in ("baseline", "P_min", "P_equ"):
            accs = acc_table[sub][prof]
            md_lines.append(
                f"| {sub} | {prof} | "
                f"{[round(a, 3) for a in accs]} |"
            )
    md_lines += [
        "",
        "## H1-H4 verdicts (synthetic substitute)",
        "",
        "| substrate | hypothesis | p-value | reject H0 |",
        "|-----------|------------|---------|-----------|",
    ]
    for sub in SUBSTRATES:
        for hname, h in per_substrate_h[sub].items():
            md_lines.append(
                f"| {sub} | {hname} | "
                f"{h['p_value']:.4f} | "
                f"{'PASS' if h['reject_h0'] else 'fail'} |"
            )
    md_lines += [
        "",
        "## Cross-substrate consistency (synthetic substitute)",
        "",
        "| hypothesis | mlx_kiki_oniric | esnn_thalamocortical | agree |",
        "|------------|-----------------|----------------------|-------|",
    ]
    for hname, c in consistency.items():
        v = c["verdicts"]
        md_lines.append(
            f"| {hname} | {v['mlx_kiki_oniric']} | "
            f"{v['esnn_thalamocortical']} | "
            f"{'YES' if c['agree'] else 'NO'} |"
        )
    md_lines += [
        "",
        f"**Fully consistent across substrates :** "
        f"{'YES' if fully_consistent else 'NO'}",
        "",
        "> All numbers in this dump are produced by mock predictors",
        "> shared across substrate labels. They validate the",
        "> *cross-substrate replication pipeline*, not consolidation",
        "> efficacy on real linguistic data or real spike-rate",
        "> dynamics. See cycle-2 spec for the real-wiring deferral.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Human report written to {md_path}")


if __name__ == "__main__":
    main()
