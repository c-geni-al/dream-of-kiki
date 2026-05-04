"""G4-septimo aggregator — H6-B verdict + H6-C conjunction with G4-sexto.

Per pre-reg sec 2 :

- **H6-B** : Welch two-sided between (P_max with mog) and
  (P_max with none) on Split-Tiny-ImageNet, alpha = 0.05 (only one
  new test ; no Bonferroni inheritance — H6-A is locked in
  G4-sexto). Failure to reject confirms H6-B.
- **H6-C** : derived conjunction
  ``H6-A_confirmed_from_g4_sexto_aggregate AND
  H6-B_confirmed_from_g4_septimo``. No additional Welch test ;
  H6-C is logical aggregation. Three resolution states :
    - confirmed : both H6-A (loaded from G4-sexto aggregate) and
      H6-B confirmed ; full universality across {Split-FMNIST,
      Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} ×
      {3-layer MLP, 5-layer MLP, small CNN, medium CNN}.
    - partial : exactly one of {H6-A, H6-B} confirmed ;
      universality scope-bounded to the confirming benchmark.
    - falsified : both Welch tests reject H0.

H6-A is ingested from
``docs/milestones/g4-sexto-aggregate-2026-05-03.json``
(``h6a_cifar100.confirmed`` flag), the canonical resolution from
the G4-sexto Step 1 run + N=95 Studio confirmatory.

Outputs :
    docs/milestones/g4-septimo-aggregate-2026-05-04.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STEP1 = (
    REPO_ROOT / "docs" / "milestones" / "g4-septimo-step1-2026-05-04.json"
)
DEFAULT_SEXTO_AGGREGATE = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-aggregate-2026-05-03.json"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-septimo-aggregate-2026-05-04.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-septimo-aggregate-2026-05-04.md"
)

HONEST_READING_H6B = (
    "Welch fail-to-reject = absence of evidence at this N for a "
    "difference between mog and none — under H6-B specifically, "
    "this **is** the predicted positive empirical claim that "
    "RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on "
    "the medium CNN substrate at Tiny-ImageNet 200-class / 64×64 "
    "RGB scale."
)


def _load_h6a_from_sexto_aggregate(path: Path) -> dict[str, Any]:
    """Load the H6-A clause from the G4-sexto aggregate JSON.

    Returns the ``h6a_cifar100`` block with the ``confirmed`` flag
    that the G4-sexto aggregator already computed (canonical
    source of truth — re-deriving it here would risk drift).
    Raises ``FileNotFoundError`` if the G4-sexto aggregate is
    missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"missing G4-sexto aggregate (H6-A source) : {path}"
        )
    payload = json.loads(path.read_text())
    return dict(payload["h6a_cifar100"])


def aggregate_g4_septimo_verdict(
    step1_path: Path,
    sexto_aggregate_path: Path,
) -> dict[str, Any]:
    """Load Step 1 + G4-sexto aggregate and return H6-B/C verdicts.

    The H6-A clause is read directly from
    ``sexto_aggregate_path``'s ``h6a_cifar100.confirmed`` flag —
    this is the canonical resolution from the G4-sexto pilot
    (Step 1 + N=95 Studio confirmatory).
    """
    s1 = json.loads(step1_path.read_text())
    h6a_block = _load_h6a_from_sexto_aggregate(sexto_aggregate_path)
    h6a_confirmed = bool(h6a_block.get("confirmed", False))

    h6b = s1["verdict"]["h6b_recombine_strategy"]
    h6b_confirmed = (
        not h6b.get("insufficient_samples", False)
        and bool(h6b.get("h6b_recombine_empty_confirmed"))
    )
    h6b_block: dict[str, Any] = {**h6b, "confirmed": h6b_confirmed}

    h6c_confirmed = h6a_confirmed and h6b_confirmed
    # H6-C "partial" : exactly one leg confirmed (XOR).
    h6c_partial = h6a_confirmed != h6b_confirmed
    h6c_falsified = (not h6a_confirmed) and (not h6b_confirmed)

    return {
        "h6a_cifar100": h6a_block,
        "h6b_tiny_imagenet": h6b_block,
        "h6c_universality": {
            "confirmed": h6c_confirmed,
            "partial": h6c_partial,
            "falsified": h6c_falsified,
        },
        "summary": {
            "h6a_confirmed_from_g4_sexto_aggregate": h6a_confirmed,
            "h6b_confirmed_from_g4_septimo": h6b_confirmed,
            "h6c_confirmed": h6c_confirmed,
            "h6c_partial": h6c_partial,
            "h6c_falsified": h6c_falsified,
            "h5c_to_h6c_universality_extension": h6c_confirmed,
        },
    }


def _render_md(verdict: dict[str, Any]) -> str:
    h6a = verdict["h6a_cifar100"]
    h6b = verdict["h6b_tiny_imagenet"]
    h6c = verdict["h6c_universality"]
    s = verdict["summary"]

    h6c_state = (
        "confirmed" if s["h6c_confirmed"]
        else "partial" if s["h6c_partial"]
        else "falsified" if s["h6c_falsified"]
        else "indeterminate"
    )

    lines: list[str] = [
        "# G4-septimo aggregate verdict",
        "",
        "**Date** : 2026-05-04",
        "**Pre-registration** : "
        "[docs/osf-prereg-g4-septimo-pilot.md]"
        "(../osf-prereg-g4-septimo-pilot.md)",
        "**Sister pilot** : G4-sexto aggregate "
        "[docs/milestones/g4-sexto-aggregate-2026-05-03.md]"
        "(./g4-sexto-aggregate-2026-05-03.md) (H6-A canonical source).",
        "",
        "## Summary",
        "",
        (
            "- H6-A (CIFAR-100, 100-class scale) confirmed "
            f"(from G4-sexto aggregate) : "
            f"**{s['h6a_confirmed_from_g4_sexto_aggregate']}**"
        ),
        (
            "- H6-B (Tiny-ImageNet, 200-class / 64×64 RGB scale) "
            f"confirmed : **{s['h6b_confirmed_from_g4_septimo']}**"
        ),
        f"- H6-C (universality conjunction) state : **{h6c_state}**",
        f"- H6-C confirmed : **{s['h6c_confirmed']}**",
        f"- H6-C partial : **{s['h6c_partial']}**",
        f"- H6-C falsified : **{s['h6c_falsified']}**",
        (
            "- H5-C → H6-C universality extension "
            "(4 benchmarks × 4 substrates) : "
            f"**{s['h5c_to_h6c_universality_extension']}**"
        ),
        "",
        "## H6-A — CIFAR-100 (n_classes=10 per task, G4SmallCNN) — from G4-sexto",
        "",
    ]
    if h6a.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (mog) : {h6a['mean_p_max_mog']:.4f}",
            f"- mean P_max (none) : {h6a['mean_p_max_none']:.4f}",
            (
                f"- Hedges' g (mog vs none) : "
                f"{h6a['hedges_g_mog_vs_none']:.4f}"
            ),
            f"- Welch t : {h6a['welch_t']:.4f}",
            (
                f"- Welch p two-sided (alpha = "
                f"{h6a['alpha_per_test']:.4f}) : "
                f"{h6a['welch_p_two_sided']:.4f}"
            ),
            (
                f"- fail_to_reject_h0 : "
                f"{h6a['fail_to_reject_h0']} -> "
                f"H6-A confirmed = {h6a['confirmed']}"
            ),
        ]
    lines += [
        "",
        "## H6-B — Tiny-ImageNet (n_classes=20 per task, G4MediumCNN)",
        "",
    ]
    if h6b.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean P_max (mog) : {h6b['mean_p_max_mog']:.4f}",
            f"- mean P_max (none) : {h6b['mean_p_max_none']:.4f}",
            (
                f"- Hedges' g (mog vs none) : "
                f"{h6b['hedges_g_mog_vs_none']:.4f}"
            ),
            f"- Welch t : {h6b['welch_t']:.4f}",
            (
                f"- Welch p two-sided (alpha = "
                f"{h6b['alpha_per_test']:.4f}) : "
                f"{h6b['welch_p_two_sided']:.4f}"
            ),
            (
                f"- fail_to_reject_h0 : "
                f"{h6b['fail_to_reject_h0']} -> "
                f"H6-B confirmed = {h6b['confirmed']}"
            ),
            "",
            f"*Honest reading* : {HONEST_READING_H6B}",
        ]
    lines += [
        "",
        "## H6-C — universality conjunction (4 benchmarks × 4 substrates)",
        "",
        f"State : **{h6c_state}**",
        "",
    ]
    if h6c["confirmed"]:
        lines.append(
            "Both H6-A (G4-sexto) and H6-B (G4-septimo) confirmed. "
            "The G4-quinto H5-C RECOMBINE-empty universality "
            "(FMNIST + CIFAR-10) is fully extended to "
            "{Split-FMNIST, Split-CIFAR-10, Split-CIFAR-100, "
            "Split-Tiny-ImageNet} × {3-layer MLP, 5-layer MLP, "
            "small CNN, medium CNN}. Framework-C claim 'richer ops "
            "yield richer consolidation' empirically refuted across "
            "the full pre-registered four-benchmark scope."
        )
    elif h6c["partial"]:
        which = (
            "H6-A only (CIFAR-100)"
            if s["h6a_confirmed_from_g4_sexto_aggregate"]
            else "H6-B only (Tiny-IN)"
        )
        lines.append(
            f"Partial : {which} confirmed. Universality is "
            "scope-bound to the confirming benchmark ; the "
            "falsifying benchmark restores the predicted DR-4 "
            "ordering at that scale."
        )
    elif h6c["falsified"]:
        lines.append(
            "Falsified : both Welch tests reject H0. RECOMBINE "
            "is empirically empty at smaller scales {FMNIST, "
            "CIFAR-10} but contributes statistically at "
            "{CIFAR-100, Tiny-IN} ; framework-C claim is "
            "rehabilitated for mid-large scales."
        )
    else:
        lines.append(
            "Indeterminate : insufficient samples on at least one leg."
        )
    lines += [
        "",
        "## Verdict — DR-4 evidence",
        "",
        (
            "Per pre-reg §6 : EC stays PARTIAL across all outcomes ; "
            "FC stays at C-v0.12.0. Under H6-C confirmed, the partial "
            "refutation of DR-4 established by G4-ter and "
            "universalised by G4-quinto / G4-sexto is extended to "
            "Tiny-ImageNet at 200-class / 64×64 RGB scale ; DR-4 "
            "evidence v0.6 amends the v0.5 G4-sexto addendum with "
            "the four-benchmark universality flag. Under H6-B "
            "falsified, the universality is shown to break at the "
            "Tiny-ImageNet scale and DR-4 evidence v0.6 records the "
            "boundary."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-septimo aggregator")
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument(
        "--sexto-aggregate",
        type=Path,
        default=DEFAULT_SEXTO_AGGREGATE,
        help=(
            "Path to G4-sexto aggregate JSON (canonical H6-A source). "
            "Required — H6-A is not re-derived in G4-septimo."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    verdict = aggregate_g4_septimo_verdict(
        args.step1, args.sexto_aggregate
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    args.out_md.write_text(_render_md(verdict))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    s = verdict["summary"]
    print(
        f"H6-A : {s['h6a_confirmed_from_g4_sexto_aggregate']}  "
        f"H6-B : {s['h6b_confirmed_from_g4_septimo']}  "
        f"H6-C : {s['h6c_confirmed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
