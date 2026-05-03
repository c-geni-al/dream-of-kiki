"""G4-sexto aggregator — load 2 step milestones and emit verdicts.

Per pre-reg sec 2 :

- **H6-A** : Welch two-sided between (P_max with mog) and
  (P_max with none) on Split-CIFAR-100, alpha = 0.0167. Failure
  to reject confirms H6-A.
- **H6-B** : same on Split-Tiny-ImageNet, alpha = 0.0167. May be
  deferred under the locked Option B path.
- **H6-C** : derived conjunction `H6-A_confirmed AND
  H6-B_confirmed`. No additional Welch test ; H6-C is logical
  aggregation. Three resolution states :
    - confirmed : both H6-A and H6-B confirmed
    - partial : exactly one of {H6-A, H6-B} confirmed AND step2
      has a verdict (i.e. NOT deferred)
    - deferred : Option B path, step2 absent

Outputs :
    docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_STEP1 = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step1-2026-05-03.json"
)
DEFAULT_STEP2 = (
    REPO_ROOT / "docs" / "milestones" / "g4-sexto-step2-2026-05-03.json"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-sexto-aggregate-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g4-sexto-aggregate-2026-05-03.md"
)

HONEST_READING_H6A = (
    "Welch fail-to-reject = absence of evidence at this N for a "
    "difference between mog and none — under H6-A specifically, "
    "this **is** the predicted positive empirical claim that "
    "RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on "
    "the small CNN substrate at CIFAR-100 100-class scale."
)
HONEST_READING_H6B = (
    "Welch fail-to-reject = absence of evidence at this N for a "
    "difference between mog and none — under H6-B specifically, "
    "this **is** the predicted positive empirical claim that "
    "RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on "
    "the medium CNN substrate at Tiny-ImageNet 200-class scale."
)


def aggregate_g4_sexto_verdict(
    step1_path: Path,
    step2_path: Path | None,
) -> dict[str, Any]:
    """Load step milestones and return aggregate H6-A/B/C verdicts.

    ``step2_path`` may be ``None`` to support compute Option B
    (Step 1 only) ; the H6-B block is then deferred and the
    H6-C confirmed flag resolves to False (deferred, not partial).
    Treat missing-on-disk paths as deferred (same as None).
    """
    s1 = json.loads(step1_path.read_text())
    s2: dict[str, Any] | None
    if step2_path is not None and step2_path.exists():
        s2 = json.loads(step2_path.read_text())
    else:
        s2 = None

    h6a = s1["verdict"]["h6a_recombine_strategy"]
    h6a_confirmed = (
        not h6a.get("insufficient_samples", False)
        and bool(h6a.get("h6a_recombine_empty_confirmed"))
    )

    h6b_block: dict[str, Any]
    h6b_deferred: bool
    h6b_confirmed: bool
    if s2 is None:
        h6b_block = {"deferred": True, "confirmed": False}
        h6b_deferred = True
        h6b_confirmed = False
    else:
        h6b = s2["verdict"]["h6b_recombine_strategy"]
        h6b_confirmed = (
            not h6b.get("insufficient_samples", False)
            and bool(h6b.get("h6b_recombine_empty_confirmed"))
        )
        h6b_block = {**h6b, "deferred": False, "confirmed": h6b_confirmed}
        h6b_deferred = False

    h6c_confirmed = h6a_confirmed and h6b_confirmed
    # H6-C "partial" : exactly one leg confirmed AND step 2 has a
    # verdict (not deferred). Under deferral the second leg is an
    # open empirical question, not a falsification, so the
    # conjunction is "deferred" (not "partial" or "confirmed").
    h6c_partial = (
        (h6a_confirmed != h6b_confirmed)  # XOR
        and not h6b_deferred
    )

    return {
        "h6a_cifar100": {**h6a, "confirmed": h6a_confirmed},
        "h6b_tiny_imagenet": h6b_block,
        "h6c_universality": {
            "confirmed": h6c_confirmed,
            "partial": h6c_partial,
            "deferred": h6b_deferred,
        },
        "summary": {
            "h6a_confirmed": h6a_confirmed,
            "h6b_confirmed": h6b_confirmed,
            "h6c_confirmed": h6c_confirmed,
            "h6c_partial": h6c_partial,
            "h6b_deferred": h6b_deferred,
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
        else "deferred" if s["h6b_deferred"]
        else "falsified"
    )

    lines: list[str] = [
        "# G4-sexto aggregate verdict",
        "",
        "**Date** : 2026-05-03",
        "**Pre-registration** : "
        "[docs/osf-prereg-g4-sexto-pilot.md]"
        "(../osf-prereg-g4-sexto-pilot.md)",
        "",
        "## Summary",
        "",
        f"- H6-A (CIFAR-100, 100-class scale) confirmed : **{s['h6a_confirmed']}**",
        (
            "- H6-B (Tiny-ImageNet, 200-class scale) confirmed : "
            f"**{s['h6b_confirmed']}**"
            + (" (deferred — Option B locked)" if s["h6b_deferred"] else "")
        ),
        f"- H6-C (universality conjunction) state : **{h6c_state}**",
        f"- H6-C confirmed : **{s['h6c_confirmed']}**",
        f"- H6-C partial : **{s['h6c_partial']}**",
        f"- H6-C deferred : **{s['h6b_deferred']}**",
        (
            "- H5-C → H6-C universality extension "
            "(4 benchmarks × 4 substrates) : "
            f"**{s['h5c_to_h6c_universality_extension']}**"
        ),
        "",
        "## H6-A — CIFAR-100 (n_classes=10 per task, G4SmallCNN)",
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
            "",
            f"*Honest reading* : {HONEST_READING_H6A}",
        ]
    lines += [
        "",
        "## H6-B — Tiny-ImageNet (n_classes=20 per task, G4MediumCNN)",
        "",
    ]
    if h6b.get("deferred"):
        lines.append(
            "DEFERRED (compute Option B locked ; Step 2 will run "
            "in a G4-septimo follow-up)."
        )
    elif h6b.get("insufficient_samples"):
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
            "Both H6-A and H6-B confirmed. The G4-quinto H5-C "
            "RECOMBINE-empty universality (FMNIST + CIFAR-10) is "
            "extended to {Split-FMNIST, Split-CIFAR-10, "
            "Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, "
            "5-layer MLP, small CNN, medium CNN}. Framework-C "
            "claim 'richer ops yield richer consolidation' "
            "empirically refuted across 4 benchmarks × 4 "
            "substrates."
        )
    elif h6c["partial"]:
        which = "H6-A only (CIFAR-100)" if s["h6a_confirmed"] else "H6-B only (Tiny-IN)"
        lines.append(
            f"Partial : {which} confirmed. Universality is "
            "scope-bound to the confirming benchmark ; the "
            "falsifying benchmark restores the predicted DR-4 "
            "ordering at mid-large scale."
        )
    elif h6c["deferred"]:
        lines.append(
            "Deferred : Option B locked at pre-reg ; Tiny-ImageNet "
            "step deferred to G4-septimo. The H6-C conjunction "
            "is an open empirical question. Under H6-A confirmed, "
            "universality is provisionally extended to "
            "{FMNIST, CIFAR-10, CIFAR-100} × {3-layer MLP, "
            "5-layer MLP, small CNN}, pending Tiny-IN evidence."
        )
    else:
        lines.append(
            "Falsified : both Welch tests reject H0. RECOMBINE "
            "is empirically empty at {FMNIST, CIFAR-10} but "
            "contributes statistically at {CIFAR-100, Tiny-IN} ; "
            "framework-C claim is rehabilitated for mid-large "
            "scales."
        )
    lines += [
        "",
        "## Verdict — DR-4 evidence",
        "",
        (
            "Per pre-reg §6 : EC stays PARTIAL across all outcomes ; "
            "FC stays at C-v0.12.0. Under H6-A confirmed (the locked "
            "Option B success path), the partial refutation of DR-4 "
            "established by G4-ter and universalised by G4-quinto "
            "is further extended to CIFAR-100 at 100-class scale. "
            "Under the deferred Option B path, the H6-C conjunction "
            "is incomplete ; STABLE promotion remains blocked "
            "pending Tiny-IN / ImageNet-1k / transformer / "
            "hierarchical E-SNN follow-ups (pre-reg §6 row 6 of "
            "G4-quinto)."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-sexto aggregator")
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument(
        "--step2",
        type=Path,
        default=DEFAULT_STEP2,
        help=(
            "Step 2 milestone path (Option A) or '/dev/null' / "
            "missing path (Option B, deferred)."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    step2: Path | None = args.step2
    # Treat /dev/null or any non-existing path as Option B deferral.
    if step2 is None or str(step2) == "/dev/null" or not step2.exists():
        step2 = None

    verdict = aggregate_g4_sexto_verdict(args.step1, step2)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    args.out_md.write_text(_render_md(verdict))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    s = verdict["summary"]
    print(
        f"H6-A : {s['h6a_confirmed']}  "
        f"H6-B : {s['h6b_confirmed']}  "
        f"H6-C : {s['h6c_confirmed']}  "
        f"deferred : {s['h6b_deferred']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
