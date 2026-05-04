"""Unit tests for the pilot-label derivation in the G6 driver.

Closes the partial-dump filename collision documented in
``docs/osf-prereg-g6-studio-path-a-star.md`` §9.2 amendment :
concurrent or sequential pilots in the G6 family must each get
a distinct namespace for their per-subdomain partial dumps.
"""
from __future__ import annotations

from pathlib import Path

from experiments.g6_studio_path_a.run_g6_studio_path_a import (
    DEFAULT_PILOT_LABEL,
    derive_pilot_label,
)


def test_derive_label_path_a() -> None:
    out = Path("docs/milestones/g6-studio-path-a-2026-05-04.json")
    assert derive_pilot_label(out) == "g6-studio-path-a"


def test_derive_label_path_a_star() -> None:
    out = Path("docs/milestones/g6-studio-path-a-star-2026-05-04.json")
    assert derive_pilot_label(out) == "g6-studio-path-a-star"


def test_derive_label_path_c() -> None:
    out = Path("docs/milestones/g6-studio-path-c-2026-05-04.json")
    assert derive_pilot_label(out) == "g6-studio-path-c"


def test_derive_label_path_d_mmlu() -> None:
    out = Path("docs/milestones/g6-m1max-path-d-mmlu-2026-05-04.json")
    assert derive_pilot_label(out) == "g6-m1max-path-d-mmlu"


def test_derive_label_path_d_synth() -> None:
    out = Path("docs/milestones/g6-m1max-path-d-synth-2026-05-04.json")
    assert derive_pilot_label(out) == "g6-m1max-path-d-synth"


def test_derive_label_unexpected_pattern_falls_back() -> None:
    out = Path("docs/milestones/no-date-suffix.json")
    assert derive_pilot_label(out) == DEFAULT_PILOT_LABEL


def test_derive_label_accepts_string() -> None:
    label = derive_pilot_label(
        "docs/milestones/g6-studio-path-c-2026-05-04.json",
    )
    assert label == "g6-studio-path-c"


def test_distinct_labels_for_distinct_pilots() -> None:
    # The collision the §9.2 amendment closes : path-a vs
    # path-a-star vs path-c vs m1max-path-d must all differ.
    labels = {
        derive_pilot_label("a/g6-studio-path-a-2026-05-04.json"),
        derive_pilot_label("a/g6-studio-path-a-star-2026-05-04.json"),
        derive_pilot_label("a/g6-studio-path-c-2026-05-04.json"),
        derive_pilot_label("a/g6-m1max-path-d-mmlu-2026-05-04.json"),
        derive_pilot_label("a/g6-m1max-path-d-synth-2026-05-04.json"),
    }
    assert len(labels) == 5
