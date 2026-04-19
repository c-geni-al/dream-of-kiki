"""Unit tests for Studyforrest BOLD loader (cycle-3 C3.15).

These tests run WITHOUT `nibabel` installed — the loader uses a
``.npy`` synthetic fixture fallback so the test path is
independent of the heavy neuroimaging dependency. The nibabel
branch is exercised only when the env has it (``pragma: no cover``
on the import-dependent code).

References :
- docs/interfaces/fmri-schema.yaml (v0.7.0+PARTIAL, schema)
- scripts/init_studyforrest_download.sh (NAS target path)
- framework-C spec §6.2 (DR-3 condition 2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.fmri.studyforrest import (
    STUDYFORREST_LOADER_VERSION,
    BoldSeries,
    StudyforrestLoader,
)


def _make_fake_studyforrest_tree(root: Path) -> None:
    """Build a minimal ds000113-shaped tree with one synthetic
    BOLD volume per subject/task. Uses numpy ``.npy`` as the
    on-disk format so nibabel is optional.
    """
    rng = np.random.default_rng(seed=0)
    ds = root / "ds000113"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "dataset_description.json").write_text('{"Name": "stub"}')
    for subject in ("sub-01",):
        func = ds / subject / "ses-movie" / "func"
        func.mkdir(parents=True, exist_ok=True)
        for task in ("task-movie", "task-retmapping"):
            bold_path = func / f"{subject}_{task}_bold.npy"
            np.save(
                bold_path,
                rng.standard_normal((4, 4, 4, 10)).astype(np.float32),
            )
            events_path = func / f"{subject}_{task}_events.tsv"
            events_path.write_text(
                "onset\tduration\n"
                "0.0\t2.0\n"
                "4.0\t2.0\n"
                "8.0\t2.0\n"
            )


def test_studyforrest_loader_validates_root_and_version(tmp_path: Path) -> None:
    """TDD-1 — loader validates required ds000113 sub-dirs, rejects
    missing / empty roots. Exports the DualVer-locked version
    constant matching fmri-schema.yaml header.
    """
    assert STUDYFORREST_LOADER_VERSION == "C-v0.7.0+PARTIAL"

    # Missing root → clear error
    with pytest.raises(FileNotFoundError, match="root"):
        StudyforrestLoader(root_path=tmp_path / "does-not-exist")

    # Non-BIDS root → rejected
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="ds000113"):
        StudyforrestLoader(root_path=empty)

    # Proper skeleton → accepted
    _make_fake_studyforrest_tree(tmp_path)
    loader = StudyforrestLoader(root_path=tmp_path)
    assert loader.root_path == tmp_path
    # dataset_description.json existence is the BIDS-ish marker
    assert (loader.root_path / "ds000113" / "dataset_description.json").exists()


def test_studyforrest_iter_bold_series_deterministic(
    tmp_path: Path,
) -> None:
    """TDD-2 — iter_bold_series yields (x, y, z, t) 4D arrays with
    TR + HRF metadata. Two iterations on the same fixture produce
    the same arrays (R1 contract). Event times are returned in
    TR-frame indices per fmri-schema.yaml.
    """
    _make_fake_studyforrest_tree(tmp_path)
    loader = StudyforrestLoader(root_path=tmp_path, tr_seconds=2.0)

    runs_a = list(loader.iter_bold_series(
        subject_id="sub-01", task="task-movie",
    ))
    runs_b = list(loader.iter_bold_series(
        subject_id="sub-01", task="task-movie",
    ))

    assert len(runs_a) == 1
    series = runs_a[0]
    assert isinstance(series, BoldSeries)
    assert series.data.ndim == 4
    assert series.data.shape == (4, 4, 4, 10)
    assert series.tr_seconds == 2.0
    assert series.subject_id == "sub-01"
    assert series.task == "task-movie"
    np.testing.assert_array_equal(series.data, runs_b[0].data)

    # Missing task → empty iteration (no exception)
    assert list(loader.iter_bold_series("sub-01", "task-missing")) == []

    # Missing subject → empty iteration
    assert list(loader.iter_bold_series("sub-99", "task-movie")) == []


def test_studyforrest_canonical_hrf_is_double_gamma_glover(
    tmp_path: Path,
) -> None:
    """TDD-3 — canonical_hrf returns a deterministic double-gamma
    HRF (Glover 1999). Peaks ~5-6 s post-stimulus, undershoots
    around 12-15 s, integrates to ~1, and is deterministic.
    """
    hrf_a = StudyforrestLoader.canonical_hrf(tr_seconds=1.0, duration_s=32.0)
    hrf_b = StudyforrestLoader.canonical_hrf(tr_seconds=1.0, duration_s=32.0)
    np.testing.assert_array_equal(hrf_a, hrf_b)

    # Shape : duration / tr samples
    assert hrf_a.shape == (32,)
    # Peak is positive and near t ~ 5-6 s
    peak_idx = int(np.argmax(hrf_a))
    assert 4 <= peak_idx <= 7
    # Double-gamma has a negative undershoot between t=10 and t=20
    assert hrf_a[10:20].min() < 0.0

    # Coarser TR halves the resolution
    hrf_coarse = StudyforrestLoader.canonical_hrf(
        tr_seconds=2.0, duration_s=32.0,
    )
    assert hrf_coarse.shape == (16,)

    # Non-positive TR rejected
    with pytest.raises(ValueError, match="tr_seconds"):
        StudyforrestLoader.canonical_hrf(tr_seconds=0.0)


def test_studyforrest_event_times_align_to_tr(tmp_path: Path) -> None:
    """TDD-4 — temporal alignment : event onsets (seconds) map to
    BOLD-frame indices via floor(onset / TR). Matches the schema
    contract ``rdm_computation.determinism: cpu_nilearn_seeded``.
    The iter_bold_series yield exposes ``event_times`` ready for
    HMM alignment downstream (C3.16). Also covers the edge cases
    of the BIDS events.tsv parser (missing / empty / malformed
    files) and the unsupported-file-suffix branch in the BOLD
    loader — all on synthetic fixtures so the production path
    stays deterministic.
    """
    _make_fake_studyforrest_tree(tmp_path)
    loader = StudyforrestLoader(root_path=tmp_path, tr_seconds=2.0)

    series = next(iter(
        loader.iter_bold_series("sub-01", "task-movie"),
    ))

    # TSV in the fixture has onsets at 0.0, 4.0, 8.0 s ; TR=2 s
    # → frames 0, 2, 4.
    np.testing.assert_array_equal(
        series.event_times, np.array([0, 2, 4], dtype=int),
    )
    # event_times are always 1-D int indices
    assert series.event_times.ndim == 1
    assert series.event_times.dtype.kind == "i"

    # Sub-frame rounding check — TR=3 s : 0→0, 4→1, 8→2
    loader3 = StudyforrestLoader(root_path=tmp_path, tr_seconds=3.0)
    series3 = next(iter(
        loader3.iter_bold_series("sub-01", "task-movie"),
    ))
    np.testing.assert_array_equal(
        series3.event_times, np.array([0, 1, 2], dtype=int),
    )

    # Missing events file : the loader yields a series with empty
    # event_times (never raises). Delete the TSV from the fixture.
    tsv = (
        tmp_path / "ds000113" / "sub-01" / "ses-movie" / "func"
        / "sub-01_task-retmapping_events.tsv"
    )
    tsv.unlink()
    series_no_events = next(iter(
        loader.iter_bold_series("sub-01", "task-retmapping"),
    ))
    assert series_no_events.event_times.shape == (0,)

    # Malformed events : header only + non-numeric onsets → empty.
    bad_tsv = (
        tmp_path / "ds000113" / "sub-01" / "ses-movie" / "func"
        / "sub-01_task-bad_events.tsv"
    )
    bad_tsv.write_text("onset\tduration\nfoo\tbar\n\n")
    bad_bold = bad_tsv.parent / "sub-01_task-bad_bold.npy"
    np.save(bad_bold, np.zeros((2, 2, 2, 3), dtype=np.float32))
    series_bad = next(iter(
        loader.iter_bold_series("sub-01", "task-bad"),
    ))
    assert series_bad.event_times.shape == (0,)

    # Empty TSV (zero bytes) → empty frames
    empty_tsv = bad_tsv.parent / "sub-01_task-empty_events.tsv"
    empty_tsv.write_text("")
    empty_bold = bad_tsv.parent / "sub-01_task-empty_bold.npy"
    np.save(empty_bold, np.zeros((2, 2, 2, 3), dtype=np.float32))
    series_empty = next(iter(
        loader.iter_bold_series("sub-01", "task-empty"),
    ))
    assert series_empty.event_times.shape == (0,)

    # Unsupported file suffix is skipped silently by iter_bold_series
    (bad_tsv.parent / "sub-01_task-skip_bold.unknown").write_text("x")
    assert list(loader.iter_bold_series("sub-01", "task-skip")) == []
