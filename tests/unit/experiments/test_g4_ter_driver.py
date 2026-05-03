"""Driver smoke tests for the G4-ter pilot (Plan Task 8)."""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

from experiments.g4_ter_hp_sweep.run_g4_ter import run_pilot


def _make_synthetic_fmnist(tmp_path: Path, n_train: int = 600) -> Path:
    """Drop a 4x4 / 10-class IDX fixture under ``tmp_path/data``."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    img_train = rng.integers(0, 256, size=(n_train, 4, 4), dtype=np.uint8)
    lbl_train = np.array([i % 10 for i in range(n_train)], dtype=np.uint8)
    img_test = rng.integers(0, 256, size=(n_train // 4, 4, 4), dtype=np.uint8)
    lbl_test = np.array(
        [i % 10 for i in range(n_train // 4)], dtype=np.uint8
    )
    for arr, kind in (
        (img_train, "train-images-idx3-ubyte.gz"),
        (lbl_train, "train-labels-idx1-ubyte.gz"),
        (img_test, "t10k-images-idx3-ubyte.gz"),
        (lbl_test, "t10k-labels-idx1-ubyte.gz"),
    ):
        with gzip.open(data_dir / kind, "wb") as fh:
            if arr.ndim == 3:
                fh.write(struct.pack(">IIII", 2051, arr.shape[0], 4, 4))
                fh.write(arr.tobytes())
            else:
                fh.write(struct.pack(">II", 2049, arr.shape[0]))
                fh.write(arr.tobytes())
    return data_dir


def test_run_pilot_smoke(tmp_path: Path) -> None:
    """Smoke run: 1 HP combo (C5) x 2 seeds x 4 arms = 8 cells, plus 1
    HP sweep cell per non-baseline arm = 3 cells. Total = 11 cells in
    < 60 s. Confirms the pilot writes both JSON and Markdown."""
    data_dir = _make_synthetic_fmnist(tmp_path)
    out_json = tmp_path / "g4-ter.json"
    out_md = tmp_path / "g4-ter.md"
    registry_db = tmp_path / "registry.sqlite"

    payload = run_pilot(
        data_dir=data_dir,
        seeds_richer=(0, 1),
        seeds_hp=(0,),
        hp_combo_ids=("C5",),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=32,
        lr=0.01,
        smoke=True,
    )
    assert out_json.exists()
    assert out_md.exists()
    cells = json.loads(out_json.read_text())["cells_richer"]
    assert len(cells) == 8  # 4 arms x 2 seeds, C5 only
    cells_hp = json.loads(out_json.read_text())["cells_hp"]
    assert len(cells_hp) == 3  # 3 dream arms x 1 seed x 1 HP combo
    # Verdict block has H1, H2, H_DR4-ter keys
    verdict = payload["verdict"]
    assert "h1_hp_artefact" in verdict
    assert "h2_substrate_richer" in verdict
    assert "h_dr4_ter_richer" in verdict
