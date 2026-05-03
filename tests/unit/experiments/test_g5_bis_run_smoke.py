"""Smoke test for the G5-bis pilot driver (Plan G5-bis Task 4)."""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np


def _make_synthetic_fmnist(tmp_path: Path, n_train: int = 200) -> Path:
    """Drop a 4x4 / 10-class IDX fixture under ``tmp_path/data``.

    The G5-bis driver reuses ``load_split_fmnist_5tasks`` (same as
    G4-ter) so a tiny IDX fixture is enough for the smoke run.
    """
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
    """4 arms x 2 seeds = 8 cells in < 60 s on synthetic 4x4 IDX."""
    from experiments.g5_bis_richer_esnn.run_g5_bis import run_pilot

    data_dir = _make_synthetic_fmnist(tmp_path)
    out_json = tmp_path / "g5-bis.json"
    out_md = tmp_path / "g5-bis.md"
    registry_db = tmp_path / "registry.sqlite"

    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=8,
        hidden_1=8,
        hidden_2=6,
        lr=0.05,
        n_steps=3,
    )
    assert out_json.exists()
    assert out_md.exists()
    cells = json.loads(out_json.read_text())["cells"]
    assert len(cells) == 8  # 4 arms x 2 seeds
    verdict = payload["verdict"]
    assert sorted(verdict["retention_by_arm"]) == [
        "P_equ",
        "P_max",
        "P_min",
        "baseline",
    ]
    assert "h7a_richer_esnn" in verdict
    assert payload["wall_time_s"] < 120
