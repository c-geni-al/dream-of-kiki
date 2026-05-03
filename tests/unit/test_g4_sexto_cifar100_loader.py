"""Unit tests for G4-sexto CIFAR-100 loader (synthetic tmp_path fixture)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.g4_sexto_test.cifar100_dataset import (
    CIFAR100_RECORD_SIZE,
    decode_cifar100_bin,
    load_split_cifar100_10tasks,
)


def _write_batch(
    path: Path, fine_labels: list[int], rng: np.random.Generator
) -> None:
    rows: list[bytes] = []
    for fl in fine_labels:
        coarse = fl // 5  # arbitrary deterministic mapping for fixture
        img = rng.integers(0, 256, size=3072, dtype=np.uint8).tobytes()
        rows.append(bytes([coarse, fl]) + img)
    path.write_bytes(b"".join(rows))


def test_decode_cifar100_bin_shape(tmp_path: Path) -> None:
    f = tmp_path / "train.bin"
    _write_batch(f, [0, 9, 10, 99], np.random.default_rng(0))
    images, fine = decode_cifar100_bin(f)
    assert images.shape == (4, 32, 32, 3)
    assert images.dtype == np.uint8
    assert fine.tolist() == [0, 9, 10, 99]


def test_decode_cifar100_bin_truncated_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * (CIFAR100_RECORD_SIZE - 1))
    with pytest.raises(ValueError, match="truncated"):
        decode_cifar100_bin(f)


def test_decode_cifar100_bin_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    images, fine = decode_cifar100_bin(f)
    assert images.shape == (0, 32, 32, 3)
    assert fine.shape == (0,)


def test_load_split_cifar100_10tasks_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bin_dir = tmp_path / "cifar-100-binary"
    bin_dir.mkdir()
    # 2 examples per fine class so each task has data.
    _write_batch(
        bin_dir / "train.bin",
        [c for c in range(100) for _ in range(2)],
        rng,
    )
    _write_batch(bin_dir / "test.bin", list(range(100)), rng)
    tasks = load_split_cifar100_10tasks(bin_dir)
    assert len(tasks) == 10
    for k, task in enumerate(tasks):
        assert task["x_train_nhwc"].shape[1:] == (32, 32, 3)
        assert task["x_train_nhwc"].dtype == np.float32
        assert task["x_train"].shape[1] == 3072
        assert set(task["y_train"].tolist()) <= set(range(10))
        # remap : fine labels {10k..10k+9} -> {0..9}
        assert task["x_train_nhwc"].shape[0] == 20
        assert task["x_test_nhwc"].shape[0] == 10
        # remap correctness : the unique sorted labels in the
        # task must be exactly {0..9}.
        assert sorted(set(task["y_train"].tolist())) == list(range(10))


def test_load_split_cifar100_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split_cifar100_10tasks(tmp_path / "nope")


def test_load_split_cifar100_missing_test_raises(tmp_path: Path) -> None:
    bin_dir = tmp_path / "cifar-100-binary"
    bin_dir.mkdir()
    _write_batch(
        bin_dir / "train.bin", [0, 1], np.random.default_rng(0)
    )
    with pytest.raises(FileNotFoundError, match="test"):
        load_split_cifar100_10tasks(bin_dir)
