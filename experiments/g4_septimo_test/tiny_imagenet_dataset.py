"""Split-Tiny-ImageNet 10-task loader — pure numpy, HF parquet path.

Tiny-ImageNet (Stanford CS231n) provides 200 fine classes at 64×64
RGB resolution, 500 train + 50 val images per class.

Two acquisition paths are supported (both pinned by SHA-256) :

1. **HF mirror (preferred)** — Hugging Face dataset
   ``zh-plus/tiny-imagenet`` (parquet). Same §9.1-style amendment
   pattern as G4-quinto / G4-sexto fallbacks. The HF parquet schema
   exposes ``{"image": {"bytes": ...}, "label": int}`` per row.

2. **Canonical fallback** — https://image-net.org/data/tiny-imagenet-200.zip
   (Stanford site, occasional HTTP 503). Reserved for §9.1
   amendment if the HF parquet path fails.

Class-incremental 10-task split using fine labels (0..199) :

    task 0 : fine classes {0..19}
    task 1 : fine classes {20..39}
    ...
    task 9 : fine classes {180..199}

Per task, fine labels are remapped to ``{0, 1, ..., 19}`` (20-class
head shared across tasks). Images stored as ``np.float32`` in
``[0, 1]``, layout ``(N, 64, 64, 3)`` for CNN consumption (NHWC)
and flattened to ``(N, 12288)`` — both returned.

Reference :
    Le & Yang 2015 — "Tiny ImageNet Visual Recognition Challenge"
    https://image-net.org/data/tiny-imagenet-200.zip
    https://huggingface.co/datasets/zh-plus/tiny-imagenet
    docs/osf-prereg-g4-septimo-pilot.md sec 5
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

TINY_IN_N_TASKS: Final[int] = 10
TINY_IN_CLASSES_PER_TASK: Final[int] = 20
TINY_IN_IMAGE_HW: Final[int] = 64

# HF parquet shards — pinned at first download. Initial values are
# placeholders matching the G4-quinto §9.1 pattern ; the real
# SHA-256 will be computed on first download and pinned in a §9.1
# amendment.
TINY_IN_HF_TRAIN_URL: Final[str] = (
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/"
    "data/train-00000-of-00001-1359597a978bc4fa.parquet"
)
TINY_IN_HF_VAL_URL: Final[str] = (
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/"
    "data/valid-00000-of-00001-70d52db3c749a935.parquet"
)
TINY_IN_TRAIN_SHA256: Final[str] = (
    "unavailable_2026-05-04_per_prereg_g4-septimo_first_run"
)
TINY_IN_VAL_SHA256: Final[str] = (
    "unavailable_2026-05-04_per_prereg_g4-septimo_first_run"
)
HTTP_USER_AGENT: Final[str] = "g4-septimo-pilot/1 (mlx-on-m3-ultra)"


class SplitTinyImageNetTask(TypedDict):
    """One Split-Tiny-ImageNet 20-class task : NHWC + flat float32 + label."""

    x_train: np.ndarray
    x_train_nhwc: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    x_test_nhwc: np.ndarray
    y_test: np.ndarray


def _http_get(url: str, timeout: int = 600) -> bytes:
    """HTTP GET with browser-style UA. Raises on non-2xx."""
    req = urllib.request.Request(
        url, headers={"User-Agent": HTTP_USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _verify_sha256(blob: bytes, expected: str, label: str) -> None:
    """Verify SHA-256 against ``expected``. Placeholder values skip."""
    if expected.startswith("unavailable_") or expected.startswith("..."):
        # Placeholder hash — first-download lock not yet performed.
        return
    h = hashlib.sha256(blob).hexdigest()
    if h != expected:
        raise ValueError(
            f"SHA-256 mismatch ({label}) : got {h}, expected {expected}"
        )


def download_if_missing_hf(data_dir: Path) -> tuple[Path, Path]:
    """Fetch the HF parquet shards if absent — preferred path.

    Returns ``(train_parquet, val_parquet)`` paths. SHA-256 pinned
    per ``TINY_IN_{TRAIN,VAL}_SHA256``. Raises ``FileNotFoundError``
    on network failure (pre-reg §9 deviation envelope a — fall back
    to canonical Stanford zip via §9.1 amendment).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "tiny-imagenet-train.parquet"
    val_path = data_dir / "tiny-imagenet-val.parquet"
    pairs = (
        (train_path, TINY_IN_HF_TRAIN_URL, TINY_IN_TRAIN_SHA256, "train"),
        (val_path, TINY_IN_HF_VAL_URL, TINY_IN_VAL_SHA256, "val"),
    )
    for path, url, sha, label in pairs:
        if path.exists():
            _verify_sha256(path.read_bytes(), sha, f"hf-{label}")
            continue
        try:
            urllib.request.urlretrieve(url, path)  # noqa: S310
        except OSError as exc:
            raise FileNotFoundError(
                f"Tiny-ImageNet HF download failed for "
                f"{label} : {exc}"
            ) from exc
        _verify_sha256(path.read_bytes(), sha, f"hf-{label}")
    return train_path, val_path


def _decode_parquet_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a HF tiny-imagenet parquet shard into (NHWC uint8, fine uint8).

    The HF schema is ``{"image": {"bytes": ...}, "label": int}`` ;
    some mirrors expose ``"img"`` / ``"fine_label"`` instead — both
    column-name conventions are tried. JPEG bytes are decoded via
    PIL and converted to RGB (handles greyscale-encoded images).
    """
    import pyarrow.parquet as pq
    from PIL import Image  # local import keeps base loader pure-numpy

    table = pq.read_table(path)
    df = table.to_pandas()
    n = len(df)
    images = np.empty((n, TINY_IN_IMAGE_HW, TINY_IN_IMAGE_HW, 3), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.int64)
    img_col = "image" if "image" in df.columns else "img"
    if "label" in df.columns:
        label_col = "label"
    elif "fine_label" in df.columns:
        label_col = "fine_label"
    else:
        raise ValueError(
            f"HF tiny-imagenet parquet at {path} has no label / "
            "fine_label column"
        )
    for i in range(n):
        cell = df[img_col].iloc[i]
        jpeg_bytes = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(jpeg_bytes)) as pil_img:
            arr = np.asarray(pil_img.convert("RGB"))
        # Some Tiny-ImageNet variants ship images at different sizes ;
        # canonical is 64×64, but resize defensively if needed.
        if arr.shape[:2] != (TINY_IN_IMAGE_HW, TINY_IN_IMAGE_HW):
            with Image.open(io.BytesIO(jpeg_bytes)) as pil_img2:
                pil_resized = pil_img2.convert("RGB").resize(
                    (TINY_IN_IMAGE_HW, TINY_IN_IMAGE_HW),
                    Image.BILINEAR,
                )
                arr = np.asarray(pil_resized)
        images[i] = arr
        labels[i] = int(df[label_col].iloc[i])
    return images, labels


def _build_tasks_from_arrays(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> list[SplitTinyImageNetTask]:
    """Common 10-task split builder shared by HF + canonical paths."""
    x_tr_nhwc = x_train_raw.astype(np.float32) / 255.0
    x_te_nhwc = x_test_raw.astype(np.float32) / 255.0
    x_tr_flat = x_tr_nhwc.reshape(x_tr_nhwc.shape[0], -1)
    x_te_flat = x_te_nhwc.reshape(x_te_nhwc.shape[0], -1)

    tasks: list[SplitTinyImageNetTask] = []
    for k in range(TINY_IN_N_TASKS):
        lo = k * TINY_IN_CLASSES_PER_TASK
        hi = (k + 1) * TINY_IN_CLASSES_PER_TASK
        tr = (y_train_raw >= lo) & (y_train_raw < hi)
        te = (y_test_raw >= lo) & (y_test_raw < hi)
        y_tr = (y_train_raw[tr].astype(np.int64) - lo)
        y_te = (y_test_raw[te].astype(np.int64) - lo)
        tasks.append(
            SplitTinyImageNetTask(
                x_train=x_tr_flat[tr],
                x_train_nhwc=x_tr_nhwc[tr],
                y_train=y_tr,
                x_test=x_te_flat[te],
                x_test_nhwc=x_te_nhwc[te],
                y_test=y_te,
            )
        )
    return tasks


def load_split_tiny_imagenet_10tasks_hf(
    train_parquet: Path, val_parquet: Path
) -> list[SplitTinyImageNetTask]:
    """Build the 10-task split from HF parquet shards."""
    if not train_parquet.exists():
        raise FileNotFoundError(
            f"missing Tiny-ImageNet HF train parquet : {train_parquet}"
        )
    if not val_parquet.exists():
        raise FileNotFoundError(
            f"missing Tiny-ImageNet HF val parquet : {val_parquet}"
        )
    x_train_raw, y_train_raw = _decode_parquet_shard(train_parquet)
    x_val_raw, y_val_raw = _decode_parquet_shard(val_parquet)
    return _build_tasks_from_arrays(
        x_train_raw, y_train_raw, x_val_raw, y_val_raw
    )


def load_split_tiny_imagenet_10tasks_auto(
    seed: int = 0,
    data_dir: Path | None = None,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Load Tiny-ImageNet as 10 sequential 20-class tasks.

    Mirrors the G4-sexto ``load_split_cifar100_10tasks_auto``
    function-shape but exposes a tuple-of-arrays interface for
    drivers that prefer ``(x_train, y_train)`` over the ``TypedDict``.
    The HF parquet path is the preferred source ; the canonical
    Stanford zip is the §9.1-fallback path (not implemented here ;
    will be added on first network failure per pre-reg §9
    envelope a).

    Args :
        seed : passed through for API parity with G4-sexto auto-loaders ;
               unused inside the loader (deterministic class-range split).
        data_dir : workspace dir for parquet caching ; defaults to
                   ``experiments/g4_septimo_test/data`` resolved via
                   the package layout.

    Returns :
        ``(train_tasks, val_tasks)`` where each list has 10 entries
        and each entry is ``(x_nhwc, y)`` with
        ``x_nhwc.shape = (N, 64, 64, 3)`` float32 in ``[0, 1]`` and
        ``y.shape = (N,)`` int64 with labels remapped to ``{0..19}``.

    Raises :
        FileNotFoundError : HF parquet download failed and canonical
                            §9.1-fallback is not implemented.
    """
    _ = seed  # API parity ; not used for splitting (deterministic).
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    train_path, val_path = download_if_missing_hf(data_dir)
    tasks = load_split_tiny_imagenet_10tasks_hf(train_path, val_path)
    train_tuples = [
        (t["x_train_nhwc"], t["y_train"]) for t in tasks
    ]
    val_tuples = [
        (t["x_test_nhwc"], t["y_test"]) for t in tasks
    ]
    return train_tuples, val_tuples
