"""Three RECOMBINE strategies for G4-quater Step 3 (H4-C test).

- ``mog`` — per-class Gaussian-MoG sampler (port of G4-ter
  ``_recombine_step`` MoG branch). Synthetic latents are drawn from
  ``N(mean_c, std_c)`` per class.
- ``ae`` — deterministic single-pass autoencoder ; encoder + decoder
  are two ``np.linalg``-style ``Linear`` blocks operating *in numpy*
  to keep the unit cheap and seedable. One MSE pass over the buffer
  produces a small reconstructed-sample set with the original
  per-record labels.
- ``none`` — placebo : returns ``None`` ; the dream-episode wrapper
  must skip the SGD pass on the output classifier when it sees
  ``None``. **This is the H4-C control isolating REPLAY+DOWNSCALE.**

The ``ae`` strategy is intentionally lightweight (numpy) to avoid
re-creating an MLX optimizer per cell ; the goal is *to expose a
non-MoG sampler* for the H4-C verdict, not to tune a strong AE.

Reference :
    docs/osf-prereg-g4-quater-pilot.md sec 2 (H4-C)
    docs/superpowers/plans/2026-05-03-g4-quater-restructure-recombine-test.md
"""
from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

RecombineStrategy = Literal["mog", "ae", "none"]


class SyntheticLatentBatch(TypedDict):
    """Synthetic latent batch returned by ``sample_synthetic_latents``."""

    x: np.ndarray
    y: np.ndarray


def _sample_mog(
    latents: np.ndarray,
    labels: np.ndarray,
    n_synthetic: int,
    seed: int,
) -> SyntheticLatentBatch | None:
    if latents.shape[0] == 0:
        return None
    classes = sorted({int(c) for c in labels.tolist()})
    if len(classes) < 2:
        return None
    rng = np.random.default_rng(seed)
    components: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for c in classes:
        mask = labels == c
        arr = latents[mask].astype(np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-6
        components[c] = (mean, std)
    per_class = max(1, n_synthetic // len(classes))
    synth_x: list[np.ndarray] = []
    synth_y: list[int] = []
    for c in classes:
        mean, std = components[c]
        for _ in range(per_class):
            noise = rng.standard_normal(size=mean.shape).astype(np.float32)
            synth_x.append(mean + std * noise)
            synth_y.append(c)
    return {
        "x": np.stack(synth_x).astype(np.float32),
        "y": np.asarray(synth_y, dtype=np.int32),
    }


def _sample_ae(
    latents: np.ndarray,
    labels: np.ndarray,
    n_synthetic: int,
    seed: int,
) -> SyntheticLatentBatch | None:
    """Deterministic 1-step AE in numpy.

    Encoder ``W_enc : (d, d/2)`` + decoder ``W_dec : (d/2, d)`` are
    initialised from ``rng.standard_normal`` with seed
    ``seed + 40_000``. One MSE-gradient step (lr = 1e-2) is taken
    over the supplied latents ; the encoder + decoder are then used
    to reconstruct ``n_synthetic`` records sampled (with
    replacement) from ``latents``. Labels are forwarded from the
    sampled records.

    Returns ``None`` if ``latents`` is empty.
    """
    n, d = latents.shape
    if n == 0:
        return None
    bottleneck = max(1, d // 2)
    rng = np.random.default_rng(seed + 40_000)
    w_enc = rng.standard_normal(size=(d, bottleneck)).astype(np.float32) * 0.1
    w_dec = rng.standard_normal(size=(bottleneck, d)).astype(np.float32) * 0.1

    # One MSE step (closed-form gradient on a tiny linear AE).
    z = latents @ w_enc
    recon = z @ w_dec
    err = recon - latents
    lr = 1e-2
    grad_dec = z.T @ err / n
    grad_enc = latents.T @ (err @ w_dec.T) / n
    w_enc -= lr * grad_enc
    w_dec -= lr * grad_dec

    # Sample n_synthetic record indices (with replacement).
    idx = rng.integers(low=0, high=n, size=n_synthetic)
    chosen = latents[idx]
    chosen_labels = labels[idx].astype(np.int32)
    z2 = chosen @ w_enc
    recon2 = z2 @ w_dec
    return {
        "x": recon2.astype(np.float32),
        "y": chosen_labels,
    }


def sample_synthetic_latents(
    *,
    strategy: RecombineStrategy,
    latents: np.ndarray,
    labels: np.ndarray,
    n_synthetic: int,
    seed: int,
) -> SyntheticLatentBatch | None:
    """Dispatch on ``strategy`` and return synthetic latents.

    ``latents`` shape : ``(N, d)`` ; ``labels`` shape : ``(N,)``.
    Returns ``None`` when no synthetic batch is produced (placebo
    ``none``, empty buffer, or single-class buffer for ``mog``).
    """
    if strategy == "none":
        return None
    if strategy == "mog":
        return _sample_mog(latents, labels, n_synthetic, seed)
    if strategy == "ae":
        return _sample_ae(latents, labels, n_synthetic, seed)
    raise ValueError(f"unknown RECOMBINE strategy : {strategy!r}")
