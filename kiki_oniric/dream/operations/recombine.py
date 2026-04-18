"""Recombine operation — C-Hobson VAE light source (creative branch).

Skeleton "light" version (S11.1): linear interpolation between two
randomly-sampled latents from `delta_latents` input. Real VAE
sampling (encoder/decoder pair) lands S13+ alongside concurrent
dream worker.

Mathematical role (per docs/proofs/op-pair-analysis.md): canonical
parallel branch (§4.3) — recombine runs in parallel with the
serial A-B-D branch to preserve generative diversity. Sampling is
non-deterministic by design; rng injection enables reproducible
tests.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class RecombineOpState:
    """Mutable state for recombine op across episodes."""

    total_episodes_handled: int = 0
    total_samples_emitted: int = 0
    last_sample: list[float] | None = None
    sample_history: list[list[float]] = field(default_factory=list)


def _interpolate(
    a: list[float], b: list[float], alpha: float
) -> list[float]:
    """Linear interpolation: alpha*a + (1-alpha)*b component-wise.

    Raises `ValueError` annotated with invariant I3 (latent
    distributional drift bounded — requires consistent latent
    dimensions) when the two latents have mismatched length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"I3: latent dimensions mismatch: {len(a)} vs {len(b)}"
        )
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)]


def recombine_handler(
    state: RecombineOpState,
    rng: random.Random | None = None,
) -> Callable[[DreamEpisode], None]:
    """Build a recombine handler bound to a state instance.

    Handler reads `delta_latents` from input_slice (must contain
    >= 2 latents), samples 2 distinct indices via rng, interpolates
    with alpha ~ U(0, 1), updates state. Real VAE sampling lands
    S13+ with MLX integration.

    Preserves : DR-0 (every call bumps `total_episodes_handled`
    and appends to `sample_history`), I3 (latent dimension
    consistency — guarded by `_interpolate`), and DR-4 (recombine
    op is part of the P_equ/P_max chain).
    """
    if rng is None:
        rng = random.Random()

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        # Invariant I3 (input shape): need >= 2 latents to
        # interpolate.
        if len(latents) < 2:
            raise ValueError(
                f"I3: delta_latents must contain at least 2 "
                f"latents, got {len(latents)}"
            )
        idx_a, idx_b = rng.sample(range(len(latents)), 2)
        alpha = rng.random()
        sample = _interpolate(latents[idx_a], latents[idx_b], alpha)
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)

    return handler


def recombine_handler_mlx(
    state: RecombineOpState,
    encoder,
    decoder,
    seed: int = 0,
) -> Callable[[DreamEpisode], None]:
    """Build a recombine handler with real MLX VAE-light sampling.

    Encoder is expected to return ``(mu, log_var)`` — i.e. the
    second tensor is the log of the *variance* of the latent
    distribution (the sigma used in reparameterization is then
    ``exp(0.5 * log_var)``). If your encoder instead returns
    ``(mu, log_std)``, wrap it in an adapter that applies
    ``log_var = 2 * log_std`` before calling this handler.

    Sampling uses the MLX reparameterization trick : ``z = mu +
    sigma * epsilon`` with ``epsilon ~ N(0, I)``. The decoder
    maps ``z`` to an output sample.

    Reproducibility is delivered by deriving epsilon from a local
    PRNG key (``mx.random.key`` + ``mx.random.split``) keyed by
    ``seed + state.total_episodes_handled``. The global MLX RNG
    is *not* mutated — concurrent dream workers can therefore run
    multiple recombine handlers without interfering.

    Skeleton handler preserved for tests / contexts not requiring
    MLX. This MLX variant produces real generative samples for the
    G4 GO-FULL gate (canal 2 output).

    Preserves : DR-0, DR-4, I3 (latent shape).

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        # Invariant I3 (input shape): need >= 2 latents to
        # interpolate / reparameterize.
        if len(latents) < 2:
            raise ValueError(
                f"I3: delta_latents must contain at least 2 "
                f"latents, got {len(latents)}"
            )

        # Local PRNG key for isolated reproducibility — does not
        # touch the process-wide MLX RNG.
        key_seed = seed + state.total_episodes_handled
        key = mx.random.key(key_seed)
        _, sample_key = mx.random.split(key)

        # Pick first latent as encoder input (deterministic choice
        # — diversity comes from sampling z, not from latent
        # selection).
        x = mx.array(latents[0])
        mu, log_var = encoder(x)
        sigma = mx.exp(0.5 * log_var)
        epsilon = mx.random.normal(shape=mu.shape, key=sample_key)
        z = mu + sigma * epsilon
        sample_arr = decoder(z)
        mx.eval(sample_arr)

        sample = [float(v) for v in sample_arr.tolist()]
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)

    return handler
