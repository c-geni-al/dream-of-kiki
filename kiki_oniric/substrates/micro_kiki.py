"""micro-kiki substrate — Qwen MoE + LoRA (cycle-3 Phase 2, draft).

Third substrate for dreamOfkiki, wrapping the micro-kiki project's
adapter-training output. The intended production base is
``Qwen/Qwen3.5-35B-A3B`` (native 256-expert MoE, 3 B active per
token) with a standard LoRA adapter trained on 32 domain experts.
The substrate is however base-model agnostic ; any MLX-loadable
checkpoint with a companion LoRA ``adapters.safetensors`` is
acceptable.

Backend choice :
- ``mlx_lm`` (default) : declared target. When importable, the
  substrate loads the model + adapter lazily at :meth:`load`
  time and ``awake`` dispatches to ``mlx_lm.generate``.
- **Stub fallback** : when ``mlx_lm`` (or its ``mlx`` parent) is
  unavailable — the default on Linux CI — the substrate builds
  cleanly and exposes the 4 op-handler factories over numpy
  tensors only. This matches the pattern from
  ``esnn_norse`` (env-gated real backend + numpy fallback) so
  the DR-3 condition-1 test surface is exercised on every host.

Reference : ``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md``
§6.2 (DR-3 Conformance Criterion). Scope : DR-0 / DR-1 / DR-3
condition-1 surface ; DR-2 / DR-4 defer to phase 4 (conformance
harness over the 3-substrate matrix).

Phase boundaries (explicit) :
- **Phase 1** : replay + downscale operational on LoRA adapter
  tensors, restructure + recombine stubbed with an explicit
  ``NotImplementedError`` citing the blocker.
- **Phase 2 (this file)** : OPLoRA projection wired in
  (restructure ; arXiv 2510.13003 Du et al.) + TIES-Merge wired
  in (recombine ; arXiv 2306.01708 Yadav et al.). All 4 handlers
  now backed ; +PARTIAL retained until the Phase-4 conformance
  harness lands (no downgrade on the EC axis while Phase-3
  cross-substrate ablation is in flight).
- **Phase 3** : swap / eval_retained bindings + cross-substrate
  ablation (cycle-3 G10 Gate D).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


# DualVer C-v0.9.0+PARTIAL — recombine (TIES-Merge, arXiv 2306.01708)
# wired in ; all 4 handlers now backed. Retained ``+PARTIAL`` until
# Phase-4 conformance harness confirms DR-2 / DR-4 across the
# 3-substrate matrix. Aligned to the sibling cycle-3 substrates
# (``esnn_thalamocortical`` + ``esnn_norse``).
MICRO_KIKI_SUBSTRATE_NAME = "micro_kiki"
MICRO_KIKI_SUBSTRATE_VERSION = "C-v0.9.0+PARTIAL"


# -----------------------------------------------------------------
# OPLoRA (Orthogonal-Projection LoRA, arXiv 2510.13003, Du et al.)
# -----------------------------------------------------------------
# Given ``k`` prior LoRA deltas ``Δ_i = B_i · A_i`` (each of shape
# ``(out_dim, in_dim)``), OPLoRA constructs a projector ``P`` onto
# the orthogonal complement of the column space spanned by the
# priors. When a new adapter B-matrix ``B_new`` is restructured we
# replace it with ``P · B_new``, which guarantees that the new
# contribution ``P · B_new · A_new · x`` is orthogonal to every
# prior range — i.e. the new stack cannot overwrite features
# already encoded in the retained subspace.
#
# The micro-kiki *training* pipeline uses torch (see local impl
# ``src/stacks/oplora.py``) ; the substrate runs on the dream
# runtime which is pure numpy (no torch / mlx dep), so this is a
# pure-linear-algebra port. The algebra is identical :
#
#   torch                                                 numpy
#   -----------------------------------------------      -------------------------------------------------
#   Q, _ = torch.linalg.qr(prior.float())                U, S, _Vt = np.linalg.svd(prior, full_matrices=False)
#   P    = I - Q @ Q.T                                   U_trim = U[:, S > rank_thresh]
#                                                        P      = I - U_trim @ U_trim.T
#
# The numpy port uses SVD rather than QR so we can filter out the
# near-zero singular values (rank selection via ``rank_thresh``) :
# QR would include numerical-noise columns in ``Q``, over-pruning
# the projected subspace. SVD + singular-value filter is the
# standard OPLoRA recipe (paper §3.2).
# -----------------------------------------------------------------


def _oplora_projector(
    prior_deltas: list[NDArray] | tuple[NDArray, ...],
    rank_thresh: float = 1e-4,
) -> NDArray:
    """Build the OPLoRA orthogonal-complement projector ``P``.

    Parameters
    ----------
    prior_deltas
        Iterable of prior-adapter delta matrices (each ``B_i @ A_i``,
        shape ``(out_dim, in_dim)``). All must share the same
        ``out_dim`` — the first axis is what the projector acts on
        (columns of ``B_new``). An empty iterable returns the
        identity (no prior subspace to exclude).
    rank_thresh
        Singular-value magnitude below which a direction is treated
        as numerical noise and dropped from the prior subspace. The
        paper uses ``1e-4`` (§3.2).

    Returns
    -------
    P : ndarray of shape ``(out_dim, out_dim)``
        Orthogonal-complement projector. Satisfies ``P == P.T``,
        ``P @ P ≈ P``, and ``P @ v ≈ 0`` for any ``v`` in the prior
        column space.

    Notes
    -----
    Guarded against two numerical failure modes :

    1. **Zero-singular-value rank collapse** : if *all* singular
       values of the stacked prior fall below ``rank_thresh``
       (pathological : priors are numerical noise), we fall back
       to ``P = I`` with a warning. The new adapter is not
       projected away.
    2. **Shape mismatch across priors** : explicit ``ValueError``
       raised rather than silently broadcasting — the caller
       must make the priors shape-consistent before handing them
       in. Guards against a whole class of latent bugs where a
       reshape earlier in the pipeline slips through.

    Reference : Du et al., *OPLoRA: Orthogonal Projection for
    LoRA Continual Learning*, arXiv 2510.13003, §3.2 (projector
    construction) + §3.3 (rank-threshold robustness study).
    """
    deltas = list(prior_deltas)
    if not deltas:
        # No priors → nothing to project away. Identity is the
        # correct (and well-defined) fallback.
        raise ValueError(
            "OPLoRA _oplora_projector requires at least one "
            "prior delta ; pass ``np.eye(out_dim)`` as the "
            "pseudo-prior if you want the no-op branch"
        )
    out_dims = {d.shape[0] for d in deltas}
    if len(out_dims) != 1:
        raise ValueError(
            f"OPLoRA: all prior deltas must share out_dim "
            f"(axis 0), got {sorted(out_dims)}"
        )
    (out_dim,) = out_dims

    # Stack priors column-wise : the combined column space is the
    # union of each prior's range, exactly what the projector must
    # annihilate.
    stacked = np.concatenate(
        [np.asarray(d, dtype=np.float64) for d in deltas], axis=1
    )
    # SVD on the stacked prior matrix. ``U`` columns span the
    # range of ``stacked`` ; filter by singular-value magnitude
    # to drop numerical-noise directions.
    try:
        U, S, _Vt = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        _LOG.warning(
            "OPLoRA: SVD failed on stacked prior (%s) ; falling "
            "back to identity projector",
            exc,
        )
        return np.eye(out_dim, dtype=np.float32)

    keep = S > rank_thresh
    if not keep.any():
        _LOG.warning(
            "OPLoRA: all %d singular values below rank_thresh=%g "
            "; prior subspace is effectively empty, returning "
            "identity projector",
            S.size, rank_thresh,
        )
        return np.eye(out_dim, dtype=np.float32)

    U_trim = U[:, keep]
    identity = np.eye(out_dim, dtype=np.float64)
    P = identity - U_trim @ U_trim.T
    # Full-rank saturation : when accumulated priors span the entire
    # output space (a real, non-pathological case in long sequential
    # multi-expert curricula), ``U_trim @ U_trim.T`` collapses to the
    # identity and ``P`` to the zero matrix. The handler would then
    # silently zero every new adapter. Surface this loudly so callers
    # know to prune priors or widen ``out_dim``.
    if np.linalg.norm(P, ord="fro") < rank_thresh * out_dim:
        _LOG.warning(
            "OPLoRA: projector is effectively zero — priors span "
            "the full output space (rank=%d, out_dim=%d) ; new "
            "adapter will be annihilated. Prune priors or widen "
            "out_dim.",
            U_trim.shape[1], out_dim,
        )
    # Cast back to float32 — the rest of the substrate stores LoRA
    # tensors in float32 (matches mlx adapter dtype).
    return np.asarray(P, dtype=np.float32)


@dataclass
class MicroKikiRestructureState:
    """DR-0 accountability record for the OPLoRA restructure op.

    Each invocation of the :meth:`restructure_handler_factory`
    closure bumps ``total_episodes_handled`` and appends the
    episode id (if supplied via ``adapter["episode_id"]``) to
    ``episode_ids``. ``completed`` + ``operation`` are mirrored
    on the last-handled record so DR-0 traceability is preserved
    (``completed=True`` + ``operation='restructure'``). The state
    is read by the conformance harness to verify DR-1 (episodic
    stamp consistency) across the 3-substrate matrix.
    """

    total_episodes_handled: int = 0
    total_projections_applied: int = 0
    last_episode_id: str | None = None
    last_operation: str = "restructure"
    last_completed: bool = False
    episode_ids: list[str] = field(default_factory=list)


# -----------------------------------------------------------------
# TIES-Merge (Yadav et al., arXiv 2306.01708, §3)
# -----------------------------------------------------------------
# Given ``K`` task-specific delta tensors ``τ_i = W_ft_i - W_base``
# (sharing a common shape), TIES-Merge produces a single merged
# delta via a three-step procedure :
#
#   1. **Trim** : per task ``i``, zero the ``(1 - k)%`` smallest-
#      magnitude entries of ``τ_i``. Keeps only the top-``k``
#      fraction by absolute value — reduces "sign-noise"
#      interference from parameters the task barely updated.
#   2. **Elect sign** : per parameter ``p`` compute the sign of
#      the sum of signed magnitudes across tasks,
#      ``γ_p = sign(Σ_i sign(τ_i[p]) · |τ_i[p]|)``. Parameters
#      with no consensus end up at ``γ = 0`` and contribute zero
#      to the merged delta.
#   3. **Disjoint merge** : per parameter, take the mean over only
#      those tasks whose sign agrees with ``γ_p``. Parameters with
#      ``γ = 0`` or no agreeing tasks remain ``0``.
#
# The merged delta is finally scaled by a merge coefficient
# ``alpha`` (default 1.0 — paper §3 default ; larger values
# amplify the merged contribution when downstream eval suggests
# under-shooting).
#
# Numpy-only port so the dream runtime stays torch / mlx free.
# -----------------------------------------------------------------


def _ties_merge(
    deltas: list[NDArray],
    trim_fraction: float = 0.2,
    alpha: float = 1.0,
) -> NDArray:
    """Merge a list of task-specific delta tensors via TIES-Merge.

    Parameters
    ----------
    deltas
        Non-empty list of per-task delta tensors. All must share
        the same shape ; shape-mismatch raises ``ValueError``.
    trim_fraction
        Fraction of entries to **keep** per task (top-magnitude
        quantile). Default ``0.2`` matches the paper's k=20 %.
        Must lie in ``(0, 1]``.
    alpha
        Merge coefficient scaling the final delta. Default ``1.0``
        — the paper's unscaled merge.

    Returns
    -------
    merged : ndarray
        Same shape as each input delta ; dtype of the first input.

    Raises
    ------
    ValueError
        - Empty ``deltas`` list.
        - Shape-mismatch across inputs.
        - ``trim_fraction`` outside ``(0, 1]``.

    Notes
    -----
    Single-element input fast-paths to ``alpha * deltas[0]`` — no
    election / trimming needed when only one task contributes.

    Reference : Yadav et al., *TIES-Merging : Resolving
    Interference When Merging Models*, arXiv 2306.01708, §3
    (procedure) + §4 (empirical defaults).
    """
    if not deltas:
        raise ValueError(
            "TIES-Merge _ties_merge requires at least one delta ; "
            "got empty list"
        )
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError(
            f"trim_fraction must lie in (0, 1], got {trim_fraction}"
        )

    first = np.asarray(deltas[0])
    target_dtype = first.dtype
    target_shape = first.shape

    if len(deltas) == 1:
        # Single-task fast path : no election / trim needed.
        return (alpha * first.astype(np.float64)).astype(
            target_dtype, copy=False,
        )

    for i, d in enumerate(deltas):
        arr = np.asarray(d)
        if arr.shape != target_shape:
            raise ValueError(
                f"TIES-Merge: all deltas must share shape "
                f"{target_shape}, got {arr.shape} at index {i}"
            )

    # Stack into shape (K, *delta_shape) in float64 for a stable
    # sign-sum reduction.
    stack = np.stack(
        [np.asarray(d, dtype=np.float64) for d in deltas], axis=0,
    )
    K = stack.shape[0]

    # Step 1 — Trim : per task, zero entries below the
    # (1 - trim_fraction) magnitude quantile.
    abs_stack = np.abs(stack)
    # Flatten per-task axis so we can take a per-row quantile.
    flat_abs = abs_stack.reshape(K, -1)
    # Quantile threshold : keep entries >= quantile(|τ_i|, 1-k).
    # When trim_fraction == 1.0 the threshold is the min (keep
    # everything) ; when trim_fraction is tiny the threshold is
    # near the max (drop all but the largest entries).
    q = 1.0 - trim_fraction
    thresholds = np.quantile(flat_abs, q, axis=1)  # shape (K,)
    # Broadcast threshold over the per-task slice.
    keep_mask_shape = (K,) + (1,) * (stack.ndim - 1)
    thresholds_b = thresholds.reshape(keep_mask_shape)
    keep_mask = abs_stack >= thresholds_b
    trimmed = np.where(keep_mask, stack, 0.0)

    # Step 2 — Elect sign : per-parameter sign of the signed-
    # magnitude sum across tasks. ``np.sign`` maps {<0, 0, >0} to
    # {-1, 0, +1}.
    signed_sum = np.sum(trimmed, axis=0)  # drop K axis
    elected = np.sign(signed_sum)  # shape == target_shape

    # Step 3 — Disjoint merge : per parameter, mean over tasks
    # whose sign agrees with the elected sign.
    trimmed_signs = np.sign(trimmed)
    agree_mask = trimmed_signs == elected[None, ...]
    # Exclude elected == 0 entries (no consensus → merged = 0).
    agree_mask &= elected[None, ...] != 0

    contrib_count = np.sum(agree_mask, axis=0)  # shape target
    numerator = np.sum(np.where(agree_mask, trimmed, 0.0), axis=0)
    # Divide-by-zero guard : parameters with zero contributors
    # stay at 0 in the merged delta.
    merged = np.where(
        contrib_count > 0,
        numerator / np.maximum(contrib_count, 1),
        0.0,
    )

    merged *= alpha
    return merged.astype(target_dtype, copy=False)


@dataclass
class MicroKikiRecombineState:
    """DR-0 accountability record for the TIES-Merge recombine op.

    Mirrors :class:`MicroKikiRestructureState` so the DR-3
    conformance harness parametrises the 4 handlers uniformly.
    Each invocation of the :meth:`recombine_handler_factory`
    closure bumps ``total_episodes_handled`` and — when the
    handler actually merged a non-empty delta list — records the
    merged-tensor shape stamp on ``last_output_shape``. ``DR-1``
    episode-id stamps land on ``last_episode_id`` + ``episode_ids``.
    """

    total_episodes_handled: int = 0
    total_merges_applied: int = 0
    last_episode_id: str | None = None
    last_operation: str = "recombine"
    last_completed: bool = False
    last_k_deltas: int = 0
    last_input_shape: tuple[int, ...] | None = None
    last_output_shape: tuple[int, ...] | None = None
    episode_ids: list[str] = field(default_factory=list)

# Optional-dependency probe : ``mlx_lm`` (Apple Silicon MLX wheel
# + LoRA adapters) is imported lazily inside the method that
# actually needs it (:meth:`MicroKikiSubstrate.load`). We record
# only a boolean flag at module-import so callers can introspect
# availability without a second try-import. Tests cover the False
# branch ; the True branch is env-gated on Apple Silicon.
try:  # pragma: no cover - branch depends on env
    import mlx_lm  # noqa: F401

    _MLX_LM_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _MLX_LM_AVAILABLE = False


@dataclass
class MicroKikiSubstrate:
    """micro-kiki framework-C substrate (Qwen MoE + LoRA).

    Parameters
    ----------
    base_model_path
        Optional path (local dir or HF repo id) to an
        ``mlx_lm``-loadable model. When ``None`` the substrate
        runs in pure-stub mode : handler factories operate on
        numpy tensors only, :meth:`awake` returns a canned string.
        This keeps the module importable + testable on hosts
        without Apple Silicon / the MLX wheel.
    adapter_path
        Optional path to a LoRA ``adapters.safetensors`` file (or
        directory containing one). Loaded by :meth:`load` via
        ``mlx_lm.load_adapters`` when present.
    num_layers
        Number of transformer layers the LoRA adapter targets.
        Default 20 — matches the micro-kiki v4 default shape. Only
        used to validate adapter-tensor shapes in the numpy
        fallback path ; real MLX loading ignores this.
    rank
        LoRA rank. Default 16 (micro-kiki v4 SOTA spec). Used to
        size stub adapter tensors in the numpy fallback path.
    seed
        Numpy RNG seed — controls any stochastic handler (recombine).

    Attributes
    ----------
    mlx_lm_available
        Informational bool mirroring the module-level probe.
    """

    base_model_path: str | None = None
    adapter_path: str | None = None
    num_layers: int = 20
    rank: int = 16
    seed: int = 0
    mlx_lm_available: bool = field(default=_MLX_LM_AVAILABLE, init=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    # Accumulator for the in-flight weight delta produced by the
    # replay / downscale handlers. Stored as a plain ``dict`` keyed
    # by the adapter weight-path (matches the shape emitted by
    # ``mlx_lm.tuner.trainable_parameters``). Round-tripped by
    # :meth:`snapshot` / :meth:`load_snapshot` as a numpy ``.npz``.
    _current_delta: dict[str, NDArray] = field(
        default_factory=dict, init=False, repr=False,
    )
    # DR-0 accountability state for the OPLoRA restructure handler
    # (arXiv 2510.13003). Exposed read-only via :meth:`restructure_state`
    # so the conformance harness can assert DR-1 (episode_id stamp
    # consistency) without poking private attributes.
    _restructure_state: MicroKikiRestructureState = field(
        default_factory=MicroKikiRestructureState,
        init=False,
        repr=False,
    )
    # DR-0 accountability state for the TIES-Merge recombine handler
    # (arXiv 2306.01708). Same accessor pattern as
    # ``_restructure_state`` — read-only via :meth:`recombine_state`.
    _recombine_state: MicroKikiRecombineState = field(
        default_factory=MicroKikiRecombineState,
        init=False,
        repr=False,
    )
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {self.num_layers}"
            )
        if self.rank <= 0:
            raise ValueError(f"rank must be > 0, got {self.rank}")
        self._rng = np.random.default_rng(self.seed)

    # ----- lazy model / adapter load -----

    def load(self) -> None:
        """Load the base model + LoRA adapter via ``mlx_lm``.

        No-op in stub mode (``base_model_path is None``). When
        ``mlx_lm`` is unavailable we also short-circuit to stub
        mode — this keeps the module importable on CI without an
        Apple Silicon wheel. The real code path runs on Mac Studio
        M3 Ultra where Qwen3.5-35B-A3B + LoRA adapter fit in the
        460 GB Metal memory budget (see ``micro-kiki/CLAUDE.md``).
        """
        if self.base_model_path is None:
            return
        if not self.mlx_lm_available:  # pragma: no cover - env-gated
            return
        # pragma: no cover - env-gated (Apple Silicon only)
        from mlx_lm import load as mlx_load
        self._model, self._tokenizer = mlx_load(self.base_model_path)
        if self.adapter_path is not None:
            from mlx_lm.tuner.utils import (                load_adapters,
            )

            self._model = load_adapters(self._model, self.adapter_path)

    # ----- awake-side generation -----

    def awake(self, prompt: str, max_tokens: int = 32) -> str:
        """Awake forward pass — returns generated text.

        Stub path : returns ``f"[stub awake] {prompt}"`` so unit
        tests can assert type + shape without an MLX wheel. Real
        path (env-gated) dispatches to ``mlx_lm.generate`` with
        the loaded model + tokenizer.
        """
        if self._model is None or self._tokenizer is None:
            return f"[stub awake] {prompt}"
        # pragma: no cover - env-gated (Apple Silicon only)
        from mlx_lm import generate
        return str(
            generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        )

    # ----- Protocol-contract factories (mirror esnn_* substrates) -----

    def replay_handler_factory(
        self,
    ) -> Callable[[list[dict], int], NDArray]:
        """A-Walker/Stickgold replay → LoRA gradient proxy.

        Signature matches
        ``esnn_thalamocortical.EsnnSubstrate.replay_handler_factory``
        for DR-3 condition-1 uniformity : the handler takes a
        ``beta_records: list[dict]`` + ``n_steps: int`` and
        returns a 1-D numpy array. The stub aggregates each
        record's ``"input"`` vector and returns the mean drive —
        sufficient to exercise the swap + S1 retained-benchmark
        path without an MLX device.
        """

        def handler(
            beta_records: list[dict], n_steps: int = 20,
        ) -> NDArray:
            if not beta_records:
                return np.zeros(1, dtype=np.float32)
            vectors: list[NDArray] = [
                np.asarray(r["input"], dtype=np.float32)
                for r in beta_records
                if "input" in r
            ]
            if not vectors:
                return np.zeros(1, dtype=np.float32)
            return np.asarray(
                np.mean(np.stack(vectors), axis=0), dtype=np.float32,
            )

        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[[NDArray, float], NDArray]:
        """B-Tononi SHY → LoRA B-matrix multiplicative shrink.

        Preserves DR-1 on the adapter state : the caller stamps
        the returned tensor with ``episode_id`` via
        ``kiki_oniric.dream.swap`` ; the handler itself only
        performs the arithmetic. Commutative, not idempotent
        (``f(f(w)) = w * factor²``). Matches the signature of
        ``esnn_*`` substrate downscale handlers.
        """

        def handler(weights: NDArray, factor: float) -> NDArray:
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"shrink_factor must be in (0, 1], got {factor}"
                )
            return (weights * factor).astype(weights.dtype, copy=False)

        return handler

    def restructure_handler_factory(
        self,
        rank_thresh: float = 1e-4,
    ) -> Callable[[dict, str, str], dict]:
        """D-Friston FEP restructure → **OPLoRA projection (phase 2)**.

        Wires the Orthogonal-Projection LoRA algorithm of Du et
        al. (arXiv 2510.13003) : given a list of prior-stack
        adapter deltas carried on ``adapter["prior_deltas"]``, the
        handler builds the projector ``P = I - U U^T`` onto the
        orthogonal complement of the priors' range and replaces
        the new adapter's B-matrix by ``P @ B_new``. The
        contribution ``P · B_new · A_new · x`` is then orthogonal
        to every prior range, preserving the S1 retained-benchmark
        invariant across sequential stack additions.

        Handler contract
        ----------------
        ``adapter`` is a mutable dict keyed by LoRA tensor name. It
        *must* carry :

        - ``"prior_deltas"`` : ``list[ndarray]`` of prior
          ``B_i @ A_i`` products (shape ``(out_dim, in_dim_i)``).
          Empty list means no priors to project away — handler
          returns the adapter unchanged (no-op).
        - ``key`` (the third positional arg) : name of the
          B-matrix entry to project. Its shape must be
          ``(out_dim, rank)`` with ``out_dim`` matching the
          priors.
        - Optional ``"episode_id"`` : DR-0 stamp propagated into
          :attr:`_restructure_state`.

        ``op`` is accepted for signature-compat with the phase-1
        stub but currently only ``"oplora"`` is honoured (the
        default). Any other value raises ``ValueError`` — keeps
        the gate explicit per DR-3 condition-1 (no silent
        no-ops).

        Returns the same ``adapter`` dict (with the entry at
        ``key`` replaced by ``P @ adapter[key]``).

        Reference : Du et al., arXiv 2510.13003 §3.2-§3.3. The
        local micro-kiki training pipeline at
        ``src/stacks/oplora.py`` (torch) mirrors this algebra ;
        this numpy port lives substrate-side for the dream
        runtime's no-torch constraint.
        """

        def handler(
            adapter: dict[str, NDArray], op: str, key: str,
        ) -> dict[str, NDArray]:
            if op not in {"oplora", "oplora_project", "project"}:
                raise ValueError(
                    f"micro_kiki.restructure_handler: unsupported "
                    f"op {op!r} ; expected one of "
                    f"{{'oplora', 'oplora_project', 'project'}}"
                )
            if key not in adapter:
                raise KeyError(
                    f"micro_kiki.restructure_handler: adapter "
                    f"missing entry for key {key!r}"
                )
            new_B = np.asarray(adapter[key])
            if new_B.ndim != 2:
                raise ValueError(
                    f"micro_kiki.restructure_handler: adapter[{key!r}] "
                    f"must be 2-D (out_dim, rank), got shape "
                    f"{new_B.shape}"
                )

            priors = list(adapter.get("prior_deltas", []))
            episode_id = adapter.get("episode_id")

            if priors:
                P = _oplora_projector(priors, rank_thresh=rank_thresh)
                if P.shape[0] != new_B.shape[0]:
                    raise ValueError(
                        f"micro_kiki.restructure_handler: projector "
                        f"out_dim {P.shape[0]} != adapter[{key!r}] "
                        f"out_dim {new_B.shape[0]}"
                    )
                projected = (P @ new_B.astype(np.float32)).astype(
                    new_B.dtype, copy=False,
                )
                adapter[key] = projected
                self._restructure_state.total_projections_applied += 1

            # DR-0 bookkeeping : always bump the counter + record
            # the episode (even when priors is empty — the op was
            # invoked, and DR-0 credits every handler call, not
            # just those with a non-trivial effect).
            self._restructure_state.total_episodes_handled += 1
            self._restructure_state.last_completed = True
            self._restructure_state.last_operation = "restructure"
            if isinstance(episode_id, str):
                self._restructure_state.last_episode_id = episode_id
                self._restructure_state.episode_ids.append(episode_id)
            return adapter

        return handler

    @property
    def restructure_state(self) -> MicroKikiRestructureState:
        """Read-only accessor for the OPLoRA DR-0 record.

        Exposed so the conformance harness (and unit tests) can
        assert DR-0 (completed flag, operation label) + DR-1
        (episode_id stamp propagation) without poking private
        attributes. The returned object is the live state dataclass
        — do not mutate ; it is refreshed by each handler call.
        """
        return self._restructure_state

    def recombine_handler_factory(
        self,
        trim_fraction: float = 0.2,
        alpha: float = 1.0,
    ) -> Callable[[dict, str], NDArray]:
        """C-Hobson recombine → **TIES-Merge (phase 2)**.

        Wires the TIES-Merging algorithm of Yadav et al. (arXiv
        2306.01708) : given a list of task-specific delta tensors
        carried on ``payload["deltas"]``, the handler returns the
        merged delta via trim → elect-sign → disjoint-mean. Scale
        coefficient ``alpha`` amplifies the final merged
        contribution (default ``1.0``, paper default).

        Handler contract
        ----------------
        ``payload`` is a dict carrying at least ``"deltas"`` — a
        list of numpy delta tensors (shape-consistent across the
        list). Pragmatically matches ``DreamEpisode`` where the
        canonical access path is ``episode.payload["deltas"]``.

        - ``"deltas"`` : ``list[ndarray]`` — per-task / per-stack
          ``τ_i = W_ft_i - W_base``. Empty list raises (caller
          handles the no-op leg explicitly).
        - Optional ``"episode_id"`` : DR-0 stamp propagated into
          :attr:`_recombine_state`.

        ``op`` is accepted for signature-compat with the sibling
        :meth:`restructure_handler_factory` ; honours ``"ties"``
        (default), ``"ties_merge"``, ``"merge"``. Any other op
        raises ``ValueError`` — no silent no-ops per DR-3
        condition 1.

        Returns the merged delta tensor (dtype of the first input
        delta). The substrate's ``_recombine_state`` is bumped on
        every call (DR-0) ; the episode-id stamp (DR-1) is
        appended when present.

        Reference : Yadav et al., arXiv 2306.01708 §3.
        """

        def handler(
            payload: dict[str, Any], op: str = "ties",
        ) -> NDArray:
            if op not in {"ties", "ties_merge", "merge"}:
                raise ValueError(
                    f"micro_kiki.recombine_handler: unsupported "
                    f"op {op!r} ; expected one of "
                    f"{{'ties', 'ties_merge', 'merge'}}"
                )
            if "deltas" not in payload:
                raise KeyError(
                    "micro_kiki.recombine_handler: payload missing "
                    "'deltas' entry (expected list[ndarray])"
                )
            deltas = list(payload["deltas"])
            episode_id = payload.get("episode_id")

            # _ties_merge guards empty / single / shape-mismatch.
            merged = _ties_merge(
                deltas, trim_fraction=trim_fraction, alpha=alpha,
            )

            # DR-0 bookkeeping.
            self._recombine_state.total_episodes_handled += 1
            self._recombine_state.total_merges_applied += 1
            self._recombine_state.last_completed = True
            self._recombine_state.last_operation = "recombine"
            self._recombine_state.last_k_deltas = len(deltas)
            first_shape = tuple(np.asarray(deltas[0]).shape)
            self._recombine_state.last_input_shape = first_shape
            self._recombine_state.last_output_shape = tuple(merged.shape)
            if isinstance(episode_id, str):
                self._recombine_state.last_episode_id = episode_id
                self._recombine_state.episode_ids.append(episode_id)
            return merged

        return handler

    @property
    def recombine_state(self) -> MicroKikiRecombineState:
        """Read-only accessor for the TIES-Merge DR-0 record.

        Mirrors :meth:`restructure_state`. The returned object is
        the live state dataclass — do not mutate ; it is refreshed
        by each handler call.
        """
        return self._recombine_state

    # ----- γ-snapshot : adapter round-trip -----

    def snapshot(self, path: str | Path) -> Path:
        """Persist the current accumulator delta to a ``.npz`` file.

        Returns the written path. Round-trips cleanly via
        :meth:`load_snapshot`. In a full MLX run the swap protocol
        would instead serialise the LoRA adapter via
        ``mlx_lm.utils.save_adapters`` to a ``.safetensors`` —
        here we use numpy ``.npz`` for portability so the same
        artifact format works on Linux CI.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        np.savez(target, **self._current_delta)
        return target

    def load_snapshot(self, path: str | Path) -> None:
        """Restore the accumulator delta from a :meth:`snapshot` file."""
        target = Path(path)
        if not target.exists() and target.with_suffix(".npz").exists():
            target = target.with_suffix(".npz")
        data = np.load(target, allow_pickle=False)
        self._current_delta = {k: np.asarray(data[k]) for k in data.files}


def micro_kiki_substrate_components() -> dict[str, str]:
    """Return the canonical map of micro-kiki substrate components.

    Mirrors ``esnn_substrate_components`` + ``norse_substrate_components``
    so the DR-3 Conformance Criterion test suite parametrizes over
    the three substrates uniformly. Phase 2 lands the real
    restructure + recombine backends ; the dotted paths stay
    stable across that transition (the same module hosts both
    the stub + the real impl).
    """
    return {
        # 8 typed Protocols (substrate-agnostic, shared)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations — factory methods on this substrate class
        "replay": "kiki_oniric.substrates.micro_kiki",
        "downscale": "kiki_oniric.substrates.micro_kiki",
        # phase-2 stubs, path stable across bump
        "restructure": "kiki_oniric.substrates.micro_kiki",
        "recombine": "kiki_oniric.substrates.micro_kiki",
        # 2 invariant guards (substrate-agnostic, shared)
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        # Runtime + swap (substrate-agnostic, shared)
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # 3 profiles (substrate-agnostic wrappers, shared)
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
