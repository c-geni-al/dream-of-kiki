# operations/ — dream primitive ops

Source files for the four dream-side primitives (DR-3 §4.2). Each op
has three variants : a skeleton (counters only, no substrate dep),
a `_real.py` (MLX-backed, cycle-3 production path), and a `_snn.py`
(Norse pure-numpy proxy exposing the same Protocol surface so the
SNN substrate satisfies DR-3 condition (1)). `concurrent.py` is the
async dispatcher orbiting the runtime, not a primitive.

## File map

| Op (source) | Skeleton | MLX | SNN | Branch | Math property |
|---|---|---|---|---|---|
| Replay (Stickgold) | `replay.py` | `replay_real.py` | `replay_snn.py` | serial A | gradient-step, **not** idempotent |
| Downscale (Tononi SHY) | `downscale.py` | `downscale_real.py` | `downscale_snn.py` | serial B | commutative, **not** idempotent (compounds) |
| Restructure (Friston FEP) | `restructure.py` | `restructure_real.py` | `restructure_snn.py` | serial D (after A-B) | non-commutative with A-B |
| Recombine (Hobson VAE) | `recombine.py` | `recombine_real.py` | `recombine_snn.py` | parallel (§4.3) | non-deterministic by design |
| Worker | `concurrent.py` | — | — | — | DR-0 single-producer |

Canonical chain : `(replay → downscale → restructure) ∥ recombine`.
Do not reorder serial branch ; do not collapse parallel branch into
the serial one.

## Op contract (all variants)

- Factory pattern : `op_handler(state, [model, …]) → Callable[[DE], None]`.
  State is an external `@dataclass`, mutated in-place by the closure.
- Validate `input_slice` keys **before** any mutation (S3 vocabulary,
  S2 finite, factor bounds). Validation errors must cite the
  invariant ID in the message (e.g. `"S3: …"`).
- DR-0 : every call appends exactly one log entry through
  `runtime.execute()`. The handler itself never logs.
- `_real` variants mutate `model.parameters()` in-place then call
  `mx.eval(...)` to force materialisation. `_snn` variants mutate
  the numpy array passed at factory time (caller keeps the ref).
- K1 tag : `_real` and `_snn` populate `state.last_compute_flops` /
  `state.total_compute_flops`. Skeletons do not — never copy the
  K1 fields into a skeleton-derived dataclass.

## Per-op invariants

| Op | Required key | Bound | Cited invariant |
|---|---|---|---|
| replay | `beta_records: list[{x, y}]` | empty → no-op (no FLOP tag) | S1, K1 |
| downscale | `shrink_factor: float` | `0 < f ≤ 1` | S2, I-Wmag |
| restructure | `topo_op ∈ {add, remove, reroute}` | + op-specific args | S3, I2 |
| recombine | `delta_latents: list[list[float]]` | latent-dim coherent | I3 |

## Anti-patterns

- **Don't add a 4th variant** (e.g. `replay_torch.py`). The three-variant
  layout is the DR-3 conformance surface ; new substrates go behind
  one of the existing variant names via a factory dispatch in
  `kiki_oniric/substrates/`.
- **Don't drop validation before mutation.** A handler that raises
  *after* touching `model.parameters()` leaves the model in a
  partially-shrunk / partially-restructured state and breaks R1.
- **Don't reorder restructure before replay/downscale.** §4.3 forbids
  restructure on yet-restructured topology — losing episodic
  specificity is a silent correctness failure, not a perf hit.
- **Don't import `numpy` into a `_real.py` hot path** ; `_snn.py`
  owns the numpy proxy by design, `_real.py` is MLX-only.
- **Don't reach for the global RNG in `recombine`.** Seed comes via
  `random.Random` injected at factory time — bypassing it breaks
  reproducibility tests under the harness run-registry.
- **Don't bypass `concurrent.py` for parallel dispatch.** Hand-rolled
  threads will silently violate the DR-0 single-producer guarantee
  and the K-QUEUE invariant.

## When adding an op

1. Update `core/primitives.py` Protocol — DualVer **formal** MAJOR.
2. Add skeleton + `_real` + `_snn`, each citing §4.2 in the docstring.
3. Add conformance test under `tests/conformance/` per parent rule.
4. Update `docs/proofs/op-pair-analysis.md` with the new op's
   commutativity / idempotence / branch-placement claim.
