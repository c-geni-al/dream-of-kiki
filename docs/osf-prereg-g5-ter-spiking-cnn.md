# OSF Pre-Registration — G5-ter pilot (spiking-CNN cross-substrate test)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G5-ter pilot — disambiguate the G5-bis H7-B verdict
  by porting the G4-quinto Step 2 small-CNN architecture onto the
  E-SNN substrate as a 4-layer spiking CNN (Conv2d-LIF + Conv2d-LIF
  + FC-LIF + Linear, STE backward) and testing whether convolutional
  inductive bias recovers the cross-arm positive effect that the
  3-layer LIF MLP failed to express.
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G5-ter run is registered in
  `harness/storage/run_registry.RunRegistry`

## 0. Background

Two prior cycle-3 results frame this pilot :

1. **G5-bis H7-B verdict** (milestone
   `docs/milestones/g5-bis-aggregate-2026-05-03.json`,
   `h7_classification == "H7-B"`, `g_h7a_esnn ≈ +0.1043`). On a
   3-layer LIF MLP (`EsnnG5BisHierarchicalClassifier`,
   784 → 32 → 16 → 2) with the four-op coupling (REPLAY +
   DOWNSCALE + RESTRUCTURE + RECOMBINE) at HP combo C5, the
   own-substrate Welch one-sided test failed to reject the null at
   α/4 = 0.0125 — i.e. the G4-ter MLX richer-head positive effect
   (`g_h2 = +2.77`) did not transfer to the E-SNN MLP. H7-B is
   classified "MLX-only artefact for the LIF MLP at this protocol
   scale". Per Critic precedent, fail-to-reject is read as
   absence-of-evidence, not evidence-of-absence.

2. **G4-quinto Step 2 H5-B (CNN MLX retention level)** (milestone
   `docs/milestones/g4-quinto-step2-2026-05-03.json`,
   `verdict.retention_by_arm` carrying 30 floats per arm in
   `{baseline, P_min, P_equ, P_max}`). The MLX small-CNN
   reproduces the cycle-3 positive-effect channel on Split-CIFAR-10
   under the four-op coupling. This pilot is the cross-substrate
   reference for the H8 Welch-two-sided gap.

The H8 question — does the E-SNN washout in G5-bis stem from (i)
the LIF rate-coding non-linearity itself, or (ii) the architectural
mismatch between the MLX dense head and the LIF MLP — is only
answerable by holding the architecture fixed (G4-quinto Step 2 CNN)
and toggling the substrate (MLX → E-SNN). G5-ter ports the
small-CNN onto E-SNN as a spiking CNN with three LIF stages and
STE backward, then runs the same 4-arm × 10-seed sweep on the
same Split-CIFAR-10 5-task benchmark at the same HP combo C5.

## 1. Purpose — three pre-registered hypotheses (locked)

The decision rule is exhaustive : every observed (`g_h8`,
own-Welch outcome, `g_p_equ_cross`) tuple maps to exactly one
classification.

### H8-A — LIF non-linearity is the load-bearing washout mechanism

**Statement** : the G4-ter / G4-quinto cycle-3 positive effect
fails to transfer through E-SNN rate-coding *regardless of
architectural depth*. The G5-bis MLP washout (H7-B) is reproduced
by the G5-ter spiking CNN, despite the convolutional inductive
bias. The substrate-level non-linearity is the load-bearing
washout mechanism for the cycle-3 positive-effect channel.

**Operationalization** :
- `g_h8 = compute_hedges_g(retention[P_equ, esnn_cnn], retention[baseline, esnn_cnn])`
- `g_h8 < H7B_G_THRESHOLD = 0.5`
- own-substrate Welch one-sided fails to reject at α/4 = 0.0125
- cross-substrate (MLX small-CNN minus E-SNN spiking-CNN) :
  `g_p_equ_cross >= H8A_G_MLX_MINUS_ESNN_MIN = 2.0` (large gap
  in absolute retention level)

### H8-B — architecture-dependent : CNN closes the gap

**Statement** : the G5-bis MLP washout was an architecture-mismatch
artefact ; once the architecture is convolutional, the LIF stack
recovers the positive cross-arm effect at near-MLX absolute level.

**Operationalization** :
- `g_h8 >= H7B_G_THRESHOLD = 0.5`
- own-substrate Welch one-sided rejects at α/4 = 0.0125
- cross-substrate gap closes :
  `g_p_equ_cross < H8B_G_MLX_MINUS_ESNN_MAX = 1.0`

### H8-C — partial : both contribute

**Statement** : architectural inductive bias contributes partially
but does not fully close the LIF gap. Either own-substrate
detects an effect but the level still diverges, or the MLP
washout is reduced but not eliminated.

**Operationalization** : anything that does not match H8-A or
H8-B exactly. Reported with the observed (`g_h8`,
`g_p_equ_cross`) pair for post-hoc inspection.

### Decision rule (locked, verbatim)

```
classify(g_h8, own_welch_reject, g_p_equ_cross):
    if g_h8 < 0.5 and not own_welch_reject and g_p_equ_cross >= 2.0:
        return "H8-A"
    if g_h8 >= 0.5 and own_welch_reject and g_p_equ_cross < 1.0:
        return "H8-B"
    return "H8-C"
```

Thresholds 0.5 / 1.0 / 2.0 are LOCKED at pre-reg time and may not
be moved post hoc. `H7B_G_THRESHOLD = 0.5` is reused verbatim from
the G5-bis pre-reg ; the new H8A / H8B knobs gate the cross-substrate
gap interpretation.

## 2. Coupling map

The four-op coupling is transposed from the G5-bis MLP onto the
spiking CNN :

| Operation | G5-bis MLP target | G5-ter CNN target |
|-----------|-------------------|-------------------|
| REPLAY | SGD-with-STE on `(W_in, W_h, W_out)` | SGD-with-STE on full network (8 tensors) |
| DOWNSCALE | multiply `(W_in, W_h, W_out)` by `factor` | multiply all 8 tensors `{W_c1, b_c1, W_c2, b_c2, W_fc1, b_fc1, W_out, b_out}` by `factor` |
| RESTRUCTURE | add `factor·sigma·N(0,1)` to `W_h` only | add `factor·sigma·N(0,1)` to `W_c2` only |
| RECOMBINE | MoG-synthetic latents → one CE-loss SGD step on `W_out` only | MoG-synthetic latents (dim 64) → one CE-loss SGD step on `(W_out, b_out)` only |

Bound checks identical to G5-bis : `factor ∈ (0, 1]` for
DOWNSCALE, `factor >= 0` for RESTRUCTURE.

DR-0 spectator runtime path is preserved : `profile.runtime.execute(
episode)` is called BEFORE any substrate-side mutation, so every
episode appends one `EpisodeLogEntry` to `profile.runtime.log`
regardless of substrate-side outcome.

## 3. Substrate

`EsnnG5TerSpikingCNN` — 4-layer pure-numpy spiking CNN. NHWC
layout throughout. Architecture :

```
(N, 32, 32, 3)
  -> Conv2d(3->16, 3x3, pad=1) -> LIF rates  (N, 32, 32, 16)
  -> Conv2d(16->32, 3x3, pad=1) -> LIF rates (N, 32, 32, 32)
  -> avg_pool 4x4 (deterministic)            (N, 8, 8, 32)
  -> flatten + Linear(2048, 64) -> LIF       (N, 64)
  -> Linear(64, 2) (no LIF)                  (N, 2) logits
```

LIF defaults : `tau = 10.0`, `threshold = 1.0`, `n_steps = 20`
(same as G5-bis). Pooling is **average-pool 4×4** — parameter-free
and fully-differentiable through the STE branch (replaces the MLX
`MaxPool2d`, which has no clean numpy backward).

Conv2d forward and backward are pure-numpy via an im2col-style
matmul (NHWC, square kernels). STE backward applies on every LIF
stage : `d_currents = d_rates` (Wu 2018). No `mlx`, no `torch`.

## 4. Dataset

Split-CIFAR-10 5-task (class-incremental binary head) :

| Task | Classes | Labels |
|------|---------|--------|
| 0 | airplane, automobile | (0, 1) |
| 1 | bird, cat | (2, 3) |
| 2 | deer, dog | (4, 5) |
| 3 | frog, horse | (6, 7) |
| 4 | ship, truck | (8, 9) |

Loader : `experiments.g4_quinto_test.cifar10_dataset.load_split_cifar10_5tasks_auto`
— SHA-256 pinned, with HF parquet fallback (uoft-cs/cifar10) per
G4-quinto §9.1 deviation envelope. Images stored as `np.float32`
in `[0, 1]`, layout `(N, 32, 32, 3)` (NHWC).

## 5. Sweep design (Option B, locked)

| Parameter | Value |
|-----------|-------|
| Arms | `["baseline", "P_min", "P_equ", "P_max"]` |
| Seeds | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (Option B, N=10) |
| Cells | 4 arms × 10 seeds × 1 HP combo = **40** |
| HP combo | `representative_combo()` C5 (downscale_factor=0.95, replay_batch=32, replay_n_steps=5, replay_lr=0.01) |
| Classifier | `EsnnG5TerSpikingCNN(n_classes=2, n_steps=20, tau=10.0, threshold=1.0)` |
| Beta buffer | `EsnnCNNBetaBuffer` capacity 256, fill 32 per task |
| RESTRUCTURE_FACTOR | 0.05 |
| RECOMBINE_N_SYNTHETIC | 16 |
| RECOMBINE_LR | 0.01 |
| Training HPs | `epochs=2, batch_size=64, lr=0.05` (matches G5-bis) |

## 6. Statistics

- Own-substrate H8-A : Welch one-sided of `baseline` against
  `P_equ` (alternative : `P_equ > baseline`) at Bonferroni
  α/4 = 0.0125. Hedges' g via `compute_hedges_g`.
- Cross-substrate per-arm : Welch two-sided between MLX small-CNN
  retention (G4-quinto Step 2 milestone, key `retention_by_arm`)
  and E-SNN spiking-CNN retention (G5-ter milestone, key
  `retention_by_arm`), four arms at α/4 = 0.0125.
- Hedges' g convention : `g_p_equ_cross = compute_hedges_g(mlx_p_equ, esnn_p_equ)`
  (positive sign means MLX exceeds E-SNN at the P_equ arm).

## 7. R1 reproducibility

- `c_version = "C-v0.12.0+PARTIAL"` (no FC bump).
- Every cell registers `(c_version, profile, seed, commit_sha)`
  via `harness.storage.run_registry.RunRegistry` — bit-stable
  `run_id`. Profile prefix : `g5-ter/spiking_cnn/{arm}`.
- Retentions are deterministic given the registered tuple :
  Conv2d uses pure-numpy float32, no MLX in the substrate path.

## 8. Outputs

- `docs/milestones/g5-ter-spiking-cnn-2026-05-03.json` — per-cell
  records + own-substrate H8-A verdict.
- `docs/milestones/g5-ter-spiking-cnn-2026-05-03.md` — markdown
  rendering of the per-cell milestone.
- `docs/milestones/g5-ter-aggregate-2026-05-03.json` —
  cross-substrate Welch two-sided table + H8-A/B/C classification.
- `docs/milestones/g5-ter-aggregate-2026-05-03.md` — markdown
  rendering of the aggregate verdict.
- `docs/proofs/dr3-substrate-evidence.md` — append-only DR-3
  evidence revision per H8 outcome.
- Paper 2 §7.1.10 (EN + FR) — section narrative anchored to the
  observed verdict.

## 9. Deviation envelope

1. **CIFAR-10 acquisition** — the canonical Toronto mirror
   (`https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`) may
   return a non-2xx response in the run window. Falls back to the
   HF parquet mirror (`uoft-cs/cifar10`) per G4-quinto pre-reg
   §9.1 deviation envelope. SHA-256 pin held by
   `experiments/g4_quinto_test/cifar10_dataset.py`.
2. **Insufficient samples after exclusions** — abort the H8
   classification (return `"ambiguous"`) if `n_p_equ < 2` or
   `n_baseline < 2` after the `acc_task1_initial < 0.5` exclusion.

No other deviation is permitted without a dated amendment under
`docs/osf-deviations-g5-ter-<date>.md`.

## 10. Cross-references

- G5-bis pre-reg : `docs/osf-prereg-g5-bis-richer-esnn.md`
- G4-quinto pre-reg : `docs/osf-prereg-g4-quinto-pilot.md`
- Plan : `docs/superpowers/plans/2026-05-03-g5-ter-spiking-cnn-washout-test.md`
- DR-3 spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- DR-3 evidence : `docs/proofs/dr3-substrate-evidence.md`

## 11. DualVer outcome rules (binding)

| Outcome | EC | FC | Rationale |
|---------|----|----|-----------|
| Any of H8-A / H8-B / H8-C | stays PARTIAL | stays C-v0.12.0 | Option B is exploratory ; no axiom or primitive change |

Confirmatory Option A (N≥30) is scheduled if H8-B or H8-C ; that
follow-up may bump EC to STABLE on H8-B. No FC bump is triggered
by this pilot regardless of H8 outcome.

## 12. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G5-ter cell is registered in the
run registry.**
