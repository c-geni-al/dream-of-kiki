# G6-M1Max Path D pilot pre-registration (Helium-1 2B cross-family substrate)

**Date:** 2026-05-04
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-regs:**
- `docs/osf-prereg-g6-studio-path-a.md` (Path A, INSUFFICIENT, synthetic 10/10 fixture).
- `docs/osf-prereg-g6-studio-path-a-star.md` (Path A*, in flight at lock time, real MMLU 50/50 fixture).
- `docs/osf-prereg-g6-studio-path-c.md` (Path C, locked but not launched, symbolic CL 50/50 fixture).
**Substrate:** **Helium-1 2B dense** (Kyutai), MLX-loadable from HF `kyutai/helium-1-2b` (~3.86 GB safetensors, single-file). MLX-native architecture supported by upstream `mlx_lm 0.31.2` ; the KIKI-Mac_tunner fork's `tuner.trainer.train` API is still used for the LoRA fine-tune (rsynced to `/tmp/mlx_lm/` on M1 Max for this pilot). Apache 2.0 license, fully open-weight (publicly reproducible, in contrast to SpikingKiki-V4 which is internal).
**Adapter stack:** **No pre-existing LoRA stack** (Helium has no shipped V4-equivalent adapters). The pilot uses fresh-init LoRA layers via `linear_to_lora_layers(model, num_layers=rank, config={"rank": 8, "scale": 2.0, "dropout": 0})` — equivalent to Path A's fresh-init path (`apply_lora_layers=True` on first cell, `idx==0`).
**Benchmark:** Both real MMLU (`tests/fixtures/mmlu_g6_real.jsonl`, 796 records) and symbolic CL (`tests/fixtures/symbolic_cl_g6.jsonl`, 500 records) — Path D runs **two steps**, one per fixture, to span the convergent-validation axes.
**Compute:** N = 5 seeds × 4 arms × 5 sub-domains × 2 fixtures = **200 measurements** (40 cells). Per-cell wall ~3-5 min on M1 Max (Helium-2B is ~17× smaller than Qwen-35B-A3B, so roughly that-much faster). Total ~2-3 h on M1 Max.
**Lock commit:** *(filled by introducing commit hash)*
**Lock timestamp:** 2026-05-04 (pre-driver-run, post Path C lock).

## §1 Background — substrate-axis convergent validation

Path A INSUFFICIENT verdict (commit `d963b24`) and Path A* / Path C in flight. The framework's RECOMBINE-empty universality is closed at the four-benchmark × four-CNN/MLP scope ceiling (G4-septimo H6-C confirmed, commit `c8dd268`). All real-LLM tier evidence comes from a *single* substrate : Qwen-35B-A3B (MoE).

Two confounding axes remain :
- **Scale axis** : <1M params (toy E-SNN G5-bis) → 35B (G6 Path A/A*/C). Five-orders-of-magnitude gap.
- **Family axis** : MoE (Qwen-35B-A3B) only at real-LLM tier. No dense LLM tested.

A null verdict on G6 Path A*/C cannot disentangle scale from family : if framework C effects do not survive at 35B-MoE, is it because of the MoE gating, or just because of the scale ?

**Path D cross-family substrate validation** : repeat the identical pipeline on **Helium-1 2B dense** (Kyutai). Three benefits :
1. **Bridge the scale gap.** 2B sits between toy E-SNN and 35B by ~3 orders of magnitude on each side ; if framework C effect emerges at 2B but disappears at 35B, the wash-out is *scale-locked*. If it disappears at both, *family* and *scale* together act as wash-out.
2. **Isolate the family effect.** Helium-1 is dense Mistral-style, NOT MoE. If the effect appears on Helium-2B-dense and not on Qwen-35B-A3B-MoE at the same N, the wash-out is *MoE-locked*.
3. **Public reproducibility.** Helium-1 2B is Apache 2.0 + open-weight on HF. Path D's run can be rerun by external reviewers without access to the SpikingKiki-V4 internal artifact.

The Hu 2020 anchor (g = 0.29) remains a directional reference, not a magnitude calibrator (cross-class biological-vs-LLM magnitude calibration is a category error).

## §2 Hypotheses (confirmatory)

- **H11-A_MMLU (Helium-2B dense + MMLU)** — `retention(P_max with mog)` is statistically distinguishable from `retention(P_max with none)` with predicted positive sign and `Hedges' g ≥ 0.5`. Welch two-sided rejects H0 at α = 0.05. H11-A_MMLU strict large-effect threshold `g ≥ 2`.
- **H11-A_synth (Helium-2B dense + symbolic CL)** — same decision rule on the symbolic CL fixture.
- **H11-A* (exploratory medium band)** : either step with `0.5 ≤ g < 2`.
- **H11-B (Helium-2B real-LLM wash-out)** — fail-to-reject H0 OR `g < 0.5` on EITHER step. Framework C effect does not survive at the dense-2B tier.
- **H11-C (negative-direction)** — rejection with negative sign on EITHER step.
- **H11-D (cross-family / cross-benchmark conjunction)** — derived. Logical aggregation of {H9_path_a_star, H10_path_c, H11_helium_mmlu, H11_helium_synth}. Resolution states :
  - `confirmed_positive_universal` iff all four return A or A* with consistent positive sign and `min(g) ≥ 0.5`.
  - `confirmed_null_universal` iff all four return B.
  - `confirmed_negative_universal` iff all four return C with consistent negative sign.
  - `scale_locked_positive` iff Helium-2B dense (H11) confirms positive but Qwen-35B (H9 + H10) returns null. Framework C has scale-bounded validity ; wash-out at MoE-35B is real.
  - `family_locked_positive` iff Helium-2B dense confirms positive AND Qwen-35B-A3B (MoE) returns null at *same N* — interpreted as MoE-gating disrupts framework C.
  - `benchmark_locked` iff one fixture (MMLU vs symbolic CL) confirms and the other does not, **on the same substrate** — benchmark-dependent effect.
  - `divergent` (catch-all) any other combination.
  - `unresolved` if any path returns INSUFFICIENT.

## §3 Power analysis

N = 5 seeds per arm at α = 0.05 detects |g| ≥ 1.85 at 80 % power. The strict H11-A confirmation threshold `g ≥ 2` is at the detection floor at this N. H11-A* (medium band) captures sub-strict positives. Sub-medium effects (g < 0.5) collapse to H11-B.

H11-D is logical aggregation, no Welch test, no Bonferroni adjustment beyond per-path α.

## §4 Exclusion criteria

Identical to Path A. Underperforming-baseline rule unchanged (`acc[S_1 after S_1] < UNDERPERFORM_THRESHOLD = 0.30`). Helium-2B is much smaller than Qwen-35B but still a real LLM — the underperforming-baseline floor should be reachable on real MMLU after 50 LoRA iters.

## §5 Substrate / driver paths

- Driver : **re-uses** `experiments/g6_studio_path_a/run_g6_studio_path_a.py` unchanged. Same fork API binding, same Metal cache config, same dream-handler wiring, same exclusion logic.
- Substrate : `--base-path /Users/electron/.cache/huggingface/hub/models--kyutai--helium-1-2b/snapshots/<sha>` (HF cache path on M1 Max). No SpikingKiki-V4-style spike-rate substrate behind it ; the dream handlers operate on the LoRA delta tensors emitted by the fork's `train` directly. The `MicroKikiSubstrate` instantiation can fall back to its env-gated synthetic-default Path 1 (`DREAM_MICRO_KIKI_REAL` UNSET) since there is no real spike-substrate to gate on.
- Adapter path : `--adapter-path /tmp/helium-fresh-adapters` (empty dir, fresh-init LoRA, mirrors Path A's `fresh_init=True` path). Created on first run.
- Fixtures :
  - Step 1 : `tests/fixtures/mmlu_g6_real.jsonl` — output `docs/milestones/g6-m1max-path-d-mmlu-2026-05-04.{json,md}`
  - Step 2 : `tests/fixtures/symbolic_cl_g6.jsonl` — output `docs/milestones/g6-m1max-path-d-synth-2026-05-04.{json,md}`
- Aggregator : a small shim `experiments/g6_studio_path_a/aggregator_h11.py` reads both Step 1 + Step 2 milestones and emits the H11-D conjunction with optional Path A* / Path C mate (if those milestones exist).
- Registry profile keys : `g6-m1max-path-d/<arm>` (distinct from the `g6-studio-path-a/<arm>` namespace for Path A/A*/C).

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H11-A_{MMLU and synth} both confirmed (g ≥ 2) | EC stays PARTIAL ; cross-family + cross-benchmark validation at 2B-dense tier ; if Path A* / Path C also positive → H11-D `confirmed_positive_universal` → DR-3 evidence v0.x major update ; framework C real-LLM tier validated. | FC stays C-v0.12.0 |
| Row 1* — H11-A* (exploratory medium band) | EC stays PARTIAL ; reported as exploratory ; needs N > 5 follow-up. | FC stays C-v0.12.0 |
| Row 2 — H11-B confirmed | EC stays PARTIAL ; G5-bis MLX-only artefact verdict extends to 2B-dense tier ; framework C wash-out is real even at smaller-than-35B real-LLM scale. | FC stays C-v0.12.0 |
| Row 3 — H11-C confirmed | EC stays PARTIAL ; framework's RECOMBINE prediction further weakened cross-family. | FC stays C-v0.12.0 |
| Row 4 — `scale_locked_positive` (Helium positive AND Qwen null) | EC stays PARTIAL ; **positive scientific finding** : framework C has a scale-bounded validity ; the 35B-MoE tier specifically washes out the effect. Honest Paper 2 §7.1.13 reports the scale boundary. | FC stays C-v0.12.0 |
| Row 5 — `family_locked_positive` (dense positive AND MoE null at same N) | EC stays PARTIAL ; **positive scientific finding** : MoE gating modulates framework C effect ; honest reporting of the family-axis interaction. | FC stays C-v0.12.0 |
| Row 6 — `benchmark_locked` (MMLU vs synth disagreement on same substrate) | EC stays PARTIAL ; benchmark-dependent effect within Helium tier ; mirrors Path C's H10-D `divergent` semantics. | FC stays C-v0.12.0 |
| Row 7 — exclusion-rate > 50 % on either step | abort and amend ; do NOT commit step milestone. | n/a |

EC stays PARTIAL across all rows. FC stays at v0.12.0. STABLE promotion blocked unless H11-D `confirmed_positive_universal`.

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome.

The convergent-validation principle (locked in Path C pre-reg §1) is preserved : a positive H11-A on Helium alone is exploratory ; only H11-D `confirmed_positive_universal` (all four cells of the {H9, H10, H11_MMLU, H11_synth} matrix consistently positive) supports a confirmatory framework C real-LLM tier claim.

Resolution states `scale_locked_positive`, `family_locked_positive`, `benchmark_locked`, `divergent` are reported as **positive scientific findings**, not aggregated into a single verdict. Each names a specific factor of effect-modulation.

## §8 Audit trail

Cells registered via `harness/storage/run_registry.py` with profile keys `g6-m1max-path-d/<arm>` (distinct namespace from Path A family).

Step 1 milestone : `docs/milestones/g6-m1max-path-d-mmlu-2026-05-04.{json,md}`
Step 2 milestone : `docs/milestones/g6-m1max-path-d-synth-2026-05-04.{json,md}`
Per-subdomain partial dumps : `docs/milestones/g6-m1max-path-d-{step}-partial-...-2026-05-04.json`

## §9 Deviations

Pre-known envelopes :

a. Helium-2B model file load fails on M1 Max (HF cache corruption, mlx_lm version drift) — abort and file §9.1 amendment.
b. Per-cell wall > 10 min sustained on M1 Max — extrapolated total > 7 h ; consider deferring to Studio.
c. `acc_initial` < 0.30 for majority on real MMLU step — this is the SAME failure mode as Path A on synthetic MMLU ; if it fires here on real MMLU 50/50, it's a Helium-2B specific limitation (model too small). Skip MMLU step, proceed with symbolic CL only ; document the Helium-MMLU floor in §9.1 amendment.
d. Conflict with PYTHONPATH=/tmp setup on M1 Max — fork copy under `/tmp/mlx_lm/` may collide with future M1-Max-side runs. Reset by `rm -rf /tmp/mlx_lm` between pilots if needed.

Any deviation outside the envelopes requires an amendment commit *before* the affected cell runs.

### §9.1 — TBD on first run if surprises surface

Reserved per the pattern.
