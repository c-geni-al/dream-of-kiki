# G5-bis pilot — port G4-ter richer hierarchical head to E-SNN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the G4-ter MLX richer-head positive effect (`g_h2 = +2.77`, REPLAY+DOWNSCALE on 3-layer hierarchy) transfers to a NON-MLX substrate by porting the same coupling pattern to a 3-layer rate-coded LIF E-SNN classifier and comparing per-arm retention against G4-ter MLX richer-head retention via Welch two-sided tests at Bonferroni α/4 = 0.0125.

**Architecture:** A new `experiments/g5_bis_richer_esnn/` module reuses `kiki_oniric.substrates.esnn_thalamocortical` and the G4-ter HP grid C5 anchor but introduces a **3-layer** rate-coded SNN classifier (`EsnnG5BisHierarchicalClassifier` : in_dim → hidden_1 → hidden_2 → output, two LIF stages, straight-through estimator) plus an E-SNN port of `dream_episode_hier()` whose REPLAY drives optimizer steps on (W_in, W_h, W_out), DOWNSCALE multiplies all three weight tensors, RESTRUCTURE perturbs W_h only, RECOMBINE injects Gaussian-MoG synthetic latents through W_out. A separate aggregator runs four Welch two-sided tests at α/4 = 0.0125 and emits an H7-A/B/C verdict to `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}`. EC stays PARTIAL, no FC bump.

**Tech Stack:** Python 3.12, numpy (LIF dynamics + STE backward), existing `kiki_oniric.substrates.esnn_thalamocortical` (numpy LIF, no `norse`), existing `experiments.g4_split_fmnist.dataset.{SplitFMNISTTask,load_split_fmnist_5tasks}`, existing `experiments.g4_ter_hp_sweep.hp_grid.representative_combo`, existing `experiments.g5_cross_substrate.esnn_dream_wrap.{PROFILE_FACTORIES,_rebind_to_esnn,ProfileT}`, existing `harness.storage.run_registry.RunRegistry`, existing `kiki_oniric.eval.statistics.{compute_hedges_g,welch_one_sided}`, pytest + Hypothesis, conventional commits.

---

## Hard prerequisites (block until satisfied)

1. `docs/milestones/g4-ter-pilot-2026-05-03.json` exists ; `verdict.h2_substrate_richer.hedges_g >= 1.0` ; `verdict.retention_richer_by_arm` is a dict with `baseline`/`P_min`/`P_equ`/`P_max` keys.
2. `experiments/g4_ter_hp_sweep/dream_wrap_hier.py` exposes `G4HierarchicalClassifier`, `BetaBufferHierFIFO`, `dream_episode_hier()`.
3. `experiments/g5_cross_substrate/esnn_dream_wrap.py` exposes `PROFILE_FACTORIES`, `_rebind_to_esnn`, `ProfileT`.
4. `kiki_oniric/substrates/esnn_thalamocortical.py` exposes `simulate_lif_step` and `LIFState`.

If any is missing : **stop and surface the blocker** before Task 1.

## Variant decision (locked) — Variant A

**Full 3-layer rate-coded LIF SNN classifier with STE backward**, not "MLX richer head wrapped in E-SNN dispatch". Same rationale as G5 plan §"Variant decision" : H7-A/B/C is only meaningful if the classifier carries the substrate's native state representation. Wrapping MLX in E-SNN dispatch only would conflate "E-SNN ops are callable" with "DR-3 holds at the positive-effect channel".

## Compute / power note

Per-cell cost ≈ 20× G4-ter MLX (two stacked numpy-LIF populations). G4-ter measured ~1.55 s/cell on M1 Max → G5-bis ~31 s/cell.

| Option | N seeds | Cells | Wall (M1 Max, est.) | Min detectable g (80 % power, two-sided α=0.0125) |
|--------|---------|-------|---------------------|-----------------------------------------------------|
| A | 30 | 120 | ~62 min | g ≈ 0.74 |
| B | 10 | 40 | ~21 min | g ≈ 1.27 |
| C | 5 (smoke) | 20 | ~10 min | g ≈ 1.85 (exploratory only) |

**Recommended : Option B (N=10) — locked in Task 0.5.** Sufficient if effect is comparable to G4-ter MLX `g_h2 = +2.77` ; positions the verdict as exploratory with a confirmatory N≥30 follow-up scheduled if H7-A or H7-C. Option C is rejected because 20 cells is the same scale that produced degenerate equal-means in G5 binary-head.

## File structure

| File | Role |
|------|------|
| `experiments/g5_bis_richer_esnn/__init__.py` (create) | Package marker |
| `experiments/g5_bis_richer_esnn/esnn_hier_classifier.py` (create) | `EsnnG5BisHierarchicalClassifier` — 3-layer LIF + STE |
| `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py` (create) | `dream_episode_hier_esnn` + `EsnnHierBetaBuffer` |
| `experiments/g5_bis_richer_esnn/run_g5_bis.py` (create) | Pilot driver, sweep + milestone dump |
| `experiments/g5_bis_richer_esnn/aggregator.py` (create) | Cross-substrate H7-A/B/C verdict |
| `tests/unit/experiments/test_g5_bis_esnn_hier_classifier.py` (create) | Classifier unit tests |
| `tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py` (create) | Wrapper unit tests |
| `tests/unit/experiments/test_g5_bis_run_smoke.py` (create) | 2-seed driver smoke (synthetic FMNIST) |
| `tests/unit/experiments/test_g5_bis_aggregator.py` (create) | Aggregator math tests |
| `docs/osf-prereg-g5-bis-richer-esnn.md` (create) | OSF pre-reg, append-only |
| `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}` (driver writes) | Per-cell + H7-A own-substrate verdict |
| `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}` (aggregator writes) | H7-A/B/C cross-substrate verdict |
| `docs/papers/paper2/results.md` (modify, add §7.1.9) | Cross-substrate richer EN |
| `docs/papers/paper2-fr/results.md` (modify, add §7.1.9) | FR mirror |
| `docs/proofs/dr3-substrate-evidence.md` (modify, append) | DR-3 evidence revision per H7 |
| `CHANGELOG.md` (modify) | `[Unreleased]` G5-bis row |
| `STATUS.md` (modify) | Gates table row + As-of update |

`experiments/` is excluded from coverage scope (per `pyproject.toml`) — pilots are not library code.

---

## Task 0: Investigate (read-only) — confirm assumptions

**Files (read only):**
- `experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
- `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- `experiments/g4_ter_hp_sweep/hp_grid.py`
- `experiments/g5_cross_substrate/esnn_classifier.py`
- `experiments/g5_cross_substrate/esnn_dream_wrap.py`
- `experiments/g5_cross_substrate/aggregator.py`
- `kiki_oniric/substrates/esnn_thalamocortical.py`
- `docs/milestones/g4-ter-pilot-2026-05-03.json` (BLOCKER)
- `docs/milestones/g5-cross-substrate-2026-05-03.md`
- `docs/osf-prereg-g4-ter-pilot.md` (template)
- `docs/proofs/dr3-substrate-evidence.md`

- [ ] **Step 1: Confirm G4-ter milestone payload schema**

Run: `python -c "import json; p = json.load(open('docs/milestones/g4-ter-pilot-2026-05-03.json')); print(sorted(p.keys())); print(sorted(p['verdict'].keys())); print(sorted(p['verdict']['retention_richer_by_arm'].keys()))"`
Expected: top-level contains `cells_richer`, `verdict`, `c_version`, `commit_sha`, `n_seeds_richer` ; `verdict` has `h2_substrate_richer`, `h_dr4_ter_richer`, `retention_richer_by_arm` ; arms `[P_equ, P_max, P_min, baseline]`.

- [ ] **Step 2: Confirm `g_h2 = +2.77` recorded**

Run: `python -c "import json; p = json.load(open('docs/milestones/g4-ter-pilot-2026-05-03.json')); print(p['verdict']['h2_substrate_richer']['hedges_g'])"`
Expected: a value `>= 1.0` (truth: ~2.766).

- [ ] **Step 3: Confirm G4-ter coupling map (REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE)**

Run: `grep -nE "_replay_optimizer_step|_downscale_step|_restructure_step|_recombine_step" experiments/g4_ter_hp_sweep/dream_wrap_hier.py`
Expected: 4 method definitions + 4 call sites in `dream_episode_hier`.

- [ ] **Step 4: Confirm E-SNN substrate factories + LIF API**

Run: `grep -nE "_handler_factory|^def simulate_lif_step|^class LIFState" kiki_oniric/substrates/esnn_thalamocortical.py`
Expected: 4 factory hits + `simulate_lif_step` + `LIFState`.

- [ ] **Step 5: Confirm `representative_combo()` returns C5**

Run: `python -c "from experiments.g4_ter_hp_sweep.hp_grid import representative_combo; c = representative_combo(); print(c.combo_id, c.downscale_factor, c.replay_batch, c.replay_n_steps, c.replay_lr)"`
Expected: `C5 0.95 32 5 0.01`.

- [ ] **Step 6: Confirm G5 wrapper exposes `PROFILE_FACTORIES` + `_rebind_to_esnn`**

Run: `grep -nE "^PROFILE_FACTORIES|^def _rebind_to_esnn|^ProfileT" experiments/g5_cross_substrate/esnn_dream_wrap.py`
Expected: 3 hits — these are reused in Task 3.

- [ ] **Step 7: No commit**

Investigation only.

---

## Task 0.5: Decision — compute budget A/B/C (locked)

- [ ] **Step 1: Lock Option B (N=10)**

Record locally : `Option B locked (N_seeds=10, 40 cells, ~21 min M1 Max).` This decision is recorded in the pre-reg in Task 1.

- [ ] **Step 2: No commit**

---

## Task 1: OSF pre-registration draft

**Files:**
- Create: `docs/osf-prereg-g5-bis-richer-esnn.md`

- [ ] **Step 1: Author the pre-reg**

Write `docs/osf-prereg-g5-bis-richer-esnn.md` mirroring the structure of `docs/osf-prereg-g4-ter-pilot.md` (sections : header → §0 background → §1 purpose → §2 sweep design → §3 effect-size anchors → §4 power → §5 exclusion → §6 outputs → §7 amendments). Required content :

- **§0 background** : cite G4-ter MLX richer-head positive finding (`g_h2 = +2.77`, p = 4.9e-14, N=30, milestone `docs/milestones/g4-ter-pilot-2026-05-03.md`) AND G5 binary-head spectator divergence (within-substrate spectator pattern + cross-substrate baseline g=+9.98 / dream g=+3.49 reject at α/4 ; absolute retention diverges). Both citations are mandatory per the brief's "Pre-reg fidelity" caveat.
- **§1 purpose** : pre-register exactly three hypotheses :
  - **H7-A (positive transfer, level-divergence)** : `g_h7a = compute_hedges_g(retention[P_equ,esnn_richer], retention[baseline,esnn_richer]) > 0` AND Welch one-sided rejects at α/4=0.0125, but cross-substrate two-sided Welch on P_equ rejects (means diverge).
  - **H7-B (MLX-only artefact)** : `|g_h7a| < 0.5` AND Welch fails to reject at α/4=0.0125.
  - **H7-C (universal cross-substrate)** : `g_h7a > 0`, Welch rejects, AND cross-substrate Welch fails to reject on P_equ (means match within factor 2 of `g_h2_mlx = 2.77`).
- **§2 sweep** : arms `["baseline","P_min","P_equ","P_max"]`, seeds `[0..9]` (Option B), 40 cells. Classifier `EsnnG5BisHierarchicalClassifier` (in_dim=784, hidden_1=32, hidden_2=16, n_classes=2, n_steps=20, tau=10.0, threshold=1.0). HP combo C5 from `representative_combo()`. Constants : `RESTRUCTURE_FACTOR=0.05`, `RECOMBINE_N_SYNTHETIC=16`, `RECOMBINE_LR=0.01`, `BETA_BUFFER_CAPACITY=256`, `BETA_BUFFER_FILL_PER_TASK=32`. Wrapper `dream_episode_hier_esnn` couples REPLAY → SGD-with-STE on (W_in, W_h, W_out) ; DOWNSCALE → multiplies (W_in, W_h, W_out) by `factor` ; RESTRUCTURE → adds `factor*sigma*N(0,1)` to W_h only ; RECOMBINE → MoG synthetic-latent injection through W_out only.
- **§3 anchors** : Hu 2020 (g=0.29, CI [0.21, 0.38]) ; G4-ter MLX `g_h2 = +2.77`; G5 cross-substrate divergence as null reference.
- **§4 power** : N=10 ; min detectable g ≈ 1.27 at 80 % power, two-sided α=0.0125. Exploratory ; confirmatory Option A scheduled if positive.
- **§5 exclusion** : `acc_task1_initial < 0.5` flagged + dropped (mirrors G4-ter / G5).
- **§6 outputs** : `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}` + `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}` + `docs/proofs/dr3-substrate-evidence.md` revision.
- **§7 amendments** : append-only ; date+sign any post-hoc change.

Header lines are `**Project**`, `**Parent registration**`, `**Amendment**`, `**PI**`, `**Date drafted**`, `**Lock target**` (= "before any G5-bis run is registered in `harness/storage/run_registry.RunRegistry`").

- [ ] **Step 2: Verify markdown is well-formed**

Run: `head -3 docs/osf-prereg-g5-bis-richer-esnn.md ; grep -c "^## " docs/osf-prereg-g5-bis-richer-esnn.md`
Expected: title line + 7+ `## ` headings.

- [ ] **Step 3: Commit**

```bash
git add docs/osf-prereg-g5-bis-richer-esnn.md
git commit -m "docs(g5bis): OSF pre-reg draft

Pre-registers H7-A/B/C cross-substrate transfer test of the
G4-ter MLX richer-head positive finding (g_h2=+2.77) onto the
E-SNN thalamocortical substrate. Locks Option B (N=10) sweep,
HP combo C5 anchor, four-op coupling on 3-layer LIF stack."
```

---

## Task 2: Stub package + 3-layer LIF classifier (TDD)

**Files:**
- Create: `experiments/g5_bis_richer_esnn/__init__.py` — single docstring line.
- Create: `experiments/g5_bis_richer_esnn/esnn_hier_classifier.py`
- Create: `tests/unit/experiments/test_g5_bis_esnn_hier_classifier.py`

- [ ] **Step 1: Write the failing test**

Create the test file. Required test cases (one assertion each, follows G5 sister test pattern in `tests/unit/experiments/test_g5_esnn_classifier.py` if present, else inline below) :

1. `test_classifier_init_shapes_match_layer_sizes` : `EsnnG5BisHierarchicalClassifier(in_dim=4, hidden_1=8, hidden_2=6, n_classes=2, seed=0)` exposes `W_in.shape == (4, 8)`, `W_h.shape == (8, 6)`, `W_out.shape == (6, 2)`.
2. `test_classifier_rejects_invalid_dims` : ValueError on `in_dim=0`, `hidden_1=0`, `hidden_2=0`, `n_classes=1`.
3. `test_predict_logits_shape_matches_n_classes` : input `(3, 4)` → logits `(3, 2)`.
4. `test_predict_logits_empty_returns_empty` : input `(0, 4)` → logits `(0, 2)`.
5. `test_eval_accuracy_in_unit_interval` : accuracy ∈ [0, 1].
6. `test_latent_returns_hidden_2_spike_rates` : `latent(x).shape == (3, 6)`, all values in [0, 1].
7. `test_train_task_changes_weights_deterministically` : two classifiers built with same seed and trained on the same `task` produce bit-identical `W_in`, `W_h`, `W_out`, AND `W_in` differs from pre-training value.

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_esnn_hier_classifier.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'experiments.g5_bis_richer_esnn'`.

- [ ] **Step 3: Implement `EsnnG5BisHierarchicalClassifier`**

Create `experiments/g5_bis_richer_esnn/esnn_hier_classifier.py`. The classifier mirrors `EsnnG5Classifier` from `experiments/g5_cross_substrate/esnn_classifier.py` but stacks **two** LIF populations between **three** linear projections.

**Architecture** (forward) :

```
x: (N, in_dim)
    -> i_h1 = x @ W_in            # (N, hidden_1)
    -> r_h1 = LIF_pop(i_h1, n_neurons=hidden_1, n_steps)    # (N, hidden_1) spike rates
    -> i_h2 = r_h1 @ W_h          # (N, hidden_2)
    -> r_h2 = LIF_pop(i_h2, n_neurons=hidden_2, n_steps)    # (N, hidden_2) spike rates
    -> logits = r_h2 @ W_out      # (N, n_classes)
```

**Backward** : straight-through estimator (Wu 2018) — the LIF non-linearity is identity in the backward pass. Concretely, given softmax-CE `d_logits = (probs - one_hot) / N` :

```
d_W_out = r_h2.T @ d_logits          ; d_r_h2 = d_logits @ W_out.T
d_W_h   = r_h1.T @ d_r_h2            ; d_r_h1 = d_r_h2 @ W_h.T   (STE: d_i_h2 = d_r_h2)
d_W_in  = x.T   @ d_r_h1                                          (STE: d_i_h1 = d_r_h1)
```

**Required public API** (mirrors G4-ter `G4HierarchicalClassifier`) :

- Frozen `@dataclass` with fields `in_dim, hidden_1, hidden_2, n_classes, seed, n_steps=20, tau=10.0, threshold=1.0` and init-only weights `W_in`, `W_h`, `W_out`.
- `__post_init__` : validate dims (>0, n_classes>=2), seed `np.random.default_rng(seed)`, Xavier-style init `scale = sqrt(2/in)` per tensor.
- `_lif_population_rates(currents, n_neurons) -> NDArray[float32]` : drive a LIF population (per-sample loop over `LIFState` + `simulate_lif_step` from `kiki_oniric.substrates.esnn_thalamocortical`), return mean spike rates of shape `(N, n_neurons)`. Identical structure to `EsnnG5Classifier._hidden_rates`.
- `latent(x) -> NDArray[float32]` of shape `(N, hidden_2)` — used by the beta buffer at push time.
- `predict_logits(x) -> NDArray[float32]` of shape `(N, n_classes)`.
- `eval_accuracy(x, y) -> float` in [0, 1].
- `_forward_with_caches(x) -> (r_h1, r_h2, logits)` for STE.
- `_ste_backward(x, y, lr)` : one SGD step with the STE gradient defined above.
- `train_task(task: dict, *, epochs, batch_size, lr)` : seeded permutation per epoch (`rng = np.random.default_rng(self.seed)`), minibatch loop calling `_ste_backward`. Mirrors `EsnnG5Classifier.train_task` determinism contract.

- [ ] **Step 4: Run tests, expect 7 PASS**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_esnn_hier_classifier.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Lint + typecheck**

Run: `uv run ruff check experiments/g5_bis_richer_esnn/`
Expected: clean.
Run: `uv run mypy experiments/g5_bis_richer_esnn/`
Expected: clean (or experiments-excluded — verify in `pyproject.toml`).

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_bis_richer_esnn/__init__.py \
        experiments/g5_bis_richer_esnn/esnn_hier_classifier.py \
        tests/unit/experiments/test_g5_bis_esnn_hier_classifier.py
git commit -m "feat(g5bis): 3-layer rate-coded LIF classifier

Adds EsnnG5BisHierarchicalClassifier with two stacked numpy LIF
populations and STE backward (Wu 2018). Mirrors G4-ter
G4HierarchicalClassifier surface so the dream wrapper and pilot
driver can transpose run_g4_ter cell logic mechanically."
```

---

## Task 3: E-SNN richer dream wrapper + buffer (TDD)

**Files:**
- Create: `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py`
- Create: `tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py`

- [ ] **Step 1: Write failing tests**

Required test cases :

1. `test_buffer_push_and_snapshot_roundtrip` : push one record with `latent=np.array([0.5, 0.6])` ; snapshot returns `[{"x":..., "y":0, "latent":[0.5, 0.6]}]`.
2. `test_buffer_capacity_evicts_fifo` : capacity=2, push 3 records with `y=0,1,2` ; snapshot ys are `[1, 2]`.
3. `test_dream_episode_runs_and_mutates_weights` : build classifier (seed=0), buffer with 8 records (latents=hidden_2 noise), call `dream_episode_hier_esnn` with `replay_lr=0.01` `downscale_factor=0.95` `restructure_factor=0.05` `recombine_n_synthetic=4` `recombine_lr=0.01` ; assert all three weight tensors changed (`W_in`, `W_h`, `W_out`).
4. `test_dream_episode_p_min_skips_restructure_recombine` : with `replay_lr=0.0`, `downscale_factor=0.5` and a P_min profile, after `dream_episode_hier_esnn` the resulting `W_h` equals `0.5 * W_h_before` exactly (no RESTRUCTURE noise added) — proves P_min only invokes REPLAY+DOWNSCALE.

- [ ] **Step 2: Run tests, expect ImportError**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py -v`
Expected: FAIL on `ImportError: cannot import name 'EsnnHierBetaBuffer'`.

- [ ] **Step 3: Implement the wrapper module**

Create `experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py`. Public surface :

- `class EsnnHierBetaBuffer(capacity: int)` : FIFO buffer with `push(x, y, latent)`, `__len__`, `snapshot()`, `sample(n, seed)`. Carries optional per-record latent (`latent: np.ndarray | None`). Mirrors `BetaBufferHierFIFO` from `experiments/g4_ter_hp_sweep/dream_wrap_hier.py:384` — same field shape, same eviction policy. `capacity <= 0` raises ValueError.
- `build_esnn_richer_profile(name: str, seed: int) -> ProfileT` : reuses `PROFILE_FACTORIES` and `_rebind_to_esnn` from `experiments/g5_cross_substrate/esnn_dream_wrap.py` to build an `ESNN-rebound` profile. Same code path as G5 binary head.
- Internal step helpers (free functions taking the classifier as first arg) :
  - `_replay_step(clf, records, *, lr, n_steps)` : if records empty or lr=0 or n_steps<=0, no-op ; else build `(x, y)` tensors from records and run `n_steps` SGD passes via `clf._ste_backward(x, y, lr)`.
  - `_downscale_step(clf, *, factor)` : validate `0 < factor <= 1`, then multiply `clf.W_in`, `clf.W_h`, `clf.W_out` by `factor` (cast to float32).
  - `_restructure_step(clf, *, factor, seed)` : if `factor < 0` raise ; if `factor == 0` no-op ; sigma = `clf.W_h.std()` ; if sigma == 0 no-op ; else `clf.W_h = clf.W_h + factor * sigma * rng.standard_normal(shape)`. Mirrors G4-ter `_restructure_step` semantics on `_l2.weight`.
  - `_recombine_step(clf, *, latents, n_synthetic, lr, seed)` : if no latents or single class → no-op ; else estimate per-class Gaussian (mean, std+1e-6) over `(latent, label)` pairs, sample `per_class = max(1, n_synthetic // n_classes)` synthetic rates per class, run **one** softmax-CE SGD step on `clf.W_out` only (linear forward `rates @ clf.W_out` ; standard CE backward ; update `clf.W_out -= lr * rates.T @ d_logits`). Mirrors G4-ter `_recombine_step` but on the LIF substrate's W_out matrix (no MLX optimizer).
- `dream_episode_hier_esnn(clf, profile, seed, *, beta_buffer, replay_n_records, replay_n_steps, replay_lr, downscale_factor, restructure_factor, recombine_n_synthetic, recombine_lr)` : transposes G4-ter `dream_episode_hier`. Steps :
  1. Determine `ops` and `channels` from `isinstance(profile, PMinProfile)` (P_min : REPLAY+DOWNSCALE only ; else all four).
  2. Build a synthetic `DreamEpisode` (mirrors G4-ter spectator runtime block — `input_slice` keys identical to `experiments/g5_cross_substrate/esnn_dream_wrap.py:dream_episode`, including `"input"` for the E-SNN replay handler) with `episode_id = f"g5bis-{type(profile).__name__}-seed{seed}"`.
  3. `profile.runtime.execute(episode)` (DR-0 logging).
  4. If REPLAY in ops : `_replay_step(clf, beta_buffer.sample(n=replay_n_records, seed=seed), lr=replay_lr, n_steps=replay_n_steps)`.
  5. If DOWNSCALE in ops : `_downscale_step(clf, factor=downscale_factor)`.
  6. If RESTRUCTURE in ops : `_restructure_step(clf, factor=restructure_factor, seed=seed + 20_000)`.
  7. If RECOMBINE in ops : iterate `beta_buffer.snapshot()`, collect `(latent, y)` pairs where latent is not None, then `_recombine_step(clf, latents=..., n_synthetic=recombine_n_synthetic, lr=recombine_lr, seed=seed + 30_000)`.

Module imports : `numpy`, `random` ; from `experiments.g5_cross_substrate.esnn_dream_wrap` import `PROFILE_FACTORIES`, `_rebind_to_esnn`, `ProfileT` ; from `experiments.g5_bis_richer_esnn.esnn_hier_classifier` import `EsnnG5BisHierarchicalClassifier` ; from `kiki_oniric.dream.episode` import `BudgetCap, DreamEpisode, EpisodeTrigger, Operation, OutputChannel` ; from `kiki_oniric.profiles.p_min` import `PMinProfile`.

- [ ] **Step 4: Run tests, expect 4 PASS**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_bis_richer_esnn/esnn_dream_wrap_hier.py \
        tests/unit/experiments/test_g5_bis_esnn_dream_wrap_hier.py
git commit -m "feat(g5bis): richer-head dream wrapper + buffer

Adds dream_episode_hier_esnn coupling REPLAY/DOWNSCALE/
RESTRUCTURE/RECOMBINE to the 3-layer LIF classifier weights.
DOWNSCALE multiplies W_in/W_h/W_out, RESTRUCTURE perturbs W_h
only, RECOMBINE injects MoG synthetic latents through W_out.
EsnnHierBetaBuffer mirrors G4-ter BetaBufferHierFIFO surface."
```

---

## Task 4: Pilot driver `run_g5_bis.py` (TDD)

**Files:**
- Create: `experiments/g5_bis_richer_esnn/run_g5_bis.py`
- Create: `tests/unit/experiments/test_g5_bis_run_smoke.py`

- [ ] **Step 1: Write the smoke test**

Required test : `test_run_pilot_smoke` (uses `tmp_path` + a `synthetic_data_dir` fixture that writes 5 `task_k.npz` fixtures). Pattern :

1. Build 5 npz tasks in `tmp_path/data` (40 train / 20 test, in_dim=4).
2. `monkeypatch.setattr(run_g5_bis, "load_split_fmnist_5tasks", _fake_loader)` returning a list of `SplitFMNISTTask` from the fixtures.
3. Call `run_g5_bis.run_pilot(data_dir=fixture_dir, seeds=(0,1), out_json=..., out_md=..., registry_db=tmp_path/"registry.sqlite", epochs=1, batch_size=8, lr=0.05, n_steps=5, hidden_1=8, hidden_2=6)`.
4. Assert : `out_json.exists()`, `out_md.exists()`, `len(payload["cells"]) == 4 * 2 == 8`, `sorted(payload["verdict"]["retention_by_arm"]) == ["P_equ","P_max","P_min","baseline"]`.

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_run_smoke.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement the driver**

Create `experiments/g5_bis_richer_esnn/run_g5_bis.py` by transposing `experiments/g4_ter_hp_sweep/run_g4_ter.py` (especially `_run_cell_richer`, `_retention_by_arm`, `_h2_verdict`, `run_pilot`, `main`) and replacing `G4HierarchicalClassifier` / `BetaBufferHierFIFO` / `dream_episode_hier` / `build_profile` with their G5-bis E-SNN equivalents. Required deltas :

| G4-ter symbol | G5-bis symbol |
|----------------|----------------|
| `G4HierarchicalClassifier` | `EsnnG5BisHierarchicalClassifier` (extra ctor arg `n_steps`) |
| `BetaBufferHierFIFO` | `EsnnHierBetaBuffer` |
| `build_profile` | `build_esnn_richer_profile` |
| `dream_episode_hier` (method) | `dream_episode_hier_esnn` (free function ; pass `clf` as first arg) |
| `HIDDEN_1=32, HIDDEN_2=16` | same defaults but exposed as `--hidden-1 --hidden-2` CLI args |
| HP sub-grid sweep | **DROPPED** — G5-bis runs only the richer sweep at C5 anchor |
| `h_dr4_ter_richer` verdict | **DROPPED** — G5-bis only computes `h7a_richer_esnn` (P_equ vs baseline) |
| `_h1_hp_verdict` | **DROPPED** with HP sub-grid |

Required module-level constants :

```
C_VERSION = "C-v0.12.0+PARTIAL"
SUBSTRATE_NAME = "esnn_thalamocortical_richer"
ARMS = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS = tuple(range(10))      # Option B
DEFAULT_DATA_DIR = REPO_ROOT/"experiments"/"g4_split_fmnist"/"data"
DEFAULT_OUT_JSON = REPO_ROOT/"docs"/"milestones"/"g5-bis-richer-esnn-2026-05-03.json"
DEFAULT_OUT_MD   = REPO_ROOT/"docs"/"milestones"/"g5-bis-richer-esnn-2026-05-03.md"
DEFAULT_REGISTRY_DB = REPO_ROOT/".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
RESTRUCTURE_FACTOR = 0.05
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
```

Required signatures :

- `_resolve_commit_sha() -> str` : copy verbatim from `run_g4_ter.py:99`.
- `_run_cell(arm, seed, tasks, *, epochs, batch_size, hidden_1, hidden_2, lr, n_steps) -> _CellPartial` : transpose `_run_cell_richer` ; build classifier with `n_steps` ; call `representative_combo()` for HP ; per-task `_push_task` captures `clf.latent(x[None,:])[0]` into the buffer ; train task 0, then for k in 1..4 : if profile not None call `dream_episode_hier_esnn(clf, profile, seed=seed+k, beta_buffer=buffer, replay_n_records=combo.replay_batch, replay_n_steps=combo.replay_n_steps, replay_lr=combo.replay_lr, downscale_factor=combo.downscale_factor, restructure_factor=RESTRUCTURE_FACTOR, recombine_n_synthetic=RECOMBINE_N_SYNTHETIC, recombine_lr=RECOMBINE_LR)`, then `clf.train_task(tasks[k], …)`, then `_push_task(tasks[k])`.
- `_h7a_verdict(retention) -> dict` : compute `g_h7a = compute_hedges_g(retention["P_equ"], retention["baseline"])`, `welch = welch_one_sided(retention["baseline"], retention["P_equ"], alpha=0.05/4)`, return `{hedges_g, above_zero, above_hu_2020_lower_ci, welch_p, welch_reject_h0, alpha_per_test=0.05/4, n_p_equ, n_base, insufficient_samples?}`.
- `_aggregate_verdict(cells) -> dict` : returns `{"h7a_richer_esnn": _h7a_verdict(...), "retention_by_arm": _retention_by_arm(cells)}`.
- `_render_md_report(payload) -> str` : sister of `run_g4_ter.py:_render_md_report` but only renders H7-A section + cells table + provenance footer (links to `g4-ter-pilot-2026-05-03.md`, `g5-cross-substrate-2026-05-03.md`, and `g5-bis-aggregate-2026-05-03.md`).
- `run_pilot(*, data_dir, seeds, out_json, out_md, registry_db, epochs, batch_size, hidden_1, hidden_2, lr, n_steps) -> dict` : sweep loop, per-cell `registry.register(c_version=C_VERSION, profile=f"g5-bis/richer/{arm}", seed=seed, commit_sha=commit_sha)`, write `{json,md}` siblings.
- `main(argv)` : argparse with `--smoke`, `--data-dir`, `--epochs=2`, `--batch-size=64`, `--hidden-1=32`, `--hidden-2=16`, `--n-steps=20`, `--lr=0.05`, `--out-json`, `--out-md`, `--registry-db`. Smoke = `seeds=(0,1)`.

- [ ] **Step 4: Run smoke test, expect PASS**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_run_smoke.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check experiments/g5_bis_richer_esnn/run_g5_bis.py tests/unit/experiments/test_g5_bis_run_smoke.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add experiments/g5_bis_richer_esnn/run_g5_bis.py \
        tests/unit/experiments/test_g5_bis_run_smoke.py
git commit -m "feat(g5bis): pilot driver run_g5_bis

Sweeps 4 arms x 10 seeds (Option B) of the richer-head E-SNN
classifier on Split-FMNIST 5-task. Per-cell pipeline transposes
G4-ter run_g4_ter._run_cell_richer onto LIF stack with C5 HP
anchor. Emits g5-bis-richer-esnn-2026-05-03.{json,md} with H7-A
own-substrate verdict. 2-seed smoke test on synthetic data."
```

---

## Task 5: Cross-substrate aggregator (G4-ter MLX vs G5-bis E-SNN, TDD)

**Files:**
- Create: `experiments/g5_bis_richer_esnn/aggregator.py`
- Create: `tests/unit/experiments/test_g5_bis_aggregator.py`

- [ ] **Step 1: Write failing tests**

Three required test cases over synthetic milestone fixtures :

1. `test_h7c_universal_when_means_match` : both fixtures have `retention[P_equ]≈0.7`, `retention[baseline]≈0.5`, N=10. `aggregate_g5bis_verdict(...)["h7_classification"] == "H7-C"` AND `per_arm["P_equ"]["consistency"] is True`.
2. `test_h7b_when_g_close_to_zero` : E-SNN P_equ ≈ baseline (≈0.5 each). `h7_classification == "H7-B"`.
3. `test_h7a_when_positive_but_diverges` : MLX P_equ=0.9, E-SNN P_equ=0.6, both have baseline=0.5. E-SNN g positive but cross-substrate Welch rejects. `h7_classification == "H7-A"`.

Fixture helpers : `_write_g4ter_fixture(path, retention)` writes `{"verdict":{"h2_substrate_richer":{"hedges_g":2.77},"retention_richer_by_arm":retention}}` ; `_write_g5bis_fixture(path, retention)` writes `{"verdict":{"h7a_richer_esnn":{"hedges_g":1.5},"retention_by_arm":retention}}`.

- [ ] **Step 2: Run tests, expect ImportError**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_aggregator.py -v`
Expected: FAIL on `ImportError: cannot import name 'aggregate_g5bis_verdict'`.

- [ ] **Step 3: Implement the aggregator**

Create `experiments/g5_bis_richer_esnn/aggregator.py`. Mirror `experiments/g5_cross_substrate/aggregator.py` (load, two-sided Welch fold, per-arm consistency table, render md) with the following deltas :

- **Different milestone keys** : MLX side reads `verdict.retention_richer_by_arm` (G4-ter), E-SNN side reads `verdict.retention_by_arm` (G5-bis driver writes this key).
- **Constants** : `REQUIRED_ARMS = ("baseline","P_min","P_equ","P_max")`, `ALPHA_PER_ARM = 0.05/4`, `H7B_G_THRESHOLD = 0.5`.
- **Helper** `_load_retention(milestone_path, key)` parameterised over the verdict key (`retention_richer_by_arm` for MLX, `retention_by_arm` for E-SNN). Validates `verdict.{key}` is a dict containing all four arms ; raises ValueError otherwise.
- **Helper** `_welch_two_sided(a, b, alpha) -> (p_two_sided, reject)` : run `welch_one_sided(a,b,alpha)` and `welch_one_sided(b,a,alpha)`, then `p = min(2*min(p_a,p_b), 1.0)`, `reject = p < alpha`.
- **`aggregate_g5bis_verdict(mlx_milestone, esnn_milestone) -> dict`** :
  1. Load MLX rets via key `"retention_richer_by_arm"`, E-SNN rets via key `"retention_by_arm"`.
  2. For each arm in REQUIRED_ARMS : if either side has < 2 samples → `{"insufficient_samples": True, "n_mlx", "n_esnn"}` ; else compute `g = compute_hedges_g(mlx_vals, esnn_vals)` + `_welch_two_sided` ; row = `{hedges_g_mlx_minus_esnn, welch_p_two_sided, reject_h0, consistency=not reject, n_mlx, n_esnn}`.
  3. Compute E-SNN own-substrate effect : `g_h7a = compute_hedges_g(esnn["P_equ"], esnn["baseline"])` ; `welch_h7a = welch_one_sided(esnn["baseline"], esnn["P_equ"], alpha=ALPHA_PER_ARM)`.
  4. Classify : if `abs(g_h7a) < H7B_G_THRESHOLD` AND `not welch_h7a.reject_h0` → `"H7-B"` ; elif `g_h7a >= H7B_G_THRESHOLD` AND `welch_h7a.reject_h0` AND P_equ row's `consistency is True` → `"H7-C"` ; elif `g_h7a >= H7B_G_THRESHOLD` AND `welch_h7a.reject_h0` → `"H7-A"` ; else `"ambiguous"`.
  5. Return `{per_arm, h7_classification, g_h7a_esnn, g_h7a_welch_p, g_h7a_welch_reject_h0, alpha_per_arm, h7b_g_threshold, mlx_milestone, esnn_milestone}`.
- **`_render_md(verdict) -> str`** : title `"# G5-bis cross-substrate aggregate - H7-A/B/C verdict"` ; render verdict + per-arm Welch table (columns : `arm`, `g (MLX - E-SNN)`, `Welch p (two-sided)`, `reject H0`, `consistent`) ; provenance footer linking DR-3 spec, DR-3 evidence, aggregator path.
- **`write_aggregate_dump(*, mlx_milestone, esnn_milestone, out_json, out_md) -> dict`** : same write contract as G5 sister (`mkdir(parents=True)`, `json.dumps(indent=2, sort_keys=True)`, then md).

- [ ] **Step 4: Run tests, expect 3 PASS**

Run: `uv run pytest tests/unit/experiments/test_g5_bis_aggregator.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/g5_bis_richer_esnn/aggregator.py \
        tests/unit/experiments/test_g5_bis_aggregator.py
git commit -m "feat(g5bis): cross-substrate H7 aggregator

Loads G4-ter MLX richer + G5-bis E-SNN richer milestones, runs
per-arm Welch two-sided at Bonferroni alpha/4, classifies into
H7-A (positive transfer, level-divergence), H7-B (MLX-only
artefact), H7-C (universal cross-substrate). H7-B threshold is
g=0.5."
```

---

## Task 6: Run pilot + emit aggregate milestone

- [ ] **Step 1: Verify run-registry baseline**

Run: `python -c "from harness.storage.run_registry import RunRegistry; r = RunRegistry('.run_registry.sqlite'); print('rows=', len(list(r.list_runs())))"`
Note the count. After Task 6 it should grow by 40.

- [ ] **Step 2: Run the pilot at Option B (N=10)**

Run: `uv run python experiments/g5_bis_richer_esnn/run_g5_bis.py`
Expected wall : ~21 min on M1 Max. Output : `docs/milestones/g5-bis-richer-esnn-2026-05-03.json` (40 cells, each with `run_id`) and `.md`.
**Abort condition** : if wall > 90 min or `verdict.h7a_richer_esnn.insufficient_samples is True` for unknown reasons, stop and surface.

- [ ] **Step 3: Run the aggregator**

```bash
uv run python -c "from pathlib import Path; from experiments.g5_bis_richer_esnn.aggregator import write_aggregate_dump; write_aggregate_dump(mlx_milestone=Path('docs/milestones/g4-ter-pilot-2026-05-03.json'), esnn_milestone=Path('docs/milestones/g5-bis-richer-esnn-2026-05-03.json'), out_json=Path('docs/milestones/g5-bis-aggregate-2026-05-03.json'), out_md=Path('docs/milestones/g5-bis-aggregate-2026-05-03.md'))"
```

Expected: writes both files. Confirm with `head -10 docs/milestones/g5-bis-aggregate-2026-05-03.md`.

- [ ] **Step 4: Read the H7 verdict (carried forward into Tasks 7-9)**

Run: `python -c "import json; p = json.load(open('docs/milestones/g5-bis-aggregate-2026-05-03.json')); print('H7=',p['h7_classification'],'g=',p['g_h7a_esnn'],'p=',p['g_h7a_welch_p'],'reject=',p['g_h7a_welch_reject_h0'])"`
Record the four values — they replace `{H7_X}`, `{G_H7A}`, `{WELCH_P}`, `{WELCH_REJ}` in Tasks 7/8/9.

- [ ] **Step 5: Commit milestones**

```bash
git add docs/milestones/g5-bis-richer-esnn-2026-05-03.json \
        docs/milestones/g5-bis-richer-esnn-2026-05-03.md \
        docs/milestones/g5-bis-aggregate-2026-05-03.json \
        docs/milestones/g5-bis-aggregate-2026-05-03.md
git commit -m "feat(g5bis): pilot milestone + aggregate

Records 4 arms x 10 seeds (Option B) E-SNN richer-head retention
on Split-FMNIST 5-task. Aggregate runs Welch two-sided per arm
vs G4-ter MLX richer milestone, classifies H7-A/B/C. Append-only
per docs/CLAUDE.md milestone discipline."
```

---

## Task 7: Update Paper 2 §7.1.9 EN + FR

**Files:**
- Modify: `docs/papers/paper2/results.md`
- Modify: `docs/papers/paper2-fr/results.md`

The G4-quinto §7.1.7 is the last existing subsection (verified in Task 0). Append §7.1.9 after the last `## 7.1.x` heading present in each file (if a §7.1.8 exists, append after it ; otherwise after §7.1.7).

- [ ] **Step 1: Append §7.1.9 EN**

Append the following block to `docs/papers/paper2/results.md`. Substitute `{H7_X}`, `{G_H7A}`, `{WELCH_P}`, `{WELCH_REJ}` with the values recorded in Task 6 Step 4.

```markdown
## 7.1.9 G5-bis pilot — richer head ported to E-SNN substrate (2026-05-03)

The G4-ter §7.1.5 MLX richer-head positive finding (`g_h2 = +2.77`, REPLAY+DOWNSCALE on a 3-layer hierarchy) was the only positive cross-arm effect surviving the cycle-3 ablation cascade. The G5 cross-substrate replication §7.1.3 transferred the *spectator pattern* but not the absolute retention level. This sub-section records the cross-substrate transfer test of the *positive* effect itself : the same coupling (REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE on richer hierarchy, HP combo C5) ported to a 3-layer rate-coded LIF SNN on the E-SNN thalamocortical substrate.

Pre-registration : [`docs/osf-prereg-g5-bis-richer-esnn.md`](../../osf-prereg-g5-bis-richer-esnn.md). Sister milestone : [`docs/milestones/g5-bis-richer-esnn-2026-05-03.md`](../../milestones/g5-bis-richer-esnn-2026-05-03.md). Cross-substrate aggregate : [`docs/milestones/g5-bis-aggregate-2026-05-03.md`](../../milestones/g5-bis-aggregate-2026-05-03.md).

E-SNN richer-head finding (Option B, N=10 per arm) :
- observed `g_h7a = {G_H7A}` (P_equ vs baseline retention),
- Welch one-sided p = `{WELCH_P}`, `reject_h0 = {WELCH_REJ}`,
- cross-substrate classification : **{H7_X}**.

**Honest reading.** {H7_X} is the verdict of the H7-A/B/C decision rule pre-registered in §1 of `docs/osf-prereg-g5-bis-richer-esnn.md`. H7-A is "positive transfer, level-divergence" ; H7-B is "MLX-only artefact" ; H7-C is "universal cross-substrate" (effect transfers and matches MLX magnitude within factor 2). The Option B pilot is exploratory ; if {H7_X} ∈ {H7-A, H7-C}, an Option A confirmatory N≥30 follow-up is scheduled.

The G4-ter MLX positive finding remains intact regardless of {H7_X} ; G5-bis only conditions the *substrate-agnosticism* claim on the positive-effect channel, separate from the qualitative-spectator-pattern claim already documented by G5 §7.1.3.
```

- [ ] **Step 2: Append §7.1.9 FR mirror**

Append to `docs/papers/paper2-fr/results.md` :

```markdown
## 7.1.9 Pilote G5-bis — tete riche portee sur substrat E-SNN (2026-05-03)

Le resultat positif G4-ter §7.1.5 sur tete hierarchique MLX (`g_h2 = +2.77`, REPLAY+DOWNSCALE sur hierarchie 3-couches) fut le seul effet inter-bras positif survivant a la cascade d'ablation du cycle-3. La replication cross-substrat G5 §7.1.3 a transfere le *pattern spectateur* mais pas le niveau absolu de retention. Cette sous-section enregistre le test de transfert cross-substrat de l'effet *positif* lui-meme : meme couplage (REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE sur hierarchie riche, combo HP C5) porte sur un classifieur SNN LIF rate-code 3-couches sur le substrat E-SNN thalamocortical.

Pre-enregistrement : [`docs/osf-prereg-g5-bis-richer-esnn.md`](../../osf-prereg-g5-bis-richer-esnn.md). Milestone soeur : [`docs/milestones/g5-bis-richer-esnn-2026-05-03.md`](../../milestones/g5-bis-richer-esnn-2026-05-03.md). Agregat cross-substrat : [`docs/milestones/g5-bis-aggregate-2026-05-03.md`](../../milestones/g5-bis-aggregate-2026-05-03.md).

Resultat tete riche E-SNN (Option B, N=10 par bras) :
- `g_h7a = {G_H7A}` observe (P_equ vs baseline retention),
- Welch unilateral p = `{WELCH_P}`, `reject_h0 = {WELCH_REJ}`,
- classification cross-substrat : **{H7_X}**.

**Lecture honnete.** {H7_X} est le verdict de la regle de decision H7-A/B/C pre-enregistree dans `docs/osf-prereg-g5-bis-richer-esnn.md` §1. H7-A = transfert positif, divergence en niveau ; H7-B = artefact MLX seulement ; H7-C = universel cross-substrat. Le pilote Option B est exploratoire ; si {H7_X} ∈ {H7-A, H7-C}, un suivi confirmatoire Option A N≥30 est programme.

Le resultat positif G4-ter MLX reste intact quel que soit {H7_X} ; G5-bis ne fait que conditionner la revendication d'*agnosticisme de substrat* au canal de l'effet positif, distincte de la revendication pattern-spectateur-qualitatif deja documentee par G5 §7.1.3.
```

- [ ] **Step 3: Verify both sections present**

Run: `grep -c "^## 7.1.9" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md`
Expected: `1` per file.

- [ ] **Step 4: Verify all placeholders substituted**

Run: `grep -nE "\{H7_X\}|\{G_H7A\}|\{WELCH_P\}|\{WELCH_REJ\}" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md`
Expected: zero hits. Fix in-place if any.

- [ ] **Step 5: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): add G5-bis section 7.1.9 EN+FR

Cross-substrate transfer test of the G4-ter MLX richer-head
positive finding onto E-SNN. Records observed g_h7a, Welch
verdict, and H7-A/B/C classification. EN+FR mirror per
docs/papers/CLAUDE.md propagation rule."
```

---

## Task 8: DR-3 evidence revision per H7 outcome

**Files:**
- Modify: `docs/proofs/dr3-substrate-evidence.md`

- [ ] **Step 1: Read current file tail**

Run: `tail -30 docs/proofs/dr3-substrate-evidence.md`
Note the last subsection heading (G5 binary-head should already be recorded there).

- [ ] **Step 2: Append the H7 entry**

Append to `docs/proofs/dr3-substrate-evidence.md` (substitute placeholders with Task 6 Step 4 values) :

```markdown
## G5-bis (2026-05-03) — positive-effect cross-substrate transfer

Pre-registration : `docs/osf-prereg-g5-bis-richer-esnn.md`.
Milestone : `docs/milestones/g5-bis-richer-esnn-2026-05-03.md`.
Aggregate : `docs/milestones/g5-bis-aggregate-2026-05-03.md`.

G4-ter MLX richer-head positive finding (`g_h2 = +2.77`,
REPLAY+DOWNSCALE) ported to a 3-layer rate-coded LIF SNN on
`kiki_oniric.substrates.esnn_thalamocortical`. Result :
`g_h7a = {G_H7A}`, Welch one-sided p = `{WELCH_P}`,
`reject_h0 = {WELCH_REJ}`. Cross-substrate classification :
**{H7_X}**.

DR-3 evidence revision (separate from the G5 binary-head
qualitative-spectator-pattern record above) :
- if {H7_X} == H7-C : "real-substrate empirical evidence" at
  the positive-effect channel ; the dream effect transfers and
  matches MLX magnitude within factor 2.
- if {H7_X} == H7-A : "positive transfer, level-divergence" ;
  effect transfers, absolute level diverges (parallel to G5
  binary-head divergence).
- if {H7_X} == H7-B : "MLX-only artefact at first-pilot scale" ;
  framework C's positive-effect channel does not transfer to
  E-SNN on this protocol. Confirmatory N>=30 follow-up scheduled.

EC stays PARTIAL ; FC stays C-v0.12.0 (no axiom statement change).
```

- [ ] **Step 3: Verify file tail**

Run: `tail -5 docs/proofs/dr3-substrate-evidence.md`
Expected: ends on the EC/FC note above with a trailing newline.

- [ ] **Step 4: Commit**

```bash
git add docs/proofs/dr3-substrate-evidence.md
git commit -m "docs(dr3): record G5-bis H7 outcome

Appends positive-effect cross-substrate evidence entry distinct
from the G5 binary-head qualitative-spectator-pattern record.
EC stays PARTIAL, FC stays C-v0.12.0 (no axiom change)."
```

---

## Task 9: CHANGELOG + STATUS updates

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `STATUS.md`

- [ ] **Step 1: Append `[Unreleased]` G5-bis row**

In `CHANGELOG.md`, locate or create the `[Unreleased]` block and append (substitute placeholders) :

```markdown
- **G5-bis (2026-05-03)** — Cross-substrate transfer test of the
  G4-ter MLX richer-head positive finding onto the E-SNN
  thalamocortical substrate. 4 arms x 10 seeds (Option B) on a
  3-layer rate-coded LIF SNN with REPLAY+DOWNSCALE+RESTRUCTURE+
  RECOMBINE coupling. Verdict : **{H7_X}** with `g_h7a = {G_H7A}`.
  EC stays PARTIAL, FC stays C-v0.12.0. Sister milestones :
  `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}` +
  `docs/milestones/g5-bis-aggregate-2026-05-03.{json,md}`.
```

- [ ] **Step 2: Add gates row + As-of update in STATUS.md**

Append to the `## Gates` table :

```markdown
| G5-bis-pilot — richer-head cross-substrate transfer | 2026-05-03 → milestone | 🔶 PARTIAL ({H7_X} verdict, g_h7a={G_H7A}, Welch p={WELCH_P}, N=10 ; Option A N≥30 confirmatory scheduled if H7-A or H7-C) |
```

Update the top `**As of**` line by prepending : `2026-05-03 G5-bis pilot ({H7_X}, g_h7a={G_H7A}) — `.

- [ ] **Step 3: Verify**

Run: `grep -n "G5-bis" STATUS.md CHANGELOG.md`
Expected: ≥ 2 hits in STATUS.md, ≥ 1 in CHANGELOG.md.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(g5bis): changelog + status update

Records G5-bis pilot result with H7 classification verdict in
the unreleased changelog and a new gates-table row in STATUS.md.
EC stays PARTIAL, FC unchanged."
```

---

## Task 10: Self-review + final test sweep

- [ ] **Step 1: Full unit-test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: every test passes (the four new test files contribute ~15 tests).

- [ ] **Step 2: Lint + typecheck**

Run: `uv run ruff check experiments/g5_bis_richer_esnn/ tests/unit/experiments/test_g5_bis_*.py`
Expected: clean.
Run: `uv run mypy harness tests`
Expected: clean (experiments/ is excluded from strict mypy).

- [ ] **Step 3: Cross-file references present**

Run: `grep -rln "g5-bis" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md docs/proofs/dr3-substrate-evidence.md CHANGELOG.md STATUS.md`
Expected: all 5 paths returned.

- [ ] **Step 4: All placeholders substituted**

Run: `grep -nE "\{H7_X\}|\{G_H7A\}|\{WELCH_P\}|\{WELCH_REJ\}" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md docs/proofs/dr3-substrate-evidence.md CHANGELOG.md STATUS.md`
Expected: zero hits. If any : fix in-place and create a NEW commit (never `--amend`).

- [ ] **Step 5: EN+FR §7.1.9 parity**

Run: `grep -c "^## 7.1.9" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md`
Expected: `1` in each file.

- [ ] **Step 6: No FC bump leaked**

Run: `grep -n "C-v0.13\|FC.*bump\|formal axis bump" CHANGELOG.md docs/proofs/dr3-substrate-evidence.md docs/papers/paper2/results.md`
Expected: zero hits referencing G5-bis. EC stays PARTIAL, FC stays C-v0.12.0.

- [ ] **Step 7: Coverage gate**

Run: `uv run pytest --cov=harness --cov=kiki_oniric --cov-fail-under=90 -q`
Expected: PASS at coverage ≥ 90 %.

- [ ] **Step 8: Pre-reg lock-target check**

Inspect `docs/osf-prereg-g5-bis-richer-esnn.md` and confirm the `**Lock target**` line says "before any G5-bis run is registered" — this anchors the OSF discipline.

- [ ] **Step 9: Self-review checklist (no commit)**

Walk through :
1. **Spec coverage** : every section of the brief maps to a Task. Task 0 (investigate), Task 0.5 (compute decision), Task 1 (pre-reg), Task 2 (classifier), Task 3 (wrapper), Task 4 (driver), Task 5 (aggregator), Task 6 (run + milestone), Task 7 (paper EN+FR), Task 8 (DR-3 revision), Task 9 (changelog+status), Task 10 (self-review).
2. **Type consistency** : `EsnnG5BisHierarchicalClassifier.W_in/W_h/W_out` referenced consistently across `_replay_step`, `_downscale_step`, `_restructure_step`, `_recombine_step` ; `dream_episode_hier_esnn` signature mirrors G4-ter `dream_episode_hier`. **OK**.
3. **Placeholder discipline** : `{H7_X}`, `{G_H7A}`, `{WELCH_P}`, `{WELCH_REJ}` are intentional run-time substitutions from the aggregate milestone, asserted absent in Step 4. **OK**.
4. **R1 fidelity** : every cell registers via `RunRegistry.register(c_version, profile, seed, commit_sha)` (Task 4 driver).
5. **EN→FR mirror** : `paper2-fr/results.md` updated in same commit as `paper2/results.md` (Task 7).
6. **Commit hygiene** : conventional commits, ≤50-char subjects, ≥3-char scopes (`g5bis`, `paper2`, `dr3`), no `Co-Authored-By`, no `--no-verify`. **OK**.
7. **Append-only milestones** : Task 6 creates the dated milestone files ; Tasks 7/8/9 modify only `docs/proofs/`, `CHANGELOG.md`, `STATUS.md`, `docs/papers/`. Milestones never re-edited.

If any item fails : fix inline + new commit. Do not amend.

- [ ] **Step 10: No commit**

Review only.

---

## Glossary (cross-references for the agent worker)

- **G4-ter** : MLX richer-head pilot (3-layer hierarchy, HP combo C5, N=30) — `experiments/g4_ter_hp_sweep/`. Result : `g_h2 = +2.77`.
- **G5** : E-SNN binary-head cross-substrate replication (1-hidden-layer LIF, N=5) — `experiments/g5_cross_substrate/`. Result : within-substrate spectator, cross-substrate level divergence.
- **G5-bis** (this plan) : E-SNN richer-head cross-substrate transfer test (3-layer LIF, N=10) — `experiments/g5_bis_richer_esnn/`.
- **HP combo C5** : `representative_combo()` from `experiments.g4_ter_hp_sweep.hp_grid` — downscale 0.95, replay_batch 32, replay_n_steps 5, replay_lr 0.01.
- **Option A/B/C** : N=30/10/5 sweep budgets. **Option B locked** for this pilot per Task 0.5.
- **H7-A/B/C** : pre-registered classification — A = positive transfer + level-divergence ; B = MLX-only artefact ; C = universal cross-substrate. See `docs/osf-prereg-g5-bis-richer-esnn.md` §1.
- **DualVer status** : EC stays PARTIAL ; FC stays C-v0.12.0. No bump triggered by this pilot regardless of H7 outcome.

End of plan.
