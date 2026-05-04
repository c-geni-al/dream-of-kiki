# G4-sexto Step 1 — H6-A CIFAR-100+RECOMBINE strategy + placebo

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `8874c7bb9b054e863e591f682c7006565070dc4a`
**Cells** : 760 (2 strategies x 4 arms x 95 seeds)
**Wall time** : 5129.2s
**Smoke** : False

**Multi-class exclusion threshold** : acc_initial < 2 × random_chance = 0.20 (random_chance = 0.10 for n_classes = 10).

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-sexto-pilot.md`

### H6-A — universality of RECOMBINE-empty (CIFAR-100, n_classes=10)

Welch two-sided test of `retention(P_max with mog)` vs `retention(P_max with none)` on the small CNN with a 10-class per-task head. **Failing** to reject H0 at alpha = 0.05 / 3 = 0.0167 confirms H6-A : the G4-quinto H5-C RECOMBINE-empty finding generalises to mid-large class count (CIFAR-100, 100 fine classes split into 10 tasks of 10 classes each).

- mean retention P_max (mog) : 0.3701 (N=76)
- mean retention P_max (none) : 0.3592 (N=76)
- Hedges' g (mog vs none) : 0.1527
- Welch t : 0.9459
- Welch p (two-sided, alpha = 0.0167) : 0.3457 -> fail_to_reject_h0 = True

**H6-A verdict** : RECOMBINE empty confirmed = True (positive empirical claim mog ≈ none if True).

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H6-A specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the small CNN substrate at CIFAR-100 100-class scale.

## Provenance

- Pre-registration : [docs/osf-prereg-g4-sexto-pilot.md](../osf-prereg-g4-sexto-pilot.md)
- Driver : `experiments/g4_sexto_test/run_step1_cifar100.py`
- Substrate : `experiments.g4_quinto_test.small_cnn.G4SmallCNN` (n_classes=10)
- Loader : `experiments.g4_sexto_test.cifar100_dataset.load_split_cifar100_10tasks_auto`
- Strategies : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
