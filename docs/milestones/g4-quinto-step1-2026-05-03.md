# G4-quinto Step 1 — H5-A MLP-on-CIFAR

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `840cc324979bdfe19f129eae9bb1260e44fe9cbd`
**Cells** : 120 (4 arms x 30 seeds x 1 HP)
**Hidden** : (256, 128, 64, 32)
**Wall time** : 229.8s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quinto-pilot.md`

### H5-A — benchmark-scale (5-layer MLP-on-CIFAR)
- mean retention P_min : 0.8713
- mean retention P_equ : 0.8754
- mean retention P_max : 0.8754
- monotonic observed P_max >= P_equ >= P_min : True
- Jonckheere J statistic : 1362.0000
- one-sided p (alpha = 0.0167) : 0.4646 -> reject_h0 = False

*Honest reading* : reject_h0 means there is evidence for the predicted ordering at this N ; failure to reject means no evidence at this N (absence of evidence vs evidence of absence).

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quinto-pilot.md](../osf-prereg-g4-quinto-pilot.md)
- Driver : `experiments/g4_quinto_test/run_step1_mlp_cifar.py`
- Substrate : `experiments.g4_quinto_test.cifar_mlp_classifier.G4HierarchicalCIFARClassifier`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
