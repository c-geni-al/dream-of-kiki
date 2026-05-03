# G5-ter pilot - spiking-CNN ported to E-SNN substrate

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `14c9d326ec7ed31044bf1904fd180fd3367a9df9`
**Substrate** : esnn_thalamocortical_spiking_cnn
**Cells** : 40 (4 arms x 10 seeds x 1 HP)
**Wall time** : 2182.1s

## Pre-registered hypothesis (own-substrate H8-A)

Pre-registration : `docs/osf-prereg-g5-ter-spiking-cnn.md`

### H8-A - E-SNN spiking-CNN (P_equ vs baseline retention)
- observed Hedges' g_h8 : **-0.1093**
- above zero : False
- above Hu 2020 lower CI 0.21 : False
- Welch one-sided p (alpha/4 = 0.0125) : 0.5992 -> reject_h0 = False

## Cells (per arm x seed)

| arm | seed | hp | acc_task1_initial | acc_task1_final | retention | excluded |
|-----|------|----|--------------------|------------------|-----------|----------|
| baseline | 0 | C5 | 0.6345 | 0.5015 | 0.7904 | False |
| baseline | 1 | C5 | 0.5940 | 0.5140 | 0.8653 | False |
| baseline | 2 | C5 | 0.6465 | 0.4995 | 0.7726 | False |
| baseline | 3 | C5 | 0.6545 | 0.5305 | 0.8105 | False |
| baseline | 4 | C5 | 0.6965 | 0.5660 | 0.8126 | False |
| baseline | 5 | C5 | 0.6335 | 0.5240 | 0.8272 | False |
| baseline | 6 | C5 | 0.7415 | 0.5875 | 0.7923 | False |
| baseline | 7 | C5 | 0.5670 | 0.5165 | 0.9109 | False |
| baseline | 8 | C5 | 0.6850 | 0.5525 | 0.8066 | False |
| baseline | 9 | C5 | 0.5655 | 0.6495 | 1.1485 | False |
| P_min | 0 | C5 | 0.6345 | 0.5005 | 0.7888 | False |
| P_min | 1 | C5 | 0.5940 | 0.5055 | 0.8510 | False |
| P_min | 2 | C5 | 0.6465 | 0.4980 | 0.7703 | False |
| P_min | 3 | C5 | 0.6545 | 0.5225 | 0.7983 | False |
| P_min | 4 | C5 | 0.6965 | 0.5800 | 0.8327 | False |
| P_min | 5 | C5 | 0.6335 | 0.6220 | 0.9818 | False |
| P_min | 6 | C5 | 0.7415 | 0.5835 | 0.7869 | False |
| P_min | 7 | C5 | 0.5670 | 0.5005 | 0.8827 | False |
| P_min | 8 | C5 | 0.6850 | 0.5325 | 0.7774 | False |
| P_min | 9 | C5 | 0.5655 | 0.5615 | 0.9929 | False |
| P_equ | 0 | C5 | 0.6345 | 0.5005 | 0.7888 | False |
| P_equ | 1 | C5 | 0.5940 | 0.5040 | 0.8485 | False |
| P_equ | 2 | C5 | 0.6465 | 0.4980 | 0.7703 | False |
| P_equ | 3 | C5 | 0.6545 | 0.5150 | 0.7869 | False |
| P_equ | 4 | C5 | 0.6965 | 0.5440 | 0.7810 | False |
| P_equ | 5 | C5 | 0.6335 | 0.6485 | 1.0237 | False |
| P_equ | 6 | C5 | 0.7415 | 0.5420 | 0.7310 | False |
| P_equ | 7 | C5 | 0.5670 | 0.5005 | 0.8827 | False |
| P_equ | 8 | C5 | 0.6850 | 0.5535 | 0.8080 | False |
| P_equ | 9 | C5 | 0.5655 | 0.5635 | 0.9965 | False |
| P_max | 0 | C5 | 0.6345 | 0.5005 | 0.7888 | False |
| P_max | 1 | C5 | 0.5940 | 0.5040 | 0.8485 | False |
| P_max | 2 | C5 | 0.6465 | 0.4980 | 0.7703 | False |
| P_max | 3 | C5 | 0.6545 | 0.5150 | 0.7869 | False |
| P_max | 4 | C5 | 0.6965 | 0.5440 | 0.7810 | False |
| P_max | 5 | C5 | 0.6335 | 0.6485 | 1.0237 | False |
| P_max | 6 | C5 | 0.7415 | 0.5420 | 0.7310 | False |
| P_max | 7 | C5 | 0.5670 | 0.5005 | 0.8827 | False |
| P_max | 8 | C5 | 0.6850 | 0.5535 | 0.8080 | False |
| P_max | 9 | C5 | 0.5655 | 0.5635 | 0.9965 | False |

## Provenance

- Pre-registration : [docs/osf-prereg-g5-ter-spiking-cnn.md](../osf-prereg-g5-ter-spiking-cnn.md)
- Sister G5-bis aggregate : [g5-bis-aggregate-2026-05-03.md](g5-bis-aggregate-2026-05-03.md)
- Sister G4-quinto Step 2 milestone : [g4-quinto-step2-2026-05-03.md](g4-quinto-step2-2026-05-03.md)
- Cross-substrate aggregate : [g5-ter-aggregate-2026-05-03.md](g5-ter-aggregate-2026-05-03.md)
- Driver : `experiments/g5_ter_spiking_cnn/run_g5_ter.py`
- Substrate : `experiments.g5_ter_spiking_cnn.spiking_cnn.EsnnG5TerSpikingCNN`
- HP combo : C5 (`representative_combo()`)
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
