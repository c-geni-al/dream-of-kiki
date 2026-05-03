# G5-ter cross-substrate aggregate - H8-A/B/C verdict

**Date** : 2026-05-03
**MLX milestone** : `docs/milestones/g4-quinto-step2-2026-05-03.json`
**E-SNN milestone** : `docs/milestones/g5-ter-spiking-cnn-2026-05-03.json`
**Bonferroni alpha / 4** : 0.0125
**Own-substrate g_h8 threshold** : 0.50
**H8-A g_mlx_minus_esnn floor** : 2.00
**H8-B g_mlx_minus_esnn ceiling** : 1.00

## Verdict : H8-C

- Observed E-SNN g_h8 (P_equ vs baseline) : **-0.1093**
- Welch one-sided p (alpha/4 = 0.0125) : 0.5992
- reject_h0 (own-substrate) : False
- Observed cross-substrate g (MLX - E-SNN, P_equ) : **+1.3125**

## Per-arm cross-substrate Welch consistency

| arm | g (MLX - E-SNN) | Welch p (two-sided) | reject H0 | consistent |
|-----|------------------|----------------------|-----------|------------|
| baseline | +1.2087 | 0.0034 | True | False |
| P_min | +1.3193 | 0.0004 | True | False |
| P_equ | +1.3125 | 0.0012 | True | False |
| P_max | +1.3125 | 0.0012 | True | False |

## Provenance

- DR-3 spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`
- Aggregator : `experiments/g5_ter_spiking_cnn/aggregator.py`
- Pre-registration : `docs/osf-prereg-g5-ter-spiking-cnn.md`
- Sister G5-bis aggregate : `docs/milestones/g5-bis-aggregate-2026-05-03.md`
