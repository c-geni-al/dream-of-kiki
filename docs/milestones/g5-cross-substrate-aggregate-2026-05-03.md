# G5 cross-substrate aggregate — DR-3 cross-substrate consistency

**Date** : 2026-05-03
**MLX milestone** : `docs/milestones/g4-pilot-2026-05-03-bis.json`
**E-SNN milestone** : `docs/milestones/g5-cross-substrate-2026-05-03.json`
**Bonferroni alpha / 4** : 0.0125

## Verdict : DR-3 cross-substrate divergence DETECTED

At least one arm shows Welch p <= alpha/4 = 0.0125 — see per-arm table below for the diverging arm(s).

## Per-arm Welch consistency

| arm | g (MLX - E-SNN) | Welch p (two-sided) | reject H0 | consistent |
|-----|------------------|----------------------|-----------|------------|
| baseline | +9.9763 | 0.0001 | True | False |
| P_min | +3.4942 | 0.0035 | True | False |
| P_equ | +3.4942 | 0.0035 | True | False |
| P_max | +3.4942 | 0.0035 | True | False |

## Provenance

- DR-3 spec : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
- DR-3 evidence record : `docs/proofs/dr3-substrate-evidence.md`
- Aggregator : `experiments/g5_cross_substrate/aggregator.py`
