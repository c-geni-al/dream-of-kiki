# G5 cross-substrate pilot — E-SNN x Split-FMNIST

**Date** : 2026-05-03
**Substrate** : `esnn_thalamocortical`
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `d475530ae448c8a44cd83cf7a0086f629867c6f7`
**Cells** : 20 (4 arms x 5 seeds)
**Wall time** : 242.9s

## Pre-registered hypotheses (E-SNN substrate)

Pre-registration : `docs/osf-prereg-g5-cross-substrate.md`

### H1 — P_equ retention vs Hu 2020 (g >= 0.21)
- observed Hedges' g : **0.0000**
- within Hu 2020 95% CI : False
- Welch one-sided p (alpha/3 = 0.0167) : 0.5000 -> reject_h0 = False

### H3 — P_min retention vs Javadi 2024 (|g| >= 0.13, decrement)
- observed Hedges' g : **0.0000**
- |g| within Javadi 2024 95% CI : False
- Welch one-sided p (alpha/3 = 0.0167) : 0.5000 -> reject_h0 = False

### H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]
- mean retention P_min : 0.5119
- mean retention P_equ : 0.5119
- mean retention P_max : 0.5119
- monotonic observed : True
- Jonckheere J : 37.5000 (one-sided p = 0.5000 -> reject_h0 = False)

## Cells (R1 traceability)

| arm | seed | acc_initial | acc_final | retention | excluded | run_id |
|-----|------|-------------|-----------|-----------|----------|--------|
| baseline | 0 | 0.9820 | 0.5015 | 0.5107 | False | `1952052d321741b0a0b7f78562dab5c9` |
| baseline | 1 | 0.9790 | 0.5030 | 0.5138 | False | `1010a575e263cc6ffba47c4e81eaf379` |
| baseline | 2 | 0.9800 | 0.5020 | 0.5122 | False | `62f1d33d48d388ac2e3d42f6e6948f3e` |
| baseline | 3 | 0.9820 | 0.5015 | 0.5107 | False | `2329b0076703974f3310cfa87b003a62` |
| baseline | 4 | 0.9790 | 0.5015 | 0.5123 | False | `3479872b20a823e51d8d6c21d2b18e49` |
| P_min | 0 | 0.9820 | 0.5015 | 0.5107 | False | `fbe8e7401b7890ce9750f441c0489bc8` |
| P_min | 1 | 0.9790 | 0.5030 | 0.5138 | False | `4d85ba6ac09dab353f51536e2818b7af` |
| P_min | 2 | 0.9800 | 0.5020 | 0.5122 | False | `50dc88474da79b6da517e14a4e16fc19` |
| P_min | 3 | 0.9820 | 0.5015 | 0.5107 | False | `0ff714c0a8257168d4afd183bc60868e` |
| P_min | 4 | 0.9790 | 0.5015 | 0.5123 | False | `b205516b79a1a155c2a0a5f76d0b687c` |
| P_equ | 0 | 0.9820 | 0.5015 | 0.5107 | False | `913e32b111b0ec24101da6b96a3372df` |
| P_equ | 1 | 0.9790 | 0.5030 | 0.5138 | False | `60e82835463c804431fe53ca3be6b38e` |
| P_equ | 2 | 0.9800 | 0.5020 | 0.5122 | False | `322e80417dce6ceceae59f0e072202b6` |
| P_equ | 3 | 0.9820 | 0.5015 | 0.5107 | False | `f3db3840361a2025d3be0cf5ab240a87` |
| P_equ | 4 | 0.9790 | 0.5015 | 0.5123 | False | `771d7c267e29f0d2ba880e7d41aae816` |
| P_max | 0 | 0.9820 | 0.5015 | 0.5107 | False | `5e5e4ac93c35efc6abb98c1b7a9bdc56` |
| P_max | 1 | 0.9790 | 0.5030 | 0.5138 | False | `af3608febf0dc73103656e3dc2a45f6d` |
| P_max | 2 | 0.9800 | 0.5020 | 0.5122 | False | `fe0db0629caf1a6cdddb957fbfd7a75d` |
| P_max | 3 | 0.9820 | 0.5015 | 0.5107 | False | `b923d1b9720152fccbec36817ffc2ba4` |
| P_max | 4 | 0.9790 | 0.5015 | 0.5123 | False | `bd832cc45445d1184933d99e4ef7fc27` |

## Provenance

- Pre-registration : [docs/osf-prereg-g5-cross-substrate.md](../osf-prereg-g5-cross-substrate.md)
- Driver : `experiments/g5_cross_substrate/run_g5.py`
- Substrate : `kiki_oniric.substrates.esnn_thalamocortical`
- Sister pilot (MLX, binary head) : [g4-pilot-2026-05-03-bis.md](g4-pilot-2026-05-03-bis.md)
- Parent richer-head positive finding (MLX, hierarchical) : [g4-ter-pilot-2026-05-03.md](g4-ter-pilot-2026-05-03.md)
- Cross-substrate aggregator output : see `docs/milestones/g5-cross-substrate-aggregate-2026-05-03.{json,md}` (Task 5 deliverable)
