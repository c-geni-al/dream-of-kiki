# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream

**Date** : 2026-05-04
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `b42f7e2296ca3dfac3cd4b787a6442f80008a2d5`
**Cells** : 20
**Wall time** : 4802.8s
**Smoke** : False

## Pre-registered hypotheses (LOCKED)

Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.
Decision rules at α/3 = 0.0167 (Bonferroni over {H9-A, H9-B, H9-C}).

### Verdict
```
{
  "bonferroni_alpha": 0.016666666666666666,
  "h9a_classification": "INSUFFICIENT",
  "h9a_g": NaN,
  "h9a_positive_sign": false,
  "h9a_strict_large_effect": false,
  "h9a_welch_p": NaN,
  "h9a_welch_reject": false,
  "h9c_classification": "INSUFFICIENT",
  "h9c_jonckheere_p": NaN,
  "h9c_jonckheere_reject": false,
  "h9c_mean_p_equ": 1.1736596736596736,
  "h9c_mean_p_max": NaN,
  "h9c_mean_p_min": NaN
}
```

## Cells (R1 traceability)

| arm | seed | retention | excluded | run_id |
|-----|------|-----------|----------|--------|
| baseline | 0 | 1.3237 | True | `3f83fe784a3860c9b00098d2759322a2` |
| baseline | 1 | 1.3014 | True | `99af58e635b9d61370a7fc778ed3c949` |
| baseline | 2 | 1.2227 | True | `a472fba07d440132cd4847896f1773b2` |
| baseline | 3 | 1.1294 | True | `b2c112fcfbcea2f33fdd53d8e5eb2a9e` |
| baseline | 4 | 1.2153 | True | `d84c3109e59b098fe398c389ff1decbf` |
| P_min | 0 | 1.1727 | True | `5622eaa042687cf1e455bedb7527e047` |
| P_min | 1 | 0.9558 | True | `53f0d470b694acab43c63ae9ac23376f` |
| P_min | 2 | 1.1240 | True | `06e7c76b6da6e0a63fc68d9bfc545529` |
| P_min | 3 | 0.9106 | True | `7b684f52f3faf5caff643adf8b79de78` |
| P_min | 4 | 1.2167 | True | `edab33ed9ea53ca9d8432b595dc8ed8c` |
| P_equ | 0 | 1.3047 | True | `b6ed447c86039a940cb9d805612f8a79` |
| P_equ | 1 | 1.1230 | True | `0c6cfa96b10c6aa8be110ab4dc9d945e` |
| P_equ | 2 | 1.0000 | True | `803a42893fcb266a368620317b73a91b` |
| P_equ | 3 | 1.1737 | False | `c0c659d6f13d8da88da89ebe6bba5f6d` |
| P_equ | 4 | 1.2631 | True | `9912b6096cca1e276afa1707b24e5101` |
| P_max | 0 | 1.1239 | True | `716e4a047c7d59df71d399240b9a6575` |
| P_max | 1 | 1.2091 | True | `11a28ec5b876d6b818ffd37719aa05a5` |
| P_max | 2 | 1.2538 | True | `12c2cee7bab3131af01754e55fb5997f` |
| P_max | 3 | 0.9501 | True | `75ea26581a08721b53214a98da868a87` |
| P_max | 4 | 1.2734 | True | `be69ab07232802c7bd8d3bf645e9c806` |

## Honest reporting

Per pre-reg §6, EC stays PARTIAL across all H9-{A,B,C} rows. FC stays at C-v0.12.0. H9-A confirmation queues an Option-A (N >= 10) follow-up pre-reg before any STABLE bump. H9-B confirms G5-bis MLX-only artefact extends from toy E-SNN to real-LLM tier. H9-C confirms DR-4 inversion universalises at real-LLM scale.