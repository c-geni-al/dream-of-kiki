# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream

**Date** : 2026-05-04
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `db44d05c37733625f955546bbfb3971cc6ea1ca0`
**Cells** : 20
**Wall time** : 4409.6s
**Smoke** : False

## Pre-registered hypotheses (LOCKED)

Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.
Decision rules at α/3 = 0.0167 (Bonferroni over {H9-A, H9-B, H9-C}).

### Verdict
```
{
  "bonferroni_alpha": 0.016666666666666666,
  "h9a_classification": "H9-B",
  "h9a_g": 0.0054288435551049985,
  "h9a_positive_sign": true,
  "h9a_strict_large_effect": false,
  "h9a_welch_p": 0.49632516063773763,
  "h9a_welch_reject": false,
  "h9c_classification": "H9-C",
  "h9c_jonckheere_p": 0.8422417645021294,
  "h9c_jonckheere_reject": false,
  "h9c_mean_p_equ": 1.0841188866049574,
  "h9c_mean_p_max": 1.032456826360917,
  "h9c_mean_p_min": 1.0915030908405272
}
```

## Cells (R1 traceability)

| arm | seed | retention | excluded | run_id |
|-----|------|-----------|----------|--------|
| baseline | 0 | 1.1077 | False | `576fcbe82e3b0bf8edc6f1ce6288c497` |
| baseline | 1 | 1.0649 | False | `d395ec153bd5263c39a7575413fc2e07` |
| baseline | 2 | 0.9814 | False | `d279ccfee84da52b25baf0e1e87c80a8` |
| baseline | 3 | 1.1335 | False | `4744223bb36e6cfb23571d66fe1cef2e` |
| baseline | 4 | 1.1312 | False | `3ac9130d7b6683d6feca57057a57f254` |
| P_min | 0 | 1.0613 | False | `ebae9fd2c5658c1bb5ad85638a1f8d1d` |
| P_min | 1 | 1.2097 | False | `b065098d9654a457b3b6a5096c404e11` |
| P_min | 2 | 1.0212 | False | `45c35aa932ab87b4f571b8ad58df2bf0` |
| P_min | 3 | 1.0204 | False | `5efb362f185079b25922c2d3847eeece` |
| P_min | 4 | 1.1449 | False | `bb7ed08cc95fcc0d7dae59675142af69` |
| P_equ | 0 | 1.1361 | False | `b95a2ace75330144fdff8dc4384d5e0e` |
| P_equ | 1 | 1.1224 | False | `9a16c78db51ef8f01836e5e8b61d8d14` |
| P_equ | 2 | 0.9793 | False | `1207081bd5862ccc91b39f3c3aeb2fcf` |
| P_equ | 3 | 1.0979 | False | `f81c8e9ae37cdc4b069d159695d4f01a` |
| P_equ | 4 | 1.0849 | False | `13c10bdde6f6a89ccb27d4076942c76d` |
| P_max | 0 | 1.0806 | False | `daa15cc75f8ba4c0db6ad8b227623aca` |
| P_max | 1 | 0.8749 | False | `110f1efdee133b66e6a31b5463684389` |
| P_max | 2 | 1.0492 | False | `80a83f6fed5f43d6c3805d3f0f067412` |
| P_max | 3 | 1.0382 | False | `ce392411efbefb2dea9d35b28d60b5ec` |
| P_max | 4 | 1.1194 | False | `a3960406278ac2fc2187c0a0b6294e36` |

## Honest reporting

Per pre-reg §6, EC stays PARTIAL across all H9-{A,B,C} rows. FC stays at C-v0.12.0. H9-A confirmation queues an Option-A (N >= 10) follow-up pre-reg before any STABLE bump. H9-B confirms G5-bis MLX-only artefact extends from toy E-SNN to real-LLM tier. H9-C confirms DR-4 inversion universalises at real-LLM scale.