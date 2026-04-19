# Cycle-2 multi-substrate ablation (synthetic substitute)

**(synthetic data — no real cohort/HW)**  Pipeline-
validation only ; substrate axis is an identity label, the
Python predictor is shared across substrate rows. See
`scripts/ablation_cycle2.py` docstring for rationale.

- harness_version : `C-v0.6.0+PARTIAL`
- cycle2_batch_id : `d25d8888c840424a5ffd03c8432a3a16`
- ablation_runner_run_id : `3cec0422f86aeeddb72f6444423ed270`
- benchmark_hash : `synthetic:c8a0712000b641...`
- seeds : `[42, 123, 7]`
- substrates : `['mlx_kiki_oniric', 'esnn_thalamocortical']`

## Per-substrate accuracy (synthetic substitute)

| substrate | profile | acc (per seed) |
|-----------|---------|----------------|
| mlx_kiki_oniric | baseline | [0.5, 0.5, 0.5] |
| mlx_kiki_oniric | P_min | [0.7, 0.7, 0.7] |
| mlx_kiki_oniric | P_equ | [0.85, 0.85, 0.85] |
| esnn_thalamocortical | baseline | [0.5, 0.5, 0.5] |
| esnn_thalamocortical | P_min | [0.7, 0.7, 0.7] |
| esnn_thalamocortical | P_equ | [0.85, 0.85, 0.85] |

## H1-H4 verdicts (synthetic substitute)

| substrate | hypothesis | p-value | reject H0 |
|-----------|------------|---------|-----------|
| mlx_kiki_oniric | H1_forgetting | 0.0000 | PASS |
| mlx_kiki_oniric | H2_equivalence_self | 0.0000 | PASS |
| mlx_kiki_oniric | H3_monotonic | 0.0248 | fail |
| mlx_kiki_oniric | H4_energy_budget | 0.0101 | PASS |
| esnn_thalamocortical | H1_forgetting | 0.0000 | PASS |
| esnn_thalamocortical | H2_equivalence_self | 0.0000 | PASS |
| esnn_thalamocortical | H3_monotonic | 0.0248 | fail |
| esnn_thalamocortical | H4_energy_budget | 0.0101 | PASS |

## Cross-substrate consistency (synthetic substitute)

| hypothesis | mlx_kiki_oniric | esnn_thalamocortical | agree |
|------------|-----------------|----------------------|-------|
| H1_forgetting | True | True | YES |
| H2_equivalence_self | True | True | YES |
| H3_monotonic | False | False | YES |
| H4_energy_budget | True | True | YES |

**Fully consistent across substrates :** YES

> All numbers in this dump are produced by mock predictors
> shared across substrate labels. They validate the
> *cross-substrate replication pipeline*, not consolidation
> efficacy on real linguistic data or real spike-rate
> dynamics. See cycle-2 spec for the real-wiring deferral.
