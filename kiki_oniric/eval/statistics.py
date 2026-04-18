"""Statistical eval module — wraps scipy.stats for H1-H4 hypotheses.

Per OSF pre-registration (docs/osf-preregistration-draft.md):
- H1 Welch's t-test (one-sided): treatment improvement vs control
- H2 TOST equivalence (bidirectional): treatment within ±epsilon
- H3 Jonckheere-Terpstra: monotonic trend across ordered groups
- H4 one-sample t-test (upper bound): sample mean below threshold

All tests return a StatTestResult with .reject_h0, .p_value, and
.test_name for uniform downstream handling.

Reference: docs/specs/2026-04-17-dreamofkiki-master-design.md §5.4
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatTestResult:
    """Uniform result type for all H1-H4 hypothesis tests."""

    test_name: str
    p_value: float
    reject_h0: bool
    statistic: float | None = None


def welch_one_sided(
    treatment: list[float],
    control: list[float],
    alpha: float = 0.05,
) -> StatTestResult:
    """H1: Welch's t-test, one-sided (treatment < control).

    H0: mean(treatment) >= mean(control)
    H1: mean(treatment) < mean(control) — i.e. treatment improves
    (lower is better, e.g. forgetting rate).
    """
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    res = stats.ttest_ind(t_arr, c_arr, equal_var=False)
    # Convert two-sided p to one-sided (treatment < control)
    if res.statistic < 0:
        p_one_sided = res.pvalue / 2
    else:
        p_one_sided = 1.0 - res.pvalue / 2
    return StatTestResult(
        test_name="Welch's t-test (one-sided)",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(res.statistic),
    )


def tost_equivalence(
    treatment: list[float],
    control: list[float],
    epsilon: float,
    alpha: float = 0.05,
) -> StatTestResult:
    """H2: Two One-Sided Tests (TOST) for equivalence.

    H0: |mean(treatment) - mean(control)| >= epsilon (not equivalent)
    H1: |mean(treatment) - mean(control)| < epsilon (equivalent)

    Returns reject_h0=True when both one-sided tests pass at alpha.
    """
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    diff_mean = float(t_arr.mean() - c_arr.mean())
    pooled_se = float(
        np.sqrt(t_arr.var(ddof=1) / len(t_arr)
                + c_arr.var(ddof=1) / len(c_arr))
    )
    df = len(t_arr) + len(c_arr) - 2  # rough Welch-Satterthwaite floor
    # Lower bound test: H0_lower: diff <= -epsilon
    t_lower = (diff_mean - (-epsilon)) / pooled_se
    p_lower = 1.0 - stats.t.cdf(t_lower, df)
    # Upper bound test: H0_upper: diff >= epsilon
    t_upper = (diff_mean - epsilon) / pooled_se
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)  # TOST: max p-value
    return StatTestResult(
        test_name="TOST equivalence",
        p_value=float(p_tost),
        reject_h0=bool(p_tost < alpha),
        statistic=diff_mean,
    )


def jonckheere_trend(
    groups: list[list[float]],
    alpha: float = 0.05,
) -> StatTestResult:
    """H3: Jonckheere-Terpstra monotonic trend test.

    H0: no ordered trend across groups
    H1: groups in increasing order (group_i < group_{i+1})

    Implementation: sum of Mann-Whitney U over ordered pairs,
    z-approx for p-value. Standard non-parametric trend test.
    """
    arrs = [np.asarray(g, dtype=float) for g in groups]
    n = sum(len(a) for a in arrs)
    j_stat = 0.0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            # Count pairs (x in arrs[i], y in arrs[j]) with x < y
            count = sum(
                1 for x in arrs[i] for y in arrs[j] if x < y
            )
            ties = sum(
                0.5 for x in arrs[i] for y in arrs[j] if x == y
            )
            j_stat += count + ties
    # Mean and variance of J under H0
    sizes = [len(a) for a in arrs]
    mean_j = (
        n ** 2 - sum(ni ** 2 for ni in sizes)
    ) / 4.0
    var_j = (
        n ** 2 * (2 * n + 3)
        - sum(ni ** 2 * (2 * ni + 3) for ni in sizes)
    ) / 72.0
    z = (j_stat - mean_j) / np.sqrt(var_j) if var_j > 0 else 0.0
    p_one_sided = 1.0 - stats.norm.cdf(z)
    return StatTestResult(
        test_name="Jonckheere-Terpstra trend",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(j_stat),
    )


def one_sample_threshold(
    sample: list[float],
    threshold: float,
    alpha: float = 0.05,
) -> StatTestResult:
    """H4: one-sample t-test against upper bound threshold.

    H0: mean(sample) >= threshold (violates budget)
    H1: mean(sample) < threshold (within budget)

    Returns reject_h0=True when sample mean is significantly below
    threshold.
    """
    arr = np.asarray(sample, dtype=float)
    res = stats.ttest_1samp(arr, popmean=threshold)
    # Two-sided p; convert to one-sided (mean < threshold)
    if res.statistic < 0:
        p_one_sided = res.pvalue / 2
    else:
        p_one_sided = 1.0 - res.pvalue / 2
    return StatTestResult(
        test_name="one-sample t-test (upper bound)",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(res.statistic),
    )
