"""A/B test statistics: two-proportion z-test, CI, sample size, peeking.

Interview goal: be the person on the team who can read an experiment
report. Cover:
  - point estimate of lift and its standard error
  - two-proportion z-test (p-value)
  - 95% confidence interval for the difference
  - required sample size for a given minimum detectable effect (MDE)
  - the danger of "peeking" / repeated testing
"""

from __future__ import annotations

import math
import random


def two_proportion_z_test(c_a, n_a, c_b, n_b):
    p_a = c_a / n_a
    p_b = c_b / n_b
    p_pool = (c_a + c_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se if se > 0 else 0.0
    # Normal CDF via erf — no scipy dependency.
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p_a, p_b, z, p_value


def diff_confidence_interval(c_a, n_a, c_b, n_b, alpha=0.05):
    p_a = c_a / n_a
    p_b = c_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z = 1.959963984540054  # 97.5% normal quantile
    diff = p_b - p_a
    return diff - z * se, diff + z * se


def required_sample_size(p_baseline, mde, alpha=0.05, power=0.8):
    """Per-group sample size for detecting an absolute lift `mde`."""
    z_alpha = 1.959963984540054
    z_beta = 0.8416212335729143  # 80% power
    p_avg = p_baseline + mde / 2
    num = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
           + z_beta * math.sqrt(p_baseline * (1 - p_baseline)
                                + (p_baseline + mde) * (1 - p_baseline - mde))) ** 2
    return math.ceil(num / (mde ** 2))


def peeking_simulation(n_per_group=2000, n_peeks=20, n_sims=2000, seed=0):
    """Two arms with the SAME true rate. How often does naive peeking lie?"""
    rng = random.Random(seed)
    p = 0.10
    false_positives = 0
    peek_at = [int(50 + i * (n_per_group - 50) / max(1, n_peeks - 1)) for i in range(n_peeks)]
    for _ in range(n_sims):
        a = [1 if rng.random() < p else 0 for _ in range(n_per_group)]
        b = [1 if rng.random() < p else 0 for _ in range(n_per_group)]
        for k in peek_at:
            _, _, _, pv = two_proportion_z_test(sum(a[:k]), k, sum(b[:k]), k)
            if pv < 0.05:
                false_positives += 1
                break
    return false_positives / n_sims


def main():
    print("=== Worked example ===")
    n_a, c_a = 10_000, 510   # 5.10% baseline
    n_b, c_b = 10_000, 575   # 5.75% variant
    p_a, p_b, z, pv = two_proportion_z_test(c_a, n_a, c_b, n_b)
    lo, hi = diff_confidence_interval(c_a, n_a, c_b, n_b)
    print(f"  Control  rate : {p_a:.4f}")
    print(f"  Variant  rate : {p_b:.4f}")
    print(f"  Lift          : {p_b - p_a:+.4f}  ({(p_b/p_a - 1)*100:+.2f}% relative)")
    print(f"  z-statistic   : {z:.3f}")
    print(f"  p-value       : {pv:.4f}")
    print(f"  95% CI on diff: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Significant?  : {'YES' if pv < 0.05 else 'NO'}")

    print("\n=== Sample-size planning ===")
    for mde in [0.01, 0.005, 0.002]:
        n = required_sample_size(p_baseline=0.05, mde=mde)
        print(f"  baseline 5%, MDE={mde:.3f} -> need {n:>7,} per arm")

    print("\n=== Peeking inflates false positives ===")
    fpr = peeking_simulation(n_per_group=1500, n_peeks=10, n_sims=500)
    print(f"  10 peeks at alpha=0.05 -> empirical false-positive rate ~= {fpr:.2f}")
    print("  (true rate should be 0.05 -- that's why you need fixed N or")
    print("   sequential corrections like Bonferroni / always-valid CIs.)")


if __name__ == "__main__":
    main()
