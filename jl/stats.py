"""Small-sample statistics for the blind test. Standard library only.

A win rate from a dozen tests is an anecdote. These helpers put an
interval around it so the Results claim says what the data supports.
"""

import math
import random


def wilson(wins, n, z=1.96):
    """95% Wilson score interval for a proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test(wins, losses):
    """Exact two-sided sign test p-value. Ties are dropped beforehand."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap_mean(values, clusters, n_boot=5000, seed=11):
    """95% CI for the mean of `values`, resampling whole clusters.

    Tests written from the same decision are not independent, so we
    resample decisions rather than individual tests. Returns
    (mean, lo, hi), or (mean, None, None) with fewer than 2 clusters.
    """
    if not values:
        return (0.0, None, None)
    mean = sum(values) / len(values)
    groups = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys = list(groups)
    if len(keys) < 2:
        return (mean, None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = []
        for _ in keys:
            sample.extend(groups[rng.choice(keys)])
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (mean, lo, hi)


def verdict(wins, losses, min_tests, alpha=0.05):
    """Plain-language call on a win/loss record, ties excluded.

    WIN or LOSS needs both enough decided tests and an exact sign test
    below alpha. A 4-0 record is a lean, not a result.
    """
    n = wins + losses
    if n == 0:
        return "NO DECIDED TESTS"
    lean = "win" if wins > losses else "loss" if losses > wins else "even"
    if n < min_tests:
        return f"TOO FEW TESTS ({n} decided, need {min_tests}+) - leaning {lean}"
    if sign_test(wins, losses) < alpha:
        return "WIN" if wins > losses else "LOSS"
    return f"INCONCLUSIVE - leaning {lean}"
