"""Shared statistics: None-skipping mean, percentile bootstraps, and multiplicity corrections.

Every registered interval in report.py and every site interval in site/derive.py comes from here,
so one seed and one percentile rule hold everywhere.
"""

import random


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
def bootstrap_interval(values, trials=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of `values` (resampling episodes). (lo, hi); (None,None) empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(trials))
    return (means[int(alpha / 2 * trials)], means[int((1 - alpha / 2) * trials)])
def clustered_binary_rate_interval(by_episode, trials=2000, seed=23, alpha=0.05):
    """Event-weighted binary rate with an episode-clustered percentile bootstrap interval."""
    groups = [list(values) for values in by_episode.values() if values]
    if not groups:
        return None, (None, None)
    estimate = sum(sum(group) for group in groups) / sum(len(group) for group in groups)
    if len(groups) == 1:
        return estimate, (estimate, estimate)
    rng = random.Random(seed)
    n = len(groups)
    samples = []
    for _ in range(trials):
        selected = [groups[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(sum(group) for group in selected) / sum(len(group) for group in selected))
    samples.sort()
    return estimate, (
        samples[int(alpha / 2 * trials)], samples[int((1 - alpha / 2) * trials)],
    )
def bootstrap_statistic_interval(items, stat, trials=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a statistic over `items` (resampling units). (None,None) if degenerate."""
    rng = random.Random(seed)
    n = len(items)
    vals = sorted(s for s in (stat([items[rng.randrange(n)] for _ in range(n)]) for _ in range(trials))
                  if s is not None)
    if len(vals) < trials * 0.5:  # too many resamples were degenerate (ties) to trust the CI
        return (None, None)
    return (vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals))])
def _resample_mean(rng, xs):
    n = len(xs)
    return sum(xs[rng.randrange(n)] for _ in range(n)) / n
def contrast_bootstrap(pos, neg=None, trials=2000, seed=0, alpha=0.05):
    """Episode-bootstrap (estimate, (lo, hi) CI, two-sided p vs 0) for a mean (neg=None) or a
    two-sample mean difference pos - neg. Both sides resample independently. The p-value is the
    percentile mass on the far side of zero, doubled — so it agrees with the CI by construction
    (the CI excludes 0 iff p < alpha). Seed-stable; (None, (None, None), None) if a side is empty."""
    pos = [x for x in pos if x is not None]
    neg = None if neg is None else [x for x in neg if x is not None]
    if not pos or (neg is not None and not neg):
        return None, (None, None), None
    est = mean(pos) if neg is None else mean(pos) - mean(neg)
    rng = random.Random(seed)
    reps = []
    for _ in range(trials):
        m = _resample_mean(rng, pos)
        reps.append(m if neg is None else m - _resample_mean(rng, neg))
    reps.sort()
    lo, hi = reps[int(alpha / 2 * trials)], reps[int((1 - alpha / 2) * trials)]
    le = sum(1 for r in reps if r <= 0) / trials
    ge = sum(1 for r in reps if r >= 0) / trials
    return est, (lo, hi), min(1.0, 2 * min(le, ge))
def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up over raw p-values -> [(adjusted_p, reject)] in input order.

    Adjusted p (the BH q-value) is the step-up-enforced minimum of p(k) * m / k, capped at 1;
    H(i) is rejected iff its adjusted p <= alpha. Empty input -> []."""
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        adj[i] = min(1.0, prev)
    return [(adj[i], adj[i] <= alpha) for i in range(m)]
def holm_bonferroni(pvals, alpha=0.05):
    """Holm family-wise adjusted p-values in input order.

    The family size is exactly ``len(pvals)``.  Callers that preregister a fixed family must pass
    ``1.0`` for an unavailable member rather than silently shrinking the denominator.
    """
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda index: pvals[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvals[index])
        adjusted[index] = min(1.0, running)
    return [(value, value <= alpha) for value in adjusted]
