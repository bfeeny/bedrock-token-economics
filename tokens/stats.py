"""Distributions, not means.

Run-to-run token spend on an identical agentic task has been measured varying
by up to 30x, so a mean over three runs carries no information. Everything here
reports percentiles and a paired bootstrap interval on the difference, which is
the honest summary when the underlying distribution is heavy-tailed.
"""
from __future__ import annotations

import random
import statistics


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def describe(values: list[float]) -> dict:
    v = [x for x in values if x == x]  # drop NaN
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": pct(v, 0.50), "p90": pct(v, 0.90),
            "min": min(v), "max": max(v),
            "mean": statistics.fmean(v),
            # The ratio the literature hides by reporting means.
            "max_over_min": round(max(v) / min(v), 2) if min(v) > 0 else None}


def paired_bootstrap(a: list[float], b: list[float], iters: int = 10000,
                     seed: int = 0, statistic=statistics.median) -> dict:
    """Interval on statistic(b) - statistic(a) over paired draws.

    Paired because the arms run the same items: pairing removes item difficulty,
    which is the dominant source of variance and would otherwise swamp the
    effect being measured.
    """
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistic([b[i] for i in idx]) - statistic([a[i] for i in idx]))
    diffs.sort()
    point = statistic(b[:n]) - statistic(a[:n])
    lo, hi = diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]
    return {"n": n, "delta": point, "ci95": [lo, hi],
            # Significant only when the interval excludes zero. The earlier
            # test compared the signs of the bounds, which called a degenerate
            # interval of [0, 0] -- two arms that are identical -- significant.
            "significant": bool(lo > 0 or hi < 0)}
