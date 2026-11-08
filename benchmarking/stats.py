"""Small, dependency-free statistics for accuracy reporting."""

from __future__ import annotations

import math
import random
import statistics


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for one run's accuracy."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def bootstrap_mean_ci(per_problem: list[float], iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap over problems (per_problem = accuracy averaged over seeds)."""
    n = len(per_problem)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(per_problem[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def paired_bootstrap_diff(a: list[float], b: list[float], iters: int = 10_000, seed: int = 0) -> dict:
    """Mean of (b - a) over the same problems, with a 95% CI and a two-sided bootstrap p-value."""
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same problems in both arms")
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return {"diff": 0.0, "ci": (0.0, 0.0), "p": 1.0}
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    observed = sum(diffs) / n
    p = 2 * min(sum(d <= 0 for d in boot), sum(d >= 0 for d in boot)) / iters
    return {"diff": observed, "ci": (boot[int(0.025 * iters)], boot[int(0.975 * iters) - 1]), "p": min(1.0, p)}


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
