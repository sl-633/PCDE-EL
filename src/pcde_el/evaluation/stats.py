"""Paired significance tests.

Queries sharing a gold entry are not independent (an entry that is easy
for one mention tends to be easy for its other mentions), so the bootstrap
resamples entries, not queries.
"""

from __future__ import annotations

from math import comb, erfc, sqrt

import numpy as np
import pandas as pd

N_BOOTSTRAP = 5000

def _cluster_indices(clusters) -> list[np.ndarray]:
    """Row indices of each cluster."""
    clusters = np.asarray(clusters)
    n = len(clusters)
    return list(pd.Series(np.arange(n)).groupby(clusters).apply(lambda x: x.to_numpy()).values)

def _resamples(clusters, n_boot: int, seed: int):
    """Yield row indices of n_boot samples, each drawing clusters with replacement."""
    idx_list = _cluster_indices(clusters)
    rng, k = np.random.default_rng(seed), len(idx_list)
    for _ in range(n_boot):
        yield np.concatenate([idx_list[j] for j in rng.integers(0, k, size=k)])

def clustered_bootstrap(a, b, clusters, n_boot: int=N_BOOTSTRAP, seed: int=0) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a 95% percentile CI. a and b are paired per-query
    values (e.g. 0/1 correctness of two systems)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot = np.array([a[i].mean() - b[i].mean() for i in _resamples(clusters, n_boot, seed)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)

def clustered_bootstrap_ratio(num_a, den_a, num_b, den_b, clusters, n_boot: int=N_BOOTSTRAP, seed: int=0) -> tuple[float, float, float]:
    """Same, for a difference of ratios such as Resolution@k (hits / covered)."""
    na, da, nb, db = (np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b))
    point = na.mean() / da.mean() - nb.mean() / db.mean()
    boot = np.array([na[i].mean() / da[i].mean() - nb[i].mean() / db[i].mean() for i in _resamples(clusters, n_boot, seed)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)

def significant(lo: float, hi: float) -> bool:
    """The CI excludes zero."""
    return lo > 0 or hi < 0

def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test; b and c count the queries only one of the
    two systems gets right. Uses the normal approximation above 1000."""
    n = b + c
    if n == 0:
        return float("nan")
    if n > 1000:
        return float(erfc(abs(b - c) / sqrt(n) / sqrt(2)))
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))