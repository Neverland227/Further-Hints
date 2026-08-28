"""Cluster-level uncertainty calculations for paired experiment summaries."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def cluster_bootstrap_interval(
    values: Sequence[float],
    rng: np.random.Generator,
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
) -> tuple[float, float]:
    """Bootstrap complete instance/list clusters, never individual candidates."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    estimates = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sample = array[rng.integers(0, len(array), size=len(array))]
        estimates[i] = statistic(sample)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def paired_cluster_bootstrap_interval(
    baseline: Sequence[float],
    exact: Sequence[float],
    rng: np.random.Generator,
    *,
    transform: Callable[[np.ndarray, np.ndarray], float],
    replicates: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap paired instance indices under an arbitrary summary transform."""

    left = np.asarray(baseline, dtype=float)
    right = np.asarray(exact, dtype=float)
    if len(left) != len(right) or not len(left):
        return math.nan, math.nan
    estimates = np.empty(replicates, dtype=float)
    for i in range(replicates):
        indices = rng.integers(0, len(left), size=len(left))
        estimates[i] = transform(left[indices], right[indices])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))

