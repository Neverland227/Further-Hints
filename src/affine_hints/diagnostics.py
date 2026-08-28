"""Statistical diagnostics with effect sizes and bounded uncertainty estimates."""

from __future__ import annotations

import collections
import hashlib
import math
from typing import Sequence

import numpy as np

from .modular import first_prime_divisor, matvec_mod


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval; zero events retain a nonzero upper bound."""

    if trials <= 0:
        return math.nan, math.nan
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def jeffreys_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Jeffreys beta-posterior equal-tailed interval."""

    if trials <= 0:
        return math.nan, math.nan
    from scipy.stats import beta

    alpha = (1.0 - confidence) / 2.0
    return (
        float(beta.ppf(alpha, successes + 0.5, trials - successes + 0.5)),
        float(beta.ppf(1.0 - alpha, successes + 0.5, trials - successes + 0.5)),
    )


def empirical_character_bias(values: Sequence[int], q: int) -> float:
    """Magnitude of the first nontrivial empirical additive character."""

    if not values:
        return math.nan
    angles = 2j * math.pi * np.asarray(values, dtype=float) / q
    return float(abs(np.exp(angles).mean()))


def uniformity_diagnostics(
    samples: Sequence[Sequence[int]],
    q: int,
    rng: np.random.Generator,
    *,
    projections: int = 16,
    include_histograms: bool = False,
) -> dict[str, float | int | list[float] | None]:
    """Large-modulus-safe coordinate/projection diagnostics.

    A full ``q^r`` chi-square is intentionally never constructed.
    """

    if not samples:
        return {"sample_count": 0}
    array = np.asarray(samples, dtype=np.int64) % q
    n_samples, dimension = array.shape
    if dimension == 0:
        return {
            "sample_count": n_samples,
            "dimension": 0,
            "applicable": False,
            "reason": "r_elim=0 has no pivot image coordinates",
        }
    coordinate_tv: list[float] = []
    coordinate_chi2: list[float] = []
    coordinate_histograms: list[list[int]] = []
    expected = n_samples / q
    for column in range(dimension):
        counts = np.bincount(array[:, column], minlength=q)
        if include_histograms:
            coordinate_histograms.append([int(value) for value in counts])
        coordinate_tv.append(float(0.5 * np.abs(counts / n_samples - 1.0 / q).sum()))
        if expected >= 5.0:
            coordinate_chi2.append(float(((counts - expected) ** 2 / expected).sum()))
    biases: list[float] = []
    for _ in range(projections):
        vector = rng.integers(0, q, size=dimension, dtype=np.int64)
        projected = (array @ vector) % q
        biases.append(empirical_character_bias(projected.tolist(), q))
    tuples = [tuple(int(value) for value in row) for row in array]
    counts = collections.Counter(tuples)
    collisions = sum(value * (value - 1) // 2 for value in counts.values())
    possible_pairs = n_samples * (n_samples - 1) // 2
    hashes = {hashlib.blake2b(repr(value).encode("ascii"), digest_size=8).digest() for value in tuples}
    return {
        "sample_count": n_samples,
        "dimension": dimension,
        "mean_coordinate_tv": float(np.mean(coordinate_tv)),
        "max_coordinate_tv": float(np.max(coordinate_tv)),
        "coordinate_chi2_mean": float(np.mean(coordinate_chi2)) if coordinate_chi2 else None,
        "coordinate_chi2": coordinate_chi2,
        "coordinate_histograms": coordinate_histograms if include_histograms else None,
        "chi2_applicable": bool(coordinate_chi2),
        "character_bias_max": float(np.max(biases)),
        "character_bias_q50": float(np.quantile(biases, 0.5)),
        "character_bias_q95": float(np.quantile(biases, 0.95)),
        "collision_rate": collisions / possible_pairs if possible_pairs else 0.0,
        "distinct_count": len(counts),
        "hash_distinct_count": len(hashes),
    }


def projective_class(vector: Sequence[int], q: int) -> tuple[int, ...] | None:
    """Normalize a nonzero vector over prime ``q`` to its projective class."""

    values = [int(value) % q for value in vector]
    first = next((value for value in values if value), None)
    if first is None:
        return None
    inverse = pow(first, -1, q)
    return tuple((inverse * value) % q for value in values)


def difference_valuation(vector: Sequence[int], q: int) -> int:
    """Minimum p-adic valuation among coordinates for prime-power diagnostics."""

    p = first_prime_divisor(q)
    valuations: list[int] = []
    for raw in vector:
        value = int(raw) % q
        if value == 0:
            continue
        valuation = 0
        while value % p == 0:
            valuation += 1
            value //= p
        valuations.append(valuation)
    if not valuations:
        exponent = 0
        value = q
        while value % p == 0:
            exponent += 1
            value //= p
        return exponent
    return min(valuations)


def candidate_correlation_diagnostics(
    differences: Sequence[Sequence[int]], passes: Sequence[bool], q: int
) -> dict[str, float | int | None]:
    """Describe projective clustering and survivor overdispersion for prime q."""

    if len(differences) != len(passes):
        raise ValueError("differences and passes have different lengths")
    classes = [projective_class(vector, q) for vector in differences]
    counts = collections.Counter(value for value in classes if value is not None)
    total_pairs = len(classes) * (len(classes) - 1) // 2
    collinear_pairs = sum(value * (value - 1) // 2 for value in counts.values())
    alpha = sum(bool(value) for value in passes) / len(passes) if passes else math.nan
    independent_variance = len(passes) * alpha * (1.0 - alpha) if passes else math.nan
    class_survivors = collections.defaultdict(int)
    for key in counts:
        class_survivors[key] = 0
    for key, passed in zip(classes, passes):
        if key is not None and passed:
            class_survivors[key] += 1
    observed_proxy = float(np.var(list(class_survivors.values()), ddof=1)) if len(class_survivors) > 1 else 0.0
    total_passes = int(sum(bool(value) for value in passes))
    joint_collinear_passes = sum(value * (value - 1) // 2 for value in class_survivors.values())
    all_joint_passes = total_passes * (total_passes - 1) // 2
    noncollinear_pairs = total_pairs - collinear_pairs
    joint_noncollinear_passes = all_joint_passes - joint_collinear_passes
    return {
        "number_projective_classes": len(counts),
        "largest_class_size": max(counts.values(), default=0),
        "fraction_collinear_pairs": collinear_pairs / total_pairs if total_pairs else 0.0,
        "survivor_count": total_passes,
        "alpha": alpha,
        "variance_under_independence": independent_variance,
        "class_survivor_variance_proxy": observed_proxy,
        "collinear_joint_pass_rate": joint_collinear_passes / collinear_pairs if collinear_pairs else None,
        "noncollinear_joint_pass_rate": joint_noncollinear_passes / noncollinear_pairs if noncollinear_pairs else None,
        "independence_joint_pass_reference": alpha * alpha if passes else None,
        "difference_valuation_min": min((difference_valuation(vector, q) for vector in differences), default=None),
    }
