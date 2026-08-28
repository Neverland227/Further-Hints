"""Continuous and integer finite-difference BKZ sensitivity diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


def root_hermite_factor(beta: float) -> float:
    """Chen--Nguyen-style asymptotic root-Hermite-factor proxy.

    This is a continuous heuristic used only for sensitivity, not a BKZ theorem.
    """

    if beta < 3:
        raise ValueError("beta must be at least 3")
    value = ((math.pi * beta) ** (1.0 / beta) * beta / (2.0 * math.pi * math.e))
    return value ** (1.0 / (2.0 * (beta - 1.0)))


def log2_radius_proxy(beta: float, effective_dimension: int) -> float:
    """Return ``(d_eff-1) log2(delta_beta)``."""

    return (effective_dimension - 1.0) * math.log2(root_hermite_factor(beta))


def solve_continuous_beta(
    *, baseline_beta: float, effective_dimension: int, delta_log2_radius: float, lower: float = 40.0
) -> float:
    """Find the smaller beta whose looser radius matches the allowed increase."""

    if baseline_beta < lower:
        raise ValueError("continuous BKZ sensitivity proxy is restricted to beta >= 40")
    target = log2_radius_proxy(baseline_beta, effective_dimension) + delta_log2_radius
    lo, hi = lower, baseline_beta
    if log2_radius_proxy(lo, effective_dimension) < target:
        return lo
    for _ in range(80):
        middle = (lo + hi) / 2.0
        # delta decreases with beta in the intended range.
        if log2_radius_proxy(middle, effective_dimension) > target:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class BetaSensitivity:
    """One sensitivity result under one effective-dimension convention."""

    model: str
    effective_dimension: int
    candidate_capacity_bits: float
    delta_log2_radius: float
    radius_ratio: float
    baseline_beta: int
    continuous_new_beta: float
    predicted_delta_beta_continuous: float
    integer_new_beta: int
    predicted_delta_beta_integer: int
    label: str = "HEURISTIC ESTIMATOR"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def beta_sensitivity_band(
    *,
    capacity_bits: float,
    baseline_beta: int,
    dimensions: Iterable[tuple[str, int]],
) -> list[BetaSensitivity]:
    """Scan lattice, beta, and sieve-proxy effective dimensions."""

    results: list[BetaSensitivity] = []
    for model, effective_dimension in dimensions:
        if effective_dimension <= 1:
            continue
        delta_radius = capacity_bits / effective_dimension
        new_beta = solve_continuous_beta(
            baseline_beta=float(baseline_beta),
            effective_dimension=effective_dimension,
            delta_log2_radius=delta_radius,
        )
        integer_new = max(40, math.ceil(new_beta))
        results.append(
            BetaSensitivity(
                model=model,
                effective_dimension=effective_dimension,
                candidate_capacity_bits=capacity_bits,
                delta_log2_radius=delta_radius,
                radius_ratio=2.0**delta_radius,
                baseline_beta=baseline_beta,
                continuous_new_beta=new_beta,
                predicted_delta_beta_continuous=baseline_beta - new_beta,
                integer_new_beta=integer_new,
                predicted_delta_beta_integer=baseline_beta - integer_new,
            )
        )
    return results
