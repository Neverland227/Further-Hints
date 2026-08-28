"""Synthetic candidate distributions for selectivity and falsification tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.posterior import reduced_log_posterior
from affine_hints.priors import SecretPrior, SupportTooLarge

from .base import CandidateList


@dataclass
class SyntheticUniformSource:
    """Uniform residual candidates over ``Z_q`` (rank-only reference source)."""

    elimination: AffineCosetElimination | None = None
    true_residual: tuple[int, ...] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        self.elimination = baseline_config["elimination"]
        self.true_residual = tuple(instance.s[i] % instance.q for i in self.elimination.residual_indices)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        if self.elimination is None:
            raise RuntimeError("prepare must be called first")
        q = self.elimination.q
        dimension = len(self.elimination.residual_indices)
        rows = rng.integers(0, q, size=(budget, dimension), dtype=np.int64)
        candidates = [tuple(int(value) for value in row) for row in rows]
        true_index: int | None = None
        if self.true_residual is not None and budget:
            true_index = int(rng.integers(0, budget))
            candidates[true_index] = self.true_residual
        return CandidateList(tuple(candidates), tuple(0.0 for _ in candidates), true_index, "SyntheticUniformSource")


@dataclass
class PriorSupportedSyntheticSource:
    """Residual coordinates obtained from fresh full-prior samples."""

    prior: SecretPrior
    elimination: AffineCosetElimination | None = None
    true_residual: tuple[int, ...] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        self.elimination = baseline_config["elimination"]
        self.true_residual = tuple(instance.s[i] % instance.q for i in self.elimination.residual_indices)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        if self.elimination is None:
            raise RuntimeError("prepare must be called first")
        n = len(self.elimination.pivot_indices) + len(self.elimination.residual_indices)
        candidates: list[tuple[int, ...]] = []
        for _ in range(budget):
            full = self.prior.sample(rng, n)
            candidates.append(tuple(int(full[i]) % self.elimination.q for i in self.elimination.residual_indices))
        true_index: int | None = None
        if self.true_residual is not None and budget:
            true_index = int(rng.integers(0, budget))
            candidates[true_index] = self.true_residual
        return CandidateList(tuple(candidates), tuple(0.0 for _ in candidates), true_index, "PriorSupportedSyntheticSource")


@dataclass
class PosteriorWeightedSyntheticSource:
    """Exact posterior sampling for tiny residual supports only."""

    secret_prior: SecretPrior
    error_prior: SecretPrior
    max_exact_states: int
    elimination: AffineCosetElimination | None = None
    instance: Any = None
    A_star: tuple[tuple[int, ...], ...] | None = None
    b_star: tuple[int, ...] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        self.instance = instance
        self.elimination = baseline_config["elimination"]
        self.A_star, self.b_star = self.elimination.transform_lwe(instance.A, instance.b)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        if self.elimination is None or self.A_star is None or self.b_star is None:
            raise RuntimeError("prepare must be called first")
        dimension = len(self.elimination.residual_indices)
        # Enumerating the full prior and projecting is exact and retains
        # fixed-weight coupling. Deduplicate residual states before weighting.
        residual_states: set[tuple[int, ...]] = set()
        n = dimension + len(self.elimination.pivot_indices)
        for full in self.secret_prior.enumerate_support(n, self.max_exact_states):
            residual_states.add(tuple(int(full[i]) % self.elimination.q for i in self.elimination.residual_indices))
        states: list[tuple[int, ...]] = []
        weights: list[float] = []
        for residual in sorted(residual_states):
            value = reduced_log_posterior(
                A_star=self.A_star,
                b_star=self.b_star,
                elimination=self.elimination,
                residual_secret=residual,
                secret_prior=self.secret_prior,
                error_prior=self.error_prior,
            )
            if math.isfinite(value):
                states.append(residual)
                weights.append(value)
        if not states:
            return CandidateList((), (), None, "PosteriorWeightedSyntheticSource", {"empty_posterior": True})
        maximum = max(weights)
        probabilities = np.exp(np.asarray(weights) - maximum)
        probabilities /= probabilities.sum()
        selected = rng.choice(len(states), size=budget, replace=True, p=probabilities)
        candidates = tuple(states[int(index)] for index in selected)
        scores = tuple(float(weights[int(index)]) for index in selected)
        truth = tuple(self.instance.s[i] % self.instance.q for i in self.elimination.residual_indices)
        true_index = next((i for i, value in enumerate(candidates) if value == truth), None)
        return CandidateList(candidates, scores, true_index, "PosteriorWeightedSyntheticSource")


@dataclass
class NormShellProxySource:
    """Matched-second-moment discrete-Gaussian/norm-shell project proxy."""

    second_moment: float
    shell_tolerance: float = 0.20
    elimination: AffineCosetElimination | None = None
    true_residual: tuple[int, ...] | None = None

    def prepare(self, instance: Any, baseline_config: dict[str, Any]) -> None:
        self.elimination = baseline_config["elimination"]
        self.true_residual = tuple(instance.s[i] % instance.q for i in self.elimination.residual_indices)

    def generate(self, budget: int, rng: np.random.Generator) -> CandidateList:
        if self.elimination is None:
            raise RuntimeError("prepare must be called first")
        dimension = len(self.elimination.residual_indices)
        target = dimension * self.second_moment
        accepted: list[tuple[int, ...]] = []
        attempts = 0
        maximum_attempts = max(100, 50 * budget)
        sigma = math.sqrt(self.second_moment)
        while len(accepted) < budget and attempts < maximum_attempts:
            attempts += 1
            vector = np.rint(rng.normal(0.0, sigma, size=dimension)).astype(np.int64)
            norm2 = float(vector @ vector)
            if abs(norm2 - target) <= max(1.0, self.shell_tolerance * target):
                accepted.append(tuple(int(value) % self.elimination.q for value in vector))
        true_index: int | None = None
        if self.true_residual is not None and accepted:
            true_index = int(rng.integers(0, len(accepted)))
            accepted[true_index] = self.true_residual
        return CandidateList(
            tuple(accepted),
            tuple(-sum(min(v, self.elimination.q - v) ** 2 for v in row) for row in accepted),
            true_index,
            "NormShellProxySource",
            {"label": "PROJECT_PROXY / B2_PROXY", "attempts": attempts, "requested_budget": budget},
        )

