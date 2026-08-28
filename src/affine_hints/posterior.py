"""Exact posterior weights before and after affine elimination."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .coset import AffineCosetElimination
from .modular import centered_vector, matvec_mod
from .priors import SecretPrior


def lwe_error(A: Sequence[Sequence[int]], b: Sequence[int], secret: Sequence[int], q: int) -> tuple[int, ...]:
    """Return the centered error candidate ``b - A s (mod q)``."""

    predicted = matvec_mod(A, secret, q)
    return tuple(centered_vector(((int(x) - y) % q for x, y in zip(b, predicted)), q))


def original_log_posterior(
    *,
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    H: Sequence[Sequence[int]],
    ell: Sequence[int],
    q: int,
    secret: Sequence[int],
    secret_prior: SecretPrior,
    error_prior: SecretPrior,
) -> float:
    """Evaluate the unnormalized exact posterior on the original variables."""

    values = [int(value) for value in secret]
    if tuple(matvec_mod(H, values, q)) != tuple(int(value) % q for value in ell):
        return -math.inf
    secret_log = secret_prior.log_prob(values)
    if not math.isfinite(secret_log):
        return -math.inf
    return secret_log + error_prior.log_prob(lwe_error(A, b, values, q))


def reduced_log_posterior(
    *,
    A_star: Sequence[Sequence[int]],
    b_star: Sequence[int],
    elimination: AffineCosetElimination,
    residual_secret: Sequence[int],
    secret_prior: SecretPrior,
    error_prior: SecretPrior,
) -> float:
    """Evaluate the exact reduced posterior, retaining all prior coupling."""

    if not elimination.remaining_hints_pass(residual_secret):
        return -math.inf
    full_mod_q = elimination.reconstruct(residual_secret)
    # Priors are distributions on small signed integers, while the affine map
    # necessarily returns canonical residues modulo q.
    full = tuple(centered_vector(full_mod_q, elimination.q))
    secret_log = secret_prior.log_prob(full)
    if not math.isfinite(secret_log):
        return -math.inf
    return secret_log + error_prior.log_prob(lwe_error(A_star, b_star, residual_secret, elimination.q))


def normalize_log_weights(items: Iterable[tuple[tuple[int, ...], float]]) -> dict[tuple[int, ...], float]:
    """Normalize finite log weights using a stable log-sum-exp calculation."""

    values = [(key, weight) for key, weight in items if math.isfinite(weight)]
    if not values:
        return {}
    maximum = max(weight for _, weight in values)
    total = sum(math.exp(weight - maximum) for _, weight in values)
    return {key: math.exp(weight - maximum) / total for key, weight in values}
