"""B1/B3 rank-only elimination and exact-prior postfilter helpers."""

from __future__ import annotations

import math
from typing import Sequence

from affine_hints.coset import AffineCosetElimination
from affine_hints.priors import SecretPrior


def exact_prior_pass(
    elimination: AffineCosetElimination, residual: Sequence[int], prior: SecretPrior
) -> tuple[bool, bool, float]:
    """Return support, remaining-hint, and exact log-prior diagnostics."""

    full = elimination.reconstruct(residual)
    support = prior.in_support(full)
    remaining = elimination.remaining_hints_pass(residual)
    log_prior = prior.log_prob(full) if support and remaining else -math.inf
    return support, remaining, log_prior

