"""Synthetic-only LWE instance generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .modular import matvec_mod
from .priors import SecretPrior


@dataclass(frozen=True)
class LWEInstance:
    """A generated LWE instance using row-oriented samples ``A: m x n``."""

    A: tuple[tuple[int, ...], ...]
    b: tuple[int, ...]
    s: tuple[int, ...]
    e: tuple[int, ...]
    q: int
    H: tuple[tuple[int, ...], ...]
    ell: tuple[int, ...]
    instance_id: str

    @property
    def n(self) -> int:
        return len(self.s)

    @property
    def m(self) -> int:
        return len(self.b)


def generate_synthetic_lwe(
    rng: np.random.Generator,
    *,
    n: int,
    m: int,
    q: int,
    secret_prior: SecretPrior,
    H: Sequence[Sequence[int]],
    error_prior: SecretPrior | None = None,
    instance_id: str = "synthetic",
) -> LWEInstance:
    """Generate ``b = A s + e (mod q)`` and exact hints ``H s = ell``.

    This is deliberately the only public construction path used by experiments;
    there is no parser for real or third-party LWE samples.
    """

    if n <= 0 or m <= 0 or q < 2:
        raise ValueError("n, m, and q must be positive with q >= 2")
    matrix = rng.integers(0, q, size=(m, n), dtype=np.int64).astype(object).tolist()
    secret = [int(value) for value in secret_prior.sample(rng, n)]
    if error_prior is None:
        errors = [int(value) for value in rng.choice(np.asarray([-1, 0, 1]), size=m)]
    else:
        errors = [int(value) for value in error_prior.sample(rng, m)]
    linear = matvec_mod(matrix, secret, q)
    b = [(value + error) % q for value, error in zip(linear, errors)]
    hints = [[int(value) % q for value in row] for row in H]
    if hints and any(len(row) != n for row in hints):
        raise ValueError("hint rows must have n columns")
    ell = matvec_mod(hints, secret, q)
    return LWEInstance(
        A=tuple(tuple(int(value) for value in row) for row in matrix),
        b=tuple(b),
        s=tuple(secret),
        e=tuple(errors),
        q=q,
        H=tuple(tuple(row) for row in hints),
        ell=tuple(ell),
        instance_id=instance_id,
    )

