from __future__ import annotations

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.modular import centered_vector


def test_rank_only_pivot_support_rate_matches_reference_at_small_q() -> None:
    rng = np.random.default_rng(14)
    q, n, r = 17, 8, 2
    hint = generate_hint_matrix(rng, n=n, r=r, q=q, hint_class="dense_random")
    secret = rng.integers(-1, 2, size=n)
    ell = [sum(hint.H[i][j] * int(secret[j]) for j in range(n)) % q for i in range(r)]
    elimination = AffineCosetElimination.build(hint.H, ell, q, r, rng=rng)
    trials = 40000
    survivors = 0
    for _ in range(trials):
        residual = tuple(int(value) % q for value in rng.integers(-1, 2, size=len(elimination.residual_indices)))
        full = centered_vector(elimination.reconstruct(residual), q)
        survivors += all(full[i] in (-1, 0, 1) for i in elimination.pivot_indices)
    observed = survivors / trials
    reference = (3 / q) ** r
    assert abs(observed - reference) < 0.008

