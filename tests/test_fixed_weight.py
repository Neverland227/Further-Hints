from __future__ import annotations

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.priors import FixedWeightTernaryPrior


def test_fixed_weight_coupling_retains_true_secret_for_every_partial_elimination() -> None:
    rng = np.random.default_rng(13)
    prior = FixedWeightTernaryPrior(2, 2)
    hint = generate_hint_matrix(rng, n=8, r=3, q=17, hint_class="dense_random")
    instance = generate_synthetic_lwe(rng, n=8, m=9, q=17, secret_prior=prior, H=hint.H)
    for r_elim in range(4):
        elimination = AffineCosetElimination.build(instance.H, instance.ell, 17, r_elim, rng=rng)
        residual = tuple(instance.s[i] % 17 for i in elimination.residual_indices)
        reconstructed = tuple(x if x <= 8 else x - 17 for x in elimination.reconstruct(residual))
        assert prior.in_support(reconstructed)

