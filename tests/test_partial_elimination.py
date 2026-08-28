from __future__ import annotations

import itertools

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector
from affine_hints.priors import UniformTernaryPrior


def test_all_representations_give_same_supported_full_set() -> None:
    rng = np.random.default_rng(11)
    prior = UniformTernaryPrior()
    hint = generate_hint_matrix(rng, n=6, r=3, q=17, hint_class="systematic_random")
    instance = generate_synthetic_lwe(rng, n=6, m=6, q=17, secret_prior=prior, H=hint.H)
    sets = []
    for r_elim in range(4):
        elimination = AffineCosetElimination.build(instance.H, instance.ell, 17, r_elim, rng=rng)
        values = set()
        for residual_signed in itertools.product((-1, 0, 1), repeat=len(elimination.residual_indices)):
            residual = tuple(x % 17 for x in residual_signed)
            if elimination.remaining_hints_pass(residual):
                full = tuple(centered_vector(elimination.reconstruct(residual), 17))
                if prior.in_support(full):
                    values.add(full)
        sets.append(values)
    assert all(value == sets[0] for value in sets[1:])

