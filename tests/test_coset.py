from __future__ import annotations

import numpy as np
import pytest

from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import centered_vector, matvec_mod
from affine_hints.priors import UniformTernaryPrior


@pytest.mark.parametrize("q", [17, 32, 256])
@pytest.mark.parametrize("r_elim", [0, 1, 2])
def test_reconstruction_and_lwe_residual_identity(q: int, r_elim: int) -> None:
    rng = np.random.default_rng(1000 + q + r_elim)
    hint = generate_hint_matrix(rng, n=7, r=2, q=q, hint_class="systematic_random")
    instance = generate_synthetic_lwe(
        rng, n=7, m=8, q=q, secret_prior=UniformTernaryPrior(), H=hint.H
    )
    elimination = AffineCosetElimination.build(instance.H, instance.ell, q, r_elim, rng=rng)
    residual = tuple(instance.s[i] % q for i in elimination.residual_indices)
    assert tuple(centered_vector(elimination.reconstruct(residual), q)) == instance.s
    assert elimination.all_hints_pass(residual)
    A_star, b_star = elimination.transform_lwe(instance.A, instance.b)
    reduced_error = tuple(
        centered_vector(((b_star[i] - matvec_mod(A_star, residual, q)[i]) % q for i in range(instance.m)), q)
    )
    assert reduced_error == instance.e

