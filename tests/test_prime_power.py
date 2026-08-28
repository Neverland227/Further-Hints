from __future__ import annotations

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.diagnostics import difference_valuation
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.modular import matvec_mod
from affine_hints.priors import UniformTernaryPrior


def test_unit_pivot_elimination_over_power_of_two() -> None:
    rng = np.random.default_rng(12)
    hint = generate_hint_matrix(rng, n=8, r=3, q=256, hint_class="coded_dual_G_transpose")
    instance = generate_synthetic_lwe(rng, n=8, m=8, q=256, secret_prior=UniformTernaryPrior(), H=hint.H)
    elimination = AffineCosetElimination.build(instance.H, instance.ell, 256, 3, rng=rng)
    residual = tuple(instance.s[i] % 256 for i in elimination.residual_indices)
    assert elimination.all_hints_pass(residual)
    assert hint.metadata["unit_minor_check"] is True


def test_prime_power_difference_image_preserves_p_adic_level() -> None:
    rng = np.random.default_rng(120)
    q = 256
    hint = generate_hint_matrix(rng, n=8, r=3, q=q, hint_class="dense_random")
    secret = UniformTernaryPrior().sample(rng, 8)
    ell = [sum(hint.H[i][j] * int(secret[j]) for j in range(8)) % q for i in range(3)]
    elimination = AffineCosetElimination.build(hint.H, ell, q, 3, rng=rng)
    for nu in (0, 1, 2):
        scale = 2**nu
        difference = [scale] + [scale * 3] * (len(elimination.residual_indices) - 1)
        image = matvec_mod(elimination.C, difference, q)
        assert difference_valuation(difference, q) == nu
        assert all(value % scale == 0 for value in image)
