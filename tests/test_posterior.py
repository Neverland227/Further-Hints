from __future__ import annotations

import math

import numpy as np

from affine_hints.coset import AffineCosetElimination
from affine_hints.hints import generate_hint_matrix
from affine_hints.lwe import generate_synthetic_lwe
from affine_hints.posterior import original_log_posterior, reduced_log_posterior
from affine_hints.priors import CBDPrior, UniformTernaryPrior


def test_original_and_reduced_posterior_are_equal() -> None:
    rng = np.random.default_rng(10)
    secret_prior = CBDPrior(2)
    error_prior = UniformTernaryPrior()
    hint = generate_hint_matrix(rng, n=5, r=2, q=17, hint_class="dense_random")
    instance = generate_synthetic_lwe(
        rng,
        n=5,
        m=6,
        q=17,
        secret_prior=secret_prior,
        error_prior=error_prior,
        H=hint.H,
    )
    elimination = AffineCosetElimination.build(instance.H, instance.ell, 17, 2, rng=rng)
    A_star, b_star = elimination.transform_lwe(instance.A, instance.b)
    residual = tuple(instance.s[i] % 17 for i in elimination.residual_indices)
    original = original_log_posterior(
        A=instance.A,
        b=instance.b,
        H=instance.H,
        ell=instance.ell,
        q=17,
        secret=instance.s,
        secret_prior=secret_prior,
        error_prior=error_prior,
    )
    reduced = reduced_log_posterior(
        A_star=A_star,
        b_star=b_star,
        elimination=elimination,
        residual_secret=residual,
        secret_prior=secret_prior,
        error_prior=error_prior,
    )
    assert math.isclose(original, reduced, abs_tol=1e-12)

